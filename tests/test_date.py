#!/usr/bin/env python3
"""
tests/test_agent6.py
=====================
Testet Agent 6 (Referenzfakten-Konsistenzprüfer) direkt auf einem Dokument.
Zeigt ob Agent 6 das injizierte Datum '2. Mai 2030' erkennt und als
Widerspruch gegen den Referenzfakt '2. Mai 2024' meldet.

Aufruf:
    python tests/test_date.py \
        --case_id case_01 \
        --modified_doc case_documents/case_01_modified.docx \
        --span_text "2. Mai 2030"

Optional: bereits gespeicherte Referenzfakten verwenden statt neu zu generieren:
    python tests/test_date.py \
        --case_id case_01 \
        --modified_doc case_documents/case_01_modified.docx \
        --span_text "2. Mai 2030" \
        --reference_facts reference_facts/reference_facts_case_01_synthetic.json
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from rag_answer_reference_facts_OHNEFIX import (
        load_dotenv, env_str, env_int, require_env,
        split_document_into_segments,
        load_reference_facts_schema,
        run_reference_facts_agent,
        run_reference_consistency_agent,
        load_taxonomy_json,
        OllamaClient,
    )
    from importDocuments_structural import normalize_text, read_docx
except ImportError as e:
    sys.exit(f"[FATAL] Import fehlgeschlagen: {e}")

load_dotenv(".env")


def main():
    ap = argparse.ArgumentParser(description="Test Agent 6: Referenzfakten-Konsistenzprüfer")
    ap.add_argument("--case_id",         required=True)
    ap.add_argument("--modified_doc",    required=True,
                    help="Modifiziertes .docx (mit injizierten Fehlern)")
    ap.add_argument("--span_text",       default="",
                    help="Erwarteter Fehler-Span den Agent 6 finden soll")
    ap.add_argument("--reference_facts", default="",
                    help="Pfad zu bereits gespeicherten Referenzfakten JSON (optional)")
    ap.add_argument("--max_chars",       type=int, default=4000,
                    help="Zeichen für Referenzfakten-Extraktion (default 4000)")
    args = ap.parse_args()

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Test Agent 6: Referenzfakten-Konsistenzprüfer")
    print(f"  Case: {args.case_id} | Dokument: {Path(args.modified_doc).name}")
    if args.span_text:
        print(f"  Erwarteter Span: '{args.span_text}'")
    print(f"{sep}\n")

    # ── 1. Dokument laden und segmentieren ────────────────────────────────────
    doc_path = Path(args.modified_doc).expanduser().resolve()
    if not doc_path.exists():
        sys.exit(f"[FATAL] Dokument nicht gefunden: {doc_path}")

    doc_text = normalize_text(read_docx(doc_path))
    segments = split_document_into_segments(
        doc_text,
        target_chars=env_int("SEG_TARGET_CHARS", 1200),
        min_chars=env_int("SEG_MIN_CHARS", 250),
        max_chars=env_int("SEG_MAX_CHARS", 2200),
    )
    print(f"[INFO] Dokument: {len(doc_text)} Zeichen | {len(segments)} Segmente")

    # Span im Dokument prüfen
    if args.span_text:
        if args.span_text in doc_text:
            idx = doc_text.index(args.span_text)
            ctx = doc_text[max(0,idx-60):idx+len(args.span_text)+60].replace("\n"," ")
            print(f"[OK] '{args.span_text}' im Dokument: ...{ctx}...")
        else:
            print(f"[WARN] '{args.span_text}' NICHT im Dokument gefunden.")

    # ── 2. LLM-Client ─────────────────────────────────────────────────────────
    llm = OllamaClient(
        base_url=require_env("OLLAMA_BASE_URL"),
        model=require_env("LLM_MODEL"),
        options={"temperature": float(env_str("LLM_TEMPERATURE", "0.1"))},
        timeout_s=env_int("LLM_TIMEOUT_S", 300),
    )

    # ── 3. Referenzfakten laden oder generieren ────────────────────────────────
    print(f"\n── Schritt 1: Referenzfakten (Agent 0) ──")

    if args.reference_facts:
        rf_path = Path(args.reference_facts).expanduser().resolve()
        if rf_path.exists():
            with rf_path.open(encoding="utf-8") as f:
                reference_facts = json.load(f)
            print(f"[OK] Referenzfakten geladen aus: {rf_path.name}")
        else:
            sys.exit(f"[FATAL] Referenzfakten nicht gefunden: {rf_path}")
    else:
        schema_path = Path(env_str("REFERENCE_FACTS_SCHEMA",
                                   "schema/reference_facts_schema.json")).expanduser()
        if not schema_path.exists():
            sys.exit(f"[FATAL] REFERENCE_FACTS_SCHEMA nicht gefunden: {schema_path}")

        print(f"[INFO] Agent 0 läuft (Referenzfakten extrahieren)...")
        schema = load_reference_facts_schema(schema_path)
        reference_facts = run_reference_facts_agent(
            llm=llm,
            doc_text=doc_text,
            case_id=args.case_id,
            schema=schema,
            max_chars=args.max_chars,
        )

    # Referenzfakten zeigen
    facts = reference_facts.get("facts", {})
    print(f"\n  Extrahierte Referenzfakten:")
    for key, values in facts.items():
        if isinstance(values, list):
            for v in values:
                conf = v.get("confidence","?")
                if conf in {"high","medium"}:
                    name = v.get("name","") or v.get("wert","") or str(v)
                    rolle = v.get("rolle","")
                    print(f"    [{conf}] {key}: {name}" + (f" ({rolle})" if rolle else ""))
        elif isinstance(values, dict):
            conf = values.get("confidence","?")
            if conf in {"high","medium"}:
                print(f"    [{conf}] {key}: {values.get('name','') or values.get('wert','')}")

    # ── 4. Agent 6 aufrufen ───────────────────────────────────────────────────
    print(f"\n── Schritt 2: Agent 6 (Konsistenzprüfer) ──")
    catalog = load_taxonomy_json(Path(env_str("TAXONOMY_JSON", "tax/taxonomy.json")))

    print(f"[INFO] Agent 6 läuft auf Volltext ({len(doc_text)} Zeichen)...")
    findings = run_reference_consistency_agent(
        llm=llm,
        doc_text=doc_text,
        segments=segments,
        reference_facts=reference_facts,
        catalog=catalog,
    )

    # ── 5. Findings analysieren ───────────────────────────────────────────────
    print(f"\n── Schritt 3: Findings von Agent 6 ({len(findings)}) ──")
    if not findings:
        print("  (keine Findings)")
    else:
        for f in findings:
            stelle   = f.get("stelle_im_segment", "")
            seg_idx  = f.get("segment_index", "?")
            subkl    = f.get("subklasse", "")
            rationale = f.get("beschreibung", f.get("rationale", ""))
            is_target = args.span_text and (
                    args.span_text in stelle or stelle in args.span_text
            )
            marker = " ◄ ZIEL-SPAN" if is_target else ""
            print(f"  [Seg {seg_idx}] [{subkl}] '{stelle}'{marker}")
            if rationale:
                print(f"           Rationale: {rationale[:100]}")

    # ── 6. Fazit ──────────────────────────────────────────────────────────────
    print(f"\n── Fazit ──")
    if not args.span_text:
        print(f"  {len(findings)} Finding(s) von Agent 6 gesamt.")
    else:
        found = any(
            args.span_text in f.get("stelle_im_segment","")
            or f.get("stelle_im_segment","") in args.span_text
            for f in findings
        )
        if found:
            print(f"[OK] '{args.span_text}' von Agent 6 als Widerspruch erkannt.")
            print(f"     → Finding wird in predictions geschrieben (Segment-Index beachten).")
        else:
            print(f"[FAIL] '{args.span_text}' nicht von Agent 6 erkannt.")
            if not reference_facts.get("facts"):
                print(f"       → Keine Referenzfakten extrahiert: Agent 6 hatte keine Basis.")
            else:
                print(f"       → Datum in Referenzfakten nicht als high/medium eingestuft,")
                print(f"          oder Agent 6 hat den Widerspruch nicht erkannt.")
    print()


if __name__ == "__main__":
    main()