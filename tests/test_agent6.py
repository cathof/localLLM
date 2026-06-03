#!/usr/bin/env python3
"""
tests/test_agent6.py
=====================
Testet Agent 6 (Referenzfakten-Konsistenzprüfer) direkt.
Zeigt ob ein bestimmter Span-Fehler (z.B. falsche Personenangabe)
vom System erkannt wird.

Aufruf mit gespeicherten Referenzfakten (schnell, kein LLM-Call für Agent 0):
    python tests/test_agent6.py \
        --case_id case_01 \
        --modified_doc case_documents/case_01_modified.docx \
        --span_text "Rolf Bosshard" \
        --reference_facts reference_facts/reference_facts_case_01_synthetic.json

Aufruf ohne Referenzfakten (Agent 0 wird aufgerufen):
    python tests/test_agent6.py \
        --case_id case_01 \
        --modified_doc case_documents/case_01_modified.docx \
        --span_text "Rolf Bosshard"
"""

import json
import sys
import argparse
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
    ap = argparse.ArgumentParser(description="Testet Agent 6 direkt auf einem Dokument.")
    ap.add_argument("--case_id",         required=True,  help="z.B. case_01")
    ap.add_argument("--modified_doc",    required=True,  help="case_XX_modified.docx")
    ap.add_argument("--span_text",       required=True,  help="Der gesuchte Fehler-Span")
    ap.add_argument("--reference_facts", default="",
                    help="Pfad zu gespeichertem reference_facts JSON (optional, sonst Agent 0)")
    ap.add_argument("--max_chars",       type=int, default=4000,
                    help="Zeichen fuer Agent-0-Kontext (default 4000)")
    args = ap.parse_args()

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Test Agent 6 — Referenzfakten-Konsistenzprüfer")
    print(f"  Case:  {args.case_id}")
    print(f"  Span:  '{args.span_text}'")
    print(f"{sep}\n")

    # Dokument laden
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
    print(f"[INFO] {doc_path.name} | {len(doc_text)} Zeichen | {len(segments)} Segmente")

    if args.span_text in doc_text:
        idx = doc_text.index(args.span_text)
        ctx = doc_text[max(0,idx-60):idx+len(args.span_text)+60].replace("\n"," ")
        print(f"[OK]   '{args.span_text}' im Dokument: ...{ctx}...")
    else:
        print(f"[WARN] '{args.span_text}' NICHT im Dokument — Injektion prüfen.")

    # LLM-Client
    llm = OllamaClient(
        base_url=require_env("OLLAMA_BASE_URL"),
        model=require_env("LLM_MODEL"),
        options={"temperature": float(env_str("LLM_TEMPERATURE", "0.1"))},
        timeout_s=env_int("LLM_TIMEOUT_S", 300),
    )

    # Referenzfakten laden oder generieren
    print(f"\n── Schritt 1: Referenzfakten ──")
    if args.reference_facts:
        rf_path = Path(args.reference_facts).expanduser().resolve()
        if not rf_path.exists():
            sys.exit(f"[FATAL] Referenzfakten nicht gefunden: {rf_path}")
        with rf_path.open(encoding="utf-8") as f:
            reference_facts = json.load(f)
        print(f"[OK] Geladen aus: {rf_path.name}")
    else:
        schema_path = Path(env_str("REFERENCE_FACTS_SCHEMA",
                                   "schema/reference_facts_schema.json")).expanduser()
        if not schema_path.exists():
            sys.exit(f"[FATAL] Schema nicht gefunden: {schema_path}")
        print(f"[INFO] Agent 0 läuft...")
        schema = load_reference_facts_schema(schema_path)
        reference_facts = run_reference_facts_agent(
            llm=llm, doc_text=doc_text, case_id=args.case_id,
            schema=schema, max_chars=args.max_chars,
        )

    # Referenzfakten ausgeben
    facts = reference_facts.get("facts", {})
    print(f"\n  Referenzfakten (high/medium):")
    for key, val in facts.items():
        if isinstance(val, dict):
            if val.get("confidence") in {"high", "medium"} and val.get("value"):
                print(f"    [{val['confidence']}] {key}: {val['value']}")
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and item.get("confidence") in {"high", "medium"}:
                    name = item.get("name") or item.get("value") or ""
                    rolle = item.get("rolle", "")
                    print(f"    [{item['confidence']}] {key}: {name}" +
                          (f" ({rolle})" if rolle else ""))

    # Agent 6 aufrufen
    print(f"\n── Schritt 2: Agent 6 läuft (Volltext, {len(doc_text)} Zeichen) ──")
    catalog = load_taxonomy_json(Path(env_str("TAXONOMY_JSON", "tax/taxonomy.json")))

    findings = run_reference_consistency_agent(
        llm=llm, doc_text=doc_text, segments=segments,
        reference_facts=reference_facts, catalog=catalog,
    )

    # Findings anzeigen
    print(f"\n── Schritt 3: Findings von Agent 6 ({len(findings)}) ──")
    if not findings:
        print("  (keine Findings)")
    for f in findings:
        stelle  = f.get("stelle_im_segment", "")
        seg_idx = f.get("segment_index", "?")
        subkl   = f.get("subklasse", "")
        ref_key = f.get("reference_key", "")
        ref_val = f.get("reference_value", "")
        beg     = f.get("begruendung", "")
        is_target = args.span_text in stelle or stelle in args.span_text
        marker = "  ◄ ZIEL" if is_target else ""
        print(f"  [Seg {seg_idx}] [{subkl}] '{stelle}'{marker}")
        if ref_key:
            print(f"           Referenz: {ref_key} = '{ref_val}'")
        if beg:
            print(f"           Begründung: {beg[:120]}")

    # Fazit
    print(f"\n── Fazit ──")
    found = any(
        args.span_text in f.get("stelle_im_segment", "")
        or f.get("stelle_im_segment", "") in args.span_text
        for f in findings
    )
    if found:
        print(f"[OK]   '{args.span_text}' von Agent 6 erkannt.")
        print(f"       → Finding landet in predictions.")
    else:
        print(f"[FAIL] '{args.span_text}' von Agent 6 nicht erkannt.")
        span_words = set(args.span_text.lower().split())
        fact_hit = False
        for key, val in facts.items():
            if isinstance(val, list):
                for item in val:
                    name = str(item.get("name") or "").lower()
                    if any(w in name for w in span_words):
                        print(f"       → Referenzfakt '{key}': '{item.get('name')}' vorhanden.")
                        fact_hit = True
        if not fact_hit:
            print(f"       → Kein ähnlicher Referenzfakt — Agent 6 hatte keine Basis.")
        else:
            print(f"       → Referenzfakt vorhanden, aber LLM hat Widerspruch übersehen.")
            print(f"         'Rolf' vs 'Roland' möglicherweise zu ähnlich für das Modell.")
    print()


if __name__ == "__main__":
    main()