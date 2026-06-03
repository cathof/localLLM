#!/usr/bin/env python3
"""

python tests/diagnose_struktur.py \
    --case_id case_01 \
    --segment_index 27 \
    --span_text "900 mm" \
    --modified_doc case_documents/case_01_modified.docx \
    --embeddings embeddings_rules.npz \
    --index index_rules.jsonl \
    --prepared prepared_rules.jsonl \
    --embeddings2 embeddings_materials.npz \
    --index2 index_materials.jsonl \
    --prepared2 prepared_materials.jsonl

"""
import argparse, sys, torch
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from rag_answer_reference_facts_OHNEFIX import (
        load_dotenv, env_str, env_int, require_env,
        split_document_into_segments,
        load_rag_store, load_hf_model, embed_e5_query,
        retrieve_from_store,
        build_agent_context_from_sources,
        build_factual_review_messages,
        build_factual_json_schema,
        normalize_factual_errors,
        _dedup_factual_findings,
        load_taxonomy_json,
        OllamaClient, EvidenceSource, SegmentEvidence,
        parse_json_response, run_factual_agent,
    )
    from importDocuments_structural import normalize_text, read_docx
except ImportError as e:
    sys.exit(f"[FATAL] Import fehlgeschlagen: {e}")

import numpy as np
load_dotenv(".env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_id",        required=True)
    ap.add_argument("--segment_index",  type=int, required=True)
    ap.add_argument("--span_text",      required=True)
    ap.add_argument("--correction",     default="")
    ap.add_argument("--modified_doc",   required=True)
    ap.add_argument("--embeddings",     required=True)
    ap.add_argument("--index",          required=True)
    ap.add_argument("--prepared",       required=True)
    ap.add_argument("--embeddings2",    required=True)
    ap.add_argument("--index2",         required=True)
    ap.add_argument("--prepared2",      required=True)
    ap.add_argument("--top_k",          type=int, default=4)
    ap.add_argument("--context_chars",  type=int, default=4000)
    ap.add_argument("--max_query_len",  type=int, default=256)
    args = ap.parse_args()

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Diagnose STRUKT_BEFUND_BESCHREIBUNG")
    print(f"  Case: {args.case_id} | Seg: {args.segment_index} | Span: '{args.span_text}'")
    print(f"{sep}\n")

    # ── 1. Segmentierung ──────────────────────────────────────────────────────
    doc_path = Path(args.modified_doc).expanduser().resolve()
    doc_text = normalize_text(read_docx(doc_path))
    segments = split_document_into_segments(
        doc_text,
        target_chars=env_int("SEG_TARGET_CHARS", 1200),
        min_chars=env_int("SEG_MIN_CHARS", 250),
        max_chars=env_int("SEG_MAX_CHARS", 2200),
    )
    print(f"[INFO] {doc_path.name} | {len(segments)} Segmente")

    if args.segment_index >= len(segments):
        sys.exit(f"[FAIL] segment_index {args.segment_index} > max {len(segments)-1}")

    seg_text = segments[args.segment_index]

    # ── 2. Span prüfen ────────────────────────────────────────────────────────
    print(f"\n── Schritt 1: Span im Segment? ──")
    if args.span_text in seg_text:
        idx = seg_text.index(args.span_text)
        ctx = seg_text[max(0,idx-60):idx+len(args.span_text)+60].replace("\n"," ")
        print(f"[OK] '{args.span_text}' gefunden. Kontext: ...{ctx}...")
    else:
        print(f"[FAIL] '{args.span_text}' NICHT in Segment {args.segment_index}!")
        for i, s in enumerate(segments):
            if args.span_text in s:
                print(f"       → liegt in Segment {i}")
                break
        sys.exit(1)

    # ── 3. RAG-Stores ─────────────────────────────────────────────────────────
    print(f"\n── Schritt 2: RAG-Stores ──")
    rules_store    = load_rag_store("rules", "rules",
                                    Path(args.embeddings), Path(args.index), Path(args.prepared))
    material_store = load_rag_store("materials", "material",
                                    Path(args.embeddings2), Path(args.index2), Path(args.prepared2))

    # ── 4. Embedding + Retrieval ──────────────────────────────────────────────
    print(f"\n── Schritt 3: Retrieval ──")
    device = torch.device("cpu")
    embed_model, embed_tok = load_hf_model(
        env_str("EMBED_MODEL", "intfloat/multilingual-e5-large"), device)
    qvec = embed_e5_query(seg_text, model=embed_model, tokenizer=embed_tok,
                          device=device, max_length=args.max_query_len)

    rules_hits    = retrieve_from_store(rules_store,    qvec, k=args.top_k, retrieval_query=seg_text)
    material_hits = retrieve_from_store(material_store, qvec, k=args.top_k, retrieval_query=seg_text)
    print(f"[OK] Rules: {len(rules_hits)} | Materials: {len(material_hits)}")

    # ── 5. Korrekter Wert im Kontext? ─────────────────────────────────────────
    print(f"\n── Schritt 4: Referenzwert '{args.correction}' im Kontext? ──")
    corr_in_rules = corr_in_material = False
    for label, hits, flag_name in [("Rules", rules_hits, "corr_in_rules"),
                                   ("Materials", material_hits, "corr_in_material")]:
        print(f"\n  {label} Top-{args.top_k}:")
        for h in hits:
            found = args.correction and args.correction in h.text
            if label == "Rules" and found:    corr_in_rules    = True
            if label == "Materials" and found: corr_in_material = True
            marker = f" ◄ '{args.correction}' HIER" if found else ""
            print(f"  [{h.score:.3f}] {h.id}: {h.text[:100].replace(chr(10),' ')}...{marker}")

    if args.correction:
        print(f"\n  '{args.correction}' in Rules:     {'[OK]' if corr_in_rules else '[FEHLT]'}")
        print(f"  '{args.correction}' in Materials: {'[OK]' if corr_in_material else '[FEHLT]'}")
        if not corr_in_rules and not corr_in_material:
            print(f"\n  [DIAGNOSE] Referenzwert fehlt → Agent 2 kann Fehler strukturell nicht erkennen.")

    # ── 6. EvidenceSource-Objekte ─────────────────────────────────────────────
    def to_ev(hits, kind):
        return [EvidenceSource(
            source_ref=h.id, source_kind=kind, chunk_id=h.id,
            document=h.meta.get("document",""), source_path=h.meta.get("source_path",""),
            case_id=h.meta.get("case_id",""), document_type=h.meta.get("document_type",""),
            chunk_index=h.meta.get("chunk_index"), score=h.score, text=h.text,
        ) for h in hits]

    evidence = SegmentEvidence(
        segment_index=args.segment_index, segment_text=seg_text,
        retrieval_queries=[seg_text],
        rules_sources=to_ev(rules_hits, "rules"),
        material_sources=to_ev(material_hits, "material"),
    )

    # ── 7. Agent 2 ────────────────────────────────────────────────────────────
    print(f"\n── Schritt 5: Agent 2 (Fachprüfer) ──")
    catalog = load_taxonomy_json(Path(env_str("TAXONOMY_JSON", "tax/taxonomy.json")))
    llm = OllamaClient(
        base_url=require_env("OLLAMA_BASE_URL"),
        model=require_env("LLM_MODEL"),
        options={"temperature": float(env_str("LLM_TEMPERATURE", "0.1"))},
        timeout_s=env_int("LLM_TIMEOUT_S", 300),
    )
    print(f"  Modell: {require_env('LLM_MODEL')} – läuft...")
    findings = run_factual_agent(
        llm=llm, evidence=evidence,
        per_agent_context_chars=args.context_chars,
        catalog=catalog,
    )

    print(f"\n── Schritt 6: Findings ──")
    if not findings:
        print("  (keine Findings)")
    for f in findings:
        stelle = f.get("stelle_im_segment","")
        is_tp  = args.span_text in stelle or stelle in args.span_text
        print(f"  [{'◄ TREFFER' if is_tp else ' '}] [{f.get('subklasse','')}] '{stelle}'")

    # ── 8. Fazit ──────────────────────────────────────────────────────────────
    found = any(
        args.span_text in f.get("stelle_im_segment","")
        or f.get("stelle_im_segment","") in args.span_text
        for f in findings
    )
    print(f"\n── Fazit ──")
    if found:
        print(f"[OK] '{args.span_text}' von Agent 2 erkannt.")
    elif args.correction and not corr_in_rules and not corr_in_material:
        print(f"[DIAGNOSE] Fehler nicht erkannt — fehlende RAG-Referenz.")
        print(f"           '{args.correction}' fehlt im Zusatzmaterial → Agent 2 hat keine Vergleichsbasis.")
    else:
        print(f"[DIAGNOSE] Referenz vorhanden, aber Agent 2 hat Fehler übersehen.")
    print()


if __name__ == "__main__":
    main()
