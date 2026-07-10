#!/usr/bin/env python3
"""
test_rag_smoke.py
=================
Schneller Smoke-Test für rag_answer_reference_facts.py:
Läuft nur die ersten N Segmente eines Dokuments, gibt Findings aus
und zählt FP-relevante Subklassen.

Ziel: nach Code-Änderungen schnell prüfen ob False-Positive-Zahl
      für bekannte Problemklassen gesunken ist — ohne einen vollen Lauf.

Aufruf (aus dem Projektverzeichnis, .env muss vorhanden sein):
    python test_rag_smoke.py \\
        --document case_documents/case_01.docx \\
        --case_id  case_01 \\
        --segments 5

Optionale Flags:
    --segments N      Anzahl Segmente (default: 5)
    --start_seg S     Ab Segment S (1-basiert, default: 1) — nützlich
                      um problematische Stellen gezielt zu testen
    --agents 2,2b,3,4,7  Komma-getrennte Liste der Agenten (default: alle)
                        2b = gezielter Material-Faktencheck (Agent 2b)
    --no_agent0       Agent 0 überspringen (spart ~4s, Referenzfakten leer)
    --out predictions/smoke_test_latest.jsonl   Predictions speichern

Umgebungsvariablen aus .env werden automatisch geladen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_THIS_DIR = Path(__file__).resolve().parent
# Projektroot (Parent des tests/-Ordners) in den Suchpfad aufnehmen,
# damit rag_answer_reference_facts.py gefunden wird unabhaengig davon
# ob das Script aus tests/ oder vom Root aus aufgerufen wird.
_PROJECT_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Infrastruktur aus dem Hauptfile laden ─────────────────────────────────────
try:
    from rag_answer_reference_facts_oldcalcagent import (
        load_dotenv, env_str, env_int, env_bool, env_json_object_optional,
        require_env,
        load_taxonomy_json, ErrorCatalog,
        load_hf_model, choose_device, make_llm_client,
        load_rag_store, RagStore,
        load_reference_facts_schema,
        run_reference_facts_agent, format_reference_facts_for_prompt,
        build_segment_evidences,
        _build_reference_words_set, build_domain_whitelist_from_store,
        _extract_person_names_spacy,
        run_factual_agent, run_factual_2b_agent,
        run_language_agent, run_calculation_agent,
        run_statement_assurance_agent, run_hypothesis_agent,
        run_reference_consistency_agent,
        check_zweifel_violations,
        _SEEN_SPANS_BY_SUBCLASS,
        _strip_internal_fields,
        split_document_into_segments,
        save_predictions_jsonl,
    )
except ImportError as e:
    sys.exit(f"[FATAL] rag_answer_reference_facts.py nicht gefunden:\n  {e}")

try:
    from importDocuments_structural import normalize_text, read_docx
except ImportError as e:
    sys.exit(f"[FATAL] importDocuments_structural.py nicht gefunden:\n  {e}")

load_dotenv(".env")

# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

_WATCH_SUBCLASSES = {
    "STRUKT_BEFUND_BESCHREIBUNG",
    "FORMALES_REDAKTION",
    "RECHT_TRENNUNG",
    "STRUKT_EVIDENZ",
    "QMQS_DOKPFLICHT",
    "RECHT_SCHLUSS",
}

_SUBCLASS_LABEL_MAP = {
    "Beschreibung von Befunden":  "STRUKT_BEFUND_BESCHREIBUNG",
    "Redaktionelle Korrektur":    "FORMALES_REDAKTION",
    "Trennung Befund/Bewertung":  "RECHT_TRENNUNG",
    "Evidenz / Belege":           "STRUKT_EVIDENZ",
    "Dokumentationspflicht":      "QMQS_DOKPFLICHT",
    "Schlussfolgerung":           "RECHT_SCHLUSS",
}


def _normalise_subclass(raw: str) -> str:
    """Mapped interne Label-Strings auf Subclass-IDs."""
    return _SUBCLASS_LABEL_MAP.get(raw, raw)


def _print_findings_table(findings: List[Dict[str, Any]], agent_label: str) -> None:
    if not findings:
        print(f"  (keine Findings von {agent_label})")
        return
    print(f"\n  {'Seg':>4}  {'Subklasse':<40}  {'Span':<50}")
    print(f"  {'─'*4}  {'─'*40}  {'─'*50}")
    for f in findings:
        seg   = str(f.get("segment_index") or "?")
        sub   = _normalise_subclass(str(f.get("subklasse") or f.get("subclass_id") or "?"))
        span  = str(f.get("stelle_im_segment") or f.get("span_text") or "")[:50]
        rat   = str(f.get("begruendung") or "")[:80]
        print(f"  {seg:>4}  {sub:<40}  {span!r}")
        if rat:
            print(f"  {'':4}  {'':40}  ↳ {rat}")


def _print_summary(all_findings: List[Dict[str, Any]], n_segments: int, elapsed: float) -> None:
    W = 70
    print("\n" + "=" * W)
    print(f"SMOKE-TEST ZUSAMMENFASSUNG  ({n_segments} Segmente, {elapsed:.1f}s)")
    print("=" * W)
    print(f"  Findings gesamt: {len(all_findings)}")

    by_agent: Counter = Counter()
    by_sub:   Counter = Counter()
    watch_count = 0

    for f in all_findings:
        agent = str(f.get("agent", f.get("agent_scope", "?")))
        sub   = _normalise_subclass(str(f.get("subklasse") or f.get("subclass_id") or "?"))
        by_agent[agent] += 1
        by_sub[sub] += 1
        if sub in _WATCH_SUBCLASSES:
            watch_count += 1

    print(f"\n  Nach Agent:")
    for agent, cnt in by_agent.most_common():
        print(f"    {agent:<45} {cnt:>3}")

    print(f"\n  Nach Subklasse:")
    for sub, cnt in by_sub.most_common():
        marker = " ← WATCH" if sub in _WATCH_SUBCLASSES else ""
        print(f"    {sub:<45} {cnt:>3}{marker}")

    print(f"\n  Watch-Subklassen gesamt: {watch_count}  "
          f"({', '.join(sorted(_WATCH_SUBCLASSES))})")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Testlogik
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(args: argparse.Namespace) -> None:
    t_total = time.perf_counter()

    # ── Taxonomie ─────────────────────────────────────────────────────────────
    catalog = load_taxonomy_json(Path(args.taxonomy_json).resolve())
    print(f"[INFO] Taxonomie: {len(catalog.main_classes)} Hauptklassen")

    # ── RAG-Stores ────────────────────────────────────────────────────────────
    stores: List[RagStore] = [
        load_rag_store(
            "rules", "rules",
            npz_path=Path(args.embeddings).resolve(),
            index_path=Path(args.index).resolve(),
            prepared_path=Path(args.prepared).resolve(),
        )
    ]
    if args.embeddings2.strip():
        stores.append(load_rag_store(
            "material", "material",
            npz_path=Path(args.embeddings2).resolve(),
            index_path=Path(args.index2).resolve(),
            prepared_path=Path(args.prepared2).resolve(),
        ))
        print(f"[INFO] Zweiter RAG-Store geladen (Fallmaterial)")
    else:
        print("[INFO] Kein zweiter RAG-Store — nur Regelwerk")

    # ── Embedding + LLM ───────────────────────────────────────────────────────
    device = choose_device(args.embed_device)
    embed_model, embed_tok = load_hf_model(args.embed_model, device)
    llm = make_llm_client()
    print(f"[INFO] LLM + Embedding bereit | device={device}")

    # ── Dokument lesen ────────────────────────────────────────────────────────
    doc_path = Path(args.document).expanduser().resolve()
    if not doc_path.exists():
        sys.exit(f"[FATAL] Dokument nicht gefunden: {doc_path}")
    doc_text = normalize_text(read_docx(doc_path))
    if not doc_text.strip():
        sys.exit(f"[FATAL] Kein Text extrahiert aus {doc_path.name}")

    all_segments = split_document_into_segments(
        doc_text,
        target_chars=env_int("SEG_TARGET_CHARS", 1200),
        min_chars=env_int("SEG_MIN_CHARS", 250),
        max_chars=env_int("SEG_MAX_CHARS", 2200),
    )
    total_segs = len(all_segments)
    start_idx  = max(1, args.start_seg)
    end_idx    = min(start_idx + args.segments - 1, total_segs)
    n_segs     = end_idx - start_idx + 1

    print(f"[INFO] Dokument: {doc_path.name} | {len(doc_text)} Zeichen | "
          f"{total_segs} Segmente gesamt")
    print(f"[INFO] Teste Segmente {start_idx}–{end_idx} ({n_segs} von {total_segs})")
    print(f"[INFO] Aktive Agenten: {args.agents}")

    active_agents: Set[str] = set(args.agents.replace(" ", "").split(","))

    # ── Agent 0: Referenzfakten ───────────────────────────────────────────────
    reference_facts: Optional[Dict[str, Any]] = None
    if "0" in active_agents and not args.no_agent0:
        print("\n[RUN] Agent 0 — Referenzfakten-Extraktor ...")
        t0 = time.perf_counter()
        reference_schema = load_reference_facts_schema(
            Path(args.reference_facts_schema).expanduser().resolve()
        )
        reference_facts = run_reference_facts_agent(
            llm, doc_text,
            case_id=args.case_id,
            schema=reference_schema,
            max_chars=args.reference_facts_context_chars,
        )
        print(f"[OK]  Agent 0 — {time.perf_counter() - t0:.1f}s")
    else:
        print("[SKIP] Agent 0 — übersprungen (--no_agent0 oder nicht in --agents)")

    # ── Whitelist aufbauen ────────────────────────────────────────────────────
    _SEEN_SPANS_BY_SUBCLASS.clear()
    _ref_words: set = _build_reference_words_set(reference_facts)
    _material_stores = [s for s in stores if s.source_kind == "material"]
    if _material_stores and args.case_id:
        for ms in _material_stores:
            wl = build_domain_whitelist_from_store(
                ms, args.case_id, min_word_length=4, min_occurrences=2)
            _ref_words |= wl
    _ner = _extract_person_names_spacy(doc_text)
    _ref_words |= _ner
    print(f"[INFO] Whitelist: {len(_ref_words)} Wörter")

    # ── Evidenzen nur für gewünschte Segmente ─────────────────────────────────
    # build_segment_evidences läuft intern auf allen Segmenten.
    # Wir holen alle Evidenzen und slicen danach — das ist am sichersten
    # weil die Segment-Indizes aus dem Retrieval stammen und nicht neu
    # berechnet werden müssen.
    print("\n[RUN] Retrieval für alle Segmente (Slicing danach) ...")
    t0 = time.perf_counter()
    all_evidences, _ = build_segment_evidences(
        doc_text, stores,
        embed_model=embed_model,
        embed_tok=embed_tok,
        device=device,
        args=args,
        vision_cfg=None,
    )
    # slice: segment_index ist 1-basiert
    evidences = [
        ev for ev in all_evidences
        if start_idx <= ev.segment_index <= end_idx
    ]
    print(f"[OK]  Retrieval — {time.perf_counter() - t0:.1f}s | "
          f"{len(evidences)} Segmente für Test")

    # ── Segment-Schleife ──────────────────────────────────────────────────────
    per_agent_context_chars = args.context_max_chars // 3

    factual_findings:           List[Dict[str, Any]] = []
    language_findings:          List[Dict[str, Any]] = []
    calculation_findings:       List[Dict[str, Any]] = []
    statement_assurance_findings: List[Dict[str, Any]] = []

    agent_times: Dict[str, float] = {}

    for ev in evidences:
        seg_label = f"S{ev.segment_index}"
        print(f"\n  ── Segment {ev.segment_index} ──────────────────────────────")
        print(f"     {ev.segment_text[:120].replace(chr(10),' ')!r}{'...' if len(ev.segment_text)>120 else ''}")

        if "2" in active_agents:
            t0 = time.perf_counter()
            try:
                new = run_factual_agent(
                    llm, ev,
                    per_agent_context_chars=per_agent_context_chars,
                    catalog=catalog,
                )
                factual_findings.extend(new)
                dt = time.perf_counter() - t0
                agent_times["factual"] = agent_times.get("factual", 0.0) + dt
                print(f"     A2 Factual:     {len(new):>2} Findings  ({dt:.1f}s)")
            except Exception as e:
                print(f"     A2 WARN: {e}")

        if "2b" in active_agents:
            t0 = time.perf_counter()
            try:
                new = run_factual_2b_agent(
                    llm, ev,
                    per_agent_context_chars=per_agent_context_chars,
                    catalog=catalog,
                )
                factual_findings.extend(new)
                dt = time.perf_counter() - t0
                agent_times["factual_2b"] = agent_times.get("factual_2b", 0.0) + dt
                print(f"     A2b Material:   {len(new):>2} Findings  ({dt:.1f}s)")
            except Exception as e:
                print(f"     A2b WARN: {e}")

        if "3" in active_agents:
            t0 = time.perf_counter()
            try:
                new = run_language_agent(
                    llm, ev, catalog=catalog, reference_words=_ref_words)
                language_findings.extend(new)
                dt = time.perf_counter() - t0
                agent_times["language"] = agent_times.get("language", 0.0) + dt
                print(f"     A3 Language:    {len(new):>2} Findings  ({dt:.1f}s)")
            except Exception as e:
                print(f"     A3 WARN: {e}")

        if "4" in active_agents:
            t0 = time.perf_counter()
            try:
                new = run_calculation_agent(llm, ev, catalog)
                calculation_findings.extend(new)
                dt = time.perf_counter() - t0
                agent_times["calculation"] = agent_times.get("calculation", 0.0) + dt
                print(f"     A4 Calculation: {len(new):>2} Findings  ({dt:.1f}s)")
            except Exception as e:
                print(f"     A4 WARN: {e}")

        if "7" in active_agents:
            t0 = time.perf_counter()
            try:
                new = run_statement_assurance_agent(llm, ev, catalog=catalog)
                statement_assurance_findings.extend(new)
                dt = time.perf_counter() - t0
                agent_times["statement_assurance"] = agent_times.get("statement_assurance", 0.0) + dt
                print(f"     A7 Statement:   {len(new):>2} Findings  ({dt:.1f}s)")
            except Exception as e:
                print(f"     A7 WARN: {e}")

        factual_findings.extend(
            check_zweifel_violations(ev.segment_text, ev.segment_index))

    # ── Dokumentweite Agenten (optional) ──────────────────────────────────────
    hypothesis_findings: List[Dict[str, Any]] = []
    reference_consistency_findings: List[Dict[str, Any]] = []
    segments_text = [ev.segment_text for ev in all_evidences]

    if "5" in active_agents:
        print(f"\n[RUN] Agent 5 — Hypothesenprüfer (Volltext) ...")
        t0 = time.perf_counter()
        try:
            hypothesis_findings = run_hypothesis_agent(
                llm, doc_text, segments_text, catalog)
            dt = time.perf_counter() - t0
            agent_times["hypothesis"] = dt
            print(f"[OK]  Agent 5 — {len(hypothesis_findings)} Findings  ({dt:.1f}s)")
        except Exception as e:
            print(f"[WARN] Agent 5: {e}")

    if "6" in active_agents and reference_facts:
        print(f"\n[RUN] Agent 6 — Referenzkonsistenz (Volltext) ...")
        t0 = time.perf_counter()
        try:
            reference_consistency_findings = run_reference_consistency_agent(
                llm, doc_text, segments_text, reference_facts, catalog)
            dt = time.perf_counter() - t0
            agent_times["reference_consistency"] = dt
            print(f"[OK]  Agent 6 — {len(reference_consistency_findings)} Findings  ({dt:.1f}s)")
        except Exception as e:
            print(f"[WARN] Agent 6: {e}")

    # ── Alle Findings zusammenführen ──────────────────────────────────────────
    language_findings = _strip_internal_fields(language_findings)

    all_findings = (
            factual_findings
            + language_findings
            + calculation_findings
            + statement_assurance_findings
            + hypothesis_findings
            + reference_consistency_findings
    )

    # ── Detail-Ausgabe pro Agent ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETAIL — FINDINGS PRO AGENT")
    print("=" * 70)

    if "2" in active_agents or "2b" in active_agents:
        label = "Agent 2+2b" if "2b" in active_agents and "2" in active_agents else ("Agent 2b" if "2b" in active_agents else "Agent 2")
        print(f"\n▶ {label} — Factual ({len(factual_findings)} Findings):")
        _print_findings_table(factual_findings, label)

    if "3" in active_agents:
        print(f"\n▶ Agent 3 — Language ({len(language_findings)} Findings):")
        _print_findings_table(language_findings, "Agent 3")

    if "4" in active_agents:
        print(f"\n▶ Agent 4 — Calculation ({len(calculation_findings)} Findings):")
        _print_findings_table(calculation_findings, "Agent 4")

    if "7" in active_agents:
        print(f"\n▶ Agent 7 — Statement Assurance ({len(statement_assurance_findings)} Findings):")
        _print_findings_table(statement_assurance_findings, "Agent 7")

    if "5" in active_agents:
        print(f"\n▶ Agent 5 — Hypothesis ({len(hypothesis_findings)} Findings):")
        _print_findings_table(hypothesis_findings, "Agent 5")

    if "6" in active_agents:
        print(f"\n▶ Agent 6 — Reference Consistency ({len(reference_consistency_findings)} Findings):")
        _print_findings_table(reference_consistency_findings, "Agent 6")

    # ── Laufzeiten ────────────────────────────────────────────────────────────
    if agent_times:
        print("\n" + "─" * 50)
        print("Laufzeiten:")
        for agent, t in agent_times.items():
            print(f"  {agent:<30} {t:>6.1f}s")

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    _print_summary(all_findings, n_segs, elapsed)

    # ── Optional: Predictions speichern ──────────────────────────────────────
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        by_seg: Dict[int, List[Dict]] = {}
        for f in all_findings:
            idx = f.get("segment_index") or 0
            by_seg.setdefault(idx, []).append(f)
        for seg_idx, seg_findings in sorted(by_seg.items()):
            record = {
                "case_id":       args.case_id or "smoke_test",
                "segment_id":    f"SPAN_ONLY",
                "segment_index": seg_idx,
                "predicted_findings": [
                    {
                        "finding_id":    f.get("finding_id", f"SMOKE-{i:04d}"),
                        "subclass_id":   _normalise_subclass(
                            str(f.get("subklasse") or f.get("subclass_id") or "")),
                        "change_type_id": str(f.get("aenderungstyp") or ""),
                        "severity_id":   str(f.get("schweregrad") or ""),
                        "span_text":     str(f.get("stelle_im_segment") or ""),
                        "correction":    str(f.get("vorschlag") or f.get("korrekter_wert") or ""),
                        "rationale":     str(f.get("begruendung") or ""),
                    }
                    for i, f in enumerate(seg_findings)
                ],
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[OK] Predictions gespeichert: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Smoke-Test: RAG-Pipeline auf N Segmenten laufen lassen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--document",  required=True, help="Pfad zum .docx Dokument")
    ap.add_argument("--case_id",   default=env_str("CASE_ID", ""), help="Case-ID")
    ap.add_argument("--segments",  type=int, default=5,
                    help="Anzahl Segmente (default: 5)")
    ap.add_argument("--start_seg", type=int, default=1,
                    help="Erstes Segment 1-basiert (default: 1)")
    ap.add_argument("--agents",    default="0,2,2b,3,4,7",
                    help="Komma-getrennte Agenten (default: 0,2,3,4,7). "
                         "5 und 6 laufen auf dem Volltext.")
    ap.add_argument("--no_agent0", action="store_true",
                    help="Agent 0 überspringen (Referenzfakten bleiben leer)")
    ap.add_argument("--out", default="",
                    help="Pfad zum Speichern der Predictions-JSONL (optional)")

    # RAG-Store 1 (Regelwerk)
    ap.add_argument("--embeddings", default=env_str("EMBED_OUT_NPZ",   "embeddings.npz"))
    ap.add_argument("--index",      default=env_str("EMBED_OUT_INDEX",  "index.jsonl"))
    ap.add_argument("--prepared",   default=env_str("OUT_JSONL",        "prepared.jsonl"))

    # RAG-Store 2 (Fallmaterial, optional)
    ap.add_argument("--embeddings2", default=env_str("EMBED_OUT_NPZ2",  ""))
    ap.add_argument("--index2",      default=env_str("EMBED_OUT_INDEX2",""))
    ap.add_argument("--prepared2",   default=env_str("OUT_JSONL2",      ""))

    # Embedding
    ap.add_argument("--embed_model",  default=env_str("EMBED_MODEL", "intfloat/multilingual-e5-large"))
    ap.add_argument("--embed_device", default=env_str("EMBED_DEVICE", "auto"))
    ap.add_argument("--query_max_length", type=int, default=env_int("QUERY_MAX_LENGTH", 256))

    # Retrieval
    ap.add_argument("--top_k",                    type=int,   default=env_int("TOP_K", 12))
    ap.add_argument("--rules_top_k",              type=int,   default=env_int("RULES_TOP_K", 8))
    ap.add_argument("--material_top_k",           type=int,   default=env_int("MATERIAL_TOP_K", 8))
    ap.add_argument("--context_max_chars",        type=int,   default=env_int("CONTEXT_MAX_CHARS", 12000))
    ap.add_argument("--multi_query_count",        type=int,   default=env_int("MULTI_QUERY_COUNT", 4))
    ap.add_argument("--mmr_lambda",               type=float, default=float(env_str("MMR_LAMBDA", "0.75")))
    ap.add_argument("--max_per_source",           type=int,   default=env_int("MAX_PER_SOURCE", 2))
    ap.add_argument("--per_segment_candidate_k",  type=int,   default=env_int("PER_SEGMENT_CANDIDATE_K", 24))
    ap.add_argument("--per_segment_rules_top_k",  type=int,   default=env_int("PER_SEGMENT_RULES_TOP_K", 8))
    ap.add_argument("--per_segment_material_top_k",type=int,  default=env_int("PER_SEGMENT_MATERIAL_TOP_K", 8))

    # Reference Facts (Agent 0)
    ap.add_argument("--reference_facts_schema",
                    default=env_str("REFERENCE_FACTS_SCHEMA", "schema/reference_facts_schema.json"))
    ap.add_argument("--reference_facts_context_chars",
                    type=int, default=env_int("REFERENCE_FACTS_CONTEXT_CHARS", 4000))

    # Unused but expected by build_segment_evidences via args.*
    ap.add_argument("--vision_model",    default=env_str("VISION_MODEL", ""))
    ap.add_argument("--vision_workers",  type=int, default=env_int("VISION_WORKERS", 3))
    ap.add_argument("--vision_timeout_s",type=int, default=env_int("VISION_TIMEOUT_S", 180))
    ap.add_argument("--print_sources",   action="store_true")
    ap.add_argument("--print_context",   action="store_true")
    ap.add_argument("--save_predictions_jsonl", default="")
    ap.add_argument("--ground_truth",    default="")
    ap.add_argument("--taxonomy_json",
                    default=env_str("TAXONOMY_JSON", "taxonomy.json"))
    ap.add_argument("--save_reference_facts_json", default="")
    ap.add_argument("--print_reference_facts", action="store_true")

    args = ap.parse_args()

    # case_id aus Dokument-Stem als Fallback
    if not args.case_id:
        args.case_id = Path(args.document).stem

    return args


if __name__ == "__main__":
    run_smoke_test(parse_args())