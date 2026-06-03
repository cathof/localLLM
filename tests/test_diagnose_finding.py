#!/usr/bin/env python3
"""
tests/test_diagnose_finding.py
===============================
Diagnostiziert warum ein bestimmter GT-Span nicht in den Predictions gefunden wurde.
Führt dieselben Schritte aus wie das Produktionssystem:
  1. Ist der Span im modifizierten Dokument vorhanden?
  2. In welchem Segment liegt er?
  3. Was haben Agent 2 und Agent 6 in diesem Segment gefunden?
  4. Warum war kein Match?

Aufruf:
    python tests/test_diagnose_finding.py \
        --span_text "Rolf Bosshard" \
        --subclass_id STRUKT_BEFUND_BESCHREIBUNG \
        --predictions predictions/predictions_case_01_synthetic.jsonl \
        --modified_doc case_documents/case_01_modified.docx \
        --min_span_score 0.20
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
        load_dotenv, env_int, env_str,
        split_document_into_segments,
    )
    from importDocuments_structural import normalize_text, read_docx
except ImportError as e:
    sys.exit(f"[FATAL] Import fehlgeschlagen: {e}")

load_dotenv(".env")


def span_match_score(a: str, b: str) -> float:
    """Token-Overlap (Jaccard) — identisch mit evaluate_predictions.py."""
    def tok(s):
        import re
        return set(re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", s.lower()))
    ta, tb = tok(a), tok(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span_text",      required=True,  help="GT span_text der nicht erkannt wurde")
    ap.add_argument("--subclass_id",    default="",     help="GT subclass_id (optional, für Subclass-Match-Check)")
    ap.add_argument("--predictions",    required=True,  help="predictions_case_XX_synthetic.jsonl")
    ap.add_argument("--modified_doc",   required=True,  help="case_XX_modified.docx")
    ap.add_argument("--min_span_score", type=float, default=0.20)
    args = ap.parse_args()

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Diagnose False Negative")
    print(f"  Span:     '{args.span_text}'")
    print(f"  Subclass: {args.subclass_id or '(nicht angegeben)'}")
    print(f"{sep}\n")

    # ── 1. Span im modifizierten Dokument? ────────────────────────────────────
    print("── Schritt 1: Span im Dokument? ──")
    doc_path = Path(args.modified_doc).expanduser().resolve()
    doc_text = normalize_text(read_docx(doc_path))

    if args.span_text in doc_text:
        idx = doc_text.index(args.span_text)
        ctx = doc_text[max(0,idx-60):idx+len(args.span_text)+60].replace("\n"," ")
        print(f"[OK] '{args.span_text}' im Dokument vorhanden.")
        print(f"     Kontext: ...{ctx}...")
    else:
        print(f"[FAIL] '{args.span_text}' NICHT im Dokument!")
        print(f"       → Die Injektion hat nicht funktioniert.")
        # Teilstring suchen
        words = args.span_text.split()
        for w in words:
            if w in doc_text:
                i = doc_text.index(w)
                ctx = doc_text[max(0,i-40):i+len(w)+40].replace("\n"," ")
                print(f"       Teilwort '{w}' gefunden: ...{ctx}...")
        return

    # ── 2. In welchem Segment? ─────────────────────────────────────────────────
    print(f"\n── Schritt 2: Segmentzuordnung ──")
    segments = split_document_into_segments(
        doc_text,
        target_chars=env_int("SEG_TARGET_CHARS", 1200),
        min_chars=env_int("SEG_MIN_CHARS", 250),
        max_chars=env_int("SEG_MAX_CHARS", 2200),
    )
    found_in_seg = None
    for i, seg in enumerate(segments):
        if args.span_text in seg:
            found_in_seg = i
            print(f"[OK] Span in Segment {i} (0-basiert).")
            print(f"     Segment-Vorschau: ...{seg[:120].replace(chr(10),' ')}...")
            break
    if found_in_seg is None:
        print(f"[WARN] Span im Volltext aber in keinem Segment — Segmentierungsparameter prüfen.")

    # ── 3. Was steht in den Predictions für dieses Segment? ───────────────────
    print(f"\n── Schritt 3: Predictions für Segment {found_in_seg} ──")
    pred_path = Path(args.predictions).expanduser().resolve()
    preds = [json.loads(l) for l in pred_path.open(encoding="utf-8") if l.strip()]

    # Produktionssystem nummeriert 1-basiert (enumerate start=1)
    seg_idx_prod = found_in_seg  # 0-basiert aus Injector
    # Prüfe beide Varianten
    for offset in [0, 1]:
        target = seg_idx_prod + offset
        seg_preds = next((p for p in preds if p["segment_index"] == target), None)
        if seg_preds:
            print(f"  Segment {target} in Predictions ({len(seg_preds.get('predicted_findings',[]))} Findings):")
            for f in seg_preds.get("predicted_findings", []):
                score = span_match_score(args.span_text, f.get("span_text",""))
                mark = " ◄" if score >= args.min_span_score else ""
                print(f"    [{f.get('subclass_id','')}] '{f.get('span_text','')[:60]}' span_score={score:.3f}{mark}")
        else:
            print(f"  Segment {target}: kein Eintrag in Predictions.")

    # ── 4. Dokumentweite Suche nach ähnlichen Predictions ─────────────────────
    print(f"\n── Schritt 4: Dokumentweite Suche (span_score ≥ {args.min_span_score}) ──")
    all_preds = [(p["segment_index"], f) for p in preds for f in p.get("predicted_findings", [])]
    hits = sorted(
        [(span_match_score(args.span_text, f.get("span_text","")), si, f)
         for si, f in all_preds],
        key=lambda x: x[0],
        reverse=True
    )
    matches = [(s, si, f) for s, si, f in hits if s >= args.min_span_score]

    if matches:
        print(f"[OK] {len(matches)} Prediction(s) mit span_score ≥ {args.min_span_score}:")
        for score, si, f in matches[:5]:
            sub_match = "✓" if f.get("subclass_id") == args.subclass_id else "✗"
            print(f"  Seg {si:3d}: '{f.get('span_text','')[:60]}'  "
                  f"subclass {sub_match}  span_score={score:.3f}")
    else:
        print(f"[FAIL] Kein dokumentweiter Match. Beste Treffer:")
        for score, si, f in hits[:5]:
            print(f"  Seg {si:3d}: '{f.get('span_text','')[:60]}'  score={score:.3f}")

    # ── 5. Ursachen-Diagnose ───────────────────────────────────────────────────
    print(f"\n── Fazit ──")
    if not matches:
        print(f"[DIAGNOSE] '{args.span_text}' wurde vom System nicht erkannt.")
        print(f"           → Agent 6 (Referenzfakten) müsste Personennamen-Fehler finden.")
        print(f"           → Prüfen: steht 'Roland Bosshard' in den Referenzfakten?")
        print(f"             python tests/test_agent6.py \\")
        print(f"               --case_id case_01 \\")
        print(f"               --modified_doc {args.modified_doc} \\")
        print(f"               --span_text '{args.span_text}'")
    else:
        best_score, best_si, best_f = matches[0]
        if best_f.get("subclass_id") != args.subclass_id and args.subclass_id:
            print(f"[DIAGNOSE] Span erkannt (score={best_score:.3f}), aber falsche Subclass:")
            print(f"           GT:   {args.subclass_id}")
            print(f"           Pred: {best_f.get('subclass_id')}")
        else:
            print(f"[OK] Span erkannt mit score={best_score:.3f} in Segment {best_si}.")
            print(f"     → Evaluations-Match sollte funktionieren.")
    print()


if __name__ == "__main__":
    main()
