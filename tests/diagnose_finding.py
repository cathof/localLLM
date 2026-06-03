#!/usr/bin/env python3
"""
diagnose_finding.py
===================
Zeigt warum ein spezifisches GT-Finding nicht erkannt wird.

Aufruf:
    python diagnose_finding.py \
        --case_id case_01 \
        --span_text "volständig" \
        --gt_jsonl ground_truth/ground_truth_case_01_synthetic.jsonl \
        --pred_jsonl predictions/predictions_case_01_synthetic.jsonl \
        --modified_doc case_documents/case_01_modified.docx

Analyse:
  1. Ist der Span im modifizierten Dokument vorhanden?
  2. In welchem Segment liegt er?
  3. Was hat das System in diesem Segment gefunden?
  4. Gibt es irgendwo eine ähnliche Prediction (span_match_score > 0)?
  5. Warum hat der Evaluation-Match nicht funktioniert?
"""

import argparse
import json
import sys
from pathlib import Path

# ── Produktionscode importieren ───────────────────────────────────────────────
# Script liegt in tests/ → Root-Verzeichnis ist eine Ebene höher
_THIS_DIR  = Path(__file__).resolve().parent
_ROOT_DIR  = _THIS_DIR.parent
for _p in [str(_THIS_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from rag_answer_reference_facts_OHNEFIX import (
        load_dotenv, env_str, env_int,
        split_document_into_segments,
    )
    from importDocuments_structural import normalize_text, read_docx
except ImportError as e:
    sys.exit(f"[FATAL] Import fehlgeschlagen: {e}")

try:
    from evaluate_predictions_mit_segment import span_match_score
except ImportError as e:
    sys.exit(f"[FATAL] evaluate_predictions.py nicht gefunden: {e}")

load_dotenv(".env")


def _load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="Diagnose: warum wird ein GT-Finding nicht erkannt?")
    ap.add_argument("--case_id",      required=True)
    ap.add_argument("--span_text",    required=True, help="span_text des GT-Findings")
    ap.add_argument("--gt_jsonl",     required=True)
    ap.add_argument("--pred_jsonl",   required=True)
    ap.add_argument("--modified_doc", required=True, help=".docx des modifizierten Dokuments")
    ap.add_argument("--min_span_score", type=float, default=0.20)
    args = ap.parse_args()

    gt_path   = Path(args.gt_jsonl).expanduser().resolve()
    pred_path = Path(args.pred_jsonl).expanduser().resolve()
    doc_path  = Path(args.modified_doc).expanduser().resolve()
    span      = args.span_text
    min_score = args.min_span_score

    print(f"\n{'='*64}")
    print(f"  Diagnose: '{span}'")
    print(f"  Case:     {args.case_id}")
    print(f"{'='*64}\n")

    # ── 1. GT-Finding suchen ─────────────────────────────────────────────────
    gt_data = _load_jsonl(gt_path)
    gt_finding = None
    gt_seg_idx = None
    for seg in gt_data:
        for f in seg.get("gold_findings", []):
            if f.get("span_text", "").strip() == span.strip():
                gt_finding = f
                gt_seg_idx = seg["segment_index"]
                break

    if not gt_finding:
        print(f"[WARN] '{span}' nicht in GT gefunden. Prüfe den span_text.\n")
        # Ähnliche Spans suchen
        print("Ähnliche Spans in GT:")
        for seg in gt_data:
            for f in seg.get("gold_findings", []):
                s = f.get("span_text", "")
                score = span_match_score(span, s)
                if score > 0.3:
                    print(f"  seg {seg['segment_index']:3d}: '{s}' (score {score:.2f})")
        return

    print(f"[OK] GT-Finding gefunden:")
    print(f"     segment_index: {gt_seg_idx}")
    print(f"     subclass_id:   {gt_finding.get('subclass_id')}")
    print(f"     span_text:     '{gt_finding.get('span_text')}'")
    print(f"     correction:    '{gt_finding.get('correction', '')}'")

    # ── 2. Span im modifizierten Dokument prüfen ──────────────────────────────
    print(f"\n── Schritt 1: Span im Dokument? ──")
    doc_text = normalize_text(read_docx(doc_path))
    if span in doc_text:
        print(f"[OK] '{span}' ist im modifizierten Dokument vorhanden.")
        # Kontext zeigen
        idx = doc_text.index(span)
        context = doc_text[max(0, idx-80):idx+len(span)+80].replace("\n", " ")
        print(f"     Kontext: ...{context}...")
    else:
        print(f"[FAIL] '{span}' NICHT im modifizierten Dokument!")
        print(f"       → Die Injektion hat nicht funktioniert oder span_text stimmt nicht.")
        return

    # ── 3. In welchem Segment liegt der Span? ────────────────────────────────
    print(f"\n── Schritt 2: Segmentzuordnung ──")
    segments = split_document_into_segments(
        doc_text,
        target_chars=env_int("SEG_TARGET_CHARS", 1200),
        min_chars=env_int("SEG_MIN_CHARS", 250),
        max_chars=env_int("SEG_MAX_CHARS", 2200),
    )
    actual_seg_idx = None
    for i, seg in enumerate(segments):
        if span in seg:
            actual_seg_idx = i
            print(f"[OK] Span in Segment {i} gefunden.")
            print(f"     GT-Segment-Index: {gt_seg_idx}")
            if i != gt_seg_idx:
                print(f"     [!] MISMATCH: GT sagt Seg {gt_seg_idx}, Dokument hat Seg {i}")
            else:
                print(f"     [OK] Segment-Index stimmt überein.")
            break

    if actual_seg_idx is None:
        print(f"[FAIL] Span in keinem Segment gefunden (obwohl im Volltext).")
        print(f"       → Segmentierungsparameter prüfen.")

    # ── 4. Was hat das System in diesem Segment gefunden? ────────────────────
    print(f"\n── Schritt 3: Predictions in Segment {gt_seg_idx} ──")
    pred_data = _load_jsonl(pred_path)
    pred_seg = next((s for s in pred_data if s["segment_index"] == gt_seg_idx), None)
    if not pred_seg:
        print(f"[WARN] Keine Predictions für Segment {gt_seg_idx}.")
        print(f"       → Das System hat in diesem Segment gar nichts gefunden.")
    else:
        pf = pred_seg.get("predicted_findings", [])
        if not pf:
            print(f"[WARN] Segment {gt_seg_idx} in Predictions vorhanden, aber keine Findings.")
        else:
            print(f"       {len(pf)} Prediction(s) in diesem Segment:")
            for p in pf:
                score = span_match_score(span, p.get("span_text", ""))
                print(f"       - '{p.get('span_text','')[:60]}' "
                      f"[{p.get('subclass_id')}] "
                      f"span_score={score:.3f}")

    # ── 5. Dokumentweite Suche nach ähnlichen Predictions ────────────────────
    print(f"\n── Schritt 4: Dokumentweite Suche nach ähnlichen Predictions ──")
    all_pf = [
        (seg["segment_index"], f)
        for seg in pred_data
        for f in seg.get("predicted_findings", [])
    ]
    hits = []
    for si, p in all_pf:
        score = span_match_score(span, p.get("span_text", ""))
        if score >= min_score:
            hits.append((score, si, p))

    if hits:
        hits.sort(reverse=True)
        print(f"[OK] {len(hits)} Prediction(s) mit span_score ≥ {min_score}:")
        for score, si, p in hits[:5]:
            subclass_match = "✓" if p.get("subclass_id") == gt_finding.get("subclass_id") else "✗"
            print(f"     seg {si:3d}: '{p.get('span_text','')[:60]}' "
                  f"[{p.get('subclass_id')}] "
                  f"subclass {subclass_match}  span_score={score:.3f}")
    else:
        print(f"[FAIL] Keine Prediction mit span_score ≥ {min_score} gefunden.")
        print(f"       → Das System hat den Span nicht erkannt.")

        # Alle Scores zeigen (auch unter Schwellwert)
        below = [(span_match_score(span, p.get("span_text","")), si, p)
                 for si, p in all_pf]
        below.sort(reverse=True)
        if below[:3]:
            print(f"       Beste Treffer unterhalb Schwellwert:")
            for score, si, p in below[:3]:
                print(f"         seg {si:3d}: '{p.get('span_text','')[:50]}' score={score:.3f}")

    # ── 6. Diagnose-Zusammenfassung ───────────────────────────────────────────
    print(f"\n── Zusammenfassung ──")
    if actual_seg_idx is not None and actual_seg_idx != gt_seg_idx:
        print(f"[DIAGNOSE] Segment-Index-Mismatch: GT={gt_seg_idx}, Dokument={actual_seg_idx}")
        print(f"           → run_validate hat den falschen Index geschrieben")
    elif not hits:
        print(f"[DIAGNOSE] Span nicht erkannt: das System hat '{span}' nicht als Fehler detektiert")
        print(f"           → Agent 3 (Sprachprüfer) hat den Tippfehler übersehen")
        print(f"           → oder LanguageTool kennt das Wort nicht als falsch")
    else:
        best_score, best_si, best_p = hits[0]
        if best_p.get("subclass_id") != gt_finding.get("subclass_id"):
            print(f"[DIAGNOSE] Span erkannt, aber falsche Subklasse:")
            print(f"           GT:   {gt_finding.get('subclass_id')}")
            print(f"           Pred: {best_p.get('subclass_id')}")
            print(f"           → Klassifikationsfehler, kein Erkennungsfehler")
        else:
            print(f"[DIAGNOSE] Span erkannt mit korrekter Subklasse.")
            print(f"           → Evaluation-Match sollte funktionieren.")
            print(f"           → Segment-Index im Evaluationsscript prüfen.")

    print()


if __name__ == "__main__":
    main()