#!/usr/bin/env python3
"""
evaluate_all_cases.py
=====================
Aggregierte Evaluation über alle Cases mit identischem Output wie evaluate_predictions.py.

Aufruf:
    # Synthetische GT, alle Cases, Modell aus Dateinamen:
    python evaluate_all_cases.py \
        --gt_dir    ./ground_truth \
        --pred_dir  ./predictions \
        --model_tag qwen2.5-32b-instruct-q4_K_M \
        --cases 01 02 03 04 05 07 08 \
        --gt_suffix synthetic \
        --output_dir ./eval_results/aggregated_qwen32b \
        --min_span_score 0.10

    # Manuelle GT (Case 06):
    python evaluate_all_cases.py \
        --gt_dir    ./ground_truth \
        --pred_dir  ./predictions \
        --cases 06 \
        --gt_suffix "" \
        --pred_suffix "" \
        --output_dir ./eval_results/aggregated_case06 \
        --min_span_score 0.10

Dateinamen-Konvention (Standard):
    GT:   ground_truth/ground_truth_case_XX_<gt_suffix>.jsonl
    Pred: predictions/predictions_case_XX_<pred_suffix>_<model_tag>.jsonl

Gibt exakt denselben Output wie evaluate_predictions.py aus —
aggregiert über alle angegebenen Cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from evaluate_predictions import (
        load_segment_records,
        evaluate_predictions,
        load_taxonomy_lookup,
        write_json,
        write_jsonl,
        write_per_subclass_csv,
        write_per_main_class_csv,
        EvaluationResult,
    )
except ImportError as e:
    sys.exit(f"[FATAL] evaluate_predictions.py nicht gefunden: {e}")


# ── Aggregation helpers ────────────────────────────────────────────────────────

def _metrics(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6)}


def _aggregate_per_class(
        results: List[EvaluationResult],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Aggregiert per_subclass und per_main_class über mehrere EvaluationResult."""
    sub_agg:  Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    main_agg: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for r in results:
        for sub_id, st in r.per_subclass.items():
            sub_agg[sub_id]["tp"] += st["tp"]
            sub_agg[sub_id]["fp"] += st["fp"]
            sub_agg[sub_id]["fn"] += st["fn"]
        for main_id, mt in r.per_main_class.items():
            main_agg[main_id]["tp"] += mt["tp"]
            main_agg[main_id]["fp"] += mt["fp"]
            main_agg[main_id]["fn"] += mt["fn"]

    per_sub  = {k: _metrics(**v) for k, v in sub_agg.items()}
    per_main = {k: _metrics(**v) for k, v in main_agg.items()}
    return per_sub, per_main


def _aggregate_summary(
        results: List[EvaluationResult],
        min_span_score: float,
) -> Dict[str, Any]:
    """Baut ein summary-Dict das identisch zu evaluate_predictions.summary ist."""
    total_tp = total_fp = total_fn = 0
    total_gold = total_pred = 0
    total_matched = 0
    sub_correct = sub_correct_ct = 0
    chg_correct = chg_correct_ct = 0
    sev_correct = sev_correct_ct = 0
    corr_correct = corr_considered = 0

    for r in results:
        s  = r.summary
        fl = s["finding_level_span_only"]
        total_tp   += fl["tp"]
        total_fp   += fl["fp"]
        total_fn   += fl["fn"]
        total_gold += s.get("gold_findings", fl["tp"] + fl["fn"])
        total_pred += s.get("pred_findings", fl["tp"] + fl["fp"])
        total_matched += s.get("matched_pairs", 0)

        mp = s.get("matched_pairs", 0)
        sub_correct   += round(s.get("subclass_accuracy_on_matched", 0) * mp)
        chg_correct   += round(s.get("change_type_accuracy_on_matched", 0) * mp)
        sev_correct   += round(s.get("severity_accuracy_on_matched", 0) * mp)
        sub_correct_ct += mp
        chg_correct_ct += mp
        sev_correct_ct += mp

        cc = s.get("correction_considered", 0)
        corr_considered += cc
        corr_correct    += round(s.get("correction_accuracy_on_considered", 0) * cc)

    prec = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    return {
        "finding_level_span_only": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6),
        },
        "gold_findings":  total_gold,
        "pred_findings":  total_pred,
        "matched_pairs":  total_matched,
        "subclass_accuracy_on_matched":
            sub_correct / sub_correct_ct if sub_correct_ct > 0 else 0.0,
        "change_type_accuracy_on_matched":
            chg_correct / chg_correct_ct if chg_correct_ct > 0 else 0.0,
        "severity_accuracy_on_matched":
            sev_correct / sev_correct_ct if sev_correct_ct > 0 else 0.0,
        "correction_accuracy_on_considered":
            corr_correct / corr_considered if corr_considered > 0 else 0.0,
        "correction_considered": corr_considered,
        "min_span_score": min_span_score,
    }


# ── Output ─────────────────────────────────────────────────────────────────────

def print_human_summary(
        summary: Dict[str, Any],
        per_subclass: Dict[str, Dict[str, Any]],
        per_main_class: Dict[str, Dict[str, Any]],
        cases: List[str],
        model_tag: str,
) -> None:
    fl   = summary["finding_level_span_only"]
    gold = summary.get("gold_findings", fl["tp"] + fl["fn"])
    pred = summary.get("pred_findings", fl["tp"] + fl["fp"])

    sep = "=" * 90
    print(sep)
    print("EVALUATION SUMMARY — Span-Only Matching (case-level, kein Segment-Index-Match)")
    print(sep)
    print(f"  Modell:              {model_tag or '(nicht angegeben)'}")
    print(f"  Cases:               {', '.join(cases)}")
    print(f"  Gold Findings (GT):  {gold}")
    print(f"  Pred Findings:       {pred}")
    print(f"  Span-Schwellwert:    {summary['min_span_score']:.2f}")
    print("")
    print(f"  TP (korrekt erkannt):      {fl['tp']}")
    print(f"  FP (False Positives):      {fl['fp']}")
    print(f"  FN (nicht erkannt):        {fl['fn']}")
    print(f"  Precision:                 {fl['precision']:.4f}")
    print(f"  Recall:                    {fl['recall']:.4f}")
    print(f"  F1:                        {fl['f1']:.4f}")
    print("")
    print(f"  Matched pairs:             {summary['matched_pairs']}")
    print(f"  Subclass-Genauigkeit:      {summary['subclass_accuracy_on_matched']:.4f}  (auf gematchten Paaren)")
    print(f"  Change-type-Genauigkeit:   {summary['change_type_accuracy_on_matched']:.4f}  (auf gematchten Paaren)")
    print(f"  Severity-Genauigkeit:      {summary['severity_accuracy_on_matched']:.4f}  (auf gematchten Paaren)")
    print(f"  Korrektur-Genauigkeit:     {summary['correction_accuracy_on_considered']:.4f}  ({summary['correction_considered']} Fälle)")
    print("")

    if per_subclass:
        print("─" * 90)
        print(f"  {'Subklasse':<40} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("─" * 90)
        for sub_id in sorted(per_subclass):
            st = per_subclass[sub_id]
            print(f"  {sub_id:<40} {st['tp']:>4} {st['fp']:>4} {st['fn']:>4} "
                  f" {st['precision']:>7.4f} {st['recall']:>7.4f} {st['f1']:>7.4f}")
        print("")

    if per_main_class:
        print("─" * 90)
        print(f"  {'Hauptklasse':<20} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("─" * 90)
        for main_id in sorted(per_main_class):
            mt = per_main_class[main_id]
            print(f"  {main_id:<20} {mt['tp']:>4} {mt['fp']:>4} {mt['fn']:>4} "
                  f" {mt['precision']:>7.4f} {mt['recall']:>7.4f} {mt['f1']:>7.4f}")
        print("")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregierte Evaluation über mehrere Cases — identischer Output wie evaluate_predictions.py."
    )
    ap.add_argument("--gt_dir",   default="./ground_truth", help="Verzeichnis mit GT-JSONL-Files")
    ap.add_argument("--pred_dir", default="./predictions",  help="Verzeichnis mit Predictions-JSONL-Files")
    ap.add_argument(
        "--cases", nargs="+",
        default=["01", "02", "03", "04", "05", "07", "08"],
        help="Case-Nummern (z.B. 01 02 03). Default: 01-05 07 08 (ohne 06)",
    )
    ap.add_argument("--gt_suffix",   default="synthetic",
                    help="Suffix im GT-Dateinamen. Default: 'synthetic' → ground_truth_case_XX_synthetic.jsonl")
    ap.add_argument("--pred_suffix", default="synthetic",
                    help="Suffix im Pred-Dateinamen vor dem Modell-Tag. Default: 'synthetic'")
    ap.add_argument("--model_tag",   default="",
                    help="Modell-Tag im Dateinamen (z.B. qwen2.5-32b-instruct-q4_K_M). "
                         "Leer → Dateiname ohne Modell-Tag (predictions_case_XX_synthetic.jsonl)")
    ap.add_argument("--taxonomy_json", default="./tax/taxonomy.json")
    ap.add_argument("--output_dir",    default="./eval_results/aggregated")
    ap.add_argument("--min_span_score", type=float, default=0.60)

    # Referenz-Fall (Case 06, manuelle GT)
    ap.add_argument("--ref_gt",   default="",
                    help="Pfad zur GT-JSONL für den Referenz-Fall (default: <gt_dir>/ground_truth_case_06.jsonl)")
    ap.add_argument("--ref_pred", default="",
                    help="Pfad zur Pred-JSONL für den Referenz-Fall (default: <pred_dir>/predictions_case_06.jsonl)")
    ap.add_argument("--ref_output_dir", default="./eval_results/case_06",
                    help="Ausgabeverzeichnis für die Referenz-Evaluation (default: eval_results/case_06)")
    ap.add_argument("--skip_reference", action="store_true",
                    help="Referenz-Fall-Evaluation überspringen")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    gt_dir   = Path(args.gt_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    out_dir  = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = (
        load_taxonomy_lookup(Path(args.taxonomy_json).resolve())
        if args.taxonomy_json.strip() and Path(args.taxonomy_json).exists()
        else None
    )

    results:        List[EvaluationResult] = []
    all_fn:         List[Dict[str, Any]]   = []
    all_fp:         List[Dict[str, Any]]   = []
    all_matches:    List[Dict[str, Any]]   = []
    processed_cases: List[str]             = []
    skipped_cases:   List[str]             = []

    for case in args.cases:
        case_id = f"case_{case}"

        # GT-Pfad
        if args.gt_suffix:
            gt_file = gt_dir / f"ground_truth_{case_id}_{args.gt_suffix}.jsonl"
        else:
            gt_file = gt_dir / f"ground_truth_{case_id}.jsonl"

        # Pred-Pfad
        if args.model_tag and args.pred_suffix:
            pred_file = pred_dir / f"predictions_{case_id}_{args.pred_suffix}_{args.model_tag}.jsonl"
        elif args.model_tag:
            pred_file = pred_dir / f"predictions_{case_id}_{args.model_tag}.jsonl"
        elif args.pred_suffix:
            pred_file = pred_dir / f"predictions_{case_id}_{args.pred_suffix}.jsonl"
        else:
            pred_file = pred_dir / f"predictions_{case_id}.jsonl"

        # Existenz prüfen
        if not gt_file.exists():
            print(f"[SKIP] GT nicht gefunden:   {gt_file}")
            skipped_cases.append(case)
            continue
        if not pred_file.exists():
            print(f"[SKIP] Pred nicht gefunden: {pred_file}")
            skipped_cases.append(case)
            continue

        print(f"[INFO] Case {case}: GT={gt_file.name}  Pred={pred_file.name}")

        gold_records = load_segment_records(gt_file,   findings_field="gold_findings",      taxonomy=taxonomy)
        pred_records = load_segment_records(pred_file, findings_field="predicted_findings",  taxonomy=taxonomy)

        # Auf diesen Case filtern
        gold_records = {k: v for k, v in gold_records.items() if k[0] == case_id}
        pred_records = {k: v for k, v in pred_records.items() if k[0] == case_id}

        if not gold_records:
            print(f"[SKIP] Keine GT-Records für {case_id}")
            skipped_cases.append(case)
            continue

        result = evaluate_predictions(
            gold_records, pred_records,
            min_span_score=args.min_span_score,
            require_exact_subclass=False,
        )

        results.append(result)
        all_fn.extend(result.false_negatives)
        all_fp.extend(result.false_positives)
        all_matches.extend(result.matches)
        processed_cases.append(case)

        fl = result.summary["finding_level_span_only"]
        print(f"       TP={fl['tp']}  FP={fl['fp']}  FN={fl['fn']}  "
              f"Prec={fl['precision']:.4f}  Rec={fl['recall']:.4f}  F1={fl['f1']:.4f}")

    if not results:
        sys.exit("[FATAL] Keine Cases konnten evaluiert werden.")

    # ── Aggregation ────────────────────────────────────────────────────────────
    summary      = _aggregate_summary(results, args.min_span_score)
    per_subclass, per_main_class = _aggregate_per_class(results)

    # ── Output ─────────────────────────────────────────────────────────────────
    print()
    print_human_summary(
        summary, per_subclass, per_main_class,
        cases=processed_cases,
        model_tag=args.model_tag,
    )

    # Dateien schreiben
    write_json(out_dir / "summary.json",       summary)
    write_json(out_dir / "per_subclass.json",  per_subclass)
    write_json(out_dir / "per_main_class.json",per_main_class)
    write_jsonl(out_dir / "false_negatives.jsonl", all_fn)
    write_jsonl(out_dir / "false_positives.jsonl", all_fp)
    write_jsonl(out_dir / "matches.jsonl",         all_matches)
    write_per_subclass_csv( out_dir / "per_subclass.csv",  per_subclass)
    write_per_main_class_csv(out_dir / "per_main_class.csv", per_main_class)

    if skipped_cases:
        print(f"[WARN] Übersprungene Cases: {skipped_cases}")
    print(f"[INFO] Wrote evaluation artifacts to: {out_dir}")

    # ── Referenz-Fall (Case 06, manuelle GT) ──────────────────────────────────
    if not args.skip_reference:
        ref_gt_path   = Path(args.ref_gt).resolve()   if args.ref_gt   else gt_dir / "ground_truth_case_06.jsonl"
        ref_pred_path = Path(args.ref_pred).resolve() if args.ref_pred else pred_dir / "predictions_case_06.jsonl"
        ref_out_dir   = Path(args.ref_output_dir).resolve()

        if not ref_gt_path.exists():
            print(f"\n[SKIP] Referenz-GT nicht gefunden: {ref_gt_path}")
        elif not ref_pred_path.exists():
            print(f"\n[SKIP] Referenz-Pred nicht gefunden: {ref_pred_path}")
        else:
            print(f"\n[INFO] Referenz Case 06: GT={ref_gt_path.name}  Pred={ref_pred_path.name}")

            ref_gold = load_segment_records(ref_gt_path,   findings_field="gold_findings",     taxonomy=taxonomy)
            ref_pred = load_segment_records(ref_pred_path, findings_field="predicted_findings", taxonomy=taxonomy)
            ref_gold = {k: v for k, v in ref_gold.items() if k[0] == "case_06"}
            ref_pred = {k: v for k, v in ref_pred.items() if k[0] == "case_06"}

            if not ref_gold:
                print("[SKIP] Keine GT-Records für case_06 gefunden.")
            else:
                ref_result = evaluate_predictions(
                    ref_gold, ref_pred,
                    min_span_score=args.min_span_score,
                    require_exact_subclass=False,
                )

                fl = ref_result.summary["finding_level_span_only"]
                print()
                print_human_summary(
                    ref_result.summary,
                    ref_result.per_subclass,
                    ref_result.per_main_class,
                    cases=["06"],
                    model_tag="(manuelle GT — Referenz)",
                )

                ref_out_dir.mkdir(parents=True, exist_ok=True)
                write_json(ref_out_dir / "summary.json",        ref_result.summary)
                write_json(ref_out_dir / "per_subclass.json",   ref_result.per_subclass)
                write_json(ref_out_dir / "per_main_class.json", ref_result.per_main_class)
                write_json(ref_out_dir / "confusion.json",      ref_result.confusion)
                write_jsonl(ref_out_dir / "matches.jsonl",           ref_result.matches)
                write_jsonl(ref_out_dir / "false_negatives.jsonl",   ref_result.false_negatives)
                write_jsonl(ref_out_dir / "false_positives.jsonl",   ref_result.false_positives)
                write_per_subclass_csv( ref_out_dir / "per_subclass.csv",   ref_result.per_subclass)
                write_per_main_class_csv(ref_out_dir / "per_main_class.csv", ref_result.per_main_class)
                print(f"[INFO] Wrote Case 06 artifacts to: {ref_out_dir}")


if __name__ == "__main__":
    main()