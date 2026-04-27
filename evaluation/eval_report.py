#!/usr/bin/env python3
"""
generate_eval_report.py — Human-readable evaluation report

Reads all files from an eval_results/<case_id>/ folder and produces
a single formatted .txt report combining all data.

Usage:
    python3 generate_eval_report.py --eval_dir ./eval_results/case_06
    python3 generate_eval_report.py --eval_dir ./eval_results/case_06 --out ./eval_results/case_06/report.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def bar(value: float, width: int = 30) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def sep(char: str = "─", width: int = 90) -> str:
    return char * width


# ── Section renderers ─────────────────────────────────────────────────────────

def render_summary(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(sep("═"))
    lines.append("EVALUATION METRICS")
    lines.append(sep("═"))

    tiers = [
        ("finding_level_strict",    "Finding level — STRICT   (exact subclass + span ≥ threshold)"),
        ("finding_level_document",  "Finding level — DOCUMENT (no segment boundary, main class + span)"),
        ("finding_level_lenient",   "Finding level — LENIENT  (main class + span, same segment)"),
        ("segment_level_detection", "Segment level detection"),
    ]
    for key, label in tiers:
        tier = summary.get(key)
        if not tier:
            continue
        tp = tier.get("tp", 0)
        fp = tier.get("fp", 0)
        fn = tier.get("fn", 0)
        p  = tier.get("precision", 0.0)
        r  = tier.get("recall",    0.0)
        f1 = tier.get("f1",        0.0)
        note = tier.get("note", "")

        lines.append("")
        lines.append(f"  {label}")
        if note:
            lines.append(f"  Note: {note}")
        lines.append(f"  {'TP':>4} {'FP':>4} {'FN':>4}   {'Precision':>10} {'Recall':>8} {'F1':>8}")
        lines.append(f"  {tp:>4} {fp:>4} {fn:>4}   {pct(p):>10} {pct(r):>8} {pct(f1):>8}")
        lines.append(f"  Precision {bar(p)} {pct(p)}")
        lines.append(f"  Recall    {bar(r)} {pct(r)}")
        lines.append(f"  F1        {bar(f1)} {pct(f1)}")

    lines.append("")
    lines.append(sep())
    lines.append("ADDITIONAL METRICS")
    lines.append(sep())
    matched = summary.get("matched_pairs", 0)
    lines.append(f"  Matched pairs:                {matched}")
    lines.append(f"  Change type accuracy:         {pct(summary.get('change_type_accuracy_on_matched', 0))}  (on {matched} matched pairs)")
    lines.append(f"  Severity accuracy:            {pct(summary.get('severity_accuracy_on_matched', 0))}  (on {matched} matched pairs)")
    corr_acc = summary.get("correction_accuracy_on_considered", 0)
    corr_n   = summary.get("correction_considered", 0)
    lines.append(f"  Correction accuracy:          {pct(corr_acc)}  (on {corr_n} cases with correction)")
    lines.append(f"  Span threshold used:          {summary.get('min_span_score', '?')}")
    lines.append(f"  Gold segments:                {summary.get('gold_segments', '?')}")
    lines.append(f"  Predicted segments:           {summary.get('pred_segments', '?')}")
    return "\n".join(lines)


def render_per_subclass(per_subclass: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(sep("═"))
    lines.append("PER-SUBCLASS BREAKDOWN")
    lines.append(sep("═"))
    lines.append(f"  {'Subclass':<45} {'Gold':>5} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4}   {'P':>7} {'R':>7} {'F1':>7}")
    lines.append(f"  {sep('─', 87)}")

    for sub_id, data in sorted(per_subclass.items()):
        label = data.get("label", sub_id)
        gold  = data.get("gold_count", 0)
        pred  = data.get("pred_count", 0)
        tp    = data.get("tp", 0)
        fp    = data.get("fp", 0)
        fn    = data.get("fn", 0)
        p     = data.get("precision", 0.0)
        r     = data.get("recall", 0.0)
        f1    = data.get("f1", 0.0)
        lines.append(
            f"  {label:<45} {gold:>5} {pred:>5} {tp:>4} {fp:>4} {fn:>4}   "
            f"{pct(p):>7} {pct(r):>7} {pct(f1):>7}"
        )
    return "\n".join(lines)


def render_confusion(confusion: Dict[str, Any]) -> str:
    if not confusion:
        return ""
    lines: List[str] = []
    lines.append("")
    lines.append(sep("═"))
    lines.append("CONFUSION MATRIX  (gold → predicted)")
    lines.append(sep("═"))
    lines.append("  Counts of how gold subclasses were mapped to predicted subclasses.")
    lines.append("  Diagonal = correct matches. Off-diagonal = misclassifications.")
    lines.append("")
    for gold_label, pred_dict in sorted(confusion.items()):
        if not pred_dict:
            continue
        lines.append(f"  GOLD: {gold_label}")
        for pred_label, count in sorted(pred_dict.items(), key=lambda x: -x[1]):
            marker = "✓" if pred_label == gold_label else " "
            lines.append(f"    {marker} → {pred_label:<45} {count:>4}×")
    return "\n".join(lines)


def render_matches(matches: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(sep("═"))
    lines.append(f"MATCHED PAIRS  ({len(matches)} total)")
    lines.append(sep("═"))
    for i, m in enumerate(matches, 1):
        gold = m.get("gold", {})
        pred = m.get("pred", {})
        lines.append(f"\n  [{i}]")
        lines.append(f"    Span score:    {m.get('span_score', 0):.3f}")
        lines.append(f"    Change match:  {'✓' if m.get('change_match') else '✗'}")
        lines.append(f"    Severity match:{'✓' if m.get('severity_match') else '✗'}")
        lines.append(f"    Gold subclass: {gold.get('subclass_id', '?')}")
        lines.append(f"    Pred subclass: {pred.get('subclass_id', '?')}")
        lines.append(f"    Gold span:     {str(gold.get('span_text', ''))[:80]!r}")
        lines.append(f"    Pred span:     {str(pred.get('span_text', ''))[:80]!r}")
        lines.append(f"    Rationale:     {str(pred.get('rationale', ''))[:600]}")
    return "\n".join(lines)


def render_false_negatives(fns: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(sep("═"))
    lines.append(f"FALSE NEGATIVES — missed by system  ({len(fns)} total)")
    lines.append(sep("═"))
    lines.append("  These gold findings were not detected by the system.")
    for i, fn in enumerate(fns, 1):
        lines.append(f"\n  [{i}]  {fn.get('segment_id', '?')}")
        lines.append(f"    Subclass:  {fn.get('subclass_id', '?')}")
        lines.append(f"    Span:      {str(fn.get('span_text', ''))[:300]!r}")
        if fn.get("rationale"):
            lines.append(f"    Rationale: {str(fn['rationale'])[:600]}")
    return "\n".join(lines)


def render_false_positives(fps: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(sep("═"))
    lines.append(f"FALSE POSITIVES — flagged but wrong  ({len(fps)} total)")
    lines.append(sep("═"))
    lines.append("  These predictions did not match any gold finding.")
    for i, fp in enumerate(fps, 1):
        lines.append(f"\n  [{i}]  {fp.get('segment_id', '?')}")
        lines.append(f"    Subclass:  {fp.get('subclass_id', '?')}")
        lines.append(f"    Span:      {str(fp.get('span_text', ''))[:300]!r}")
        if fp.get("rationale"):
            lines.append(f"    Rationale: {str(fp['rationale'])[:600]}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a human-readable report from eval_results/<case_id>/"
    )
    ap.add_argument("--eval_dir", required=True,
                    help="Path to eval_results/<case_id>/ directory")
    ap.add_argument("--out", default="",
                    help="Output .txt path (default: <eval_dir>/report.txt)")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    if not eval_dir.is_dir():
        raise SystemExit(f"Directory not found: {eval_dir}")

    out_path = Path(args.out).resolve() if args.out else eval_dir / "report.txt"

    case_id = eval_dir.name

    # ── Load files ────────────────────────────────────────────────────────────
    summary      = load_json(eval_dir / "summary.json")       if (eval_dir / "summary.json").exists()       else {}
    per_subclass = load_json(eval_dir / "per_subclass.json")  if (eval_dir / "per_subclass.json").exists()  else {}
    confusion    = load_json(eval_dir / "confusion.json")     if (eval_dir / "confusion.json").exists()     else {}
    matches      = list(iter_jsonl(eval_dir / "matches.jsonl"))       if (eval_dir / "matches.jsonl").exists()       else []
    fns          = list(iter_jsonl(eval_dir / "false_negatives.jsonl")) if (eval_dir / "false_negatives.jsonl").exists() else []
    fps          = list(iter_jsonl(eval_dir / "false_positives.jsonl")) if (eval_dir / "false_positives.jsonl").exists() else []

    # ── Build report ──────────────────────────────────────────────────────────
    sections: List[str] = []

    header = [
        sep("═"),
        f"EVALUATION REPORT — {case_id.upper()}",
        sep("═"),
        f"  Eval directory: {eval_dir}",
        f"  Gold findings:  {summary.get('gold_segments', '?')} annotated segments",
        f"  Pred findings:  {summary.get('pred_segments', '?')} predicted segments",
    ]
    sections.append("\n".join(header))

    if summary:
        sections.append(render_summary(summary))
    if per_subclass:
        sections.append(render_per_subclass(per_subclass))
    if confusion:
        sections.append(render_confusion(confusion))
    if matches:
        sections.append(render_matches(matches))
    if fns:
        sections.append(render_false_negatives(fns))
    if fps:
        sections.append(render_false_positives(fps))

    sections.append("")
    sections.append(sep("═"))
    sections.append("END OF REPORT")
    sections.append(sep("═"))

    report = "\n".join(sections)
    out_path.write_text(report, encoding="utf-8")
    print(f"[OK] Report written to: {out_path}")
    print(f"     {len(report.splitlines())} lines, {len(report)} chars")


if __name__ == "__main__":
    main()