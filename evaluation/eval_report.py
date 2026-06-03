#!/usr/bin/env python3
"""
eval_report.py — Human-readable evaluation report

Reads all files from an eval_results/<case_id>/ folder and produces
a single formatted .txt report combining all data.

Usage:
    python3 eval_report.py --eval_dir ./eval_results/case_01_synthetic_qwen2.5-32b
    python3 eval_report.py --eval_dir ./eval_results/case_01_synthetic_qwen2.5-32b --out ./report.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


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


def _opt(d: Dict[str, Any], key: str, default: str = "?") -> str:
    v = d.get(key)
    return str(v) if v is not None else default


# ── Section renderers ─────────────────────────────────────────────────────────

def render_summary(summary: Dict[str, Any]) -> str:
    """
    Renders the top-level summary.json.
    Supports both old-style tier keys (finding_level_strict, …) and
    the current flat span-only key (finding_level_span_only).
    """
    lines: List[str] = []
    lines.append(sep("═"))
    lines.append("EVALUATION METRICS")
    lines.append(sep("═"))

    # Current output: single span-only tier
    span_only = summary.get("finding_level_span_only")
    if span_only:
        tp = span_only.get("tp", 0)
        fp = span_only.get("fp", 0)
        fn = span_only.get("fn", 0)
        p  = span_only.get("precision", 0.0)
        r  = span_only.get("recall",    0.0)
        f1 = span_only.get("f1",        0.0)
        acc = span_only.get("accuracy", 0.0)
        lines.append("")
        lines.append("  Finding level — SPAN-ONLY  (case-level, no segment boundary)")
        lines.append(f"  {'TP':>4} {'FP':>4} {'FN':>4}   {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>9}")
        lines.append(f"  {tp:>4} {fp:>4} {fn:>4}   {pct(p):>10} {pct(r):>8} {pct(f1):>8} {pct(acc):>9}")
        lines.append(f"  Precision {bar(p)} {pct(p)}")
        lines.append(f"  Recall    {bar(r)} {pct(r)}")
        lines.append(f"  F1        {bar(f1)} {pct(f1)}")

    # Legacy tier keys (kept for backwards compatibility)
    legacy_tiers = [
        ("finding_level_strict",    "Finding level — STRICT   (exact subclass + span ≥ threshold)"),
        ("finding_level_document",  "Finding level — DOCUMENT (no segment boundary, main class + span)"),
        ("finding_level_lenient",   "Finding level — LENIENT  (main class + span, same segment)"),
        ("segment_level_detection", "Segment level detection"),
    ]
    for key, label in legacy_tiers:
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
        if note and len(note) < 60:
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
    matched    = summary.get("matched_pairs", 0)
    gold_total = summary.get("gold_findings", "?")
    pred_total = summary.get("pred_findings", "?")
    lines.append(f"  Gold findings:                {gold_total}")
    lines.append(f"  Predicted findings:           {pred_total}")
    lines.append(f"  Matched pairs:                {matched}")
    lines.append(f"  Subclass accuracy:            {pct(summary.get('subclass_accuracy_on_matched', 0))}  (on {matched} matched pairs)")
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
    # gold_count/pred_count may not be present in current output — show TP/FP/FN only
    lines.append(f"  {'Subclass':<40} {'TP':>4} {'FP':>4} {'FN':>4}   {'Prec':>7} {'Rec':>7} {'F1':>7}")
    lines.append(f"  {sep('─', 85)}")

    for sub_id, data in sorted(per_subclass.items()):
        tp = data.get("tp", 0)
        fp = data.get("fp", 0)
        fn = data.get("fn", 0)
        p  = data.get("precision", 0.0)
        r  = data.get("recall", 0.0)
        f1 = data.get("f1", 0.0)
        lines.append(
            f"  {sub_id:<40} {tp:>4} {fp:>4} {fn:>4}   "
            f"{pct(p):>7} {pct(r):>7} {pct(f1):>7}"
        )
        note = data.get("note", "")
        if note and len(note) < 60:
            lines.append(f"  {'':40}   {note}")
    return "\n".join(lines)


def render_per_main_class(per_main: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(sep("═"))
    lines.append("PER-MAIN-CLASS BREAKDOWN")
    lines.append(sep("═"))
    lines.append(f"  {'Main class':<20} {'TP':>4} {'FP':>4} {'FN':>4}   {'Prec':>7} {'Rec':>7} {'F1':>7}")
    lines.append(f"  {sep('─', 65)}")
    for main_id, data in sorted(per_main.items()):
        tp = data.get("tp", 0)
        fp = data.get("fp", 0)
        fn = data.get("fn", 0)
        p  = data.get("precision", 0.0)
        r  = data.get("recall", 0.0)
        f1 = data.get("f1", 0.0)
        lines.append(
            f"  {main_id:<20} {tp:>4} {fp:>4} {fn:>4}   "
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


def render_matches(matches: List[Dict[str, Any]], title: str = "MATCHED PAIRS") -> str:
    """
    Renders matches from matches.jsonl, matches_document_strict.jsonl,
    or matches_document_lenient.jsonl.

    Current schema: flat keys gold_span_text, pred_span_text, gold_subclass_id, etc.
    Legacy schema:  nested gold={...}, pred={...} dicts.
    Both are supported.
    """
    lines: List[str] = []
    lines.append("")
    lines.append(sep("═"))
    lines.append(f"{title}  ({len(matches)} total)")
    lines.append(sep("═"))

    for i, m in enumerate(matches, 1):
        # Support both flat (current) and nested (legacy) schema
        if "gold_span_text" in m:
            # Current flat schema
            gold_span     = str(m.get("gold_span_text", ""))
            pred_span     = str(m.get("pred_span_text", ""))
            gold_sub      = m.get("gold_subclass_id", "?")
            pred_sub      = m.get("pred_subclass_id", "?")
            gold_corr     = str(m.get("gold_correction", ""))
            pred_corr     = str(m.get("pred_correction", ""))
            gold_rat      = str(m.get("gold_rationale", ""))
            pred_rat      = str(m.get("pred_rationale", ""))
            gold_scope    = m.get("gold_agent_scope", "")
            pred_scope    = m.get("pred_agent_scope", "")
            change_match  = m.get("change_match", "?")
            sev_match     = m.get("severity_match", "?")
            corr_match    = m.get("correction_match", "?")
            span_score    = m.get("span_score", 0.0)
            sub_match     = m.get("subclass_match", "?")
        else:
            # Legacy nested schema
            gold = m.get("gold", {})
            pred = m.get("pred", {})
            gold_span  = str(gold.get("span_text", ""))
            pred_span  = str(pred.get("span_text", ""))
            gold_sub   = gold.get("subclass_id", "?")
            pred_sub   = pred.get("subclass_id", "?")
            gold_corr  = str(gold.get("correction", ""))
            pred_corr  = str(pred.get("correction", ""))
            gold_rat   = str(gold.get("rationale", ""))
            pred_rat   = str(pred.get("rationale", ""))
            gold_scope = gold.get("agent_scope", "")
            pred_scope = pred.get("agent_scope", "")
            change_match = m.get("change_match", "?")
            sev_match    = m.get("severity_match", "?")
            corr_match   = m.get("correction_match", "?")
            span_score   = m.get("span_score", 0.0)
            sub_match    = m.get("subclass_match", "?")

        def bool_icon(v: Any) -> str:
            if v is True:  return "✓"
            if v is False: return "✗"
            return str(v)

        lines.append(f"\n  [{i}]  seg: {m.get('segment_id', '?')}")
        lines.append(f"    Span score:      {span_score:.3f}")
        lines.append(f"    Subclass match:  {bool_icon(sub_match)}"
                     f"   Gold: {gold_sub}  →  Pred: {pred_sub}")
        lines.append(f"    Change match:    {bool_icon(change_match)}")
        lines.append(f"    Severity match:  {bool_icon(sev_match)}")
        lines.append(f"    Correction match:{bool_icon(corr_match)}")
        lines.append(f"    Gold span:       {gold_span[:100]!r}")
        lines.append(f"    Pred span:       {pred_span[:100]!r}")
        if gold_corr:
            lines.append(f"    Gold correction: {gold_corr[:100]!r}")
        if pred_corr:
            lines.append(f"    Pred correction: {pred_corr[:100]!r}")
        if gold_scope:
            lines.append(f"    Gold scope:      {gold_scope}")
        if pred_scope:
            lines.append(f"    Pred scope:      {pred_scope}")
        if pred_rat:
            lines.append(f"    Pred rationale:  {pred_rat[:400]}")
    return "\n".join(lines)


def render_false_negatives(fns: List[Dict[str, Any]]) -> str:
    """
    False negatives: gold findings not detected by the system.

    Current schema: flat finding dict with finding_id, subclass_id, span_text, rationale, etc.
    """
    lines: List[str] = []
    lines.append("")
    lines.append(sep("═"))
    lines.append(f"FALSE NEGATIVES — missed by system  ({len(fns)} total)")
    lines.append(sep("═"))
    if not fns:
        lines.append("  (none — all gold findings were matched)")
        return "\n".join(lines)
    lines.append("  These gold findings were not detected by the system.")
    for i, fn in enumerate(fns, 1):
        seg  = fn.get("segment_id", "?")
        fid  = fn.get("finding_id", fn.get("gold_finding_id", "?"))
        sub  = fn.get("subclass_id", fn.get("gold_subclass_id", "?"))
        span = str(fn.get("span_text", fn.get("gold_span_text", "")))
        rat  = str(fn.get("rationale", fn.get("gold_rationale", "")))
        lines.append(f"\n  [{i}]  {seg}  |  {fid}")
        lines.append(f"    Subclass:  {sub}")
        lines.append(f"    Span:      {span[:300]!r}")
        if rat:
            lines.append(f"    Rationale: {rat[:400]}")
    return "\n".join(lines)


def render_false_positives(fps: List[Dict[str, Any]]) -> str:
    """
    False positives: predictions that did not match any gold finding.

    Current schema: flat finding dict with finding_id, subclass_id, span_text,
    rationale, agent_scope, best_unmatched_gold_in_case.
    """
    lines: List[str] = []
    lines.append("")
    lines.append(sep("═"))
    lines.append(f"FALSE POSITIVES — flagged but unmatched  ({len(fps)} total)")
    lines.append(sep("═"))
    if not fps:
        lines.append("  (none)")
        return "\n".join(lines)
    lines.append("  These predictions did not match any gold finding.")
    for i, fp in enumerate(fps, 1):
        seg   = fp.get("segment_id", "?")
        fid   = fp.get("finding_id", fp.get("pred_finding_id", "?"))
        sub   = fp.get("subclass_id", fp.get("pred_subclass_id", "?"))
        span  = str(fp.get("span_text", fp.get("pred_span_text", "")))
        rat   = str(fp.get("rationale", fp.get("pred_rationale", "")))
        scope = fp.get("agent_scope", "")
        corr  = str(fp.get("correction", ""))
        best  = fp.get("best_unmatched_gold_in_case", [])

        lines.append(f"\n  [{i}]  {seg}  |  {fid}")
        lines.append(f"    Subclass:  {sub}")
        if scope:
            lines.append(f"    Scope:     {scope}")
        lines.append(f"    Span:      {span[:300]!r}")
        if corr:
            lines.append(f"    Correction:{corr[:100]!r}")
        if rat:
            lines.append(f"    Rationale: {rat[:400]}")
        if best:
            lines.append(f"    Closest gold candidates:")
            for cand in best[:3]:
                csub  = cand.get("candidate_subclass_id", "?")
                cspan = str(cand.get("candidate_span_text", ""))
                cscore= cand.get("span_score", 0.0)
                lines.append(f"      score={cscore:.3f}  sub={csub}  span={cspan[:80]!r}")
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
    def _load_json_if_exists(name: str) -> Any:
        p = eval_dir / name
        return load_json(p) if p.exists() else {}

    def _load_jsonl_if_exists(name: str) -> List[Dict[str, Any]]:
        p = eval_dir / name
        return list(iter_jsonl(p)) if p.exists() else []

    summary        = _load_json_if_exists("summary.json")
    per_subclass   = _load_json_if_exists("per_subclass.json")
    per_main_class = _load_json_if_exists("per_main_class.json")
    confusion      = _load_json_if_exists("confusion.json")
    matches        = _load_jsonl_if_exists("matches.jsonl")
    doc_strict     = _load_jsonl_if_exists("matches_document_strict.jsonl")
    doc_lenient    = _load_jsonl_if_exists("matches_document_lenient.jsonl")
    fns            = _load_jsonl_if_exists("false_negatives.jsonl")
    fps            = _load_jsonl_if_exists("false_positives.jsonl")

    # ── Build report ──────────────────────────────────────────────────────────
    sections: List[str] = []

    gold_total = summary.get("gold_findings", "?") if summary else "?"
    pred_total = summary.get("pred_findings", "?") if summary else "?"
    header = [
        sep("═"),
        f"EVALUATION REPORT — {case_id.upper()}",
        sep("═"),
        f"  Eval directory:  {eval_dir}",
        f"  Gold findings:   {gold_total}",
        f"  Pred findings:   {pred_total}",
        f"  Matches:         {len(matches)}",
        f"  False positives: {len(fps)}",
        f"  False negatives: {len(fns)}",
    ]
    sections.append("\n".join(header))

    if summary:
        sections.append(render_summary(summary))
    if per_subclass:
        sections.append(render_per_subclass(per_subclass))
    if per_main_class:
        sections.append(render_per_main_class(per_main_class))
    if confusion:
        sections.append(render_confusion(confusion))
    if matches:
        sections.append(render_matches(matches, title="MATCHED PAIRS (span-only)"))
    if doc_strict:
        sections.append(render_matches(doc_strict, title="MATCHED PAIRS — DOCUMENT STRICT"))
    if doc_lenient and doc_lenient != doc_strict:
        sections.append(render_matches(doc_lenient, title="MATCHED PAIRS — DOCUMENT LENIENT"))
    sections.append(render_false_negatives(fns))
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
