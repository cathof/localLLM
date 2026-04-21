#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


# ── IO helpers ────────────────────────────────────────────────────────────────

def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object on line {line_no} in {path}")
            yield obj


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected top-level JSON object in {path}")
    return obj


# ── Text normalization / fuzzy matching ───────────────────────────────────────

_WS_RE = re.compile(r"\s+")
_PUNCT_STRIP_RE = re.compile(r"[^\wäöüÄÖÜß]+")


def normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip()).strip()


def simplify_text(text: str) -> str:
    s = normalize_text(text).casefold()
    s = _PUNCT_STRIP_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def token_set(text: str) -> set[str]:
    s = simplify_text(text)
    return {tok for tok in s.split(" ") if tok}


def jaccard_similarity(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def span_match_score(gold_span: str, pred_span: str) -> float:
    g = simplify_text(gold_span)
    p = simplify_text(pred_span)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    if g == p:
        return 1.0
    if g in p or p in g:
        return 0.9
    return jaccard_similarity(g, p)


# ── Taxonomy validation ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaxonomyLookup:
    subclass_ids: set[str]
    change_type_ids: set[str]
    severity_ids: set[str]


def load_taxonomy_lookup(path: Optional[Path]) -> Optional[TaxonomyLookup]:
    if path is None:
        return None
    raw = load_json(path)

    subclass_ids: set[str] = set()
    for main in raw.get("main_classes", []):
        if not isinstance(main, dict):
            continue
        for sub in main.get("subclasses", []):
            if isinstance(sub, dict):
                sub_id = str(sub.get("id") or "").strip()
                if sub_id:
                    subclass_ids.add(sub_id)

    change_type_ids = {
        str(x.get("id") or "").strip()
        for x in raw.get("change_types", [])
        if isinstance(x, dict) and str(x.get("id") or "").strip()
    }
    severity_ids = {
        str(x.get("id") or "").strip()
        for x in raw.get("severity_levels", [])
        if isinstance(x, dict) and str(x.get("id") or "").strip()
    }

    return TaxonomyLookup(
        subclass_ids=subclass_ids,
        change_type_ids=change_type_ids,
        severity_ids=severity_ids,
    )


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Finding:
    finding_id: str
    subclass_id: str
    change_type_id: str
    severity_id: str
    span_text: str
    correction: str = ""
    rationale: str = ""
    source: str = ""

    @property
    def normalized_span(self) -> str:
        return simplify_text(self.span_text)

    @property
    def normalized_correction(self) -> str:
        return simplify_text(self.correction)


@dataclass(frozen=True)
class SegmentRecord:
    case_id: str
    segment_id: str
    segment_index: Optional[int]
    findings: Tuple[Finding, ...]
    raw: Dict[str, Any] = field(repr=False)

    @property
    def key(self) -> Tuple[str, str]:
        return self.case_id, self.segment_id


# ── Loaders ───────────────────────────────────────────────────────────────────

def _coerce_segment_index(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    return None


def _validate_finding_ids(f: Finding, taxonomy: Optional[TaxonomyLookup], path: Path, segment_id: str) -> None:
    if taxonomy is None:
        return
    if f.subclass_id and f.subclass_id not in taxonomy.subclass_ids:
        raise RuntimeError(f"{path}: unknown subclass_id '{f.subclass_id}' in segment {segment_id}")
    if f.change_type_id and f.change_type_id not in taxonomy.change_type_ids:
        raise RuntimeError(f"{path}: unknown change_type_id '{f.change_type_id}' in segment {segment_id}")
    if f.severity_id and f.severity_id not in taxonomy.severity_ids:
        raise RuntimeError(f"{path}: unknown severity_id '{f.severity_id}' in segment {segment_id}")


def load_segment_records(
    path: Path,
    *,
    findings_field: str,
    taxonomy: Optional[TaxonomyLookup] = None,
) -> Dict[Tuple[str, str], SegmentRecord]:
    records: Dict[Tuple[str, str], SegmentRecord] = {}
    for obj in iter_jsonl(path):
        case_id = str(obj.get("case_id") or "").strip()
        segment_id = str(obj.get("segment_id") or "").strip()
        segment_index = _coerce_segment_index(obj.get("segment_index"))

        if not case_id or not segment_id:
            raise RuntimeError(f"{path}: each row needs case_id and segment_id")

        findings_raw = obj.get(findings_field, [])
        if not isinstance(findings_raw, list):
            raise RuntimeError(f"{path}: field '{findings_field}' must be a list")

        findings: List[Finding] = []
        for item in findings_raw:
            if not isinstance(item, dict):
                raise RuntimeError(f"{path}: findings must contain objects")
            finding = Finding(
                finding_id=str(item.get("finding_id") or "").strip(),
                subclass_id=str(item.get("subclass_id") or "").strip(),
                change_type_id=str(item.get("change_type_id") or "").strip(),
                severity_id=str(item.get("severity_id") or "").strip(),
                span_text=str(item.get("span_text") or "").strip(),
                correction=str(item.get("correction") or "").strip(),
                rationale=str(item.get("rationale") or "").strip(),
                source=str(item.get("source") or "").strip(),
            )
            if not finding.finding_id:
                raise RuntimeError(f"{path}: missing finding_id in segment {segment_id}")
            if not finding.subclass_id:
                raise RuntimeError(f"{path}: missing subclass_id in segment {segment_id}")
            if not finding.change_type_id:
                raise RuntimeError(f"{path}: missing change_type_id in segment {segment_id}")
            if not finding.severity_id:
                raise RuntimeError(f"{path}: missing severity_id in segment {segment_id}")
            if not finding.span_text:
                raise RuntimeError(f"{path}: missing span_text in segment {segment_id}")
            _validate_finding_ids(finding, taxonomy, path, segment_id)
            findings.append(finding)

        rec = SegmentRecord(
            case_id=case_id,
            segment_id=segment_id,
            segment_index=segment_index,
            findings=tuple(findings),
            raw=obj,
        )
        if rec.key in records:
            raise RuntimeError(f"{path}: duplicate segment key {rec.key}")
        records[rec.key] = rec
    return records


# ── Matching ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MatchCandidate:
    gold_idx: int
    pred_idx: int
    score: float
    span_score: float
    change_match: bool
    severity_match: bool
    correction_match: Optional[bool]


@dataclass(frozen=True)
class MatchedPair:
    gold: Finding
    pred: Finding
    score: float
    span_score: float
    change_match: bool
    severity_match: bool
    correction_match: Optional[bool]


def build_candidates(
    gold_findings: Sequence[Finding],
    pred_findings: Sequence[Finding],
    *,
    min_span_score: float,
    require_exact_subclass: bool = True,
) -> List[MatchCandidate]:
    candidates: List[MatchCandidate] = []
    for gi, gold in enumerate(gold_findings):
        for pi, pred in enumerate(pred_findings):
            if require_exact_subclass and gold.subclass_id != pred.subclass_id:
                continue

            s_score = span_match_score(gold.span_text, pred.span_text)
            if s_score < min_span_score:
                continue

            change_match = gold.change_type_id == pred.change_type_id
            severity_match = gold.severity_id == pred.severity_id

            if gold.correction or pred.correction:
                correction_match: Optional[bool] = gold.normalized_correction == pred.normalized_correction
            else:
                correction_match = None

            score = (
                100.0
                + 20.0 * s_score
                + (5.0 if change_match else 0.0)
                + (2.0 if severity_match else 0.0)
                + (1.0 if correction_match is True else 0.0)
            )
            candidates.append(
                MatchCandidate(
                    gold_idx=gi,
                    pred_idx=pi,
                    score=score,
                    span_score=s_score,
                    change_match=change_match,
                    severity_match=severity_match,
                    correction_match=correction_match,
                )
            )

    candidates.sort(
        key=lambda c: (
            -c.score,
            -c.span_score,
            -int(c.change_match),
            -int(c.severity_match),
            c.gold_idx,
            c.pred_idx,
        )
    )
    return candidates


def greedy_match_findings(
    gold_findings: Sequence[Finding],
    pred_findings: Sequence[Finding],
    *,
    min_span_score: float,
    require_exact_subclass: bool = True,
) -> Tuple[List[MatchedPair], List[Finding], List[Finding]]:
    candidates = build_candidates(
        gold_findings,
        pred_findings,
        min_span_score=min_span_score,
        require_exact_subclass=require_exact_subclass,
    )

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matches: List[MatchedPair] = []

    for cand in candidates:
        if cand.gold_idx in used_gold or cand.pred_idx in used_pred:
            continue
        used_gold.add(cand.gold_idx)
        used_pred.add(cand.pred_idx)
        matches.append(
            MatchedPair(
                gold=gold_findings[cand.gold_idx],
                pred=pred_findings[cand.pred_idx],
                score=cand.score,
                span_score=cand.span_score,
                change_match=cand.change_match,
                severity_match=cand.severity_match,
                correction_match=cand.correction_match,
            )
        )

    unmatched_gold = [g for i, g in enumerate(gold_findings) if i not in used_gold]
    unmatched_pred = [p for i, p in enumerate(pred_findings) if i not in used_pred]
    return matches, unmatched_gold, unmatched_pred


# ── Metrics ───────────────────────────────────────────────────────────────────

def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    summary: Dict[str, Any]
    false_negatives: List[Dict[str, Any]]
    false_positives: List[Dict[str, Any]]
    matches: List[Dict[str, Any]]
    per_subclass: Dict[str, Dict[str, Any]]
    confusion: Dict[str, Dict[str, int]]


def evaluate_predictions(
    gold_records: Dict[Tuple[str, str], SegmentRecord],
    pred_records: Dict[Tuple[str, str], SegmentRecord],
    *,
    min_span_score: float,
) -> EvaluationResult:
    keys = sorted(set(gold_records.keys()) | set(pred_records.keys()))

    tp = fp = fn = 0
    segment_tp = segment_fp = segment_fn = 0

    matched_rows: List[Dict[str, Any]] = []
    false_negatives: List[Dict[str, Any]] = []
    false_positives: List[Dict[str, Any]] = []

    change_correct = 0
    severity_correct = 0
    correction_considered = 0
    correction_correct = 0

    subclass_tp = Counter()
    subclass_fp = Counter()
    subclass_fn = Counter()
    confusion: Dict[str, Counter] = defaultdict(Counter)

    for key in keys:
        gold_seg = gold_records.get(key)
        pred_seg = pred_records.get(key)

        gold_findings = list(gold_seg.findings if gold_seg else ())
        pred_findings = list(pred_seg.findings if pred_seg else ())

        if gold_findings and pred_findings:
            segment_tp += 1
        elif gold_findings and not pred_findings:
            segment_fn += 1
        elif pred_findings and not gold_findings:
            segment_fp += 1

        matches, unmatched_gold, unmatched_pred = greedy_match_findings(
            gold_findings,
            pred_findings,
            min_span_score=min_span_score,
            require_exact_subclass=True,
        )

        tp += len(matches)
        fn += len(unmatched_gold)
        fp += len(unmatched_pred)

        for match in matches:
            subclass_tp[match.gold.subclass_id] += 1
            change_correct += int(match.change_match)
            severity_correct += int(match.severity_match)
            if match.correction_match is not None:
                correction_considered += 1
                correction_correct += int(match.correction_match)

            matched_rows.append({
                "case_id": key[0],
                "segment_id": key[1],
                "gold_finding_id": match.gold.finding_id,
                "pred_finding_id": match.pred.finding_id,
                "gold_subclass_id": match.gold.subclass_id,
                "pred_subclass_id": match.pred.subclass_id,
                "gold_change_type_id": match.gold.change_type_id,
                "pred_change_type_id": match.pred.change_type_id,
                "gold_severity_id": match.gold.severity_id,
                "pred_severity_id": match.pred.severity_id,
                "gold_span_text": match.gold.span_text,
                "pred_span_text": match.pred.span_text,
                "span_score": round(match.span_score, 4),
                "change_match": match.change_match,
                "severity_match": match.severity_match,
                "correction_match": match.correction_match,
            })

        for gold in unmatched_gold:
            subclass_fn[gold.subclass_id] += 1
            false_negatives.append({
                "case_id": key[0],
                "segment_id": key[1],
                "finding_id": gold.finding_id,
                "subclass_id": gold.subclass_id,
                "change_type_id": gold.change_type_id,
                "severity_id": gold.severity_id,
                "span_text": gold.span_text,
                "correction": gold.correction,
                "rationale": gold.rationale,
            })

        for pred in unmatched_pred:
            subclass_fp[pred.subclass_id] += 1
            false_positives.append({
                "case_id": key[0],
                "segment_id": key[1],
                "finding_id": pred.finding_id,
                "subclass_id": pred.subclass_id,
                "change_type_id": pred.change_type_id,
                "severity_id": pred.severity_id,
                "span_text": pred.span_text,
                "correction": pred.correction,
                "rationale": pred.rationale,
            })

        if unmatched_gold and unmatched_pred:
            for gold in unmatched_gold:
                scored = sorted(
                    (
                        (span_match_score(gold.span_text, pred.span_text), pred)
                        for pred in unmatched_pred
                    ),
                    key=lambda x: x[0],
                    reverse=True,
                )
                if scored and scored[0][0] >= min_span_score:
                    confusion[gold.subclass_id][scored[0][1].subclass_id] += 1

    overall = prf(tp, fp, fn)
    segment_overall = prf(segment_tp, segment_fp, segment_fn)

    per_subclass: Dict[str, Dict[str, Any]] = {}
    all_subclasses = sorted(set(subclass_tp) | set(subclass_fp) | set(subclass_fn))
    for subclass_id in all_subclasses:
        sub_tp = subclass_tp[subclass_id]
        sub_fp = subclass_fp[subclass_id]
        sub_fn = subclass_fn[subclass_id]
        metrics = prf(sub_tp, sub_fp, sub_fn)
        per_subclass[subclass_id] = {
            "tp": sub_tp,
            "fp": sub_fp,
            "fn": sub_fn,
            **metrics,
        }

    summary = {
        "finding_level": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **overall,
        },
        "segment_level_detection": {
            "tp": segment_tp,
            "fp": segment_fp,
            "fn": segment_fn,
            **segment_overall,
        },
        "matched_pairs": len(matched_rows),
        "change_type_accuracy_on_matched": safe_div(change_correct, len(matched_rows)),
        "severity_accuracy_on_matched": safe_div(severity_correct, len(matched_rows)),
        "correction_accuracy_on_considered": safe_div(correction_correct, correction_considered),
        "correction_considered": correction_considered,
        "min_span_score": min_span_score,
        "gold_segments": len(gold_records),
        "pred_segments": len(pred_records),
    }

    confusion_out = {gold_cls: dict(preds) for gold_cls, preds in confusion.items()}

    return EvaluationResult(
        summary=summary,
        false_negatives=false_negatives,
        false_positives=false_positives,
        matches=matched_rows,
        per_subclass=per_subclass,
        confusion=confusion_out,
    )


# ── Output writers ────────────────────────────────────────────────────────────

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_per_subclass_csv(path: Path, per_subclass: Dict[str, Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subclass_id", "tp", "fp", "fn", "precision", "recall", "f1"])
        for subclass_id, stats in sorted(per_subclass.items()):
            writer.writerow([
                subclass_id,
                stats["tp"],
                stats["fp"],
                stats["fn"],
                f"{stats['precision']:.4f}",
                f"{stats['recall']:.4f}",
                f"{stats['f1']:.4f}",
            ])


def print_human_summary(result: EvaluationResult) -> None:
    s = result.summary
    fl = s["finding_level"]
    seg = s["segment_level_detection"]

    print("=" * 90)
    print("EVALUATION SUMMARY")
    print("=" * 90)
    print("Finding level")
    print(f"  TP: {fl['tp']}")
    print(f"  FP: {fl['fp']}")
    print(f"  FN: {fl['fn']}")
    print(f"  Precision: {fl['precision']:.4f}")
    print(f"  Recall:    {fl['recall']:.4f}")
    print(f"  F1:        {fl['f1']:.4f}")
    print("")
    print("Segment level detection")
    print(f"  TP: {seg['tp']}")
    print(f"  FP: {seg['fp']}")
    print(f"  FN: {seg['fn']}")
    print(f"  Precision: {seg['precision']:.4f}")
    print(f"  Recall:    {seg['recall']:.4f}")
    print(f"  F1:        {seg['f1']:.4f}")
    print("")
    print(f"Matched pairs: {s['matched_pairs']}")
    print(f"Change type accuracy on matched: {s['change_type_accuracy_on_matched']:.4f}")
    print(f"Severity accuracy on matched:    {s['severity_accuracy_on_matched']:.4f}")
    print(f"Correction accuracy considered:  {s['correction_accuracy_on_considered']:.4f}")
    print(f"Correction cases considered:     {s['correction_considered']}")
    print(f"Span threshold:                  {s['min_span_score']:.2f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate prediction JSONL against ground truth JSONL for document findings."
    )
    ap.add_argument("--ground_truth_jsonl", required=True, help="Path to ground truth JSONL")
    ap.add_argument("--predictions_jsonl", required=True, help="Path to predictions JSONL")
    ap.add_argument("--taxonomy_json", default="", help="Optional taxonomy JSON for ID validation")
    ap.add_argument(
        "--min_span_score",
        type=float,
        default=0.60,
        help="Minimum fuzzy span score for matching within a segment (default: 0.60)",
    )
    ap.add_argument(
        "--output_dir",
        default="eval_results",
        help="Directory for evaluation artifacts (default: eval_results)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    gt_path = Path(args.ground_truth_jsonl).resolve()
    pred_path = Path(args.predictions_jsonl).resolve()
    out_dir = Path(args.output_dir).resolve()
    taxonomy = load_taxonomy_lookup(Path(args.taxonomy_json).resolve()) if args.taxonomy_json.strip() else None

    gold_records = load_segment_records(gt_path, findings_field="gold_findings", taxonomy=taxonomy)
    pred_records = load_segment_records(pred_path, findings_field="predicted_findings", taxonomy=taxonomy)

    result = evaluate_predictions(
        gold_records,
        pred_records,
        min_span_score=args.min_span_score,
    )

    print_human_summary(result)

    write_json(out_dir / "summary.json", result.summary)
    write_json(out_dir / "per_subclass.json", result.per_subclass)
    write_json(out_dir / "confusion.json", result.confusion)
    write_jsonl(out_dir / "matches.jsonl", result.matches)
    write_jsonl(out_dir / "false_negatives.jsonl", result.false_negatives)
    write_jsonl(out_dir / "false_positives.jsonl", result.false_positives)
    write_per_subclass_csv(out_dir / "per_subclass.csv", result.per_subclass)

    print("")
    print(f"[INFO] Wrote evaluation artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
