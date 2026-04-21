
#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object at line {line_no} in {path}")
            yield obj


@dataclass(frozen=True)
class ErrorCatalog:
    raw: Dict[str, Any]
    main_classes: Dict[str, Dict[str, Any]]
    sub_by_main: Dict[str, Dict[str, Dict[str, Any]]]
    change_types: Dict[str, Dict[str, Any]]
    severity_levels: Dict[str, Dict[str, Any]]

    @property
    def all_subclass_ids(self) -> Set[str]:
        return {sub_id for sub_map in self.sub_by_main.values() for sub_id in sub_map.keys()}

    @property
    def all_change_type_ids(self) -> Set[str]:
        return set(self.change_types.keys())

    @property
    def all_severity_ids(self) -> Set[str]:
        return set(self.severity_levels.keys())


@dataclass(frozen=True)
class GoldFinding:
    finding_id: str
    subclass_id: str
    change_type_id: str
    severity_id: str
    span_text: str
    correction: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class GroundTruthSegment:
    case_id: str
    segment_id: str
    segment_index: Optional[int]
    gold_findings: List[GoldFinding]


def load_taxonomy_json(path: Path) -> ErrorCatalog:
    if not path.exists():
        raise SystemExit(f"Taxonomy file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("Taxonomy JSON must be a top-level object.")

    main_classes_list = raw.get("main_classes")
    change_types_list = raw.get("change_types")
    severity_levels_list = raw.get("severity_levels")

    if not isinstance(main_classes_list, list) or not isinstance(change_types_list, list) or not isinstance(severity_levels_list, list):
        raise SystemExit("Taxonomy JSON is missing required arrays: main_classes, change_types, severity_levels")

    main_classes: Dict[str, Dict[str, Any]] = {}
    sub_by_main: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for entry in main_classes_list:
        if not isinstance(entry, dict):
            continue
        main_id = str(entry.get("id") or "").strip()
        label = str(entry.get("label") or "").strip()
        if not main_id or not label:
            continue
        main_classes[main_id] = entry
        sub_by_main[main_id] = {}
        for sub in entry.get("subclasses", []):
            if not isinstance(sub, dict):
                continue
            sub_id = str(sub.get("id") or "").strip()
            sub_label = str(sub.get("label") or "").strip()
            if sub_id and sub_label:
                sub_by_main[main_id][sub_id] = sub

    change_types: Dict[str, Dict[str, Any]] = {}
    for entry in change_types_list:
        if isinstance(entry, dict):
            _id = str(entry.get("id") or "").strip()
            label = str(entry.get("label") or "").strip()
            if _id and label:
                change_types[_id] = entry

    severity_levels: Dict[str, Dict[str, Any]] = {}
    for entry in severity_levels_list:
        if isinstance(entry, dict):
            _id = str(entry.get("id") or "").strip()
            label = str(entry.get("label") or "").strip()
            if _id and label:
                severity_levels[_id] = entry

    if not main_classes or not change_types or not severity_levels:
        raise SystemExit("Taxonomy JSON contains no valid classes, change types, or severity levels.")

    return ErrorCatalog(
        raw=raw,
        main_classes=main_classes,
        sub_by_main=sub_by_main,
        change_types=change_types,
        severity_levels=severity_levels,
    )


def load_ground_truth_jsonl(path: Path, catalog: ErrorCatalog) -> List[GroundTruthSegment]:
    if not path.exists():
        raise SystemExit(f"Ground truth JSONL not found: {path}")

    subclass_ids = catalog.all_subclass_ids
    change_type_ids = catalog.all_change_type_ids
    severity_ids = catalog.all_severity_ids

    rows: List[GroundTruthSegment] = []
    for obj in iter_jsonl(path):
        case_id = str(obj.get("case_id") or "").strip()
        segment_id = str(obj.get("segment_id") or "").strip()
        seg_index_raw = obj.get("segment_index")
        segment_index = int(seg_index_raw) if isinstance(seg_index_raw, int) else None

        if not case_id or not segment_id:
            continue

        findings_raw = obj.get("gold_findings", [])
        if not isinstance(findings_raw, list):
            continue

        findings: List[GoldFinding] = []
        for item in findings_raw:
            if not isinstance(item, dict):
                continue

            finding_id = str(item.get("finding_id") or "").strip()
            subclass_id = str(item.get("subclass_id") or "").strip()
            change_type_id = str(item.get("change_type_id") or "").strip()
            severity_id = str(item.get("severity_id") or "MEDIUM").strip()
            span_text = str(item.get("span_text") or "").strip()
            correction = str(item.get("correction") or "").strip()
            rationale = str(item.get("rationale") or "").strip()

            if not finding_id or not subclass_id or not change_type_id or not span_text:
                continue
            if subclass_id not in subclass_ids:
                continue
            if change_type_id not in change_type_ids:
                continue
            if severity_id not in severity_ids:
                severity_id = "MEDIUM"

            findings.append(
                GoldFinding(
                    finding_id=finding_id,
                    subclass_id=subclass_id,
                    change_type_id=change_type_id,
                    severity_id=severity_id,
                    span_text=span_text,
                    correction=correction,
                    rationale=rationale,
                )
            )

        rows.append(
            GroundTruthSegment(
                case_id=case_id,
                segment_id=segment_id,
                segment_index=segment_index,
                gold_findings=findings,
            )
        )

    return rows
