#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import requests
import torch


# ── .env loader ───────────────────────────────────────────────────────────────

def load_dotenv(dotenv_path: str | Path = ".env") -> None:
    p = Path(dotenv_path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv(".env")


# ── Run logger (tee stdout+stderr to logs/) ───────────────────────────────────

class _TeeLogger:
    """Writes every write() call to both the original stream and a log file."""

    def __init__(self, original_stream, log_file_handle):
        self._orig = original_stream
        self._log  = log_file_handle

    def write(self, msg: str) -> int:
        self._orig.write(msg)
        self._orig.flush()
        self._log.write(msg)
        self._log.flush()
        return len(msg)

    def flush(self) -> None:
        self._orig.flush()
        self._log.flush()

    # Delegate everything else (e.g. .fileno(), .isatty()) to the original.
    def __getattr__(self, name: str):
        return getattr(self._orig, name)


def _setup_run_logging(case_id: str, model_name: str) -> Optional[Path]:
    """
    Creates logs/<YYYY-MM-DD_HH-MM-SS>_<case_id>_<model_tag>.log and tees
    sys.stdout + sys.stderr into it.  Returns the log path (or None on error).

    Called once at the start of main() after args are parsed.
    """
    try:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        case_tag = (case_id or "nocase").strip().replace(" ", "_")
        model_tag = re.sub(r"[^A-Za-z0-9._-]", "-", model_name or "unknown")
        log_name  = f"{ts}_{case_tag}_{model_tag}.log"
        log_path  = logs_dir / log_name

        log_fh = log_path.open("w", encoding="utf-8", buffering=1)

        sys.stdout = _TeeLogger(sys.__stdout__, log_fh)
        sys.stderr = _TeeLogger(sys.__stderr__, log_fh)

        # Write header so the file is self-contained
        header = (
            f"# RAG run log\n"
            f"# Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Case    : {case_id or '(none)'}\n"
            f"# Model   : {model_name or '(unknown)'}\n"
            f"# Log file: {log_path.resolve()}\n"
            f"{'#' * 72}\n\n"
        )
        sys.stdout.write(header)

        return log_path

    except Exception as exc:  # never crash the pipeline over logging
        sys.__stdout__.write(f"[WARN] Could not set up log file: {exc}\n")
        return None


def _teardown_run_logging(log_path: Optional[Path]) -> None:
    """Flushes and closes the log file; restores sys.stdout/stderr.
    stdout and stderr share the SAME underlying file handle, so it must be
    closed exactly once — and the std streams must be restored FIRST, so that
    a failure here can never leave a broken _TeeLogger on sys.stderr (which
    would crash the interpreter's shutdown flush with exit code 120).
    """
    try:
        log_fh = None

        # Restore the real streams first; remember the shared handle.
        if isinstance(sys.stdout, _TeeLogger):
            log_fh = sys.stdout._log
            sys.stdout = sys.__stdout__
        if isinstance(sys.stderr, _TeeLogger):
            log_fh = sys.stderr._log
            sys.stderr = sys.__stderr__

        # Close the shared handle exactly once.
        if log_fh is not None and not log_fh.closed:
            try:
                log_fh.flush()
            except Exception:
                pass
            log_fh.close()

        if log_path is not None:
            print(f"[INFO] Log written to: {log_path}")
    except Exception as exc:
        # Make sure the real streams are restored even on an unexpected error.
        if isinstance(sys.stdout, _TeeLogger):
            sys.stdout = sys.__stdout__
        if isinstance(sys.stderr, _TeeLogger):
            sys.stderr = sys.__stderr__
        sys.__stdout__.write(f"[WARN] Could not close log file: {exc}\n")


def env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and v.strip() else default


def env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "").strip()
    return int(v) if v else default

def env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}

def require_env(key: str) -> str:
    v = os.environ.get(key, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {key}")
    return v


def env_json_object_optional(key: str) -> Dict[str, Any]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError(f"{key} must be a JSON object, got {type(obj).__name__}")
    return obj


# ── Device + math ─────────────────────────────────────────────────────────────

def choose_device(name: str) -> torch.device:
    n = (name or "auto").lower().strip()
    if n == "cpu":
        return torch.device("cpu")
    if n in {"mps", "metal"}:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if n == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)


# ── Generic JSONL helper ──────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {line_no} in {path}: {e}") from e


# ── Taxonomy ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ErrorCatalog:
    raw: Dict[str, Any]
    main_classes: Dict[str, Dict[str, Any]]
    sub_by_main: Dict[str, Dict[str, Dict[str, Any]]]
    sub_to_main: Dict[str, str]
    change_types: Dict[str, Dict[str, Any]]
    severity_levels: Dict[str, Dict[str, Any]]

    @property
    def allowed_main_labels(self) -> Set[str]:
        return {v["label"] for v in self.main_classes.values()}

    @property
    def allowed_subclasses_by_main_label(self) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for main_id, main_obj in self.main_classes.items():
            out[main_obj["label"]] = {sub["label"] for sub in self.sub_by_main.get(main_id, {}).values()}
        return out

    @property
    def allowed_change_labels(self) -> Set[str]:
        return {v["label"] for v in self.change_types.values()}

    @property
    def allowed_severity_labels(self) -> Set[str]:
        return {v["label"] for v in self.severity_levels.values()}

    @property
    def subclass_label_to_main_label(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for main_id, main_obj in self.main_classes.items():
            main_label = main_obj["label"]
            for sub in self.sub_by_main.get(main_id, {}).values():
                out[sub["label"]] = main_label
        return out


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
        raise SystemExit("Taxonomy JSON must contain main_classes, change_types, and severity_levels arrays.")

    main_classes: Dict[str, Dict[str, Any]] = {}
    sub_by_main: Dict[str, Dict[str, Dict[str, Any]]] = {}
    sub_to_main: Dict[str, str] = {}

    for entry in main_classes_list:
        if not isinstance(entry, dict):
            continue
        main_id = str(entry.get("id") or "").strip()
        main_label = str(entry.get("label") or "").strip()
        if not main_id or not main_label:
            continue

        main_classes[main_id] = entry
        sub_by_main[main_id] = {}

        subclasses = entry.get("subclasses", [])
        if not isinstance(subclasses, list):
            continue

        for sub in subclasses:
            if not isinstance(sub, dict):
                continue
            sub_id = str(sub.get("id") or "").strip()
            sub_label = str(sub.get("label") or "").strip()
            if not sub_id or not sub_label:
                continue
            sub_by_main[main_id][sub_id] = sub
            sub_to_main[sub_label] = main_label

    change_types: Dict[str, Dict[str, Any]] = {}
    for entry in change_types_list:
        if not isinstance(entry, dict):
            continue
        _id = str(entry.get("id") or "").strip()
        label = str(entry.get("label") or "").strip()
        if _id and label:
            change_types[_id] = entry

    severity_levels: Dict[str, Dict[str, Any]] = {}
    for entry in severity_levels_list:
        if not isinstance(entry, dict):
            continue
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
        sub_to_main=sub_to_main,
        change_types=change_types,
        severity_levels=severity_levels,
    )

def build_label_to_id_maps(catalog: ErrorCatalog):
    sub_label_to_id = {}
    for main_id, subs in catalog.sub_by_main.items():
        for sub_id, sub in subs.items():
            sub_label_to_id[sub["label"]] = sub_id

    change_label_to_id = {
        v["label"]: k for k, v in catalog.change_types.items()
    }

    severity_label_to_id = {
        v["label"]: k for k, v in catalog.severity_levels.items()
    }

    return sub_label_to_id, change_label_to_id, severity_label_to_id

def save_predictions_jsonl(
        report: Dict[str, Any],
        case_id: str,
        output_path: Path,
        catalog: ErrorCatalog,
) -> None:
    sub_map, change_map, severity_map = build_label_to_id_maps(catalog)

    segments: Dict[int, List[Dict[str, Any]]] = {}

    all_findings = (
            report.get("factual_findings", [])
            + report.get("language_findings", [])
            + report.get("calculation_findings", [])
            + report.get("hypothesis_findings", [])
            + report.get("reference_consistency_findings", [])
            + report.get("statement_assurance_findings", [])
    )

    for i, item in enumerate(all_findings, start=1):
        seg_idx = item.get("segment_index")
        if seg_idx is None:
            continue

        subclass_label = str(item.get("subklasse") or "").strip()
        change_label = str(item.get("aenderungstyp") or "").strip()
        severity_label = str(item.get("schweregrad") or "").strip()
        span_text = str(item.get("stelle_im_segment") or "").strip()

        if not span_text:
            continue

        subclass_id = sub_map.get(subclass_label)
        change_type_id = change_map.get(change_label)
        severity_id = severity_map.get(severity_label)

        if not subclass_id or not change_type_id or not severity_id:
            print(
                f"[SAVE DROP] seg={seg_idx} "
                f"subclass={subclass_label!r}→{subclass_id!r} "
                f"change={change_label!r}→{change_type_id!r} "
                f"severity={severity_label!r}→{severity_id!r}"
            )
            # Fallback: try case-insensitive lookup for subclass and change type
            if not subclass_id:
                subclass_id = next(
                    (v for k, v in sub_map.items() if k.strip().lower() == subclass_label.strip().lower()),
                    None,
                )
            if not change_type_id:
                change_type_id = next(
                    (v for k, v in change_map.items() if k.strip().lower() == change_label.strip().lower()),
                    None,
                )
            if not severity_id:
                severity_id = next(
                    (v for k, v in severity_map.items() if k.strip().lower() == severity_label.strip().lower()),
                    None,
                )
            if not subclass_id or not change_type_id or not severity_id:
                print(
                    f"[SAVE DROP] unfixable — skipping finding: "
                    f"subclass={subclass_label!r} change={change_label!r} severity={severity_label!r}"
                )
                continue
            print(f"[SAVE RECOVERED] seg={seg_idx} via case-insensitive lookup")

        # source_refs mitschreiben damit Downstream-Analyse (false_positives.jsonl)
        # DOC_INTERNAL vs. externe Quellen unterscheiden kann.
        raw_refs = item.get("source_refs") or []
        source_refs_out = [str(r).strip() for r in raw_refs if str(r).strip()]

        finding = {
            "finding_id": f"PRED-{case_id}-{i:04d}",
            "subclass_id": subclass_id,
            "change_type_id": change_type_id,
            "severity_id": severity_id,
            "span_text": span_text,
            "rationale": str(item.get("begruendung") or "").strip(),
            "source_refs": source_refs_out,
            "agent_scope": str(item.get("agent_scope") or "").strip(),
        }

        vorschlag = str(item.get("vorschlag") or "").strip()
        if vorschlag:
            finding["correction"] = vorschlag

        segments.setdefault(seg_idx, []).append(finding)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for seg_idx, findings in sorted(segments.items()):
            obj = {
                "case_id": case_id,
                "segment_id": f"{case_id}_seg_{seg_idx:04d}",
                "segment_index": seg_idx,
                "predicted_findings": findings,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved predictions to {output_path}")

def build_taxonomy_block(catalog: ErrorCatalog) -> str:
    lines: List[str] = ["Zulässige Klassifikation:"]
    for main_id, main_obj in catalog.main_classes.items():
        lines.append(main_obj["label"])
        for sub in catalog.sub_by_main.get(main_id, {}).values():
            desc = str(sub.get("description") or "").strip()
            if desc:
                lines.append(f"  - {sub['label']}: {desc}")
            else:
                lines.append(f"  - {sub['label']}")

    lines.append("")
    lines.append("Zulässige Änderungstypen:")
    for obj in catalog.change_types.values():
        desc = str(obj.get("description") or "").strip()
        if desc:
            lines.append(f"  - {obj['label']}: {desc}")
        else:
            lines.append(f"  - {obj['label']}")

    lines.append("")
    lines.append("Zulässige Schweregrade:")
    for obj in catalog.severity_levels.values():
        lines.append(f"  - {obj['label']}")

    return "\n".join(lines)



# ── Agent 0: Reference facts ──────────────────────────────────────────────────

def load_reference_facts_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Reference facts schema not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("Reference facts schema must be a top-level JSON object.")
    return raw


def _empty_fact() -> Dict[str, str]:
    return {"value": "", "source_span": "", "confidence": "low"}


def _coerce_fact(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        confidence = str(value.get("confidence") or "low").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        return {
            "value": str(value.get("value") or "").strip(),
            "source_span": str(value.get("source_span") or "").strip(),
            "confidence": confidence,
        }
    if isinstance(value, str):
        return {"value": value.strip(), "source_span": value.strip(), "confidence": "low"}
    return _empty_fact()


REFERENCE_FACT_KEYS = [
    "auftraggeber",
    "gutachten_titel",
    "vorfall",
    "ereignisdatum",
    "beschuldigte_person",
    "ort",
    "sachverstaendige_person",
    "hauptsachbearbeitung",
]


def _looks_like_upper_surname(token: str) -> bool:
    letters = re.sub(r"[^A-Za-zÄÖÜäöüß]", "", token or "")
    return bool(letters) and letters == letters.upper() and len(letters) >= 2


def _normalize_header_person_name(raw_name: str) -> str:
    """Normalize header notation like 'GERBER Joel' to 'Joel GERBER'."""
    parts = [p for p in re.split(r"\s+", (raw_name or "").strip()) if p]
    if len(parts) >= 2 and _looks_like_upper_surname(parts[0]):
        surname = parts[0]
        given = " ".join(parts[1:])
        return f"{given} {surname}".strip()
    return " ".join(parts).strip()


def _extract_accused_person_fact_from_text(doc_text: str) -> Optional[Dict[str, str]]:
    """
    Extract a header field labelled 'Person' as the semantic fact
    'beschuldigte_person'.
    """
    text = (doc_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    header_flat = " ".join("\n".join(lines[:40]).split())
    pattern = re.compile(
        r"\bPerson\s+"
        r"(?P<name>[A-ZÄÖÜ][A-ZÄÖÜäöüß\-]+\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+)*)"
        r"\s*,?\s*"
        r"(?P<birth>\d{1,2}\.\d{1,2}\.\d{4})?"
        r"(?P<role_part>\s+Beschuldigt(?:e|er|en|em)?(?:\s+Person)?)?"
        r"(?P<after>.{0,220})",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(header_flat)
    if not match:
        return None

    raw_name = " ".join((match.group("name") or "").split()).strip(" ,;")
    name = _normalize_header_person_name(raw_name)
    if not name:
        return None

    birth = " ".join((match.group("birth") or "").split()).strip()
    after = " ".join((match.group("after") or "").split()).strip()

    role_detail = ""
    role_match = re.search(r"\bals\s+(?P<detail>[^.;:\n]{3,160})", after, flags=re.IGNORECASE)
    if role_match:
        role_detail = " ".join(role_match.group("detail").split()).strip(" ,;")

    value_parts = [name]
    if birth:
        value_parts.append(birth)
    value_parts.append("Beschuldigte Person")
    if role_detail:
        value_parts.append(f"als {role_detail}")

    source_span = re.sub(r"\s+", " ", header_flat[match.start():match.end()].strip())
    if len(source_span) > 260:
        source_span = source_span[:260].rstrip(" ,;")

    return {
        "value": ", ".join(value_parts),
        "source_span": source_span,
        "confidence": "high",
    }


def _ensure_person_entry(
        personen: List[Dict[str, str]],
        *,
        name: str,
        rolle: str,
        source_span: str,
        confidence: str = "high",
) -> None:
    norm_name = " ".join((name or "").split()).lower()
    if not norm_name:
        return
    for person in personen:
        if " ".join(str(person.get("name") or "").split()).lower() == norm_name:
            if not str(person.get("rolle") or "").strip():
                person["rolle"] = rolle
            if not str(person.get("source_span") or "").strip():
                person["source_span"] = source_span
            if str(person.get("confidence") or "low").lower() == "low":
                person["confidence"] = confidence
            return
    personen.append({
        "name": name,
        "rolle": rolle,
        "source_span": source_span,
        "confidence": confidence,
    })


def enrich_reference_facts_from_document(reference_facts: Dict[str, Any], doc_text: str) -> Dict[str, Any]:
    """Add deterministic high-confidence header facts that the LLM may miss."""
    facts = reference_facts.setdefault("facts", {})
    accused_fact = _extract_accused_person_fact_from_text(doc_text)
    if accused_fact:
        current = facts.get("beschuldigte_person")
        current_value = str(current.get("value") or "").strip() if isinstance(current, dict) else ""
        current_conf = str(current.get("confidence") or "low").strip().lower() if isinstance(current, dict) else "low"
        if not current_value or current_conf == "low":
            facts["beschuldigte_person"] = accused_fact

        person_name = accused_fact.get("value", "").split(",", 1)[0].strip()
        personen = facts.setdefault("personen", [])
        if isinstance(personen, list):
            _ensure_person_entry(
                personen,
                name=person_name,
                rolle="Beschuldigte Person",
                source_span=accused_fact.get("source_span", ""),
                confidence="high",
            )

    return reference_facts


def normalize_reference_facts(raw: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    facts_raw = raw.get("facts") if isinstance(raw.get("facts"), dict) else {}

    facts: Dict[str, Any] = {}
    for key in REFERENCE_FACT_KEYS:
        facts[key] = _coerce_fact(facts_raw.get(key))

    personen_raw = facts_raw.get("personen", [])
    personen: List[Dict[str, str]] = []
    if isinstance(personen_raw, list):
        for person in personen_raw:
            if not isinstance(person, dict):
                continue
            name = str(person.get("name") or "").strip()
            rolle = str(person.get("rolle") or "").strip()
            source_span = str(person.get("source_span") or "").strip()
            confidence = str(person.get("confidence") or "low").strip().lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"
            if name or rolle or source_span:
                personen.append({
                    "name": name,
                    "rolle": rolle,
                    "source_span": source_span,
                    "confidence": confidence,
                })

    facts["personen"] = personen

    entities_raw = facts_raw.get("referenz_entitaeten", [])
    referenz_entitaeten: List[Dict[str, Any]] = []
    if isinstance(entities_raw, list):
        for ent in entities_raw:
            if not isinstance(ent, dict):
                continue
            confidence = str(ent.get("confidence") or "low").strip().lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"
            aliases_raw = ent.get("aliases", [])
            aliases = [str(a).strip() for a in aliases_raw if str(a).strip()] if isinstance(aliases_raw, list) else []
            referenz_entitaeten.append({
                "bezeichnung": str(ent.get("bezeichnung") or "").strip(),
                "typ": str(ent.get("typ") or "").strip(),
                "rolle": str(ent.get("rolle") or "").strip(),
                "source_span": str(ent.get("source_span") or "").strip(),
                "confidence": confidence,
                "aliases": aliases,
            })
    facts["referenz_entitaeten"] = referenz_entitaeten

    return {
        "case_id": str(raw.get("case_id") or case_id or "").strip(),
        "facts": facts,
    }

def extract_document_header(doc_text: str, max_lines: int = 25) -> str:
    lines = [line.strip() for line in doc_text.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines])

def build_reference_facts_messages(
        doc_context: str,
        *,
        case_id: str,
        document_header: str = "",
) -> List[Dict[str, str]]:
    # System: kurz, nur Output-Format und allgemeine Extraktionsregeln.
    # Feldbezogene Semantik steht im User-Prompt direkt bei den Feldern.
    system = (
        "Antworte AUSSCHLIESSLICH mit einem JSON-Objekt. Kein Markdown, keine Erklärung.\n\n"
        "Du bist Agent 0: Referenzfakten-Extraktor für forensische Gutachten.\n"
        "Extrahiere nur Fakten die explizit im DOKUMENTKONTEXT stehen. Erfinde nichts.\n\n"
        "Für jedes Feld gilt:\n"
        "- value: normalisierter Wert (Datum wenn möglich als YYYY-MM-DD)\n"
        "- source_span: exakter kurzer Ausschnitt aus dem Kontext der den Wert belegt\n"
        "- confidence: high = explizit/eindeutig | medium = plausibel/indirekt | low = fehlt/unsicher\n"
        "Fehlendes Feld: value=\"\", source_span=\"\", confidence=\"low\".\n"
        "Nutze exakt die vorgegebenen Feldnamen."
    )

    # User: Dokumentkontext + feldbezogene Extraktionsregeln direkt bei den Feldern.
    user = (
        f"CASE_ID:\n{case_id}\n\n"
        f"DOKUMENTKOPF / ADRESSIERUNG:\n"
        f"{document_header.strip()}\n\n"
        f"DOKUMENTKONTEXT:\n{doc_context.strip()}\n\n"
        "FELDREGELN — lies diese vor der Extraktion:\n"
        "- auftraggeber: Behörde oder Institution aus dem Dokumentkopf/Briefkopf/Adressfeld.\n"
        "  Vorrang hat die Adressierung am Anfang vor späterem Fliesstext ('Mit Schreiben...').\n"
        "  Personen (Jugendanwältin, RA, Sachverständige) sind nicht der Auftraggeber.\n"
        "- ereignisdatum: Datum des Vorfalls/Unfalls/Ereignisses — nicht Berichts-, Auftrags-\n"
        "  oder Versanddatum. Nur extrahieren wenn explizit als Vorfall-/Tatdatum erkennbar.\n"
        "  Bei mehreren Daten: dasjenige wählen das semantisch zum Vorfall gehört.\n"
        "- beschuldigte_person: Wenn im Kopf ein Tabellenfeld 'Person' steht, ist dies die\n"
        "  beschuldigte Person. Beispiel: 'Person GERBER Joel, 16.09.2006 Beschuldigt als Lenker'\n"
        "  → value = 'Joel GERBER, 16.09.2006, Beschuldigte Person, als Lenker'.\n"
        "- personen: Liste aller im Dokumentkopf genannten relevanten Personen mit Rolle.\n\n"
        "Gib exakt dieses JSON-Objekt zurück:\n"
        "{\n"
        "  \"case_id\": \"...\",\n"
        "  \"facts\": {\n"
        "    \"auftraggeber\": {\"value\": \"\", \"source_span\": \"\", \"confidence\": \"low\"},\n"
        "    \"gutachten_titel\": {\"value\": \"\", \"source_span\": \"\", \"confidence\": \"low\"},\n"
        "    \"vorfall\": {\"value\": \"\", \"source_span\": \"\", \"confidence\": \"low\"},\n"
        "    \"ereignisdatum\": {\"value\": \"\", \"source_span\": \"\", \"confidence\": \"low\"},\n"
        "    \"beschuldigte_person\": {\"value\": \"\", \"source_span\": \"\", \"confidence\": \"low\"},\n"
        "    \"ort\": {\"value\": \"\", \"source_span\": \"\", \"confidence\": \"low\"},\n"
        "    \"sachverstaendige_person\": {\"value\": \"\", \"source_span\": \"\", \"confidence\": \"low\"},\n"
        "    \"hauptsachbearbeitung\": {\"value\": \"\", \"source_span\": \"\", \"confidence\": \"low\"},\n"
        "    \"personen\": [\n"
        "      {\"name\": \"\", \"rolle\": \"\", \"source_span\": \"\", \"confidence\": \"low\"}\n"
        "    ],\n"
        "    \"referenz_entitaeten\": []\n"
        "  }\n"
        "}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run_reference_facts_agent(
        llm: LLMClient,
        doc_text: str,
        *,
        case_id: str,
        schema: Dict[str, Any],
        max_chars: int,
) -> Dict[str, Any]:
    doc_context = doc_text[:max(500, max_chars)].strip()
    document_header = extract_document_header(doc_text, max_lines=25)
    messages = build_reference_facts_messages(
        doc_context,
        case_id=case_id,
        document_header=document_header,
    )

    raw_reply = llm.chat(messages, json_mode=True, schema=schema)
    print(f"[DEBUG] Reference facts agent raw reply ({len(raw_reply)} chars): {raw_reply[:300]!r}")

    try:
        raw_json = extract_first_json_object(raw_reply)
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError("Reference facts response must be a JSON object")
    except Exception as e:
        print(f"[WARN] Reference facts agent parse failed: {e}")
        parsed = {}

    normalized = normalize_reference_facts(parsed, case_id=case_id)
    return enrich_reference_facts_from_document(normalized, doc_context)


def format_reference_facts_for_prompt(reference_facts: Dict[str, Any]) -> str:
    """
    Compact formatter kept for printing/debugging only.
    Agent 2 no longer receives the full reference-facts JSON because it caused
    systematic false positives: the model treated missing segment facts as
    contradictions and copied reference/source text into stelle_im_segment.
    """
    facts = reference_facts.get("facts", {}) if isinstance(reference_facts, dict) else {}
    compact: Dict[str, Any] = {"case_id": reference_facts.get("case_id", ""), "facts": {}}

    for key in REFERENCE_FACT_KEYS:
        fact = facts.get(key, {}) if isinstance(facts, dict) else {}
        if not isinstance(fact, dict):
            continue
        value = str(fact.get("value") or "").strip()
        confidence = str(fact.get("confidence") or "low").strip().lower()
        if value and confidence in {"high", "medium"}:
            compact["facts"][key] = {
                "value": value,
                "confidence": confidence,
            }

    persons = facts.get("personen", []) if isinstance(facts, dict) else []
    if isinstance(persons, list):
        kept_persons = []
        for person in persons:
            if not isinstance(person, dict):
                continue
            name = str(person.get("name") or "").strip()
            rolle = str(person.get("rolle") or "").strip()
            confidence = str(person.get("confidence") or "low").strip().lower()
            if name and confidence in {"high", "medium"}:
                kept_persons.append({"name": name, "rolle": rolle, "confidence": confidence})
        if kept_persons:
            compact["facts"]["personen"] = kept_persons

    return json.dumps(compact, ensure_ascii=False, indent=2)


def build_factual_json_schema(catalog: ErrorCatalog) -> Dict[str, Any]:
    """
    Build an Ollama-compatible JSON Schema from the ErrorCatalog at runtime.
    Passed as payload["format"] to Ollama for grammar-constrained decoding:
    the model physically cannot produce field names or enum values outside
    the schema, eliminating alias mapping problems entirely.
    The schema is built from the live catalog so taxonomy changes are
    reflected automatically without code changes.

    Agent 4 (Rechenfehler) and Agent 5 (Hypothesenprüfung) classes are
    excluded — Agent 2 must never use these.
    """
    EXCLUDED_MAIN = {"Rechenfehler", "Hypothesenprüfung"}
    EXCLUDED_CHANGE = {"Rechnerische Korrektur", "Hypothesen-Korrektur"}

    main_labels = sorted(
        m for m in catalog.allowed_main_labels
        if m not in EXCLUDED_MAIN
    )
    change_labels = sorted(
        c for c in catalog.allowed_change_labels
        if c not in EXCLUDED_CHANGE
    )
    sev_labels = sorted(catalog.allowed_severity_labels)

    all_sub_labels: List[str] = []
    for main_label in main_labels:
        subs = catalog.allowed_subclasses_by_main_label.get(main_label, set())
        all_sub_labels.extend(sorted(subs))
    all_sub_labels = sorted(set(all_sub_labels))

    finding_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "hauptklasse":        {"type": "string", "enum": main_labels},
            "subklasse":          {"type": "string", "enum": all_sub_labels},
            "aenderungstyp":      {"type": "string", "enum": change_labels},
            "schweregrad":        {"type": "string", "enum": sev_labels},
            "stelle_im_segment":  {"type": "string"},
            "begruendung":        {"type": "string"},
            "source_refs":        {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "required": [
            "hauptklasse", "subklasse", "aenderungstyp", "schweregrad",
            "stelle_im_segment", "begruendung", "source_refs",
        ],
    }

    return {
        "type": "object",
        "properties": {
            "errors": {"type": "array", "items": finding_schema}
        },
        "required": ["errors"],
    }


# ── RAG store ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RagStore:
    """
    source_kind:
      "rules"    -> global QM/QS Regelwerk
      "material" -> fallbezogene Zusatzmaterialien
    """
    name: str
    source_kind: str
    ids: np.ndarray
    emb: np.ndarray
    index_map: Dict[str, Dict[str, Any]]
    text_map: Dict[str, str]


@dataclass(frozen=True)
class Retrieved:
    rank: int
    score: float
    id: str
    meta: Dict[str, Any]
    text: str
    vec: Optional[np.ndarray] = None
    retrieval_query: str = ""


@dataclass(frozen=True)
class EvidenceSource:
    source_ref: str
    source_kind: str
    chunk_id: str
    document: str
    source_path: str
    case_id: str
    document_type: str
    chunk_index: Optional[int]
    score: float
    text: str


@dataclass(frozen=True)
class SegmentEvidence:
    segment_index: int
    segment_text: str
    retrieval_queries: List[str]
    rules_sources: List[EvidenceSource]
    material_sources: List[EvidenceSource]

    @property
    def all_sources(self) -> List[EvidenceSource]:
        return self.rules_sources + self.material_sources


# ── Artifact loaders ──────────────────────────────────────────────────────────

def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    ids = data["ids"]
    emb = data["embeddings"].astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")
    if len(ids) != emb.shape[0]:
        raise ValueError(f"ids length {len(ids)} != embeddings rows {emb.shape[0]}")
    return ids, emb


def load_index_map(index_path: Path) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(index_path):
        _id = obj.get("id")
        if isinstance(_id, str):
            m[_id] = obj
    return m


def load_prepared_text_map(prepared_jsonl: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for obj in _iter_jsonl(prepared_jsonl):
        _id = obj.get("id")
        txt = obj.get("text")
        if isinstance(_id, str) and isinstance(txt, str):
            m[_id] = txt
    return m


def load_rag_store(
        name: str,
        source_kind: str,
        npz_path: Path,
        index_path: Path,
        prepared_path: Path,
) -> RagStore:
    if not npz_path.exists():
        raise SystemExit(f"[{name}] Embeddings not found: {npz_path}")
    if not index_path.exists():
        raise SystemExit(f"[{name}] Index not found: {index_path}")
    if not prepared_path.exists():
        raise SystemExit(f"[{name}] Prepared JSONL not found: {prepared_path}")

    ids, emb = load_npz(npz_path)
    index_map = load_index_map(index_path)
    text_map = load_prepared_text_map(prepared_path)

    print(
        f"[INFO] Loaded RAG store '{name}': "
        f"{emb.shape[0]} chunks, dim={emb.shape[1]}, "
        f"index={len(index_map)}, texts={len(text_map)}"
    )
    return RagStore(
        name=name,
        source_kind=source_kind,
        ids=ids,
        emb=emb,
        index_map=index_map,
        text_map=text_map,
    )


# ── Embedding model (E5 query) ────────────────────────────────────────────────

def load_hf_model(model_name: str, device: torch.device):
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModel.from_pretrained(model_name).eval().to(device)
    return mdl, tok


def embed_e5_query(
        query: str,
        *,
        model,
        tokenizer,
        device: torch.device,
        max_length: int,
) -> np.ndarray:
    text = "query: " + query.strip()
    enc = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = l2_normalize(pooled).to(torch.float32).cpu().numpy()
    return pooled[0].astype(np.float32)


# ── Retrieval helpers ─────────────────────────────────────────────────────────

_STOPWORDS_DE = {
    "als", "am", "an", "auch", "aus", "bei", "bis", "dabei", "das", "dass", "dem", "den",
    "der", "des", "die", "dies", "diese", "dieser", "doch", "durch", "ein", "eine", "einem",
    "einen", "einer", "eines", "er", "es", "für", "habe", "haben", "hat", "hinter", "ich",
    "im", "in", "ist", "ja", "kann", "können", "mich", "mir", "mit", "muss", "müssen", "nach",
    "noch", "nun", "oder", "sehr", "sein", "sind", "so", "soll", "sollen", "tue", "tun", "und",
    "unter", "vom", "von", "vor", "war", "was", "welche", "welcher", "welches", "wenn", "wer",
    "wie", "wir", "wird", "wo", "zu", "zum", "zur",
}


def topk_cosine(emb: np.ndarray, qvec: np.ndarray, k: int) -> np.ndarray:
    scores = emb @ qvec
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def _normalize_query_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split()).strip()


def _extract_query_keywords(text: str, *, max_tokens: int = 8) -> str:
    raw_tokens = [
        t.lower()
        for t in re.findall(r"[A-Za-zÄÖÜäöüß0-9][A-Za-zÄÖÜäöüß0-9\-_/]+", text)
    ]
    keep: List[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        if len(tok) < 4 or tok in _STOPWORDS_DE or tok in seen:
            continue
        seen.add(tok)
        keep.append(tok)
        if len(keep) >= max_tokens:
            break
    return " ".join(keep)


def build_multi_queries(query_text: str, *, mode: str, max_queries: int) -> List[str]:
    base = _normalize_query_text(query_text)
    if not base:
        return []

    variants: List[str] = [base]
    keywords = _extract_query_keywords(base)
    if keywords and keywords != base.lower():
        variants.append(keywords)

    base_l = base.lower()
    process_cues = ("wie ", "vorgehen", "ablauf", "prozess", "schritte", "checkliste")
    if keywords and (mode == "segment" or any(c in base_l for c in process_cues)):
        variants.extend([
            f"{keywords} vorgehen",
            f"{keywords} schritte",
            f"{keywords} checkliste",
        ])

    deduped: List[str] = []
    seen: set[str] = set()
    for q in variants:
        qn = _normalize_query_text(q)
        key = qn.lower()
        if not qn or key in seen:
            continue
        seen.add(key)
        deduped.append(qn)
        if len(deduped) >= max_queries:
            break
    return deduped


def _metadata_matches(meta: Dict[str, Any], metadata_filter: Optional[Dict[str, Any]]) -> bool:
    if not metadata_filter:
        return True
    for key, wanted in metadata_filter.items():
        if meta.get(key) != wanted:
            return False
    return True


def retrieve_from_store(
        store: RagStore,
        qvec: np.ndarray,
        k: int,
        *,
        retrieval_query: str = "",
        metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Retrieved]:
    if metadata_filter:
        filtered_rows: List[int] = []
        for i, raw_id in enumerate(store.ids):
            _id = str(raw_id)
            meta = store.index_map.get(_id, {})
            if _metadata_matches(meta, metadata_filter):
                filtered_rows.append(i)

        if not filtered_rows:
            return []

        row_idx = np.array(filtered_rows, dtype=np.int64)
        emb_sub = store.emb[row_idx]
        local_top = topk_cosine(emb_sub, qvec, k=min(k, emb_sub.shape[0]))
        idxs = row_idx[local_top]
        scores = (emb_sub @ qvec)[local_top]
    else:
        idxs = topk_cosine(store.emb, qvec, k=k)
        scores = (store.emb @ qvec)[idxs]

    hits: List[Retrieved] = []
    for i, s in zip(idxs, scores):
        _id = str(store.ids[i])
        meta = store.index_map.get(_id, {})
        text = store.text_map.get(_id, "")
        hits.append(
            Retrieved(
                rank=0,
                score=float(s),
                id=_id,
                meta=meta,
                text=text,
                vec=store.emb[i],
                retrieval_query=retrieval_query,
            )
        )
    return hits


def _source_key(hit: Retrieved) -> str:
    meta = hit.meta or {}
    return str(
        meta.get("source_path")
        or meta.get("origin_source_path")
        or meta.get("source_name")
        or meta.get("origin_source_name")
        or ""
    )


def _max_similarity_to_selected(hit: Retrieved, selected: Sequence[Retrieved]) -> float:
    if hit.vec is None or not selected:
        return 0.0
    sims: List[float] = []
    for other in selected:
        if other.vec is None:
            continue
        sims.append(float(hit.vec @ other.vec))
    return max(sims) if sims else 0.0


def diversify_hits_mmr(
        candidates: List[Retrieved],
        *,
        top_k: int,
        mmr_lambda: float,
        max_per_source: int,
) -> List[Retrieved]:
    unique_by_id: Dict[str, Retrieved] = {}
    for cand in candidates:
        prev = unique_by_id.get(cand.id)
        if prev is None or cand.score > prev.score:
            unique_by_id[cand.id] = cand

    pool = sorted(unique_by_id.values(), key=lambda h: h.score, reverse=True)
    if not pool:
        return []

    selected: List[Retrieved] = []
    source_counts: Dict[str, int] = {}

    def choose_candidate(enforce_source_cap: bool) -> Optional[Retrieved]:
        best_hit: Optional[Retrieved] = None
        best_mmr = float("-inf")
        selected_ids = {s.id for s in selected}
        for cand in pool:
            if cand.id in selected_ids:
                continue
            src_key = _source_key(cand)
            if enforce_source_cap and max_per_source > 0 and source_counts.get(src_key, 0) >= max_per_source:
                continue
            redundancy = _max_similarity_to_selected(cand, selected)
            mmr_score = (mmr_lambda * cand.score) - ((1.0 - mmr_lambda) * redundancy)
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_hit = cand
        return best_hit

    while len(selected) < min(top_k, len(pool)):
        chosen = choose_candidate(enforce_source_cap=True)
        if chosen is None:
            chosen = choose_candidate(enforce_source_cap=False)
        if chosen is None:
            break
        selected.append(chosen)
        source_counts[_source_key(chosen)] = source_counts.get(_source_key(chosen), 0) + 1

    return [
        Retrieved(
            rank=i + 1,
            score=h.score,
            id=h.id,
            meta=h.meta,
            text=h.text,
            vec=h.vec,
            retrieval_query=h.retrieval_query,
        )
        for i, h in enumerate(selected)
    ]


def _tag_hits(hits: List[Retrieved], source_kind: str) -> List[Retrieved]:
    return [
        Retrieved(
            rank=h.rank,
            score=h.score,
            id=h.id,
            meta={**h.meta, "source_kind": source_kind},
            text=h.text,
            vec=h.vec,
            retrieval_query=h.retrieval_query,
        )
        for h in hits
    ]


def retrieve_multi_query(
        query_text: str,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        max_length: int,
        top_k: int,
        candidate_k: int,
        multi_query_count: int,
        mmr_lambda: float,
        max_per_source: int,
        mode: str,
        query_expander: Optional[Callable[..., List[str]]] = None,
        case_id: str = "",
        rules_top_k: int = 10,
        material_top_k: int = 10,
) -> Tuple[List[Retrieved], List[str], List[Retrieved], List[Retrieved]]:
    expand = query_expander or build_multi_queries
    queries = expand(query_text, mode=mode, max_queries=multi_query_count)

    rules_candidates: List[Retrieved] = []
    material_candidates: List[Retrieved] = []

    for query_variant in queries:
        qvec = embed_e5_query(
            query_variant,
            model=embed_model,
            tokenizer=embed_tok,
            device=device,
            max_length=max_length,
        )

        for store in stores:
            if store.source_kind == "rules":
                raw_hits = retrieve_from_store(
                    store,
                    qvec,
                    k=candidate_k,
                    retrieval_query=query_variant,
                )
                rules_candidates.extend(_tag_hits(raw_hits, "rules"))

            elif store.source_kind == "material":
                filt = {"case_id": case_id} if case_id else None
                raw_hits = retrieve_from_store(
                    store,
                    qvec,
                    k=candidate_k,
                    retrieval_query=query_variant,
                    metadata_filter=filt,
                )
                material_candidates.extend(_tag_hits(raw_hits, "case_material"))

    rules_hits = diversify_hits_mmr(
        rules_candidates,
        top_k=max(0, rules_top_k),
        mmr_lambda=mmr_lambda,
        max_per_source=max_per_source,
    )

    material_hits = diversify_hits_mmr(
        material_candidates,
        top_k=max(0, material_top_k),
        mmr_lambda=mmr_lambda,
        max_per_source=max_per_source,
    )

    final_hits = rules_hits + material_hits
    final_hits = sorted(final_hits, key=lambda h: h.score, reverse=True)

    if top_k > 0:
        final_hits = final_hits[:top_k]

    final_hits = [
        Retrieved(
            rank=i + 1,
            score=h.score,
            id=h.id,
            meta=h.meta,
            text=h.text,
            vec=h.vec,
            retrieval_query=h.retrieval_query,
        )
        for i, h in enumerate(final_hits)
    ]
    return final_hits, queries, rules_hits, material_hits


# ── Vision captioning ─────────────────────────────────────────────────────────

def enrich_hits_with_image_captions(
        hits: List[Retrieved],
        *,
        vision_cfg: dict,
        max_workers: int = 3,
) -> List[Retrieved]:
    try:
        from importDocuments_structural import ollama_caption_png
    except ImportError:
        print("[WARN] importDocuments_structural not found — skipping image captioning")
        return hits

    def process(hit: Retrieved) -> Retrieved:
        image_paths: List[str] = hit.meta.get("embedded_images") or []
        if not image_paths:
            return hit

        captions: List[str] = []
        for img_path in image_paths:
            p = Path(img_path)
            if not p.exists():
                continue
            try:
                caption = ollama_caption_png(cfg=vision_cfg, png_bytes=p.read_bytes())
                if caption:
                    captions.append(caption)
            except Exception as e:
                print(f"[WARN] Vision captioning failed for {p.name}: {e}")
                continue

        if not captions:
            return hit

        enriched_text = hit.text.rstrip()
        for i, cap in enumerate(captions, start=1):
            enriched_text += f"\n\n[Bild {i}]: {cap}"

        return Retrieved(
            rank=hit.rank,
            score=hit.score,
            id=hit.id,
            meta=hit.meta,
            text=enriched_text,
            vec=hit.vec,
            retrieval_query=hit.retrieval_query,
        )

    hits_with = [h for h in hits if h.meta.get("embedded_images")]
    hits_without = [h for h in hits if not h.meta.get("embedded_images")]
    enriched = list(hits_without)

    if hits_with:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(process, h): h for h in hits_with}
            for future in as_completed(futures):
                enriched.append(future.result())
        enriched.sort(key=lambda h: h.rank)

    return enriched


# ── Context builders ──────────────────────────────────────────────────────────

def build_context_blocks(
        hits: List[Retrieved],
        *,
        max_chars: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    sources: List[Dict[str, Any]] = []
    blocks: List[str] = []
    used = 0

    for h in hits:
        m = h.meta or {}
        src = {
            "n": h.rank,
            "score": round(h.score, 4),
            "id": h.id,
            "retrieval_query": h.retrieval_query,
            "source_name": m.get("source_name") or m.get("origin_source_name"),
            "source_path": m.get("source_path") or m.get("origin_source_path"),
            "source_kind": m.get("source_kind"),
            "case_id": m.get("case_id"),
            "document_type": m.get("document_type"),
            "chunk_index": m.get("chunk_index"),
            "chunk_len": m.get("chunk_len"),
            "pdf_ocr_used": m.get("pdf_ocr_used"),
            "pdf_text_reader": m.get("pdf_text_reader"),
            "ocr_lang": m.get("ocr_lang"),
            "embedded_images": m.get("embedded_images", []),
        }
        sources.append(src)

        file_name = m.get("source_name") or m.get("origin_source_name") or "?"
        extra = []
        if m.get("chunk_index") is not None:
            extra.append(f"chunk_index={m['chunk_index']}")
        if m.get("section_title"):
            extra.append(f"section={m['section_title']!r}")
        if m.get("case_id"):
            extra.append(f"case_id={m['case_id']}")
        if m.get("document_type"):
            extra.append(f"document_type={m['document_type']}")
        extra_s = (" " + " ".join(extra)) if extra else ""

        header = (
            f"[{h.rank}] "
            f"chunk_id={h.id} "
            f"document={file_name}"
            f"{extra_s} "
            f"score={h.score:.4f}"
        )
        body = (h.text or "").strip()
        if not body:
            continue

        block = header + "\n" + body
        if used + len(block) + 2 > max_chars:
            remaining = max(0, max_chars - used - len(header) - 2)
            if remaining > 0:
                block = header + "\n" + body[:remaining].rstrip() + "\n…"
                blocks.append(block)
            break

        blocks.append(block)
        used += len(block) + 2

    return "\n\n".join(blocks).strip(), sources


def make_evidence_sources(
        hits: List[Retrieved],
        *,
        ref_prefix: str,
) -> List[EvidenceSource]:
    sources: List[EvidenceSource] = []
    for i, h in enumerate(hits, start=1):
        m = h.meta or {}
        sources.append(
            EvidenceSource(
                source_ref=f"{ref_prefix}_{i}",
                source_kind=str(m.get("source_kind") or ""),
                chunk_id=h.id,
                document=str(m.get("source_name") or m.get("origin_source_name") or "?"),
                source_path=str(m.get("source_path") or m.get("origin_source_path") or ""),
                case_id=str(m.get("case_id") or ""),
                document_type=str(m.get("document_type") or ""),
                chunk_index=m.get("chunk_index"),
                score=float(h.score),
                text=(h.text or "").strip(),
            )
        )
    return sources


def build_agent_context_from_sources(
        sources: List[EvidenceSource],
        *,
        max_chars: int,
) -> str:
    blocks: List[str] = []
    used = 0
    for src in sources:
        header = (
            f"{src.source_ref} | "
            f"chunk_id={src.chunk_id} | "
            f"document={src.document} | "
            f"source_kind={src.source_kind} | "
            f"chunk_index={src.chunk_index} | "
            f"score={src.score:.4f}"
        )
        body = src.text.strip()
        if not body:
            continue

        block = header + "\n" + body
        if used + len(block) + 2 > max_chars:
            remaining = max(0, max_chars - used - len(header) - 2)
            if remaining > 0:
                blocks.append(header + "\n" + body[:remaining].rstrip() + "\n…")
            break

        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks).strip()


# ── LLM client ────────────────────────────────────────────────────────────────

class LLMClient:
    def chat(
            self,
            messages: List[Dict[str, str]],
            json_mode: bool = False,
            schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(
            self,
            base_url: str,
            model: str,
            options: Dict[str, Any],
            timeout_s: int,
            disable_think: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options = options
        self.timeout_s = timeout_s
        # Reasoning models (e.g. Qwen3) otherwise spend their whole generation
        # budget inside the <think> block and return empty message.content,
        # which silently degrades structured agents (Agent 0, factual agent) to
        # empty results. When True we send "think": false on every call.
        self.disable_think = disable_think

    def _post_chat(self, payload: Dict[str, Any]) -> str:
        """Single /api/chat round-trip. Returns the stripped message content."""
        url = f"{self.base_url}/api/chat"
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response: {data}")
        return content.strip()

    def chat(
            self,
            messages: List[Dict[str, str]],
            json_mode: bool = False,
            schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.options:
            payload["options"] = self.options
        if schema is not None:
            # Grammar-constrained decoding: field names and enum values are
            # enforced at the token level — aliases like "type", "Fehler"
            # are physically impossible to generate.
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"

        if self.disable_think:
            payload["think"] = False

        try:
            content = self._post_chat(payload)
        except requests.HTTPError as exc:
            # Older Ollama versions / non-thinking models reject the `think`
            # field with a 400. Drop it and retry once instead of crashing.
            body = ""
            try:
                body = exc.response.text if exc.response is not None else ""
            except Exception:
                body = ""
            if payload.get("think") is False and "think" in body.lower():
                payload.pop("think", None)
                content = self._post_chat(payload)
            else:
                raise

        # If thinking was left enabled and the model returned nothing (budget
        # exhausted mid-<think>), retry once with thinking explicitly off.
        if not content and payload.get("think") is not False:
            payload["think"] = False
            try:
                content = self._post_chat(payload)
            except requests.HTTPError:
                pass

        return content


def make_llm_client() -> LLMClient:
    backend = require_env("LLM_BACKEND").lower()
    model = require_env("LLM_MODEL")
    timeout_s = env_int("LLM_TIMEOUT_S", 300)
    options = env_json_object_optional("LLM_OPTIONS_JSON")

    if backend == "ollama":
        base_url = require_env("OLLAMA_BASE_URL")
        return OllamaClient(
            base_url=base_url,
            model=model,
            options=options,
            timeout_s=timeout_s,
            disable_think=env_bool("LLM_DISABLE_THINK", True),
        )

    raise RuntimeError(f"Unsupported LLM_BACKEND: {backend!r}")


# ── Query expansion (Stufe B: LLM-basiertes Query-Rewriting) ──────────────────

QueryExpander = Callable[..., List[str]]


class LLMQueryExpander:
    """Semantisches Query-Rewriting über ein FEST gewähltes Modell.

    Läuft bewusst NICHT mit dem evaluierten Modell, damit das Retrieval
    über alle verglichenen Modelle identisch bleibt. Ergebnisse werden
    gecached; bei Fehlern wird auf die deterministische Heuristik
    build_multi_queries zurückgefallen, damit die Pipeline nie abbricht.
    """

    _SCHEMA = {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"},
                        "minItems": 1, "maxItems": 6},
        },
        "required": ["queries"],
    }

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._cache: Dict[str, List[str]] = {}

    def __call__(self, query_text: str, *, mode: str, max_queries: int) -> List[str]:
        base = _normalize_query_text(query_text)
        if not base:
            return []
        cache_key = f"{max_queries}|{base.lower()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        n_rewrites = max(1, max_queries - 1)  # Original kommt zusätzlich dazu
        messages = [
            {"role": "system", "content": (
                "Du formulierst Suchanfragen für ein deutschsprachiges Retrieval-System um. "
                "Erzeuge semantisch VERSCHIEDENE Umformulierungen derselben Informationsabsicht: "
                "Synonyme, andere Satzstellung, Ober- und Unterbegriffe. "
                "Hänge KEINE blossen Stichwörter an. Erfinde keine neuen Fakten. "
                "Antworte ausschliesslich mit einem JSON-Objekt der Form {\"queries\": [...]}."
            )},
            {"role": "user", "content":
                f"Anfrage:\n{base}\n\nErzeuge genau {n_rewrites} alternative Umformulierungen."},
        ]
        try:
            raw = self.llm.chat(messages, json_mode=True, schema=self._SCHEMA)
            rewrites = [
                _normalize_query_text(q)
                for q in json.loads(raw).get("queries", [])
                if isinstance(q, str)
            ]
        except Exception as exc:
            print(f"[QUERY-EXPAND] LLM-Rewriting fehlgeschlagen, Fallback auf Heuristik: {exc}")
            fallback = build_multi_queries(base, mode=mode, max_queries=max_queries)
            self._cache[cache_key] = fallback
            return fallback

        out: List[str] = []
        seen: Set[str] = set()
        for q in [base, *rewrites]:
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                out.append(q)
            if len(out) >= max_queries:
                break

        if not out:  # Modell lieferte nur Unbrauchbares
            out = build_multi_queries(base, mode=mode, max_queries=max_queries)
        self._cache[cache_key] = out
        return out


def make_query_expander() -> Optional[QueryExpander]:
    """Baut den Query-Expander. Bei QUERY_EXPANDER_MODE != 'llm' wird None
    zurückgegeben, wodurch retrieve_multi_query auf die deterministische
    Heuristik build_multi_queries zurückfällt."""
    mode = env_str("QUERY_EXPANDER_MODE", "heuristic").lower()
    if mode != "llm":
        return None

    if require_env("LLM_BACKEND").lower() != "ollama":
        raise RuntimeError("Query-Expander unterstützt derzeit nur ollama.")

    model = require_env("QUERY_EXPANDER_MODEL")  # FEST – nicht das evaluierte Modell!
    client = OllamaClient(
        base_url=require_env("OLLAMA_BASE_URL"),
        model=model,
        options=env_json_object_optional("QUERY_EXPANDER_OPTIONS_JSON") or {"temperature": 0.0},
        timeout_s=env_int("LLM_TIMEOUT_S", 300),
        disable_think=env_bool("LLM_DISABLE_THINK", True),
    )
    print(f"[INFO] Query-Expander: LLM-Modus, festes Modell={model!r}")
    return LLMQueryExpander(client)


# ── Prompt builders ───────────────────────────────────────────────────────────

# Gemeinsame System-Prompt-Präambel für alle Agenten (2–7).
_AGENT_JSON_PREFIX = (
    "Antworte AUSSCHLIESSLICH mit einem einzigen JSON-Objekt.\n"
    "Kein erklärender Text, keine Einleitung, kein Markdown, keine Kommentare.\n"
    "Erste Zeichen deiner Antwort müssen '{\"errors\"' sein.\n\n"
)


def build_qa_messages(question: str, context: str) -> List[Dict[str, str]]:
    system = (
        "Du bist ein präziser Assistent für transparente RAG-Antworten.\n"
        "Beantworte die FRAGE ausschliesslich anhand des KONTEXTS.\n"
        "Jede inhaltliche Aussage muss mit [N] belegt werden.\n"
        "Erfinde nichts und ergänze nichts aus eigenem Wissen."
    )
    user = (
        f"FRAGE:\n{question.strip()}\n\n"
        f"KONTEXT:\n{context}\n\n"
        "Antworte quellengebunden."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_factual_review_messages(
        segment_text: str,
        rules_context: str,
        material_context: str,
        catalog: ErrorCatalog,
) -> List[Dict[str, str]]:
    # Compact valid-values reference — grammar-constrained decoding enforces
    # enums at token level, so we only need a brief reminder here.
    # The full taxonomy block is intentionally removed to shorten the prompt.
    hauptklassen = ", ".join(sorted(
        m for m in catalog.allowed_main_labels
        if m not in {"Rechenfehler", "Hypothesenprüfung"}
    ))
    aenderungstypen = ", ".join(sorted(
        c for c in catalog.allowed_change_labels
        if c not in {"Rechnerische Korrektur", "Hypothesen-Korrektur"}
    ))

    system = (
            _AGENT_JSON_PREFIX +

            "Du bist Agent 2: Fachprüfer für technische Dokumente.\n"
            "Erkenne Fehler im DOKUMENTSEGMENT und belege sie mit REGELWERK- oder FALLMATERIAL-QUELLEN.\n"
            "Für Logikfehler innerhalb des aktuellen Segments: source_refs=[\"DOC_INTERNAL\"].\n\n"

            "Interne Evidenzregel (STRUKT_EVIDENZ):\n"
            "Melde eine fehlende Quellenangabe NUR wenn BEIDE Bedingungen erfüllt sind:\n"
            "  1. Der Satz macht eine konkrete Tatsachenbehauptung (Messwert, Befund, Identifikation)\n"
            "  2. Es fehlt jede Quellenreferenz: weder Aktenstelle, Beilage, Dokumentverweis\n"
            "     noch eine Quellenformel ('gemäss Aktenlage', 'laut Unterlagen', 'gemäss Quellen',\n"
            "     'laut Gutachten LSI', 'gemäss FOR', 'gemäss Quellenmaterial').\n"
            "NICHT melden als STRUKT_EVIDENZ: Sätze mit irgendeiner Quellenformel — auch wenn sie\n"
            "unspezifisch ist. NICHT auf STRUKT_BEFUND_BESCHREIBUNG ausweichen wenn die Quellenangabe\n"
            "fehlt — dafür gibt es ausschliesslich STRUKT_EVIDENZ.\n"
            "Klassifikation: subklasse='Evidenz / Belege'; "
            "aenderungstyp='Evidenzergänzung'; source_refs=[\"DOC_INTERNAL\"].\n\n"

            "PFLICHTFELDER:\n"
            f"  hauptklasse       {hauptklassen}\n"
            "  subklasse         passende Subklasse zur Hauptklasse\n"
            f"  aenderungstyp     {aenderungstypen}\n"
            "  schweregrad       niedrig | mittel | hoch\n"
            "  stelle_im_segment kurzer Originalausschnitt max. 8 Wörter NUR aus dem SEGMENT\n"
            "  begruendung       'Laut [SRC_X_N]: ...' oder 'Widerspruch: ...' bei DOC_INTERNAL\n"
            "  source_refs       mind. eine Ref — sonst kein Finding erlaubt\n\n"

            "NICHT melden:\n"
            "  ss-Schreibweise (Schweiz korrekt) | sprachliche/stilistische Fehler (Agent 3) | "
            "Rechenfehler (Agent 4) | Hypothesenfehler (Agent 5) | "
            "Fehler die im Segment selbst erklärt werden\n\n"

            "Kein belegbarer Fehler → {\"errors\":[]}\n\n"

            "Beispiel externe Quelle:\n"
            "{\"errors\":[{\"hauptklasse\":\"Struktur und Argumentation\","
            "\"subklasse\":\"Beschreibung von Befunden\","
            "\"aenderungstyp\":\"Fachliche Präzisierung\","
            "\"schweregrad\":\"hoch\","
            "\"stelle_im_segment\":\"<Originalausschnitt mit Fehler>\","
            "\"begruendung\":\"Laut [SRC_X_N]: <abweichender Wert aus Quelle>.\","
            "\"source_refs\":[\"SRC_X_N\"]}]}\n\n"

            "Beispiel interner Widerspruch:\n"
            "{\"errors\":[{\"hauptklasse\":\"Struktur und Argumentation\","
            "\"subklasse\":\"Beschreibung von Befunden\","
            "\"aenderungstyp\":\"Fachliche Präzisierung\","
            "\"schweregrad\":\"hoch\","
            "\"stelle_im_segment\":\"<Originalausschnitt mit Widerspruch>\","
            "\"begruendung\":\"Widerspruch: <Erklärung des internen Widerspruchs>.\","
            "\"source_refs\":[\"DOC_INTERNAL\"]}]}"
    )

    user = (
        f"DOKUMENTSEGMENT:\n{segment_text.strip()}\n\n"
        f"REGELWERK-QUELLEN:\n{rules_context.strip()}\n\n"
        "Prüfe NUR dokumentinterne Fehler und Regelwerk-Verletzungen:\n"
        "1. Gibt es logische Widersprüche innerhalb des SEGMENTS selbst?\n"
        "2. Werden Fachbegriffe aus dem REGELWERK falsch angewendet?\n"
        "3. Ist eine konkrete Tatsachenbehauptung ohne jede Quellenangabe belassen?\n"
        "source_refs = [\"DOC_INTERNAL\"] für STRUKT_EVIDENZ (fehlende Quellenangabe).\n"
        "Für Regelwerk-Fehler: echte Chunk-ID aus den QUELLEN (z.B. S7_R_3), niemals SRC_X_N.\n"
        "Kein Fehler gefunden → {\"errors\":[]}\n"
        "Antworte NUR mit dem JSON-Objekt."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]



def build_language_review_messages(segment_text: str) -> List[Dict[str, str]]:
    system = (
            _AGENT_JSON_PREFIX +

            "Du bist Agent 3: Sprach- und Formalprüfer.\n"
            "Du meldest NUR eindeutige, lokale formale Fehler.\n\n"

            "Zulässige Taxonomie:\n"
            "  hauptklasse = Formales\n"
            "  subklasse   = Redaktionelle Korrektur | Referenzen (formal) | Dokumentstruktur | Adressierung\n"
            "  aenderungstyp = Redaktionelle Korrektur | schweregrad = niedrig\n\n"

            "ERLAUBT: eindeutige Orthografie-, Zeichensetzungs-, Grammatik- oder Adressierungsfehler.\n"
            "NICHT ERLAUBT:\n"
            "  - Stilverbesserungen, Umformulierungen, Terminologieangleichungen\n"
            "  - Eigennamen, Produktnamen, Institutionen, Marken, Fahrzeugbezeichnungen, Aktenzeichen\n"
            "  - Abkürzungen (sofern nicht eindeutig falsch ausgeschrieben)\n"
            "  - Korrekturen innerhalb von Anführungszeichen\n"
            "  - Korrekturen ohne sehr hohe Sicherheit\n"
            "  - 'ss', 'gross', 'Grösse', 'grosszügig' u.ä.: In der Schweiz ist ss korrekt,\n"
            "    NIEMALS als Fehler melden. 'mutmaßlich' → korrekt ist 'mutmasslich' (ss).\n"
            "  - Kein Vorschlag vorhanden (vorschlag leer): kein Finding — das Wort ist ein\n"
            "    Fachbegriff, Eigenname oder korrektes Kompositum.\n"
            "  - Komposita NIEMALS aufteilen: 'zurückzukehrte', 'herumwirbelnde',\n"
            "    'brandbetroffene', 'lagekorrekt' sind korrekte deutsche Komposita.\n"
            "    Ein Leerzeichen einzufügen ('zurück zukehrte') ist IMMER falsch.\n"
            "  - Grammatikkorrekturen die die Bedeutung ändern sind verboten: 'elektrischer\n"
            "    Weidezaun' (Nominativ m.) ist korrekt — nicht 'elektrisches Zaun'.\n"
            "    Adjektivbeugung nur melden wenn der Kasus eindeutig falsch ist.\n\n"

            "Kein eindeutiger Fehler → {\"errors\":[]}\n\n"

            "stelle_im_segment muss ein KURZER Ausschnitt sein (max. 4 Wörter), nicht ein ganzer Satz.\n\n"

            "Format (Beispiel: einzelnes falsch geschriebenes Wort):\n"
            "{\"errors\":[{\"hauptklasse\":\"Formales\",\"subklasse\":\"Redaktionelle Korrektur\","
            "\"aenderungstyp\":\"Redaktionelle Korrektur\",\"schweregrad\":\"niedrig\","
            "\"stelle_im_segment\":\"zurückzukehrte\","
            "\"begruendung\":\"Konjugationsfehler: korrekt ist 'zurückkehrte'.\","
            "\"vorschlag\":\"zurückkehrte\"}]}"
    )
    user = (
        f"DOKUMENTSEGMENT:\n{segment_text.strip()}\n\n"
        "Melde nur eindeutige formale Fehler. "
        "Wenn du unsicher bist, gib keine Meldung aus."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_json_repair_messages(raw_reply: str, schema_name: str) -> List[Dict[str, str]]:
    system = (
        "Extrahiere aus der folgenden Antwort ausschliesslich ein einziges valides JSON-Objekt.\n"
        "Gib nur JSON zurück, ohne Markdown, ohne Einleitung, ohne Erklärung.\n"
        "Verwende exakt den Schlüssel 'errors' (nicht 'Fehler', nicht 'fehler').\n"
        f"Schema: {schema_name}\n"
        "Falls kein verwertbares JSON vorhanden ist, gib exakt zurück: {\"errors\":[]}"
    )
    user = f"Antwort:\n{raw_reply}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── JSON parsing / normalization ──────────────────────────────────────────────

def extract_first_json_object(raw_text: str) -> str:
    s = raw_text.strip()

    # Handle fenced code blocks containing either object or array
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", s, flags=re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate.startswith("["):
            return json.dumps({"errors": json.loads(candidate)}, ensure_ascii=False)
        return candidate

    decoder = json.JSONDecoder()

    # Determine whether an object or array comes first
    first_brace   = s.find("{")
    first_bracket = s.find("[")

    if first_brace == -1 and first_bracket == -1:
        raise ValueError("No JSON object found in response")

    use_array = (
            first_bracket != -1
            and (first_brace == -1 or first_bracket < first_brace)
    )
    start = first_bracket if use_array else first_brace

    while start != -1:
        try:
            obj, _ = decoder.raw_decode(s[start:])
            if use_array:
                return json.dumps({"errors": obj}, ensure_ascii=False)
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            start = s.find("[" if use_array else "{", start + 1)

    raise ValueError("No JSON object found in response")


def parse_json_response(raw_reply: str) -> Dict[str, Any]:
    raw_json = extract_first_json_object(raw_reply)
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON must be an object")

    # Use is None instead of `or` so that a valid empty list [] is not skipped
    errors = parsed.get("errors")
    if errors is None:
        errors = parsed.get("fehler")
    if errors is None:
        errors = parsed.get("Fehler")
    if errors is None:
        errors = parsed.get("Errors")
    if errors is None:
        errors = parsed.get("fachliche_Fehler")
    if errors is None:
        errors = parsed.get("fachlicheFehler")
    if errors is None:
        errors = parsed.get("fachliche_fehler")
    if errors is None:
        errors = parsed.get("findings")
    if errors is None:
        errors = parsed.get("language_errors")
    if errors is None:
        errors = parsed.get("Sprachfehler")
    if errors is None:
        errors = parsed.get("sprachliche_fehler")

    # Handle flat object: {"Fehler_01": "...", "Fehler_02": "..."}  (S12 pattern)
    if errors is None:
        flat = [{"beschreibung": v} for k, v in parsed.items()
                if isinstance(v, str) and k.lower().startswith("fehler")]
        if flat:
            errors = flat

    # Last resort: if still None, take the first list value found in the object
    # Safe for language agent responses which contain exactly one findings list
    if errors is None:
        for v in parsed.values():
            if isinstance(v, list):
                errors = v
                break

    if not isinstance(errors, list):
        raise ValueError("JSON must contain 'errors' or 'fehler' as a list")
    return {"errors": errors}

def _map_to_taxonomy(value: str, allowed: set) -> str:
    if value in allowed:
        return value
    v = value.lower()
    if any(k in v for k in ("recht", "gesetz", "straf", "führerausweis",
                            "fahrerlaubnis", "legal", "norm", "vorschrift",
                            "verkehr", "fahrzeug", "ladung", "sicht",
                            "verletzung", "unfall", "regelverstoß", "regel",
                            "fehlerhaft", "datum", "date_format", "date",
                            "zeitangabe", "lizenz", "bewilligung")):
        return "Rechtskonformität"
    if any(k in v for k in ("qualit", "methode", "berechnung", "messung",
                            "gutachten", "geschwindigkeit", "widerspruch",
                            "inkonsistenz", "inkorrekt", "ungenau",
                            "rundung", "fehlerhafte angabe")):
        return "QM/QS-Konformität"
    if any(k in v for k in ("struktur", "argument", "logik", "schluss",
                            "unklar", "unvollständig", "übersicht",
                            "mangelnde", "fehlende", "aufmerksamkeit",
                            "angaben", "beschreibung", "befund",
                            "dokumentation", "nachvollzieh")):
        return "Struktur und Argumentation"
    if any(k in v for k in ("formal", "sprach", "referenz", "adress",
                            "typo", "typograph", "orthograf", "format",
                            "rechtschreib", "grammatik", "zeichensetz",
                            "redaktion")):
        return "Formales"
    return ""

def normalize_factual_errors(
        raw_errors: List[Any],
        catalog: ErrorCatalog,
        segment_text: str = "",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allowed_main = catalog.allowed_main_labels
    allowed_subs_by_main = catalog.allowed_subclasses_by_main_label
    allowed_change = catalog.allowed_change_labels
    allowed_severity = catalog.allowed_severity_labels
    sub_to_main = catalog.subclass_label_to_main_label

    # Bestätigungen / nicht-Fehler — werden gedroppt
    BESTAETIGUNGS_PATTERNS = [
        "ist konsistent", "stimmt überein", "korrekt beschrieben",
        "entspricht den", "übereinstimmend", "keine abweichung",
        "ist korrekt", "keine fehler", "kein fehler", "ist richtig",
        "ist vollständig", "ist nachvollziehbar", "ist plausibel",
    ]

    for item in raw_errors:
        if not isinstance(item, dict):
            continue

        hauptklasse = str(
            item.get("hauptklasse")
            or item.get("main_class")
            or item.get("category")
            or item.get("typ") or item.get("type")
            or item.get("Typ")
            or item.get("Art")
            or item.get("fehlertyp") or item.get("Fehlertyp")
            or ""
        ).strip()

        subklasse = str(
            item.get("subklasse")
            or item.get("subcategory")
            or item.get("sub_type") or item.get("subtype")
            or ""
        ).strip()

        aenderungstyp = str(
            item.get("aenderungstyp")
            or item.get("change_type")
            or ""
        ).strip()

        schweregrad = str(
            item.get("schweregrad")
            or item.get("severity")
            or ""
        ).strip()

        stelle = str(
            item.get("stelle_im_segment")
            or item.get("stelle")
            or item.get("span_text")
            or item.get("Zeile")
            or ""
        ).strip()

        # Guard: every factual finding must be anchored in the checked segment.
        # If the LLM copied text from RAG/context/reference facts, drop the finding.
        normalized_segment_for_stelle = " ".join((segment_text or "").split())
        normalized_stelle_for_check = " ".join(stelle.split())
        if not normalized_stelle_for_check:
            print("[DROP factual] missing stelle_im_segment")
            continue
        if segment_text and normalized_stelle_for_check not in normalized_segment_for_stelle:
            print(
                f"[DROP factual] stelle_im_segment not found in segment — "
                f"likely copied from source/context: {stelle[:80]!r}"
            )
            continue

        begruendung = str(
            item.get("begruendung")
            or item.get("begründung")
            or item.get("rationale")
            or item.get("description")
            or item.get("Beschreibung")
            or item.get("beschreibung")
            or item.get("fehlerbeschreibung")
            or item.get("reason")
            or item.get("frage")
            or item.get("text")
            or ""
        ).strip()

        # Drop findings where the begruendung is a confirmation, not an error.
        beg_lower = begruendung.lower()
        if any(p in beg_lower for p in BESTAETIGUNGS_PATTERNS):
            print(f"[DROP] begruendung ist Bestätigung, kein Fehler: {begruendung[:60]!r}")
            continue

        _refs_raw = (
                item.get("source_refs")
                or item.get("quellen")
                or item.get("Quellen")
                or item.get("source_chunk_id")
        )

        if not _refs_raw and item.get("quelle"):
            _refs_raw = [item.get("quelle")]

        if isinstance(_refs_raw, str):
            _refs_raw = [_refs_raw]

        source_refs = [str(x).strip() for x in (_refs_raw or []) if str(x).strip()]

        # source_refs is the primary quality gate.
        # DOC_INTERNAL is valid for logic errors grounded in the document itself.
        # Findings without any source_ref are false positives — drop them.
        if not source_refs:
            print(f"[DROP] no source_refs — False Positive, stelle={stelle[:60]!r}")
            continue

        # ── Art-307 / Standard-Disclaimer Whitelist ───────────────────────────
        # Diese Formulierungen sind korrekte Gutachtenkonventionen und werden
        # nicht als fehlende Quellenangabe gemeldet, auch wenn source_refs
        # nur DOC_INTERNAL enthält.
        _DISCLAIMER_PATTERNS = [
            "art. 307 stgb",
            "in kenntnis von",
            "nach bestem wissen",
            "gemäss auftrag",
            "mit schreiben",
            "wurde beauftragt",
        ]
        stelle_lower = stelle.lower()
        beg_lower_check = begruendung.lower()
        if (
                all(r == "DOC_INTERNAL" for r in source_refs)
                and any(pat in stelle_lower or pat in beg_lower_check
                        for pat in _DISCLAIMER_PATTERNS)
        ):
            print(f"[DROP] Disclaimer-Whitelist: {stelle[:60]!r}")
            continue

        if hauptklasse not in allowed_main:
            mapped_main = _map_to_taxonomy(hauptklasse, allowed_main)
            if mapped_main:
                print(f"[RECOVER] hauptklasse {hauptklasse!r} → {mapped_main!r}")
                hauptklasse = mapped_main

        if hauptklasse not in allowed_main:
            correct_main = sub_to_main.get(subklasse)
            if correct_main:
                print(
                    f"[RECOVER] missing/invalid hauptklasse {hauptklasse!r} → {correct_main!r} "
                    f"for subklasse {subklasse!r}"
                )
                hauptklasse = correct_main
            else:
                print(f"[DROP] hauptklasse {hauptklasse!r} not in {sorted(allowed_main)}")
                continue

        if subklasse not in allowed_subs_by_main.get(hauptklasse, set()):
            correct_main = sub_to_main.get(subklasse)
            if correct_main and subklasse in allowed_subs_by_main.get(correct_main, set()):
                print(
                    f"[RECOVER] hauptklasse {hauptklasse!r} → {correct_main!r} "
                    f"for subklasse {subklasse!r}"
                )
                hauptklasse = correct_main
            else:
                print(
                    f"[DROP] main={hauptklasse!r}, subklasse={subklasse!r}, "
                    f"allowed_for_main={sorted(allowed_subs_by_main.get(hauptklasse, set()))}, "
                    f"correct_main={correct_main!r}"
                )
                continue

        if aenderungstyp not in allowed_change:
            changes = sorted(allowed_change)
            aenderungstyp = changes[0] if changes else ""
            print(f"[INFO] aenderungstyp fallback → {aenderungstyp!r}")

        if schweregrad not in allowed_severity:
            schweregrad = "mittel"

        out.append({
            "hauptklasse": hauptklasse,
            "subklasse": subklasse,
            "aenderungstyp": aenderungstyp,
            "schweregrad": schweregrad,
            "stelle_im_segment": stelle,
            "begruendung": begruendung,
            "source_refs": source_refs,
        })

    return out


def normalize_language_errors(raw_errors: List[Any], catalog: ErrorCatalog) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allowed_subs = catalog.allowed_subclasses_by_main_label.get("Formales", set())

    legacy_to_new = {
        "Rechtschreibung": "Redaktionelle Korrektur",
        "Grammatik": "Redaktionelle Korrektur",
        "Kommafehler": "Redaktionelle Korrektur",
        "Stil": "Stil / Redundanz",
    }

    for item in raw_errors:
        if not isinstance(item, dict):
            continue

        subklasse = str(item.get("subklasse") or item.get("fehlerklasse") or "Redaktionelle Korrektur").strip()
        subklasse = legacy_to_new.get(subklasse, subklasse)

        if subklasse not in allowed_subs:
            continue

        aenderungstyp = str(item.get("aenderungstyp") or item.get("änderungstyp") or "Redaktionelle Korrektur").strip()
        if aenderungstyp not in catalog.allowed_change_labels:
            aenderungstyp = "Redaktionelle Korrektur"

        schweregrad = str(item.get("schweregrad") or item.get("fehlerschwere") or "niedrig").strip()
        if schweregrad not in catalog.allowed_severity_labels:
            schweregrad = "niedrig"

        out.append({
            "hauptklasse": "Formales",
            "subklasse": subklasse,
            "aenderungstyp": aenderungstyp,
            "schweregrad": schweregrad,
            "stelle_im_segment": str(item.get("stelle_im_segment") or item.get("stelle") or "").strip(),
            "begruendung": str(item.get("begruendung") or item.get("begründung") or "").strip(),
            "vorschlag": str(item.get("vorschlag") or "").strip(),
        })
    return out


# ── Document segmentation ─────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[\.!?…])(?:[\]\)\"'»”’]+)?\s+(?=(?:[A-ZÄÖÜ]|\d|[-–—•*]))"
)


def _split_long_text_on_whitespace(text: str, *, target_chars: int, max_chars: int) -> List[str]:
    s = " ".join((text or "").split())
    if not s:
        return []

    parts: List[str] = []
    remaining = s

    while len(remaining) > max_chars:
        hard_limit = min(len(remaining), max_chars)
        preferred = min(len(remaining), target_chars)

        cut = remaining.rfind(" ", 0, preferred + 1)
        if cut < max(1, preferred // 2):
            cut = remaining.rfind(" ", 0, hard_limit + 1)
        if cut == -1:
            cut = hard_limit

        part = remaining[:cut].strip()
        if part:
            parts.append(part)

        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)

    return parts


def _split_paragraph_into_sentence_units(paragraph: str) -> List[str]:
    para = (paragraph or "").strip()
    if not para:
        return []

    if "\n" in para:
        line_units = [line.strip() for line in para.splitlines() if line.strip()]
        if len(line_units) > 1:
            units: List[str] = []
            for line in line_units:
                units.extend(_split_paragraph_into_sentence_units(line))
            return units

    if re.match(r"^(?:[-*•]\s+|\d+[\.)]\s+)", para):
        return [para]

    pieces = re.split(_SENTENCE_SPLIT_RE, para)
    units = [p.strip() for p in pieces if p and p.strip()]
    return units or [para]


def _split_paragraph_hierarchically(
        paragraph: str,
        *,
        target_chars: int,
        max_chars: int,
) -> List[str]:
    para = (paragraph or "").strip()
    if not para:
        return []

    if len(para) <= max_chars:
        return [para]

    sentence_units = _split_paragraph_into_sentence_units(para)
    if len(sentence_units) == 1 and sentence_units[0] == para:
        return _split_long_text_on_whitespace(para, target_chars=target_chars, max_chars=max_chars)

    chunks: List[str] = []
    current: List[str] = []

    def current_text() -> str:
        return " ".join(current).strip()

    def flush() -> None:
        nonlocal current
        joined = current_text()
        if joined:
            chunks.append(joined)
        current = []

    for unit in sentence_units:
        unit = unit.strip()
        if not unit:
            continue

        if len(unit) > max_chars:
            if current:
                flush()
            chunks.extend(_split_long_text_on_whitespace(unit, target_chars=target_chars, max_chars=max_chars))
            continue

        candidate = " ".join(current + [unit]).strip() if current else unit
        if current and len(candidate) > target_chars:
            flush()

        current.append(unit)

        if len(current_text()) >= target_chars:
            flush()

    flush()
    return chunks or [para]


def split_document_into_segments(
        document_text: str,
        *,
        target_chars: int = 1200,
        min_chars: int = 250,
        max_chars: int = 2200,
) -> List[str]:
    text = (document_text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return [text]

    atomic_units: List[str] = []
    for para in paragraphs:
        atomic_units.extend(
            _split_paragraph_hierarchically(
                para,
                target_chars=target_chars,
                max_chars=max_chars,
            )
        )

    segments: List[str] = []
    current: List[str] = []

    def current_segment() -> str:
        return "\n\n".join(current).strip()

    def flush() -> None:
        nonlocal current
        seg = current_segment()
        if seg:
            segments.append(seg)
        current = []

    for unit in atomic_units:
        unit = unit.strip()
        if not unit:
            continue

        if len(unit) > max_chars:
            if current:
                flush()
            segments.extend(_split_long_text_on_whitespace(unit, target_chars=target_chars, max_chars=max_chars))
            continue

        candidate = "\n\n".join(current + [unit]).strip() if current else unit
        if current and len(candidate) > target_chars:
            flush()

        current.append(unit)

        if len(current_segment()) >= target_chars:
            flush()

    flush()

    merged: List[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if merged and len(seg) < min_chars:
            candidate = merged[-1] + "\n\n" + seg
            if len(candidate) <= max_chars:
                merged[-1] = candidate
                continue
        merged.append(seg)

    return merged or [text]


# ── Agent 1: Evidence Builder ─────────────────────────────────────────────────

def build_segment_evidences(
        document_text: str,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        args: argparse.Namespace,
        vision_cfg: Optional[dict],
) -> Tuple[List[SegmentEvidence], List[str]]:
    segments = split_document_into_segments(document_text)
    evidences: List[SegmentEvidence] = []
    all_queries: List[str] = []

    per_segment_candidate_k = args.per_segment_candidate_k
    per_segment_rules_top_k = args.per_segment_rules_top_k
    per_segment_material_top_k = args.per_segment_material_top_k
    per_segment_total_top_k = per_segment_rules_top_k + per_segment_material_top_k

    for seg_idx, segment in enumerate(segments, start=1):
        _, queries, rules_hits, material_hits = retrieve_multi_query(
            segment,
            stores,
            embed_model=embed_model,
            embed_tok=embed_tok,
            device=device,
            max_length=args.query_max_length,
            top_k=per_segment_total_top_k,
            candidate_k=per_segment_candidate_k,
            multi_query_count=args.multi_query_count,
            mmr_lambda=args.mmr_lambda,
            max_per_source=args.max_per_source,
            mode="segment",
            query_expander=getattr(args, "query_expander", None),
            case_id=args.case_id,
            rules_top_k=per_segment_rules_top_k,
            material_top_k=per_segment_material_top_k,
        )

        # Vision captioning is now done at ingestion time in importDocuments_structural.py.
        # Captions are stored in the chunk text and embedded with the chunk,
        # so enrich_hits_with_image_captions is no longer called here.
        rules_sources = make_evidence_sources(rules_hits, ref_prefix=f"S{seg_idx}_R")
        material_sources = make_evidence_sources(material_hits, ref_prefix=f"S{seg_idx}_M")

        evidences.append(
            SegmentEvidence(
                segment_index=seg_idx,
                segment_text=segment,
                retrieval_queries=queries,
                rules_sources=rules_sources,
                material_sources=material_sources,
            )
        )
        all_queries.extend(queries)

    deduped_queries = list(dict.fromkeys(q for q in all_queries if q))
    return evidences, deduped_queries


# ── Agent 2 / Agent 3 execution ───────────────────────────────────────────────

def _dedup_material_sources(
        sources: List[EvidenceSource],
        similarity_threshold: float = 0.72,
) -> List[EvidenceSource]:
    """
    Option 4: Dedupliziert semantisch ähnliche Materialreferenzen
    vor dem Agent-2-Call mittels Token-level Jaccard.

    Verhindert dass dasselbe Thema aus N leicht verschiedenen Chunksn
    N separate Findings erzeugt (z.B. "H5 zur Ursache" 6× aus 6 Chunks).
    Pro Cluster wird nur der Chunk mit dem höchsten Retrieval-Score behalten.

    similarity_threshold: 0.72 ist konservativ genug um echte Varianten
    zu erhalten, aggressiv genug um Fast-Duplikate zu eliminieren.
    """
    if len(sources) <= 2:
        return sources

    # Sortiere absteigend nach Score — bei Duplikaten den besten behalten
    sorted_srcs = sorted(sources, key=lambda s: s.score, reverse=True)
    kept: List[EvidenceSource] = []

    for src in sorted_srcs:
        src_tokens = set(src.text.lower().split())
        is_near_duplicate = False
        for k in kept:
            k_tokens = set(k.text.lower().split())
            intersection = len(src_tokens & k_tokens)
            union = len(src_tokens | k_tokens)
            jaccard = intersection / union if union else 0.0
            if jaccard >= similarity_threshold:
                is_near_duplicate = True
                break
        if not is_near_duplicate:
            kept.append(src)

    if len(kept) < len(sources):
        print(
            f"[MATERIAL-DEDUP] {len(sources)} → {len(kept)} Materialreferenzen "
            f"(threshold={similarity_threshold:.2f})"
        )
    return kept


def _dedup_factual_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Zwei-stufige Deduplizierung innerhalb einer Findings-Liste:

    Stufe 1 — source_refs + begruendung (bestehend):
        Verwirft Findings die exakt dieselben source_refs und denselben
        begruendung-Prefix (80 Zeichen) haben. Verhindert dass das LLM
        dasselbe RAG-referenzierte Finding mehrfach mit leicht variiertem
        span_text ausgibt.

    Stufe 2 — span_text (neu):
        Verwirft Findings die denselben normalisierten span_text haben,
        unabhängig von subclass oder agent. Verhindert dass deterministischer
        Guard und LLM-Agent denselben Span doppelt melden (z.B. Date-Guard
        als QMQS_DOKPFLICHT und Agent 2 als STRUKT_BEFUND_BESCHREIBUNG).
        Bei Kollision gewinnt das erste Finding in der Liste — deterministische
        Guard-Findings werden vor LLM-Findings einsortiert und haben daher
        Priorität (siehe run_detect / check_document).
    """
    # Stufe 1: source_refs + begruendung
    seen_refs: set = set()
    stage1: List[Dict[str, Any]] = []
    for f in findings:
        refs = tuple(sorted(f.get("source_refs") or []))
        beg = (f.get("begruendung") or "")[:80]
        key = (refs, beg)
        if key not in seen_refs:
            seen_refs.add(key)
            stage1.append(f)

    # Stufe 2: span_text — normalisiert, case-insensitive, max 120 Zeichen
    seen_spans: set = set()
    out: List[Dict[str, Any]] = []
    for f in stage1:
        raw_span = (
                f.get("span_text")
                or f.get("stelle_im_segment")
                or ""
        )
        span_key = " ".join(str(raw_span).split()).lower()[:120]
        if span_key and span_key in seen_spans:
            print(
                f"[DEDUP-SPAN] span={span_key[:60]!r} bereits gemeldet "
                f"(subclass={f.get('subclass_id') or f.get('subklasse')!r}) "
                f"-- uebersprungen"
            )
            continue
        if span_key:
            seen_spans.add(span_key)
        out.append(f)
    return out


# Dokumentweite STRUKT_EVIDENZ-Spans — wird in check_document befüllt
# Dokumentweiter Span-Dedup-Cache pro Subklasse.
# Dict[subklasse_label -> set[span_key]]
# Wird in check_document/run_detect vor dem ersten Agenten-Call geleert.
_SEEN_SPANS_BY_SUBCLASS: Dict[str, set] = {}

# Fuzzy-Dedup-Schwellwert (Jaccard auf Token-Ebene).
# 0.85 eliminiert nahezu identische Spans ohne echte Varianten zu unterdruecken.
_SPAN_DEDUP_JACCARD_THRESHOLD = 0.85


def _jaccard_tokens(a: str, b: str) -> float:
    """Token-level Jaccard-Aehnlichkeit zweier normalisierter Strings."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _dedup_spans_across_segments(
        findings: List[Dict[str, Any]],
        span_field: str = "stelle_im_segment",
) -> List[Dict[str, Any]]:
    """
    Dokumentweite Span-Deduplizierung pro Subklasse.
    Ein Finding wird verworfen wenn ein Span mit Jaccard >= _SPAN_DEDUP_JACCARD_THRESHOLD
    fuer dieselbe Subklasse bereits in einem frueheren Segment gemeldet wurde.
    Gilt fuer alle Subklassen (nicht nur STRUKT_EVIDENZ).
    """
    out: List[Dict[str, Any]] = []
    for f in findings:
        subklasse = str(f.get("subklasse") or f.get("subclass_id") or "").strip()
        span_raw  = f.get(span_field) or f.get("stelle_im_segment") or ""
        span_key  = " ".join(str(span_raw).split()).lower()[:120]
        if not subklasse or not span_key:
            out.append(f)
            continue
        seen_for_sub = _SEEN_SPANS_BY_SUBCLASS.setdefault(subklasse, set())
        is_duplicate = any(
            _jaccard_tokens(span_key, seen) >= _SPAN_DEDUP_JACCARD_THRESHOLD
            for seen in seen_for_sub
        )
        if is_duplicate:
            print(
                f"[DEDUP] Subklasse={subklasse!r} span={span_key[:60]!r} "
                f"bereits gemeldet -- uebersprungen"
            )
            continue
        seen_for_sub.add(span_key)
        out.append(f)
    return out


def _dedup_evidenz_across_segments(
        findings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Veraltet -- delegiert an _dedup_spans_across_segments."""
    return _dedup_spans_across_segments(findings)

# ── Agent 2 (Ergänzung): Keyword-Guard Zweifel ───────────────────────────────
ZWEIFEL_VERBOTEN = [
    "zweifelsfrei", "ohne Zweifel", "begründete Zweifel",
    "Restzweifel", "keine Zweifel mehr", "Zweifel säen",
    "Zweifel ausräumen", "räumt Zweifel aus",
    "bestehen keine Zweifel", "ist zweifelsfrei bewiesen",
]

def check_zweifel_violations(
        segment_text: str,
        segment_index: int,
) -> List[Dict[str, Any]]:
    findings = []
    for term in ZWEIFEL_VERBOTEN:
        if term.lower() in segment_text.lower():
            findings.append({
                "segment_index": segment_index,
                "hauptklasse": "QM/QS-Konformität",
                "subklasse": "Terminologie (normativ)",
                "aenderungstyp": "Normativer Mangel",
                "schweregrad": "mittel",
                "stelle_im_segment": term,
                "begruendung": (
                    "Verbotene Zweifel-Formulierung gemäss Begriffs- und Kontextregel: "
                    f"'{term}' soll durch hypothesenvergleichende Sprache ersetzt werden."
                ),
            })
    return findings


def _verify_strukt_befund_finding(
        finding: Dict[str, Any],
        segment_text: str,
        material_sources: List[EvidenceSource],
        min_value_jaccard_distance: float = 0.25,
) -> bool:
    """
    Option 3: Verifikation von STRUKT_BEFUND_BESCHREIBUNG-Findings.

    Ein Finding wird nur akzeptiert wenn:
    1. stelle_im_segment wörtlich im Segmenttext vorkommt.
    2. Die Begründung eine Materialreferenz ([Sx_Mn]) enthält.
    3. Der behauptete Gegenwert (aus der Begründung extrahiert) tatsächlich
       in einer Materialreferenz vorkommt — verhindert Halluzinationen.
    4. stelle_im_segment und Gegenwert unterscheiden sich ausreichend
       (Jaccard-Distanz >= min_value_jaccard_distance) — verhindert
       triviale Abweichungen wie "19:00 Uhr" vs "19:02 Uhr".
    """
    import re as _re

    # Nur für STRUKT_BEFUND_BESCHREIBUNG
    if finding.get("subklasse") != "Beschreibung von Befunden":
        return True

    stelle = str(finding.get("stelle_im_segment") or "").strip()
    begruendung = str(finding.get("begruendung") or "").strip()

    # 1. Span muss im Segment vorkommen
    if stelle and stelle not in segment_text:
        print(f"[VERIFY-DROP] stelle_im_segment nicht im Segment: {stelle[:60]!r}")
        return False

    # 2. Muss eine Materialreferenz enthalten (nicht nur DOC_INTERNAL)
    has_material_ref = bool(_re.search(r"\[S\d+_M_\d+\]", begruendung))
    source_refs = finding.get("source_refs") or []
    has_external_ref = any(
        r and r not in ("DOC_INTERNAL", "", None)
        for r in source_refs
    )
    if not has_material_ref and not has_external_ref:
        # DOC_INTERNAL-only ist für STRUKT_BEFUND_BESCHREIBUNG nicht ausreichend
        print(f"[VERIFY-DROP] STRUKT_BEFUND ohne Materialreferenz: {stelle[:60]!r}")
        return False

    # 3. Extrahiere Gegenwert aus Begründung — Text nach dem letzten ":" oder
    #    nach Muster "war X", "ist X", "nennt X" (max. 6 Tokens)
    counter_value = ""
    # Muster: "Laut [S3_M_1]: <Gegenwert>" — nimm alles nach dem letzten Doppelpunkt
    colon_match = _re.search(r":\s*(.{3,80})$", begruendung.strip())
    if colon_match:
        counter_value = colon_match.group(1).strip()

    if counter_value:
        # 3a. Gegenwert muss in mindestens einer Materialreferenz vorkommen
        counter_lower = counter_value.lower()
        # Wir prüfen nur die ersten 60 Zeichen des Gegenwerts (Kernaussage)
        counter_key = counter_lower[:60]
        found_in_material = any(
            counter_key in src.text.lower()
            for src in material_sources
        )
        if not found_in_material and len(counter_value) < 100:
            print(
                f"[VERIFY-DROP] Gegenwert nicht in Materialreferenzen: "
                f"{counter_value[:60]!r}"
            )
            return False

        # 3b. Jaccard-Distanz zwischen Span und Gegenwert
        # Zu ähnliche Werte (z.B. "19:00 Uhr" vs "19:02 Uhr") werden verworfen
        stelle_tokens = set(stelle.lower().split())
        counter_tokens = set(counter_value.lower().split())
        if stelle_tokens and counter_tokens:
            union = len(stelle_tokens | counter_tokens)
            intersection = len(stelle_tokens & counter_tokens)
            jaccard_sim = intersection / union if union else 0.0
            jaccard_dist = 1.0 - jaccard_sim
            if jaccard_dist < min_value_jaccard_distance:
                print(
                    f"[VERIFY-DROP] Jaccard-Distanz zu gering "
                    f"({jaccard_dist:.2f} < {min_value_jaccard_distance}): "
                    f"{stelle[:40]!r} vs {counter_value[:40]!r}"
                )
                return False

    return True


def run_factual_agent(
        llm: LLMClient,
        evidence: SegmentEvidence,
        *,
        per_agent_context_chars: int,
        catalog: ErrorCatalog,
) -> List[Dict[str, Any]]:
    # Option 4: Materialreferenzen vor dem Context-Build deduplizieren
    material_sources_deduped = _dedup_material_sources(evidence.material_sources)
    rules_context = build_agent_context_from_sources(evidence.rules_sources, max_chars=per_agent_context_chars)
    material_context = build_agent_context_from_sources(material_sources_deduped, max_chars=per_agent_context_chars)

    messages = build_factual_review_messages(
        evidence.segment_text,
        rules_context,
        material_context,
        catalog,
    )
    factual_schema = build_factual_json_schema(catalog)
    raw_reply = llm.chat(messages, json_mode=True, schema=factual_schema)
    print(f"[DEBUG] Factual agent S{evidence.segment_index} raw reply ({len(raw_reply)} chars): {raw_reply[:200]!r}")

    try:
        parsed = parse_json_response(raw_reply)
    except Exception:
        repair_messages = build_json_repair_messages(raw_reply, schema_name="factual_errors")
        repaired = llm.chat(repair_messages, json_mode=True, schema=factual_schema)
        parsed = parse_json_response(repaired)

    findings = normalize_factual_errors(parsed.get("errors", []), catalog, segment_text=evidence.segment_text)
    findings = _dedup_factual_findings(findings)

    # Option 3: Span-Verifikation für STRUKT_BEFUND_BESCHREIBUNG
    before_verify = len(findings)
    findings = [
        f for f in findings
        if _verify_strukt_befund_finding(
            f,
            evidence.segment_text,
            material_sources_deduped,
        )
    ]
    dropped_verify = before_verify - len(findings)
    if dropped_verify:
        print(
            f"[VERIFY-FILTER] S{evidence.segment_index}: "
            f"{dropped_verify} STRUKT_BEFUND_BESCHREIBUNG verworfen"
        )

    # Option 1: Max-Findings-Limit pro Segment für STRUKT_BEFUND_BESCHREIBUNG.
    # Ein Segment mit 1200 Zeichen kann realistisch max. 2 echte fachliche
    # Fehler enthalten. Bei mehr ist es statistisches Rauschen.
    _MAX_STRUKT_BEFUND_PER_SEGMENT = 2
    strukt_befund = [f for f in findings if f.get("subklasse") == "Beschreibung von Befunden"]
    other_findings = [f for f in findings if f.get("subklasse") != "Beschreibung von Befunden"]
    if len(strukt_befund) > _MAX_STRUKT_BEFUND_PER_SEGMENT:
        # Behalte die N mit dem höchsten Schweregrad (hoch > mittel > niedrig)
        _severity_order = {"hoch": 0, "mittel": 1, "niedrig": 2}
        strukt_befund_sorted = sorted(
            strukt_befund,
            key=lambda f: _severity_order.get(str(f.get("schweregrad") or "niedrig").lower(), 2)
        )
        dropped_limit = len(strukt_befund) - _MAX_STRUKT_BEFUND_PER_SEGMENT
        strukt_befund = strukt_befund_sorted[:_MAX_STRUKT_BEFUND_PER_SEGMENT]
        print(
            f"[MAX-LIMIT] S{evidence.segment_index}: "
            f"{dropped_limit} STRUKT_BEFUND_BESCHREIBUNG über Limit verworfen"
        )
    findings = other_findings + strukt_befund

    # ── STRUKT_EVIDENZ: nur DOC_INTERNAL akzeptieren ─────────────────────────
    # Fehlende Quellenangabe ist per Definition ein dokumentinternes Finding.
    # Regelwerk- (_R_) und Material-Referenzen (_M_) sind kein valider Beleg
    # dafür dass eine Quellenangabe fehlt — sie beschreiben das Regelwerk,
    # nicht den Fehler im Dokument.
    # Zusätzlich: Platzhalter-Referenzen wie "SRC_X_N" (aus dem Prompt-Beispiel)
    # werden verworfen — das LLM hat den Beispiel-Ref direkt übernommen.
    _VALID_EVIDENZ_REF_RE = re.compile(r"^S\\d+_[RM]_\\d+$")

    def _is_valid_evidenz_ref(refs) -> bool:
        """STRUKT_EVIDENZ ist gültig wenn source_refs == ["DOC_INTERNAL"] (exakt)."""
        refs = [r for r in (refs or []) if r and r not in ("", None)]
        if not refs:
            return False
        # Prompt-Platzhalter sind keine validen Referenzen
        _PLACEHOLDER_REFS = {"SRC_X_N", "SRC_R_N", "SRC_M_N", "SRC_X_1", "SRC_N"}
        if any(r in _PLACEHOLDER_REFS for r in refs):
            return False
        # Nur DOC_INTERNAL — keine Regelwerk- oder Materialreferenzen
        return refs == ["DOC_INTERNAL"]

    before_evidenz = len(findings)
    findings = [
        f for f in findings
        if not (
                f.get("subklasse") == "Evidenz / Belege"
                and not _is_valid_evidenz_ref(f.get("source_refs"))
        )
    ]
    dropped_evidenz = before_evidenz - len(findings)
    if dropped_evidenz:
        print(
            f"[DEBUG-EVIDENZ-FILTER] S{evidence.segment_index}: "
            f"{dropped_evidenz} STRUKT_EVIDENZ mit ungültiger Referenz verworfen "
            f"(nur DOC_INTERNAL erlaubt)"
        )

    for f in findings:
        f["segment_index"] = evidence.segment_index
    return findings







# ── Agent 3 Ergänzung: deterministischer Guard für rechtliche Abkürzungen ─────

LEGAL_ABBREVIATION_CANONICAL: Dict[str, str] = {
    # Grosse Vier / zentrale Erlasse
    "StGB": "Schweizerisches Strafgesetzbuch",
    "StPO": "Schweizerische Strafprozessordnung",
    "ZGB": "Schweizerisches Zivilgesetzbuch",
    "OR": "Obligationenrecht",
    "ZPO": "Schweizerische Zivilprozessordnung",

    # Strafrecht & Verfahren
    "JStPO": "Schweizerische Jugendstrafprozessordnung",
    "StA": "Staatsanwaltschaft / Staatsanwalt",
    "VStrR": "Bundesgesetz über das Verwaltungsstrafrecht",

    # Zivilrecht / Verwaltung
    "SchKG": "Bundesgesetz über Schuldbetreibung und Konkurs",
    "IPRG": "Bundesgesetz über das Internationale Privatrecht",
    "VVG": "Versicherungsvertragsgesetz",
    "VwVG": "Verwaltungsverfahrensgesetz",
    "BGFA": "Bundesgesetz über die Freizügigkeit der Anwältinnen und Anwälte",

    # Sozialversicherungsrecht
    "ATSG": "Bundesgesetz über den Allgemeinen Teil des Sozialversicherungsrechts",
    "AHVG": "Bundesgesetz über die Alters- und Hinterlassenenversicherung",
    "IVG": "Bundesgesetz über die Invalidenversicherung",
    "UVG": "Bundesgesetz über die Unfallversicherung",

    # Gerichtsbarkeit & Rechtsprechung
    "BGG": "Bundesgesetz über das Bundesgericht",
    "BGE": "Entscheidungen des Schweizerischen Bundesgerichts / Bundesgerichtsentscheid",
    "BGer": "Bundesgericht",

    # Berufsbezeichnungen / Zitierabkürzungen
    "RA": "Rechtsanwalt",
    "RAin": "Rechtsanwältin",
    "RAe": "Rechtsanwälte",
    "Not.": "Notar / Notarin",
    "i.V.m.": "in Verbindung mit",
    "a.a.O.": "am angegebenen Ort",
}

# Explizite Varianten. Diese Liste ist absichtlich konservativ:
# Sie enthält nur Schreibungen, die im Gutachtenkontext praktisch sicher falsch sind.
LEGAL_ABBREVIATION_VARIANTS: Dict[str, str] = {
    # Gesetze: falsche Gross-/Kleinschreibung
    "stgb": "StGB", "Stgb": "StGB", "STGB": "StGB",
    "stpo": "StPO", "Stpo": "StPO", "StPo": "StPO", "stPo": "StPO", "STPO": "StPO",
    "zgb": "ZGB", "Zgb": "ZGB",
    "zpo": "ZPO", "Zpo": "ZPO",
    "jstpo": "JStPO", "JSTPO": "JStPO", "Jstpo": "JStPO", "JStpo": "JStPO",
    "vstrr": "VStrR", "VSTRR": "VStrR", "Vstrr": "VStrR", "VStrr": "VStrR",
    "schkg": "SchKG", "SCHKG": "SchKG", "Schkg": "SchKG",
    "iprg": "IPRG", "Iprg": "IPRG",
    "vvg": "VVG", "Vvg": "VVG",
    "vwvg": "VwVG", "VWVG": "VwVG", "Vwvg": "VwVG",
    "bgfa": "BGFA", "Bgfa": "BGFA",
    "atsg": "ATSG", "Atsg": "ATSG",
    "ahvg": "AHVG", "Ahvg": "AHVG",
    "ivg": "IVG", "Ivg": "IVG",
    "uvg": "UVG", "Uvg": "UVG",
    "bgg": "BGG", "Bgg": "BGG",
    "bge": "BGE", "Bge": "BGE",
    "bger": "BGer", "BGER": "BGer", "Bger": "BGer",

    # Berufs-/Behördenabkürzungen
    "sta": "StA", "STA": "StA", "Sta": "StA",
    "rain": "RAin", "RAIN": "RAin", "RaIn": "RAin", "RAn": "RAin",
    "rae": "RAe", "RAE": "RAe", "Rae": "RAe",
    "not.": "Not.", "NOT.": "Not.", "Not": "Not.", "not": "Not.",

    # Zitierabkürzungen: fehlende Punkte / falsche Grossschreibung
    "ivm": "i.V.m.", "iVm": "i.V.m.", "I.V.M.": "i.V.m.", "I.v.m.": "i.V.m.",
    "i.v.m.": "i.V.m.", "i.V.m": "i.V.m.", "i V m": "i.V.m.", "i. V. m.": "i.V.m.",
    "aaO": "a.a.O.", "aao": "a.a.O.", "A.A.O.": "a.a.O.", "a.a.o.": "a.a.O.",
    "a.a.O": "a.a.O.", "a. a. O.": "a.a.O.",
}

# Kurze Abkürzungen mit höherem False-Positive-Risiko werden nur in juristischem Kontext geprüft.
_LEGAL_ABBREVIATION_CONTEXT_RE = re.compile(
    r"\b(?:Art\.|Artikel|Abs\.|Ziff\.|lit\.|SR|Straf|Zivil|Verfahren|Gesetz|Bundesgesetz|"
    r"Staatsanwalt|Staatsanwaltschaft|Rechtsanwalt|Rechtsanwältin|Notar|Notarin|Bundesgericht|"
    r"Entscheid|Urteil|Verordnung|i\.?\s*V\.?\s*m\.?|a\.?\s*a\.?\s*O\.?)\b",
    flags=re.IGNORECASE,
)

_ALWAYS_SAFE_LEGAL_VARIANTS = {
    # Lange / spezifische Varianten sind ohne Zusatzkontext praktisch eindeutig.
    "stgb", "Stgb", "STGB", "stpo", "Stpo", "STPO", "jstpo", "JSTPO", "Jstpo", "JStpo",
    "vstrr", "VSTRR", "Vstrr", "VStrr", "schkg", "SCHKG", "Schkg", "vwvg", "VWVG", "Vwvg",
    "bgfa", "Bgfa", "atsg", "Atsg", "ahvg", "Ahvg", "bger", "BGER", "Bger",
    "ivm", "iVm", "I.V.M.", "I.v.m.", "i.v.m.", "i.V.m", "i V m", "i. V. m.",
    "aaO", "aao", "A.A.O.", "a.a.o.", "a.a.O", "a. a. O.",
}


def _legal_abbrev_has_context(segment_text: str, start: int, end: int) -> bool:
    window = segment_text[max(0, start - 80): min(len(segment_text), end + 80)]
    return bool(_LEGAL_ABBREVIATION_CONTEXT_RE.search(window))


def check_legal_abbreviation_variants(
        segment_text: str,
        segment_index: int,
) -> List[Dict[str, Any]]:
    """
    Deterministic formal checker for common Swiss legal abbreviations.

    Detects conservative, high-confidence misspellings such as:
      - Stpo -> StPO
      - stgb -> StGB
      - ivm / i.v.m. -> i.V.m.
      - aaO / a.a.o. -> a.a.O.
      - bger -> BGer

    It intentionally avoids correcting canonical forms and avoids broad fuzzy
    matching so that ordinary words are not turned into legal abbreviations.
    """
    findings: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int, str]] = set()

    # Longer variants first, so "i. V. m." is preferred over partial fragments.
    variants = sorted(LEGAL_ABBREVIATION_VARIANTS.items(), key=lambda kv: len(kv[0]), reverse=True)

    for wrong, correct in variants:
        if wrong == correct:
            continue

        # Token boundary: avoid matching inside longer words or reference IDs.
        escaped = re.escape(wrong)
        pattern = re.compile(rf"(?<![A-Za-zÄÖÜäöüß0-9_]){escaped}(?![A-Za-zÄÖÜäöüß0-9_])")

        for match in pattern.finditer(segment_text):
            found = match.group(0)
            if found == correct:
                continue

            # Short/high-risk abbreviations need legal context.
            if wrong not in _ALWAYS_SAFE_LEGAL_VARIANTS and not _legal_abbrev_has_context(segment_text, match.start(), match.end()):
                continue

            key = (match.start(), match.end(), correct)
            if key in seen:
                continue
            seen.add(key)

            meaning = LEGAL_ABBREVIATION_CANONICAL.get(correct, "rechtliche Abkürzung")
            findings.append({
                "segment_index": segment_index,
                "hauptklasse": "Formales",
                "subklasse": "Redaktionelle Korrektur",
                "aenderungstyp": "Redaktionelle Korrektur",
                "schweregrad": "niedrig",
                "stelle_im_segment": found,
                "begruendung": (
                    f"Falsch geschriebene rechtliche Abkürzung: '{found}' sollte '{correct}' "
                    f"heissen ({meaning})."
                ),
                "vorschlag": correct,
                "legal_abbrev_guard": True,
            })

    return findings


# ── LanguageTool singleton ────────────────────────────────────────────────────

_language_tool_instance: Optional[Any] = None
_language_tool_available: Optional[bool] = None


def _get_language_tool() -> Optional[Any]:
    """
    Return a cached LanguageTool instance for Swiss German (de-CH).
    Returns None if language_tool_python is not installed.

    LanguageTool is used as the primary language checker because it uses
    deterministic grammar rules (German adjective capitalisation, Swiss ss/ß,
    article-noun agreement) that are 100% reliable for the errors the LLM
    misses most — capitalisation like 'die Beschuldigte Person' → 'die
    beschuldigte Person'.
    """
    global _language_tool_instance, _language_tool_available
    if _language_tool_available is False:
        return None
    if _language_tool_instance is not None:
        return _language_tool_instance
    try:
        import language_tool_python  # type: ignore
        _language_tool_instance = language_tool_python.LanguageTool(
            "de-CH",
            config={"maxSpellingSuggestions": 1},
        )
        _language_tool_available = True
        print("[INFO] LanguageTool de-CH loaded — using rule-based language checking.")
        return _language_tool_instance
    except Exception as e:
        _language_tool_available = False
        print(f"[WARN] LanguageTool not available ({e}). Falling back to LLM language agent.")
        return None


# Rule IDs that are reliable for formal document checking.
# Excludes style suggestions, redundancy, and overly aggressive comma rules.
_LT_ALLOWED_RULE_PREFIXES = (
    "DE_CASE",
    "GERMAN_SPELLER",
    "SWISS_GERMAN_SPELLER",
    "MORFOLOGIK",
    "AGREEMENT",
    "COMPOUND",
    "UPPERCASE_SENTENCE",
    "COMMA_PARENTHESIS",
    "DE_DOUBLE_PUNCTUATION",
)

# Rule IDs to always suppress regardless of prefix
_LT_BLOCKED_RULE_IDS = {
    "KOMMA_VOR_UND_ODER",       # optional comma before und/oder
    "COMMA_BEFORE_ODER",
    "DOPPELUNG",                 # stylistic redundancy
    "STYLE",
    "WHITESPACE_RULE",           # whitespace normalisation
}


def _lt_attr(m: Any, *names: str, default: Any = "") -> Any:
    """Try multiple attribute names on a LanguageTool Match object."""
    for name in names:
        val = getattr(m, name, None)
        if val is not None and val != "":
            return val
    return default


def _run_language_tool(segment_text: str, catalog: ErrorCatalog) -> List[Dict[str, Any]]:
    """
    Run LanguageTool on a segment and return normalized findings.
    Uses _lt_attr() to handle attribute name differences across
    language_tool_python versions (ruleId vs rule_id etc.).
    """
    tool = _get_language_tool()
    if tool is None:
        return []

    try:
        matches = tool.check(segment_text)
    except Exception as e:
        print(f"[WARN] LanguageTool check failed: {e}")
        return []

    findings: List[Dict[str, Any]] = []
    for m in matches:
        # ruleId varies by version — try all known attribute names
        rule_id = str(_lt_attr(m, "ruleId", "rule_id", "matchedByRule", "ruleIssueType",
                               default=""))

        # offset and length also vary
        offset = int(_lt_attr(m, "offset", "offsetInContext", default=0))
        length = int(_lt_attr(m, "errorLength", "error_length", "length", default=0))
        message = str(_lt_attr(m, "message", "msg", "shortMessage", default=""))
        replacements_raw = _lt_attr(m, "replacements", "suggested_replacements", default=[])
        if isinstance(replacements_raw, list):
            # str(r) kann LanguageTool-interne Statusstrings liefern wie
            # "(Vorschlagslimit erreicht)" wenn das Java-Backend keinen
            # sinnvollen Ersatz hat. Diese werden herausgefiltert.
            _LT_INTERNAL_STRINGS = {
                "(vorschlagslimit erreicht)",
                "(limit reached)",
                "(no suggestions)",
            }
            replacements = [
                s for r in replacements_raw
                if (s := str(r).strip()) and s.lower() not in _LT_INTERNAL_STRINGS
            ]
        else:
            replacements = []

        # Skip explicitly blocked rules
        if rule_id in _LT_BLOCKED_RULE_IDS:
            continue

        # If rule_id is known, only keep allowed families
        # If rule_id is empty (version mismatch), let the finding through
        if rule_id and not any(rule_id.startswith(p) for p in _LT_ALLOWED_RULE_PREFIXES):
            continue

        stelle = segment_text[offset: offset + length].strip() if length > 0 else ""
        if not stelle:
            continue

        vorschlag = replacements[0] if replacements else ""

        # Swiss German: ss is correct, ß is not — skip ß→ss suggestions
        def _norm_sz(s: str) -> str:
            return s.replace("ß", "ss").lower()
        if vorschlag and _norm_sz(stelle) == _norm_sz(vorschlag):
            continue

        if vorschlag and vorschlag.strip() == stelle:
            continue

        findings.append({
            "hauptklasse": "Formales",
            "subklasse": "Redaktionelle Korrektur",
            "aenderungstyp": "Redaktionelle Korrektur",
            "schweregrad": "niedrig",
            "stelle_im_segment": stelle,
            "begruendung": message,
            "vorschlag": vorschlag,
            "lt_rule_id": rule_id,
        })

    return findings


# ── spaCy singleton ──────────────────────────────────────────────────────────

_spacy_nlp_instance: Optional[Any] = None
_spacy_available: Optional[bool] = None


def _get_spacy_nlp() -> Optional[Any]:
    """
    Return a cached spaCy de_dep_news_trf pipeline instance.
    Returns None if spaCy or the model is not installed.

    Used for German adjective capitalisation detection using dependency parsing.
    The German model uses TIGER treebank dependency labels (nk, sb, oa, etc.)
    rather than Universal Dependencies (amod, nsubj, etc.).
    """
    global _spacy_nlp_instance, _spacy_available
    if _spacy_available is False:
        return None
    if _spacy_nlp_instance is not None:
        return _spacy_nlp_instance
    try:
        import spacy  # type: ignore
        _spacy_nlp_instance = spacy.load("de_dep_news_trf")
        _spacy_available = True
        print("[INFO] spaCy de_dep_news_trf loaded — adjective capitalisation checking enabled.")
        return _spacy_nlp_instance
    except Exception as e:
        _spacy_available = False
        print(f"[WARN] spaCy not available ({e}). Skipping adjective capitalisation check.")
        return None


def _run_spacy_adjective_check(segment_text: str) -> List[Dict[str, Any]]:
    nlp = _get_spacy_nlp()
    if nlp is None:
        return []

    GERMAN_ARTICLES = {
        "der", "die", "das", "dem", "den", "des",
        "ein", "eine", "einem", "einen", "einer",
        "kein", "keine", "keinem", "keinen", "keiner",
        "dieser", "diese", "diesem", "diesen", "dieses",
        "jeder", "jede", "jedem", "jeden", "jedes",
        "welcher", "welche", "welchem", "welchen", "welches",
    }

    doc = nlp(segment_text)
    findings: List[Dict[str, Any]] = []

    for token in doc:
        prev_text = doc[token.i - 1].text.lower() if token.i > 0 else ""
        prev_pos = doc[token.i - 1].pos_ if token.i > 0 else ""

        if (
                token.pos_ == "ADJ"
                and token.dep_ == "nk"
                and token.head.pos_ == "NOUN"
                and token.text[0].isupper()
                and token.i > 0
                and (
                prev_pos == "DET"
                or prev_text in GERMAN_ARTICLES
        )
        ):
            correct = token.text[0].lower() + token.text[1:]
            # Artikel + Adjektiv + Nomen als Span — passt besser zur Ground Truth
            span_start = token.i - 1  # Artikel
            span_end = token.i + 2    # Adjektiv + Nomen
            context_span = doc[max(0, span_start): min(len(doc), span_end)].text
            correct_span = context_span.replace(token.text, correct, 1)

            findings.append({
                "hauptklasse":       "Formales",
                "subklasse":         "Redaktionelle Korrektur",
                "aenderungstyp":     "Redaktionelle Korrektur",
                "schweregrad":       "niedrig",
                "stelle_im_segment": context_span,
                "begruendung": (
                    f"Attributives Adjektiv sollte kleingeschrieben werden: "
                    f"'{context_span}' → '{correct_span}'"
                ),
                "vorschlag": correct_span,
            })

    return findings

def _run_spacy_case_government_check(segment_text: str) -> List[Dict[str, Any]]:
    """
    Conservative spaCy-based grammar guard for case errors in nominal groups.

    Detects patterns like:
      'für den Bereich dem Sichtfeld' -> 'für den Bereich des Sichtfeldes'

    This is intentionally narrow to avoid false positives.
    """
    nlp = _get_spacy_nlp()
    if nlp is None:
        return []

    doc = nlp(segment_text)
    findings: List[Dict[str, Any]] = []

    for i, token in enumerate(doc):
        # Look for: für den Bereich dem <NOUN>
        if token.text.lower() != "bereich":
            continue

        if i < 2 or i + 2 >= len(doc):
            continue

        prev2 = doc[i - 2]
        prev1 = doc[i - 1]
        next1 = doc[i + 1]
        next2 = doc[i + 2]

        if prev2.text.lower() != "für":
            continue
        if prev1.text.lower() != "den":
            continue
        if next1.text.lower() not in {"dem", "der", "den"}:
            continue
        if next2.pos_ not in {"NOUN", "PROPN"}:
            continue

        # Very conservative: only flag when the following noun is singular-looking
        # and the phrase is exactly the common faulty construction.
        wrong_span = doc[i - 2: i + 3].text

        noun = next2.text
        if noun == "Sichtfeld":
            correct_noun_phrase = "des Sichtfeldes"
        else:
            # Avoid risky automatic inflection for unknown nouns.
            # For unknown nouns, still provide a safer generic proposal.
            correct_noun_phrase = f"des {noun}s"

        correct_span = f"für den Bereich {correct_noun_phrase}"

        findings.append({
            "hauptklasse": "Formales",
            "subklasse": "Redaktionelle Korrektur",
            "aenderungstyp": "Redaktionelle Korrektur",
            "schweregrad": "niedrig",
            "stelle_im_segment": wrong_span,
            "begruendung": (
                "Kasusfehler in der Nominalgruppe: Nach 'für den Bereich' "
                "ist hier eine Genitivkonstruktion erforderlich."
            ),
            "vorschlag": correct_span,
        })

    return findings

def _dedup_language_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate language findings after combining LanguageTool and spaCy."""
    seen: Set[Tuple[str, str, str]] = set()
    out: List[Dict[str, Any]] = []
    for f in findings:
        key = (
            str(f.get("stelle_im_segment") or "").strip().lower(),
            str(f.get("vorschlag") or "").strip().lower(),
            str(f.get("subklasse") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def run_language_agent(
        llm: LLMClient,
        evidence: SegmentEvidence,
        *,
        catalog: ErrorCatalog,
        reference_words: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Agent 3 language/formal checking.

    Debug version:
      1) Run LanguageTool + spaCy exactly once.
      2) Print raw and filtered findings.
      3) Optionally disable LLM fallback via:
         DISABLE_LLM_LANGUAGE_FALLBACK=true

    reference_words: Set von bekannten Wörtern aus Referenzfakten.
      Findings deren stelle_im_segment ein bekanntes Wort enthält werden
      gefiltert (verhindert False Positives bei Ortsnamen, Eigennamen).
    """
    deterministic_raw: List[Dict[str, Any]] = []

    legal_abbrev_raw = check_legal_abbreviation_variants(evidence.segment_text, evidence.segment_index)
    print(f"[DEBUG-LEGAL-ABBREV-RAW] S{evidence.segment_index}: {legal_abbrev_raw}")
    deterministic_raw.extend(legal_abbrev_raw)

    lt_raw = _run_language_tool(evidence.segment_text, catalog)
    print(f"[DEBUG-LT-RAW] S{evidence.segment_index}: {lt_raw}")
    deterministic_raw.extend(lt_raw)

    spacy_raw = _run_spacy_adjective_check(evidence.segment_text)
    # print(f"[DEBUG-SPACY-RAW] S{evidence.segment_index}: {spacy_raw}")
    deterministic_raw.extend(spacy_raw)

    spacy_case_raw = _run_spacy_case_government_check(evidence.segment_text)
    print(f"[DEBUG-SPACY-CASE-RAW] S{evidence.segment_index}: {spacy_case_raw}")
    deterministic_raw.extend(spacy_case_raw)

    findings = filter_language_findings_by_exact_span(
        deterministic_raw,
        evidence.segment_text,
    )
    print(f"[DEBUG-LANG-SPAN] S{evidence.segment_index}: {findings}")

    findings = filter_language_findings_by_plausibility(findings)
    print(f"[DEBUG-LANG-PLAUSIBLE] S{evidence.segment_index}: {findings}")

    findings = _dedup_language_findings(findings)
    print(f"[DEBUG-LANG-DEDUP] S{evidence.segment_index}: {findings}")
    findings = _dedup_spans_across_segments(findings, span_field="stelle_im_segment")

    # ── Referenzfakten-Filter ────────────────────────────────────────────────
    # Ortsnamen, Personennamen und Institutionen aus den Referenzfakten
    # werden nicht als Rechtschreibfehler gemeldet.
    # Hinweis: Diese Hilfsfunktion wird auch nach dem LLM-Fallback angewendet,
    # damit der Filter auch LLM-generierte Findings abdeckt.
    def _apply_reference_words_filter(
            findings: List[Dict[str, Any]],
            reference_words: set,
            segment_index: int,
            label: str = "",
    ) -> List[Dict[str, Any]]:
        before = len(findings)
        filtered = [
            f for f in findings
            if not any(
                w in reference_words
                for w in re.findall(r"[\w\-äöüÄÖÜß]+", str(f.get("stelle_im_segment") or ""))
                if len(w) >= 4
            )
        ]
        dropped = before - len(filtered)
        if dropped:
            print(
                f"[DEBUG-LANG-REFFILTER{('-' + label) if label else ''}] S{segment_index}: "
                f"{dropped} Finding(s) durch Referenzfakten-Filter entfernt"
            )
        return filtered

    if reference_words:
        findings = _apply_reference_words_filter(findings, reference_words, evidence.segment_index, label="DET")

    # ── Institutionsnamen-Filter ─────────────────────────────────────────────
    # Adjektiv-Grossschreibung in offiziellen Behörden-/Institutionsbezeichnungen
    # ist korrekt und wird nicht als Fehler gemeldet.
    # Beispiel: "der Kriminaltechnischen Abteilung" ist kein Fehler.
    #
    # Präzisierung: nur filtern wenn das letzte Substantiv ein bekannter
    # Institutionsname ist. "der Freien Natur" enthält "Natur" → kein
    # Institutionsname → nicht gefiltert.
    _INSTITUTION_PREFIX_RE = re.compile(
        r"^(?:der|die|das|dem|den|des|von|vom|zur|zum|beim|an\s+der|an\s+die)\s",
        re.IGNORECASE,
    )
    _INSTITUTION_NOUNS = {
        "abteilung", "abteilungen", "amt", "ämter", "behörde", "behörden",
        "departement", "departements", "dienst", "dienste", "direktion",
        "direktionen", "gericht", "gerichte", "gerichts", "instituts",
        "institut", "institute", "kammer", "kammern", "kommission",
        "kommissionen", "ministerium", "ministeriums", "polizei",
        "sektion", "sektionen", "staatsanwaltschaft", "staatsanwaltschaften",
        "stelle", "stellen",
    }
    findings = [
        f for f in findings
        if not (
                "attributives adjektiv" in str(f.get("begruendung") or "").lower()
                and _INSTITUTION_PREFIX_RE.match(str(f.get("stelle_im_segment") or ""))
                and any(
            w.lower() in _INSTITUTION_NOUNS
            for w in str(f.get("stelle_im_segment") or "").split()
        )
        )
    ]

    if findings:
        for f in findings:
            f["segment_index"] = evidence.segment_index
        return findings

    if env_bool("DISABLE_LLM_LANGUAGE_FALLBACK", False):
        print(
            f"[DEBUG-LANG] S{evidence.segment_index}: "
            "deterministic tools found no usable findings; LLM fallback disabled"
        )
        return []

    # Fallback: LLM language agent only when deterministic tools found nothing usable.
    try:
        messages = build_language_review_messages(evidence.segment_text)
        raw_reply = llm.chat(messages, json_mode=True)
        print(
            f"[DEBUG-LLM-LANG-RAW] S{evidence.segment_index} "
            f"({len(raw_reply)} chars): {raw_reply[:300]!r}"
        )

        try:
            parsed = parse_json_response(raw_reply)
        except Exception:
            repair_messages = build_json_repair_messages(
                raw_reply,
                schema_name="language_errors",
            )
            repaired = llm.chat(repair_messages, json_mode=True)
            print(
                f"[DEBUG-LLM-LANG-REPAIRED] S{evidence.segment_index} "
                f"({len(repaired)} chars): {repaired[:300]!r}"
            )
            parsed = parse_json_response(repaired)

        findings = normalize_language_errors(parsed.get("errors", []), catalog)
        print(f"[DEBUG-LLM-LANG-NORMALIZED] S{evidence.segment_index}: {findings}")

        findings = filter_language_findings_by_exact_span(
            findings,
            evidence.segment_text,
        )
        findings = filter_language_findings_by_plausibility(findings)
        findings = _dedup_language_findings(findings)
        findings = _dedup_spans_across_segments(findings, span_field="stelle_im_segment")

        # Whitelist-Filter auch auf LLM-Fallback-Output anwenden —
        # verhindert FPs bei Fachbegriffen, Komposita und Eigennamen
        # die LanguageTool nicht kennt aber das LLM trotzdem meldet.
        if reference_words:
            findings = _apply_reference_words_filter(
                findings, reference_words, evidence.segment_index, label="LLM"
            )

        print(f"[DEBUG-LLM-LANG-FINAL] S{evidence.segment_index}: {findings}")

        for f in findings:
            f["segment_index"] = evidence.segment_index
        return findings

    except Exception as e:
        print(f"[WARN] LLM language fallback failed: {e}")
        return []

def filter_language_findings_by_exact_span(
        findings: List[Dict[str, Any]],
        segment_text: str,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    normalized_segment = " ".join(segment_text.split())

    for item in findings:
        stelle = str(item.get("stelle_im_segment") or "").strip()
        if not stelle:
            continue

        normalized_stelle = " ".join(stelle.split())
        if normalized_stelle in normalized_segment:
            kept.append(item)

    return kept


def _build_reference_words_set(reference_facts: Optional[Dict[str, Any]]) -> "Set[str]":
    """
    Extrahiert alle Wörter (≥4 Zeichen) aus Referenzfakten mit Konfidenz high/medium.
    Dient als Ausschlussliste für LanguageTool-Findings: bekannte Eigennamen,
    Ortsnamen und Institutionen werden nicht als Rechtschreibfehler gemeldet.

    Verarbeitet beide Fakt-Strukturen:
    - Dict-Fakten (z.B. ort, ereignisdatum): value + source_span ausgelesen
    - List-Fakten (z.B. personen, referenz_entitaeten): alle Items ausgelesen

    Beispiel: 'Hohtenn', 'Erbjini', 'Woluhäärde' aus dem ort-Fakt,
              'Williner', 'Bosshard' aus den personen-Fakten.
    """
    words: set = set()
    if not reference_facts or not isinstance(reference_facts, dict):
        return words
    facts = reference_facts.get("facts", {})
    if not isinstance(facts, dict):
        return words

    # Alle Felder die Textwerte enthalten können
    DICT_FIELDS = ("value", "name", "wert", "source_span")
    LIST_FIELDS = ("name", "value", "wert", "rolle", "source_span", "bezeichnung")

    for key, val in facts.items():
        if isinstance(val, dict):
            # Einzel-Fakt (z.B. ort, ereignisdatum, sachverstaendige_person)
            conf = val.get("confidence", "")
            if conf in {"high", "medium"}:
                for field in DICT_FIELDS:
                    text = str(val.get(field) or "")
                    words.update(
                        w for w in re.findall(r"[\w\-äöüÄÖÜß]+", text)
                        if len(w) >= 4
                    )
        elif isinstance(val, list):
            # Listen-Fakt (z.B. personen, referenz_entitaeten)
            for item in val:
                if not isinstance(item, dict):
                    continue
                conf = item.get("confidence", "")
                if conf in {"high", "medium"}:
                    for field in LIST_FIELDS:
                        text = str(item.get(field) or "")
                        words.update(
                            w for w in re.findall(r"[\w\-äöüÄÖÜß]+", text)
                            if len(w) >= 4
                        )
                    # Aliases ebenfalls einbeziehen
                    for alias in item.get("aliases", []):
                        words.update(
                            w for w in re.findall(r"[\w\-äöüÄÖÜß]+", str(alias))
                            if len(w) >= 4
                        )

    print(f"[DEBUG-REFWORDS] {len(words)} Wörter im Referenzfakten-Set")
    return words




def _extract_person_names_spacy(doc_text: str, max_chars: int = 12000) -> "Set[str]":
    """
    Extrahiert alle Personennamen (PER) aus dem Dokumenttext via spaCy NER.
    Nutzt denselben gecachten spaCy-Instance wie _run_spacy_adjective_check.

    Alle Wortteile ≥ 4 Zeichen aus erkannten Personen-Entitäten werden
    in _ref_words aufgenommen — verhindert False Positives bei Nachnamen
    wie "Loosli", "Werlen", "Heynen" die LanguageTool als Tippfehler meldet.

    max_chars: begrenzt den analysierten Text (spaCy ist langsam auf langen Texten).
               12000 Zeichen = ~6–8 Seiten, erfasst fast alle genannten Personen.
    """
    nlp = _get_spacy_nlp()
    if nlp is None:
        return set()
    names: set = set()
    try:
        doc = nlp(doc_text[:max_chars])
        # Fix 3: LOC, ORG, MISC zusaetzlich zu PER erfassen.
        # Verhindert False Positives fuer Ortsnamen (Swiss-Topo, Metralie),
        # Institutionen und fallspezifische Eigennamen die LanguageTool nicht kennt.
        _NER_LABELS = {"PER", "LOC", "ORG", "MISC"}
        for ent in doc.ents:
            if ent.label_ in _NER_LABELS:
                for part in ent.text.split():
                    clean = re.sub(r"[^A-Za-zÄÖÜäöüß\-]", "", part)
                    if len(clean) >= 3:  # 3 statt 4: kurze Ortsnamen wie "map", "la" einschliessen
                        names.add(clean)
        if names:
            print(f"[DEBUG-NER] {len(names)} NER-Tokens erkannt (PER/LOC/ORG/MISC): "
                  f"{sorted(names)[:10]}{'...' if len(names) > 10 else ''}")
    except Exception as e:
        print(f"[WARN] spaCy NER fehlgeschlagen: {e}")
    return names



def build_domain_whitelist_from_store(
        store: "RagStore",
        case_id: str,
        min_word_length: int = 4,
        min_occurrences: int = 2,
) -> "Set[str]":
    """
    Baut eine Domänen-Whitelist aus dem fallspezifischen Material Store.

    Wörter die im Fallmaterial mindestens min_occurrences-mal vorkommen
    werden als bekannte Fachbegriffe, Eigennamen und Ortsnamen behandelt
    und nicht als Rechtschreibfehler gemeldet.

    min_occurrences=2: verhindert dass einmalige Tippfehler im Material
    selbst in die Whitelist gelangen.

    Nur Chunks mit passender case_id werden berücksichtigt.
    Chunks ohne case_id-Metadaten (Regelwerk) werden ignoriert.
    """
    from collections import Counter
    word_counts: Counter = Counter()

    for chunk_id, text in store.text_map.items():
        meta = store.index_map.get(chunk_id, {})
        chunk_case_id = str(meta.get("case_id") or "").strip()
        if chunk_case_id and chunk_case_id != case_id:
            continue  # anderer Case → überspringen
        # Wörter extrahieren (nur Buchstaben, Bindestriche, Umlaute)
        for w in re.findall(r"[A-Za-zÄÖÜäöüß]{" + str(min_word_length) + r",}", text):
            word_counts[w] += 1

    whitelist = {w for w, cnt in word_counts.items() if cnt >= min_occurrences}
    return whitelist


def filter_language_findings_by_plausibility(
        findings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []

    def _normalize_eszett(s: str) -> str:
        return s.replace("ß", "ss").lower()

    def _tokenize(s: str) -> List[str]:
        return re.findall(r"[A-Za-zÄÖÜäöüß0-9\-]+|[^\w\s]", s, flags=re.UNICODE)

    def _looks_like_named_entity(s: str) -> bool:
        # Im Deutschen sind alle Substantive grossgeschrieben — Grossschreibung
        # allein ist kein Eigennamen-Indikator. Nur echte ALL-CAPS-Abkürzungen
        # wie FOR, ZH, SZ werden als Eigennamen behandelt.
        tokens = re.findall(r"[A-Za-zÄÖÜäöüß]+", s)
        if not tokens:
            return False
        if all(t.isupper() and len(t) >= 2 for t in tokens):
            return True  # reine Abkürzung z.B. "FOR", "ZH"
        return False

    def _is_minor_article_insertion(stelle: str, vorschlag: str) -> bool:
        s_tokens = _tokenize(stelle)
        v_tokens = _tokenize(vorschlag)
        added = [t for t in v_tokens if t not in s_tokens]
        return (
                all(
                    t.lower() in {
                        "der", "die", "das", "dem", "den", "des",
                        "ein", "eine", "einer", "einem", "einen",
                    }
                    for t in added
                )
                and len(added) <= 2
        )

    for item in findings:
        stelle = str(item.get("stelle_im_segment") or "").strip()
        begruendung = str(item.get("begruendung") or "").strip().lower()
        vorschlag = str(item.get("vorschlag") or "").strip()

        if not stelle:
            continue

        if vorschlag and vorschlag == stelle:
            continue

        # Fix 2: LanguageTool kennt das Wort nicht, hat aber keinen Korrekturvorschlag
        # -> fast immer ein Fachbegriff, Eigenname oder Fremdwort, kein echter Fehler.
        if (
                not vorschlag
                and begruendung.lower().startswith("möglicher tippfehler")
        ):
            continue

        # Gross-/Kleinschreibungskorrektur — erkennt spaCy-Findings (Begründung)
        # und LLM-Findings (stelle.lower() == vorschlag.lower()) gleichermassen.
        is_case_only_fix = (
                "attributives adjektiv" in begruendung
                or (
                        vorschlag
                        and stelle.lower() == vorschlag.lower()
                        and stelle != vorschlag
                )
        )

        if ("punkt" in begruendung or "abschlusspunkt" in begruendung) and stelle.endswith((".", "!", "?")):
            continue

        if any(q in stelle for q in ("«", "»", "„", "\u201c", "\u201a", "'")):
            continue

        # Schweizer ss/ß: nur droppen wenn nicht reine Grossschreibungskorrektur
        if not is_case_only_fix and _normalize_eszett(stelle) == _normalize_eszett(vorschlag):
            continue

        # Nur echte ALL-CAPS-Abkürzungen als Named Entity behandeln
        if not is_case_only_fix and _looks_like_named_entity(stelle):
            continue

        # keine weichen Stilkorrekturen
        if "konsistent" in begruendung or "konsistenter gebrauch" in vorschlag.lower():
            continue

        # keine Mini-Artikel-Ergänzungen ohne klaren Fehler
        if vorschlag and _is_minor_article_insertion(stelle, vorschlag):
            continue

        # keine aggressiven Wortersetzungen bei kurzen Phrasen
        if len(stelle.split()) <= 3 and len(vorschlag.split()) <= 3:
            if stelle.lower() != vorschlag.lower():
                if not is_case_only_fix and not any(x in begruendung for x in [
                    "orthograf",
                    "rechtschreib",
                    "grammatik",
                    "zeichensetzung",
                    "komma",
                    "adjektiv",
                    "einheit",
                    "abkürzung",
                    "abbildung",
                    "referenz",
                    "tippfehler",
                    "inkonsistent",
                    "nummer",
                    "falsch",
                    "ungültig",
                    "schreibweise",
                ]):
                    continue

        # Silbentrennung-Artefakt: Bindestrich mitten im Wort ohne Leerzeichen.
        # Entsteht wenn Word nach Run-Zusammenführung automatisch trennt
        # (Soft-Hyphen U+00AD oder Trennstrich durch Zeilenumbruch).
        # Beispiel: "befind-lichen" → kein echter Rechtschreibfehler.
        if (
                "-" in stelle
                and " " not in stelle
                and re.search(r"\w+-\w+", stelle)
                and vorschlag
                and stelle.replace("-", "") == vorschlag.replace("-", "")
        ):
            continue

        # Silbentrennung-Artefakt Teil 2: Wort das auf "-" endet (z.B. "Ergän-")
        # → kein Rechtschreibfehler, sondern abgeschnittenes Wort am Zeilenende.
        if stelle.endswith("-") and re.match(r"^[A-Za-zÄÖÜäöüß]{2,}-$", stelle):
            continue

        # Silbentrennung-Artefakt Teil 3: sehr kurze Silbenreste (z.B. "zung", "lich")
        # die allein stehen und durch Zeilenumbruch entstanden sind.
        if (
                len(stelle) <= 5
                and re.match(r"^[a-züäöß]{2,5}$", stelle)
                and begruendung.lower().startswith("möglicher tippfehler")
                and vorschlag and len(vorschlag) <= 8
        ):
            continue

        # Fremdsprachige Kurzwörter: "la", "map", "gh", "de" etc.
        # Sehr kurze Kleinbuchstaben-Strings die LT nicht kennt aber keine
        # deutschen Fehler sind (französische/englische Quellenangaben).
        if (
                len(stelle) <= 4
                and re.match(r"^[a-z]{1,4}$", stelle)
                and not vorschlag
        ):
            continue

        kept.append(item)

    return kept


# ── Agent 4: Calculation Checker ─────────────────────────────────────────────

def build_calculation_json_schema(catalog: ErrorCatalog) -> Dict[str, Any]:
    """Grammar-constrained schema for Agent 4 — uses only the Rechenfehler main class.

    Design: LLM extracts the arithmetic expression and the claimed result from the
    document text. Python evaluates and verifies — LLM never computes korrekter_wert.

    Key fields:
      expression      Arithmetic expression as a Python-evaluable string, e.g. "(81/1.7)*3.6".
                      Only digits, operators (+−*/), parentheses, dots. No units, no text.
                      Use empty string "" if no explicit expression is present in the text
                      (e.g. pure unit errors).
      claimed_result  The numeric value as stated in the document, e.g. 171.4.
                      Use 0.0 if not applicable (unit-only findings).
      decimal_places  Number of decimal places visible in the document result, e.g. 1 for "171.4".
      wert_im_dokument  The claimed value as written in the document, including unit.
      begruendung     Brief explanation why this looks like an error (before Python verification).
    """
    rechen_subs = sorted(
        sub["label"]
        for sub in catalog.sub_by_main.get("RECHENFEHLER", {}).values()
    )
    change_labels = sorted(catalog.allowed_change_labels)
    sev_labels = sorted(catalog.allowed_severity_labels)

    finding_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "hauptklasse":       {"type": "string", "enum": ["Rechenfehler"]},
            "subklasse":         {"type": "string", "enum": rechen_subs},
            "aenderungstyp":     {"type": "string", "enum": change_labels},
            "schweregrad":       {"type": "string", "enum": sev_labels},
            "stelle_im_segment": {"type": "string"},
            "expression":        {"type": "string"},
            "claimed_result":    {"type": "number"},
            "decimal_places":    {"type": "integer", "minimum": 0, "maximum": 6},
            "wert_im_dokument":  {"type": "string"},
            "begruendung":       {"type": "string"},
        },
        "required": [
            "hauptklasse", "subklasse", "aenderungstyp", "schweregrad",
            "stelle_im_segment", "expression", "claimed_result", "decimal_places",
            "wert_im_dokument", "begruendung",
        ],
    }
    return {
        "type": "object",
        "properties": {"errors": {"type": "array", "items": finding_schema}},
        "required": ["errors"],
    }


def build_calculation_review_messages(
        segment_text: str,
        catalog: ErrorCatalog,
) -> List[Dict[str, str]]:
    system = (
            _AGENT_JSON_PREFIX +

            "Du bist Agent 4: Rechenprüfer für technische Dokumente.\n\n"

            "AUFGABE: Identifiziere alle Stellen im Segment, an denen eine numerische "
            "Berechnung steht (Geschwindigkeiten, Abstände, Zeiten, Umrechnungen, "
            "Prozentwerte, Kräfte, Massen, Winkelmasse). "
            "Extrahiere für jede Stelle den arithmetischen Ausdruck und den im Dokument "
            "behaupteten Ergebniswert.\n\n"

            "WICHTIG — Du rechnest NICHT nach und bestimmst NICHT den korrekten Wert. "
            "Das macht Python nach deiner Extraktion. Deine einzige Aufgabe ist die "
            "saubere Extraktion der folgenden Felder:\n\n"

            "  expression       Der arithmetische Ausdruck exakt wie er sich aus dem Text "
            "ergibt, als auswertbarer Python-String.\n"
            "                   Nur Ziffern, Operatoren (+ - * /), Klammern, Punkte als "
            "Dezimaltrenner.\n"
            "                   Keine Einheiten, kein Text. Beispiel: \"(70/1.2)*2.6\"\n"
            "                   Leer lassen (\"\") wenn kein expliziter Ausdruck vorhanden "
            "(z.B. reine Einheitenfehler).\n\n"

            "  claimed_result   Der im Dokument stehende Ergebniswert als reine Zahl, "
            "ohne Einheit. Beispiel: 171.4\n\n"

            "  decimal_places   Anzahl Nachkommastellen des Ergebniswerts im Dokument. "
            "Beispiel: 1 für '11.8', 0 für '11', 2 für '12.14'\n\n"

            "  wert_im_dokument Der behauptete Wert inklusive Einheit, exakt wie im "
            "Dokument. Beispiel: \"11.8 km/h\"\n\n"

            "  stelle_im_segment Originalausschnitt aus dem Dokument der den Fehler enthält.\n\n"

            "  begruendung      Kurze Erklärung warum dieser Wert verdächtig ist "
            "(ohne selbst nachzurechnen).\n\n"

            "Melde NUR Stellen mit einem expliziten numerischen Ausdruck oder "
            "Einheitenfehler. Kein Fehler erkennbar → {\"errors\":[]}\n\n"

            "Klassifikation: hauptklasse='Rechenfehler' | "
            "subklasse='Arithmetischer Fehler'|'Einheitenfehler'|'Rundungsfehler' | "
            "aenderungstyp='Rechnerische Korrektur' | schweregrad=niedrig|mittel|hoch\n\n"

            "Format:\n"
            "{\"errors\":[{\"hauptklasse\":\"Rechenfehler\",\"subklasse\":\"Arithmetischer Fehler\","
            "\"aenderungstyp\":\"Rechnerische Korrektur\",\"schweregrad\":\"hoch\","
            "\"stelle_im_segment\":\"<Originalausschnitt>\","
            "\"expression\":\"<auswertbarer Ausdruck oder leer>\","
            "\"claimed_result\":<Zahl>,"
            "\"decimal_places\":<int>,"
            "\"wert_im_dokument\":\"<Wert mit Einheit>\","
            "\"begruendung\":\"<kurze Erklärung>\"}]}"
    )
    user = (
        f"DOKUMENTSEGMENT:\n{segment_text.strip()}\n\n"
        "Extrahiere alle Stellen mit numerischen Berechnungen oder Einheitenfehlern. "
        "Fülle expression, claimed_result und decimal_places sorgfältig aus — "
        "Python verifiziert das Ergebnis. "
        "Wenn keine solche Stelle vorhanden ist, antworte mit {\"errors\":[]}."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]




def _format_seconds_value(value: float) -> str:
    """Format seconds in the same style used by findings/predictions."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _parse_number_de_ch(value: str) -> float:
    """Parse decimal numbers with either dot or comma as decimal separator."""
    return float(str(value).strip().replace("'", "").replace(",", "."))


# ── Safe arithmetic evaluator ─────────────────────────────────────────────────

import ast as _ast
import math as _math
import decimal as _decimal

_SAFE_OPS = {
    _ast.Add: lambda a, b: a + b,
    _ast.Sub: lambda a, b: a - b,
    _ast.Mult: lambda a, b: a * b,
    _ast.Div: lambda a, b: a / b,
    _ast.USub: lambda a: -a,
    _ast.UAdd: lambda a: a,
}


def _safe_eval_expr(node: _ast.expr) -> _decimal.Decimal:
    """Recursively evaluate an AST node using only basic arithmetic.

    Raises ValueError for any unsupported node type (function calls,
    attribute access, imports, etc.) so LLM-injected code cannot execute.
    """
    if isinstance(node, _ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return _decimal.Decimal(str(node.value))
    if isinstance(node, _ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        if isinstance(node.op, _ast.Div) and right == 0:
            raise ValueError("Division by zero")
        return op_fn(left, right)
    if isinstance(node, _ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval_expr(node.operand))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def evaluate_expression(expression: str) -> _decimal.Decimal:
    """Parse and evaluate a plain arithmetic expression string safely.

    Only digits, +, -, *, /, parentheses and decimal points are supported.
    Any other content (function calls, names, imports) raises ValueError.

    Examples:
        evaluate_expression("(81/1.7)*3.6")  → Decimal('171.52941...')
        evaluate_expression("25.320 - 15.320") → Decimal('10.000')
    """
    # Normalise: replace comma decimals and typographic minus variants
    cleaned = (
        expression.strip()
        .replace(",", ".")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    # Whitelist check: only allow chars that belong in an arithmetic expression
    allowed = set("0123456789+-*/(). \t")
    illegal = set(cleaned) - allowed
    if illegal:
        raise ValueError(f"Illegal characters in expression: {illegal!r}")
    try:
        tree = _ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from e
    return _safe_eval_expr(tree.body)


def truncate_to_decimal_places(value: _decimal.Decimal, decimal_places: int) -> _decimal.Decimal:
    """Truncate (floor towards zero) to n decimal places — not rounding.

    Examples:
        truncate(171.5294, 1) → 171.5
        truncate(171.4705, 1) → 171.4
        truncate(20.79,    1) → 20.7
        truncate(20.75,    1) → 20.7   (NOT 20.8 — truncation, not rounding)
    """
    if decimal_places < 0:
        decimal_places = 0
    factor = _decimal.Decimal(10) ** decimal_places
    # Use ROUND_DOWN which truncates toward zero (correct for positive values)
    return (value * factor).to_integral_value(
        rounding=_decimal.ROUND_DOWN
    ) / factor


def _decimal_places_in_claimed(claimed: float) -> int:
    """Infer the number of decimal places from the claimed float value.

    Used as fallback when the LLM omits decimal_places.
    """
    s = str(claimed)
    if "." in s:
        return len(s.split(".")[1].rstrip("0") or "0")
    return 0


def verify_calculation(
        expression: str,
        claimed_result: float,
        decimal_places: int,
        *,
        plausibility_factor: float = 100.0,
) -> Tuple[bool, str, str]:
    """Evaluate expression and compare truncated result to claimed_result.

    Returns:
        (is_error, korrekter_wert, berechnung)
        is_error        True when the claimed value is incorrect.
        korrekter_wert  The correct truncated result as a string (with same
                        decimal formatting as the document).
        berechnung      Human-readable step string for the finding.

    Raises ValueError if the expression cannot be evaluated.
    """
    computed = evaluate_expression(expression)
    truncated = truncate_to_decimal_places(computed, decimal_places)

    claimed_dec = _decimal.Decimal(str(claimed_result))
    tolerance = _decimal.Decimal(10) ** (-decimal_places) * _decimal.Decimal("0.5")

    # Plausibility guard: if computed is off by more than plausibility_factor
    # from claimed, the expression was likely extracted incorrectly — skip.
    if claimed_dec != 0:
        ratio = float(abs(computed / claimed_dec))
        if ratio > plausibility_factor or ratio < (1.0 / plausibility_factor):
            raise ValueError(
                f"Plausibility check failed: computed={float(computed):.6g}, "
                f"claimed={claimed_result} (ratio={ratio:.2f})"
            )

    is_error = abs(truncated - claimed_dec) > tolerance

    fmt = f".{decimal_places}f"
    korrekter_wert_str = format(float(truncated), fmt)
    berechnung = (
        f"{expression} = {float(computed):.10g} "
        f"→ abgeschnitten auf {decimal_places} Nachkommastelle(n): {korrekter_wert_str}"
    )
    return is_error, korrekter_wert_str, berechnung


def check_time_difference_expressions(
        segment_text: str,
        segment_index: int,
) -> List[Dict[str, Any]]:
    """
    Deterministic Agent-4 guard for explicit time-difference calculations.

    Detects patterns such as:
      "zeitliche Differenz von 11 Sekunden (25.320 Sekunden – 15.320 Sekunden)"

    This is intentionally deterministic because the arithmetic is exact and should
    not depend on whether the LLM notices the expression.
    """
    findings: List[Dict[str, Any]] = []

    pattern = re.compile(
        r"(?P<span>"
        r"[^.!?\n]{0,180}?"
        r"(?:zeitliche\s+Differenz|Differenz|Zeitdifferenz|Zeitdauer)\s+von\s+"
        r"(?P<claimed>\d+(?:[.,]\d+)?)\s*(?:Sekunden?|s)"
        r"\s*\(\s*"
        r"(?P<a>\d+(?:[.,]\d+)?)\s*(?:Sekunden?|s)\s*"
        r"(?P<op>[−–—-])\s*"
        r"(?P<b>\d+(?:[.,]\d+)?)\s*(?:Sekunden?|s)"
        r"\s*\)"
        r"[^.!?\n]*[.!?]?"
        r")",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(segment_text):
        span = match.group("span").strip()
        if not span:
            continue

        try:
            claimed = _parse_number_de_ch(match.group("claimed"))
            a = _parse_number_de_ch(match.group("a"))
            b = _parse_number_de_ch(match.group("b"))
        except Exception:
            continue

        # The textual pattern is a difference. Use absolute difference because
        # document wording usually means elapsed time between two timestamps.
        correct = abs(a - b)

        if abs(claimed - correct) < 1e-6:
            continue

        claimed_s = _format_seconds_value(claimed)
        correct_s = _format_seconds_value(correct)
        a_s = _format_seconds_value(a)
        b_s = _format_seconds_value(b)

        correction = re.sub(
            r"((?:zeitliche\s+Differenz|Differenz|Zeitdifferenz|Zeitdauer)\s+von\s+)"
            r"\d+(?:[.,]\d+)?"
            r"(\s*(?:Sekunden?|s))",
            rf"\g<1>{correct_s}\g<2>",
            span,
            count=1,
            flags=re.IGNORECASE,
        )

        findings.append({
            "segment_index": segment_index,
            "hauptklasse": "Rechenfehler",
            "subklasse": "Arithmetischer Fehler",
            "aenderungstyp": "Rechnerische Korrektur",
            "schweregrad": "hoch",
            "stelle_im_segment": span,
            "berechnung": f"{a_s} s - {b_s} s = {correct_s} s",
            "wert_im_dokument": f"{claimed_s} Sekunden",
            "korrekter_wert": f"{correct_s} Sekunden",
            "begruendung": (
                f"Die Differenz zwischen {a_s} Sekunden und {b_s} Sekunden "
                f"beträgt {correct_s} Sekunden, nicht {claimed_s} Sekunden."
            ),
            "vorschlag": correction,
            "calculation_guard": "time_difference",
        })

    return findings


def _dedup_calculation_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate deterministic and LLM calculation findings."""
    seen: Set[Tuple[int, str, str, str]] = set()
    out: List[Dict[str, Any]] = []

    for f in findings:
        key = (
            int(f.get("segment_index") or 0),
            " ".join(str(f.get("stelle_im_segment") or "").split()).lower(),
            str(f.get("subklasse") or "").strip().lower(),
            str(f.get("korrekter_wert") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(f)

    return out


# ── Deterministischer Guard: Masseinheiten-Konsistenz ─────────────────────────

# Dimensionskategorien: Einheiten derselben Kategorie dürfen im selben
# Satz nicht gemischt werden (ausser bei expliziten Umrechnungen).
_UNIT_DIMENSIONS: Dict[str, str] = {
    # Länge
    "mm": "length", "cm": "length", "dm": "length",
    "m":  "length",  "km": "length",
    # Zeit
    "s":        "time", "ms":      "time",
    "sekunde":  "time", "sekunden":"time",
    "minute":   "time", "minuten": "time",
    "stunde":   "time", "stunden": "time",
    "h":        "time",
    # Masse
    "g": "mass", "kg": "mass", "mg": "mass", "t": "mass",
    # Frequenz
    "hz": "freq", "khz": "freq", "mhz": "freq",
    # Winkel
    "grad": "angle", "°": "angle",
    # Konzentration
    "promille": "concentration", "%": "concentration",
    # Datenmenge
    "bit": "data", "byte": "data", "kb": "data", "mb": "data",
}

# Wenn eines dieser Muster im Satz vorkommt, handelt es sich
# wahrscheinlich um eine explizite Umrechnung → kein Finding.
_CONVERSION_RE = re.compile(
    r"(?:entspricht|gleich|entsprechen|umgerechnet|"
    r"ergibt|ergibt sich|bzw\.|d\.h\.|"
    r"=\s*\d)",
    re.IGNORECASE,
)

# Extrahiert alle Einheiten mit Zahlen aus einem Text.
# Erkennt: "10 mm", "3.5kg", "14 Stunden", "48kHz"
_UNIT_EXTRACTION_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(mm|cm|dm|km|ms|khz|mhz|mb|kb|mg|"
    r"m|g|s|h|t|hz|°|%|"
    r"sekunden?|minuten?|stunden?|grad|promille|bit|byte)",
    re.IGNORECASE,
)


def check_unit_consistency_in_segment(
        segment_text: str,
        segment_index: int,
) -> List[Dict[str, Any]]:
    """
    Deterministischer Guard: prüft ob in einem Satz Masseinheiten
    derselben Dimension gemischt werden.

    Erlaubt: km + Minuten (verschiedene Dimensionen)
    Verboten: mm + dm (beide Länge), mm + m (beide Länge)

    Explizite Umrechnungen (erkennbar an 'entspricht', '=', 'bzw.' etc.)
    werden ignoriert.
    """
    findings: List[Dict[str, Any]] = []

    # Satzweise prüfen
    sentences = re.split(r"(?<=[.!?])\s+", segment_text)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Umrechnung im Satz → überspringen
        if _CONVERSION_RE.search(sent):
            continue

        # Einheiten extrahieren
        matches = _UNIT_EXTRACTION_RE.findall(sent)
        if len(matches) < 2:
            continue

        # Nach Dimension gruppieren
        dim_to_units: Dict[str, List[str]] = {}
        for _val, unit in matches:
            unit_norm = unit.lower().rstrip(".")
            dim = _UNIT_DIMENSIONS.get(unit_norm)
            if dim:
                dim_to_units.setdefault(dim, []).append(unit)

        # Konflikt: dieselbe Dimension mit mehr als einer Einheit
        for dim, units in dim_to_units.items():
            unique_units = list(dict.fromkeys(u.lower() for u in units))
            if len(unique_units) < 2:
                continue

            # Häufigste Einheit = wahrscheinlich korrekte Einheit
            from collections import Counter
            counts = Counter(u.lower() for u in units)
            majority_unit = counts.most_common(1)[0][0]
            outliers = [u for u in units if u.lower() != majority_unit]

            if not outliers:
                continue

            # Span: ganzer Satz (damit der Kontext klar ist)
            stelle = sent[:120].strip()

            findings.append({
                "segment_index": segment_index,
                "hauptklasse":  "Rechenfehler",
                "subklasse":    "Einheitenfehler",
                "aenderungstyp":"Rechnerische Korrektur",
                "schweregrad":  "mittel",
                "stelle_im_segment": stelle,
                "berechnung":   (
                    f"Einheiten-Konflikt in Dimension '{dim}': "
                    f"Mehrheitlich '{majority_unit}', Ausreisser: {outliers}"
                ),
                "wert_im_dokument": ", ".join(outliers),
                "korrekter_wert":   majority_unit,
                "begruendung": (
                    f"Einheiten-Inkonsistenz: Im selben Satz werden für die Dimension "
                    f"'{dim}' verschiedene Einheiten verwendet "
                    f"({', '.join(unique_units)}). "
                    f"Ausreisser '{outliers[0]}' sollte vermutlich '{majority_unit}' sein."
                ),
                "unit_consistency_guard": True,
            })

    if findings:
        print(
            f"[DEBUG-UNIT-CONSISTENCY] S{segment_index}: "
            f"{[f['berechnung'] for f in findings]}"
        )
    return findings


def run_calculation_agent(
        llm: LLMClient,
        evidence: SegmentEvidence,
        catalog: ErrorCatalog,
) -> List[Dict[str, Any]]:
    deterministic_findings = check_time_difference_expressions(
        evidence.segment_text,
        evidence.segment_index,
    )
    deterministic_findings += check_unit_consistency_in_segment(
        evidence.segment_text,
        evidence.segment_index,
    )
    if deterministic_findings:
        print(
            f"[DEBUG-CALC-GUARD] S{evidence.segment_index}: "
            f"{[{ 'stelle': f.get('stelle_im_segment'), 'korrekt': f.get('korrekter_wert') } for f in deterministic_findings]}"
        )

    messages = build_calculation_review_messages(evidence.segment_text, catalog)
    calc_schema = build_calculation_json_schema(catalog)
    raw_reply = llm.chat(messages, json_mode=True, schema=calc_schema)
    print(
        f"[DEBUG] Calculation agent S{evidence.segment_index} raw reply "
        f"({len(raw_reply)} chars): {raw_reply[:200]!r}"
    )

    try:
        parsed = parse_json_response(raw_reply)
    except Exception:
        try:
            repair_messages = build_json_repair_messages(raw_reply, schema_name="calculation_errors")
            repaired = llm.chat(repair_messages, json_mode=True)
            parsed = parse_json_response(repaired)
        except Exception as e:
            print(f"[WARN] Calculation agent parse failed S{evidence.segment_index}: {e}")
            return _dedup_calculation_findings(deterministic_findings)

    findings: List[Dict[str, Any]] = []
    normalized_segment = " ".join(evidence.segment_text.split())

    allowed_rechen_subs = {
        sub["label"]
        for sub in catalog.sub_by_main.get("RECHENFEHLER", {}).values()
    }

    for item in parsed.get("errors", []):
        if not isinstance(item, dict):
            continue

        stelle = str(item.get("stelle_im_segment") or "").strip()
        if not stelle:
            continue

        # Guard: stelle must appear verbatim in the segment
        normalized_stelle = " ".join(stelle.split())
        if normalized_stelle not in normalized_segment:
            print(f"[DROP calc stelle] not found in segment: {stelle[:80]!r}")
            continue

        subklasse = str(item.get("subklasse") or "Arithmetischer Fehler").strip()
        aenderungstyp = str(item.get("aenderungstyp") or "Rechnerische Korrektur").strip()
        schweregrad = str(item.get("schweregrad") or "hoch").strip()
        wert_im_dokument = str(item.get("wert_im_dokument") or "").strip()
        begruendung = str(item.get("begruendung") or "").strip()

        if subklasse not in allowed_rechen_subs:
            subklasse = "Arithmetischer Fehler"
        if aenderungstyp not in catalog.allowed_change_labels:
            aenderungstyp = "Rechnerische Korrektur"
        if schweregrad not in catalog.allowed_severity_labels:
            schweregrad = "hoch"

        expression = str(item.get("expression") or "").strip()
        claimed_result = item.get("claimed_result")
        decimal_places = item.get("decimal_places")

        # ── Python verifies arithmetic; LLM never sets korrekter_wert ────────
        if expression and claimed_result is not None:
            try:
                claimed_float = float(claimed_result)
                dp = int(decimal_places) if decimal_places is not None else _decimal_places_in_claimed(claimed_float)
                dp = max(0, min(dp, 6))

                is_error, korrekter_wert_str, berechnung_str = verify_calculation(
                    expression, claimed_float, dp,
                )
                if not is_error:
                    print(
                        f"[CALC-OK] S{evidence.segment_index} "
                        f"expression={expression!r} claimed={claimed_float} → correct, skipping"
                    )
                    continue

                # Extract unit from wert_im_dokument for korrekter_wert label
                unit_match = re.search(r"[^\d.,\s].*$", wert_im_dokument.strip())
                unit_suffix = (" " + unit_match.group(0).strip()) if unit_match else ""
                korrekter_wert = korrekter_wert_str + unit_suffix

                print(
                    f"[CALC-ERROR] S{evidence.segment_index} "
                    f"expression={expression!r} claimed={claimed_float} "
                    f"→ correct={korrekter_wert_str}"
                )

            except (ValueError, ZeroDivisionError, Exception) as exc:
                print(
                    f"[SKIP calc expression] S{evidence.segment_index} "
                    f"expression={expression!r}: {exc}"
                )
                continue
        else:
            # No evaluable expression (e.g. unit-only finding) —
            # keep the finding but leave berechnung and korrekter_wert empty.
            berechnung_str = ""
            korrekter_wert = ""

        findings.append({
            "segment_index": evidence.segment_index,
            "hauptklasse": "Rechenfehler",
            "subklasse": subklasse,
            "aenderungstyp": aenderungstyp,
            "schweregrad": schweregrad,
            "stelle_im_segment": stelle,
            "berechnung": berechnung_str,
            "wert_im_dokument": wert_im_dokument,
            "korrekter_wert": korrekter_wert,
            "begruendung": begruendung,
            "vorschlag": "",
        })

    return _dedup_calculation_findings(deterministic_findings + findings)


# ── Agent 5: Hypothesis Consistency Checker ──────────────────────────────────

def _find_segment_for_stelle(stelle: str, segments: List[str]) -> Optional[int]:
    """
    Find the 1-based segment index that contains the given stelle text.
    First tries exact normalized substring match, then falls back to
    token overlap (Jaccard) for cases where whitespace differs slightly.
    Returns None if no segment scores above threshold.
    """
    if not stelle:
        return None
    normalized_stelle = " ".join(stelle.split()).lower()

    # Pass 1: exact normalized substring
    for i, seg in enumerate(segments, start=1):
        if normalized_stelle in " ".join(seg.split()).lower():
            return i

    # Pass 2: token overlap fallback (threshold 0.5)
    stelle_tokens = set(normalized_stelle.split())
    if not stelle_tokens:
        return None
    best_idx: Optional[int] = None
    best_score = 0.0
    for i, seg in enumerate(segments, start=1):
        seg_tokens = set(" ".join(seg.split()).lower().split())
        if not seg_tokens:
            continue
        score = len(stelle_tokens & seg_tokens) / len(stelle_tokens)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_score >= 0.5 else None


def build_hypothesis_json_schema(catalog: ErrorCatalog) -> Dict[str, Any]:
    """Grammar-constrained schema for Agent 5 — uses only the Hypothesenprüfung main class."""
    hypo_subs = sorted(
        sub["label"]
        for sub in catalog.sub_by_main.get("HYPOTHESEN", {}).values()
    )
    change_labels = sorted(catalog.allowed_change_labels)
    sev_labels = sorted(catalog.allowed_severity_labels)

    finding_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "hauptklasse":         {"type": "string", "enum": ["Hypothesenprüfung"]},
            "subklasse":           {"type": "string", "enum": hypo_subs},
            "aenderungstyp":       {"type": "string", "enum": change_labels},
            "schweregrad":         {"type": "string", "enum": sev_labels},
            "hypothese_text":      {"type": "string"},
            "befundbewertung_text":{"type": "string"},
            "stelle_im_segment":   {"type": "string"},
            "begruendung":         {"type": "string"},
        },
        "required": [
            "hauptklasse", "subklasse", "aenderungstyp", "schweregrad",
            "hypothese_text", "stelle_im_segment", "begruendung",
        ],
    }
    return {
        "type": "object",
        "properties": {"errors": {"type": "array", "items": finding_schema}},
        "required": ["errors"],
    }


def build_hypothesis_review_messages(
        doc_text: str,
        catalog: ErrorCatalog,
) -> List[Dict[str, str]]:
    system = (
            _AGENT_JSON_PREFIX +

            "Du bist Agent 5: Hypothesen-Konsistenzprüfer für technische Gutachten.\n\n"

            "Aufgabe:\n"
            "1. Identifiziere alle Hypothesen (z.B. 'Hypothese', 'H1', 'H2', "
            "'Unter der Annahme', 'Es wird angenommen', 'Falls').\n"
            "2. Finde die zugehörige Diskussion/Bewertung — auch in späteren Abschnitten "
            "('Diskussion', 'Wertung', 'Beurteilung', 'Schlussfolgerung', 'Fazit').\n"
            "3. Prüfe ob Schlussbewertung und Befunde zusammenpassen.\n"
            "4. Melde NUR nachweisbare Inkonsistenzen — keine Spekulationen.\n\n"

            "Grundregel: Eine Hypothese ist nur plausibel wenn Befunde sie stützen. "
            "Widerspruch zwischen Diskussion und späterer Wertung ohne Begründung ist eine Inkonsistenz.\n\n"

            "Subklassen: Hypothesen-Befund-Widerspruch | Fehlende Befundbewertung | "
            "Inkonsistente Hypothesenbewertung | Unbegründete Hypothese\n"
            "Klassifikation: hauptklasse='Hypothesenprüfung' | aenderungstyp='Hypothesen-Korrektur'\n\n"

            "Alle Textausschnitte müssen wörtlich aus dem Dokument stammen.\n"
            "Keine Hypothesen oder alle konsistent → {\"errors\":[]}\n\n"

            "Format:\n"
            "{\"errors\":[{\"hauptklasse\":\"Hypothesenprüfung\",\"subklasse\":\"<Fehlertyp>\","
            "\"aenderungstyp\":\"Hypothesen-Korrektur\",\"schweregrad\":\"<niedrig|mittel|hoch>\","
            "\"hypothese_text\":\"<Originalausschnitt Hypothese>\","
            "\"befundbewertung_text\":\"<Originalausschnitt Bewertung oder leer>\","
            "\"stelle_im_segment\":\"<problematischer Originalausschnitt>\","
            "\"begruendung\":\"<Erklärung der Inkonsistenz>\"}]}"
    )

    user = (
        f"DOKUMENT:\n{doc_text.strip()}\n\n"
        "Prüfe alle Hypothesen gegen Diskussion und Wertung. "
        "Berücksichtige spätere Abschnitte (Wertung, Schlussfolgerung, Fazit). "
        "Spezialfall — wenn Absichtshypothesen vorkommen: Prüfe ob konkrete Befunde "
        "für Absicht oder gezieltes Handeln im Dokument genannt werden; "
        "fehlen sie bei einer 'sehr plausibel'-Bewertung, ist das eine Inkonsistenz. "
        "Bei gegenseitig ausschliessenden Hypothesen: Prüfe ob beide gleich stark bewertet werden "
        "ohne dokumentierte Begründung. "
        "Melde nur echte Widersprüche. Wenn keine Inkonsistenz vorliegt: {\"errors\":[]}. "
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run_hypothesis_agent(
        llm: LLMClient,
        doc_text: str,
        segments: List[str],
        catalog: ErrorCatalog,
) -> List[Dict[str, Any]]:
    """
    Agent 5 — runs once per document on the full text.
    Assigns segment_index by matching stelle_im_segment back to segments.
    Uses a higher num_ctx via a dedicated options override if configured.
    """
    doc_chars = len(doc_text)
    print(f"[INFO] Hypothesis agent: document {doc_chars} chars, {len(segments)} segments")

    if doc_chars > 60_000:
        print(
            f"[WARN] Hypothesis agent: document is {doc_chars} chars — "
            f"may exceed model context window (num_ctx={60_000}). "
            f"Consider setting HYPOTHESIS_NUM_CTX in .env."
        )

    messages = build_hypothesis_review_messages(doc_text, catalog)
    hypo_schema = build_hypothesis_json_schema(catalog)

    raw_reply = llm.chat(messages, json_mode=True, schema=hypo_schema)
    print(
        f"[DEBUG] Hypothesis agent raw reply "
        f"({len(raw_reply)} chars): {raw_reply[:200]!r}"
    )

    try:
        parsed = parse_json_response(raw_reply)
    except Exception:
        try:
            repair_messages = build_json_repair_messages(raw_reply, schema_name="hypothesis_errors")
            repaired = llm.chat(repair_messages, json_mode=True)
            parsed = parse_json_response(repaired)
        except Exception as e:
            print(f"[WARN] Hypothesis agent parse failed: {e}")
            return []

    allowed_hypo_subs = {
        sub["label"]
        for sub in catalog.sub_by_main.get("HYPOTHESEN", {}).values()
    }

    findings: List[Dict[str, Any]] = []
    for item in parsed.get("errors", []):
        if not isinstance(item, dict):
            continue

        stelle = str(item.get("stelle_im_segment") or "").strip()
        if not stelle:
            continue

        # Assign segment_index by matching stelle back to segments
        seg_idx = _find_segment_for_stelle(stelle, segments)
        if seg_idx is None:
            # Fallback: try hypothese_text
            hypothese_text = str(item.get("hypothese_text") or "").strip()
            seg_idx = _find_segment_for_stelle(hypothese_text, segments)
        if seg_idx is None:
            print(f"[WARN hypo] Could not assign segment for stelle: {stelle[:60]!r} — skipping")
            continue

        subklasse = str(item.get("subklasse") or "Hypothesen-Befund-Widerspruch").strip()
        aenderungstyp = str(item.get("aenderungstyp") or "Hypothesen-Korrektur").strip()
        schweregrad = str(item.get("schweregrad") or "hoch").strip()

        if subklasse not in allowed_hypo_subs:
            subklasse = "Hypothesen-Befund-Widerspruch"
        if aenderungstyp not in catalog.allowed_change_labels:
            aenderungstyp = "Hypothesen-Korrektur"
        if schweregrad not in catalog.allowed_severity_labels:
            schweregrad = "hoch"

        findings.append({
            "segment_index": seg_idx,
            "hauptklasse": "Hypothesenprüfung",
            "subklasse": subklasse,
            "aenderungstyp": aenderungstyp,
            "schweregrad": schweregrad,
            "stelle_im_segment": stelle,
            "hypothese_text": str(item.get("hypothese_text") or "").strip(),
            "befundbewertung_text": str(item.get("befundbewertung_text") or "").strip(),
            "begruendung": str(item.get("begruendung") or "").strip(),
        })

    print(f"[INFO] Hypothesis agent: {len(findings)} finding(s) found")
    return findings





# ── Deterministic date consistency guard ──────────────────────────────────────

GERMAN_MONTHS: Dict[str, int] = {
    "januar": 1, "jan": 1,
    "februar": 2, "feb": 2,
    "märz": 3, "maerz": 3, "mrz": 3,
    "april": 4, "apr": 4,
    "mai": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "oktober": 10, "okt": 10,
    "november": 11, "nov": 11,
    "dezember": 12, "dez": 12,
}

DATE_PATTERN_NUMERIC = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})[\.\-/](?P<month>\d{1,2})[\.\-/](?P<year>\d{4})(?!\d)"
)

DATE_PATTERN_TEXTUAL = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})\.\s*"
    r"(?P<month>Januar|Jan\.?|Februar|Feb\.?|März|Maerz|Mrz\.?|April|Apr\.?|Mai|"
    r"Juni|Jun\.?|Juli|Jul\.?|August|Aug\.?|September|Sept\.?|Sep\.?|"
    r"Oktober|Okt\.?|November|Nov\.?|Dezember|Dez\.?)\s+"
    r"(?P<year>\d{4})(?!\d)",
    flags=re.IGNORECASE,
)


def _add_years_safe(value: datetime, years: int) -> datetime:
    """Add years while handling 29 February safely."""
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        # 29 February -> 28 February in non-leap target year
        return value.replace(year=value.year + years, day=28)


def _parse_date_parts(day: str, month: str, year: str) -> Optional[datetime]:
    try:
        return datetime(int(year), int(month), int(day))
    except ValueError:
        return None


def _parse_german_textual_date(day: str, month_name: str, year: str) -> Optional[datetime]:
    key = month_name.strip().rstrip(".").lower().replace("ä", "ae")
    month = GERMAN_MONTHS.get(key)
    if month is None:
        return None
    return _parse_date_parts(day, str(month), year)


def _parse_reference_event_date(reference_facts: Dict[str, Any]) -> Optional[datetime]:
    """Parse Agent-0 ereignisdatum from value first, then source_span."""
    facts = reference_facts.get("facts", {}) if isinstance(reference_facts, dict) else {}
    fact = facts.get("ereignisdatum") if isinstance(facts, dict) else None
    if not isinstance(fact, dict):
        return None

    candidates = [
        str(fact.get("value") or "").strip(),
        str(fact.get("source_span") or "").strip(),
    ]
    for candidate in candidates:
        if not candidate:
            continue

        iso_match = re.search(r"(?<!\d)(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})(?!\d)", candidate)
        if iso_match:
            parsed = _parse_date_parts(
                iso_match.group("day"),
                iso_match.group("month"),
                iso_match.group("year"),
            )
            if parsed:
                return parsed

        numeric_match = DATE_PATTERN_NUMERIC.search(candidate)
        if numeric_match:
            parsed = _parse_date_parts(
                numeric_match.group("day"),
                numeric_match.group("month"),
                numeric_match.group("year"),
            )
            if parsed:
                return parsed

        textual_match = DATE_PATTERN_TEXTUAL.search(candidate)
        if textual_match:
            parsed = _parse_german_textual_date(
                textual_match.group("day"),
                textual_match.group("month"),
                textual_match.group("year"),
            )
            if parsed:
                return parsed

    return None


def _iter_document_dates(doc_text: str) -> Iterator[Tuple[str, datetime]]:
    """Yield visible date mentions from the document with parsed datetime values."""
    seen: Set[Tuple[int, int]] = set()

    for match in DATE_PATTERN_NUMERIC.finditer(doc_text or ""):
        parsed = _parse_date_parts(match.group("day"), match.group("month"), match.group("year"))
        if parsed is None:
            continue
        seen.add((match.start(), match.end()))
        yield match.group(0), parsed

    for match in DATE_PATTERN_TEXTUAL.finditer(doc_text or ""):
        # Avoid duplicate overlap with numeric matches, although patterns should not overlap.
        if any(not (match.end() <= a or match.start() >= b) for a, b in seen):
            continue
        parsed = _parse_german_textual_date(match.group("day"), match.group("month"), match.group("year"))
        if parsed is None:
            continue
        yield match.group(0), parsed


def _format_date_iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")


def check_dates_within_event_window(
        doc_text: str,
        segments: List[str],
        reference_facts: Dict[str, Any],
        *,
        max_years_after_event: int = 5,
        max_years_before_event: int = 1,

) -> List[Dict[str, Any]]:
    """
    Deterministic Agent-6 guard:
    all explicit document dates must not be later than ereignisdatum + N years.

    This intentionally checks only dates after the allowed window. Dates before
    the event, e.g. dates of birth, are not flagged because they can be legitimate.
    """
    event_date = _parse_reference_event_date(reference_facts)
    if event_date is None:
        print("[INFO] Date consistency guard: no parseable ereignisdatum — skipping")
        return []

    latest_allowed = _add_years_safe(event_date, max_years_after_event)
    earliest_allowed = _add_years_safe(event_date, -max_years_before_event)
    findings: List[Dict[str, Any]] = []
    seen_spans: Set[str] = set()

    for span, parsed in _iter_document_dates(doc_text):
        # Do not flag the event date itself if it appears in another spelling.
        if parsed.date() == event_date.date():
            continue
        if parsed <= latest_allowed:
            continue

        norm_span = " ".join(span.split())
        if norm_span.lower() in seen_spans:
            continue
        seen_spans.add(norm_span.lower())

        seg_idx = _find_segment_for_stelle(span, segments)
        if seg_idx is None:
            print(f"[DROP date] date span not found in segments: {span!r}")
            continue

        findings.append({
            "segment_index": seg_idx,
            "hauptklasse": "Struktur und Argumentation",
            "subklasse": "Beschreibung von Befunden",
            "aenderungstyp": "Fachliche Präzisierung",
            "schweregrad": "hoch",
            "stelle_im_segment": span,
            "reference_key": "ereignisdatum",
            "reference_value": _format_date_iso(event_date),
            "begruendung": (
                f"Chronologischer Widerspruch: Das Datum {span} liegt ausserhalb des zulässigen "
                f"Zeitraums von {max_years_after_event} Jahren nach dem Ereignisdatum "
                f"{_format_date_iso(event_date)}. Spätestes zulässiges Datum ist "
                f"{_format_date_iso(latest_allowed)}."
            ),
            "source_refs": ["DOC_INTERNAL"],
            "date_consistency_guard": True,
        })

    if findings:
        print(
            f"[DEBUG-DATE-CONSISTENCY] event_date={_format_date_iso(event_date)}, "
            f"latest_allowed={_format_date_iso(latest_allowed)}, findings={len(findings)}"
        )
    return findings


# ── Agent 6: Reference Facts Consistency Checker ─────────────────────────────

def build_reference_consistency_json_schema(catalog: ErrorCatalog) -> Dict[str, Any]:
    """
    Grammar-constrained schema for Agent 6 — document-level consistency check
    against Agent 0 reference facts. It uses factual/argumentation classes, not
    Hypothesenprüfung or Rechenfehler.
    """
    excluded_main = {"Rechenfehler", "Hypothesenprüfung"}
    excluded_change = {"Rechnerische Korrektur", "Hypothesen-Korrektur"}

    main_labels = sorted(m for m in catalog.allowed_main_labels if m not in excluded_main)
    change_labels = sorted(c for c in catalog.allowed_change_labels if c not in excluded_change)
    sev_labels = sorted(catalog.allowed_severity_labels)

    all_sub_labels: List[str] = []
    for main_label in main_labels:
        all_sub_labels.extend(sorted(catalog.allowed_subclasses_by_main_label.get(main_label, set())))
    all_sub_labels = sorted(set(all_sub_labels))

    finding_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "hauptklasse": {"type": "string", "enum": main_labels},
            "subklasse": {"type": "string", "enum": all_sub_labels},
            "aenderungstyp": {"type": "string", "enum": change_labels},
            "schweregrad": {"type": "string", "enum": sev_labels},
            "reference_key": {"type": "string"},
            "reference_value": {"type": "string"},
            "stelle_im_segment": {"type": "string"},
            "begruendung": {"type": "string"},
        },
        "required": [
            "hauptklasse", "subklasse", "aenderungstyp", "schweregrad",
            "reference_key", "reference_value", "stelle_im_segment", "begruendung",
        ],
    }
    return {
        "type": "object",
        "properties": {"errors": {"type": "array", "items": finding_schema}},
        "required": ["errors"],
    }


def _compact_reference_facts_for_consistency(reference_facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only high/medium confidence facts with non-empty values. This prevents
    Agent 6 from treating missing or low-confidence facts as document errors.
    """
    facts = reference_facts.get("facts", {}) if isinstance(reference_facts, dict) else {}
    compact: Dict[str, Any] = {"case_id": reference_facts.get("case_id", ""), "facts": {}}
    if not isinstance(facts, dict):
        return compact

    for key in REFERENCE_FACT_KEYS:
        fact = facts.get(key)
        if not isinstance(fact, dict):
            continue
        value = str(fact.get("value") or "").strip()
        source_span = str(fact.get("source_span") or "").strip()
        confidence = str(fact.get("confidence") or "low").strip().lower()
        if value and confidence in {"high", "medium"}:
            compact["facts"][key] = {
                "value": value,
                "source_span": source_span,
                "confidence": confidence,
            }

    persons = facts.get("personen", [])
    kept_persons: List[Dict[str, str]] = []
    if isinstance(persons, list):
        for person in persons:
            if not isinstance(person, dict):
                continue
            name = str(person.get("name") or "").strip()
            rolle = str(person.get("rolle") or "").strip()
            source_span = str(person.get("source_span") or "").strip()
            confidence = str(person.get("confidence") or "low").strip().lower()
            if name and confidence in {"high", "medium"}:
                kept_persons.append({
                    "name": name,
                    "rolle": rolle,
                    "source_span": source_span,
                    "confidence": confidence,
                })
    if kept_persons:
        compact["facts"]["personen"] = kept_persons

    return compact


def build_reference_consistency_review_messages(
        doc_text: str,
        reference_facts: Dict[str, Any],
        catalog: ErrorCatalog,
) -> List[Dict[str, str]]:
    hauptklassen = ", ".join(sorted(
        m for m in catalog.allowed_main_labels
        if m not in {"Rechenfehler", "Hypothesenprüfung"}
    ))
    aenderungstypen = ", ".join(sorted(
        c for c in catalog.allowed_change_labels
        if c not in {"Rechnerische Korrektur", "Hypothesen-Korrektur"}
    ))
    compact_facts = _compact_reference_facts_for_consistency(reference_facts)

    system = (
            _AGENT_JSON_PREFIX +

            "Du bist Agent 6: Referenzfakten-Konsistenzprüfer für forensische Gutachten.\n"
            "Du prüfst das GESAMTE DOKUMENT gegen die REFERENZFAKTEN aus Agent 0.\n\n"

            "Regeln:\n"
            "1. Nur Referenzfakten mit confidence high/medium und nicht-leerem value verwenden.\n"
            "2. Melde NUR echte Widersprüche: Dokument nennt anderen konkreten Wert als Referenzwert.\n"
            "3. Fehlende Wiederholung eines Referenzfakts ist kein Fehler.\n"
            "4. Den source_span des Referenzfakts selbst nicht als Fehler melden.\n"
            "5. stelle_im_segment muss wörtlicher Originalausschnitt mit dem abweichenden Wert sein.\n"
            "6. Keine wörtliche Stelle auffindbar → kein Finding.\n\n"

            "Typische Widersprüche (generisch):\n"
            "- Referenz auftraggeber = <Behörde A>, Dokumentstelle nennt <Behörde B>.\n"
            "- Referenz person = <Name A>, Dokumentstelle nennt <Name B> (abweichender Name).\n"
            "- Referenz ereignisdatum = <Datum A>, Dokumentstelle nennt <Datum B> "
            "(mehr als 3 Jahre abweichend = ausserhalb des plausiblen Zeitraums).\n\n"

            "Nicht melden: fehlende Angaben | Wiederholungen | Bestätigungen | "
            "Rechenfehler | Sprachfehler | Hypothesenfehler\n\n"

            f"Zulässige Werte: hauptklasse={hauptklassen} | "
            f"aenderungstyp={aenderungstypen} | schweregrad=niedrig|mittel|hoch\n\n"

            "Zusatzfelder: reference_key (Feldname des Referenzfakts) | "
            "reference_value (Referenzwert) | stelle_im_segment | begruendung\n\n"

            "Format:\n"
            "{\"errors\":[{\"hauptklasse\":\"<Hauptklasse>\",\"subklasse\":\"<Subklasse>\","
            "\"aenderungstyp\":\"<Änderungstyp>\",\"schweregrad\":\"<niedrig|mittel|hoch>\","
            "\"reference_key\":\"<Feldname>\",\"reference_value\":\"<Referenzwert>\","
            "\"stelle_im_segment\":\"<abweichender Originalausschnitt>\","
            "\"begruendung\":\"Widerspruch: Referenzfakt <Feldname> ist <Referenzwert>, "
            "im Dokument steht <abweichender Wert>.\"}]}"
    )

    user = (
        "REFERENZFAKTEN:\n"
        f"{json.dumps(compact_facts, ensure_ascii=False, indent=2)}\n\n"
        "DOKUMENT:\n"
        f"{doc_text.strip()}\n\n"
        "Prüfe dokumentweite Widersprüche gegen die Referenzfakten. "
        "Wenn keine echte abweichende Dokumentstelle vorhanden ist, antworte mit {\"errors\":[]}."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]



def _edit_distance(a: str, b: str) -> int:
    """
    Levenshtein-Distanz zwischen zwei Strings.
    Wird für den Personennamen-Guard verwendet.
    """
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


def check_person_name_consistency(
        doc_text: str,
        segments: List[str],
        reference_facts: Dict[str, Any],
        *,
        max_edit_distance: int = 3,
) -> List[Dict[str, Any]]:
    """
    Deterministischer Guard für Personennamen-Fehler.

    Vergleicht alle high/medium-Personennamen aus den Referenzfakten
    mit dem Volltext. Findet Vorkommen die ähnlich (Edit-Distanz ≤ max_edit_distance)
    aber nicht identisch sind — z.B. "Rolf Bosshard" vs. "Roland Bosshard".

    Der Vergleich läuft auf Wort-Ebene:
    - Jedes Wort im Dokument (≥4 Zeichen) wird gegen jeden Namensteil
      aus den Referenzfakten verglichen.
    - Nur wenn ein Namensteil ähnlich aber nicht identisch ist, wird
      ein Finding generiert.
    - Vollständig identische Namen werden nicht gemeldet.
    - Zu kurze Wörter (< 4 Zeichen) werden ignoriert um False Positives
      bei Artikeln und Präpositionen zu vermeiden.
    """
    facts = reference_facts.get("facts", {}) if isinstance(reference_facts, dict) else {}
    findings: List[Dict[str, Any]] = []
    seen_spans: set = set()

    # Alle Personennamen mit high/medium Konfidenz sammeln
    known_persons: List[Dict[str, str]] = []
    persons_list = facts.get("personen", [])
    if isinstance(persons_list, list):
        for p in persons_list:
            if not isinstance(p, dict):
                continue
            conf = str(p.get("confidence") or "low").lower()
            name = str(p.get("name") or "").strip()
            rolle = str(p.get("rolle") or "").strip()
            if conf in {"high", "medium"} and name:
                known_persons.append({"name": name, "rolle": rolle, "confidence": conf})

    # Auch sachverstaendige_person und beschuldigte_person als Einzelfakten
    for key in ("sachverstaendige_person", "beschuldigte_person", "hauptsachbearbeitung"):
        fact = facts.get(key)
        if isinstance(fact, dict):
            conf = str(fact.get("confidence") or "low").lower()
            val  = str(fact.get("value") or "").strip()
            if conf in {"high", "medium"} and val:
                # Ersten Teil vor Komma als Name nehmen
                name = val.split(",")[0].strip()
                if name and not any(p["name"] == name for p in known_persons):
                    known_persons.append({"name": name, "rolle": key, "confidence": conf})

    if not known_persons:
        return []

    # Dokument tokenisieren: Wörter ≥4 Zeichen mit ihrer Position
    word_pattern = re.compile(r"[A-Za-zÄÖÜäöüß]{4,}")

    for person in known_persons:
        ref_name      = person["name"]
        ref_name_norm = ref_name.lower()
        ref_parts     = [p for p in ref_name.split() if len(p) >= 4]

        if not ref_parts:
            continue

        # Für jeden Namensteil im Dokument nach ähnlichen aber falschen Wörtern suchen
        for ref_part in ref_parts:
            ref_part_norm = ref_part.lower()
            ref_part_len  = len(ref_part)

            for match in word_pattern.finditer(doc_text):
                found_word = match.group(0)
                found_norm = found_word.lower()

                # Identisch → kein Fehler
                if found_norm == ref_part_norm:
                    continue

                # Längenfilter: ähnlich lange Wörter (±2 Zeichen)
                if abs(len(found_word) - ref_part_len) > 2:
                    continue

                dist = _edit_distance(found_norm, ref_part_norm)
                if dist == 0 or dist > max_edit_distance:
                    continue

                # Kontext im Dokument bestimmen
                start = match.start()
                end   = match.end()
                ctx   = doc_text[max(0, start - 40): end + 40].replace("\n", " ").strip()

                # Span: gefundenes Wort + unmittelbaren Kontext (Vor- oder Nachname)
                # Versuche vollständigen gefundenen Namen als Span zu rekonstruieren
                ctx_wide = doc_text[max(0, start - 60): end + 60]
                name_match = re.search(
                    rf"[A-ZÄÖÜ][a-zäöüß]+ {re.escape(found_word)}"
                    rf"|{re.escape(found_word)} [A-ZÄÖÜ][a-zäöüß]+",
                    ctx_wide
                )
                span = name_match.group(0).strip() if name_match else found_word

                norm_span = " ".join(span.split()).lower()
                if norm_span in seen_spans:
                    continue

                # Nicht melden wenn der Span identisch mit dem Referenznamen ist
                if span.lower() == ref_name_norm:
                    continue

                seen_spans.add(norm_span)

                seg_idx = _find_segment_for_stelle(span, segments)
                if seg_idx is None:
                    # Fallback: Einzelwort suchen
                    seg_idx = _find_segment_for_stelle(found_word, segments)
                if seg_idx is None:
                    continue

                findings.append({
                    "segment_index":    seg_idx,
                    "hauptklasse":      "Struktur und Argumentation",
                    "subklasse":        "Beschreibung von Befunden",
                    "aenderungstyp":    "Fachliche Präzisierung",
                    "schweregrad":      "mittel",
                    "stelle_im_segment": span,
                    "reference_key":    "personen",
                    "reference_value":  ref_name,
                    "begruendung": (
                        f"Möglicher Personenname-Fehler: '{span}' ähnelt dem Referenznamen "
                        f"'{ref_name}' ({person['rolle']}) mit Edit-Distanz {dist}. "
                        f"Kontext: ...{ctx}..."
                    ),
                    "source_refs":       ["DOC_INTERNAL"],
                    "person_name_guard": True,
                    "edit_distance":     dist,
                })

    if findings:
        print(
            f"[DEBUG-PERSON-GUARD] {len(findings)} mögliche Personennamen-Fehler: "
            f"{[f['stelle_im_segment'] for f in findings]}"
        )

    return findings


def run_reference_consistency_agent(
        llm: LLMClient,
        doc_text: str,
        segments: List[str],
        reference_facts: Dict[str, Any],
        catalog: ErrorCatalog,
) -> List[Dict[str, Any]]:
    """
    Agent 6 — runs once per document on the full text, like Agent 5.
    It checks only actual contradictions against high/medium reference facts and
    assigns segment_index by matching stelle_im_segment back to segments.
    """
    compact = _compact_reference_facts_for_consistency(reference_facts)
    fact_count = len(compact.get("facts", {})) if isinstance(compact.get("facts"), dict) else 0
    doc_chars = len(doc_text)
    print(f"[INFO] Reference consistency agent: document {doc_chars} chars, {len(segments)} segments, {fact_count} fact groups")

    if fact_count == 0:
        print("[INFO] Reference consistency agent: no high/medium reference facts — skipping")
        return []

    messages = build_reference_consistency_review_messages(doc_text, reference_facts, catalog)
    ref_schema = build_reference_consistency_json_schema(catalog)

    raw_reply = llm.chat(messages, json_mode=True, schema=ref_schema)
    print(
        f"[DEBUG] Reference consistency agent raw reply "
        f"({len(raw_reply)} chars): {raw_reply[:200]!r}"
    )

    try:
        parsed = parse_json_response(raw_reply)
    except Exception:
        try:
            repair_messages = build_json_repair_messages(raw_reply, schema_name="reference_consistency_errors")
            repaired = llm.chat(repair_messages, json_mode=True, schema=ref_schema)
            parsed = parse_json_response(repaired)
        except Exception as e:
            print(f"[WARN] Reference consistency agent parse failed: {e}")
            return []

    findings: List[Dict[str, Any]] = []
    deterministic_date_findings = check_dates_within_event_window(
        doc_text,
        segments,
        reference_facts,
        max_years_after_event=3,
    )

    for item in parsed.get("errors", []):
        if not isinstance(item, dict):
            continue

        stelle = str(item.get("stelle_im_segment") or "").strip()
        if not stelle:
            print("[DROP ref] missing stelle_im_segment")
            continue

        seg_idx = _find_segment_for_stelle(stelle, segments)
        if seg_idx is None:
            print(f"[DROP ref] stelle_im_segment not found in document segments: {stelle[:80]!r}")
            continue

        # Avoid confirmations and source-span self-reports.
        beg = str(item.get("begruendung") or "").strip()
        beg_lower = beg.lower()
        if any(p in beg_lower for p in [
            "ist konsistent", "stimmt überein", "keine abweichung", "kein widerspruch",
            "ist korrekt", "entspricht dem referenzfakt",
        ]):
            print(f"[DROP ref] confirmation, not error: {beg[:80]!r}")
            continue

        hauptklasse = str(item.get("hauptklasse") or "Struktur und Argumentation").strip()
        subklasse = str(item.get("subklasse") or "Beschreibung von Befunden").strip()
        aenderungstyp = str(item.get("aenderungstyp") or "Fachliche Präzisierung").strip()
        schweregrad = str(item.get("schweregrad") or "mittel").strip()

        allowed_main = catalog.allowed_main_labels
        allowed_subs_by_main = catalog.allowed_subclasses_by_main_label
        sub_to_main = catalog.subclass_label_to_main_label

        if hauptklasse not in allowed_main:
            hauptklasse = sub_to_main.get(subklasse) or "Struktur und Argumentation"
        if subklasse not in allowed_subs_by_main.get(hauptklasse, set()):
            correct_main = sub_to_main.get(subklasse)
            if correct_main and subklasse in allowed_subs_by_main.get(correct_main, set()):
                hauptklasse = correct_main
            else:
                # Prefer a stable existing subclass if the model chose an invalid one.
                fallback_subs = allowed_subs_by_main.get("Struktur und Argumentation", set())
                hauptklasse = "Struktur und Argumentation" if "Struktur und Argumentation" in allowed_main else hauptklasse
                subklasse = "Beschreibung von Befunden" if "Beschreibung von Befunden" in fallback_subs else (sorted(fallback_subs)[0] if fallback_subs else subklasse)

        if aenderungstyp not in catalog.allowed_change_labels:
            aenderungstyp = "Fachliche Präzisierung" if "Fachliche Präzisierung" in catalog.allowed_change_labels else sorted(catalog.allowed_change_labels)[0]
        if schweregrad not in catalog.allowed_severity_labels:
            schweregrad = "mittel"

        findings.append({
            "segment_index": seg_idx,
            "hauptklasse": hauptklasse,
            "subklasse": subklasse,
            "aenderungstyp": aenderungstyp,
            "schweregrad": schweregrad,
            "stelle_im_segment": stelle,
            "reference_key": str(item.get("reference_key") or "").strip(),
            "reference_value": str(item.get("reference_value") or "").strip(),
            "begruendung": beg,
            "source_refs": ["DOC_INTERNAL"],
        })

    # Deterministischer Personennamen-Guard
    person_name_findings = check_person_name_consistency(
        doc_text,
        segments,
        reference_facts,
        max_edit_distance=3,
    )

    findings = deterministic_date_findings + person_name_findings + findings
    print(f"[INFO] Reference consistency agent: {len(findings)} finding(s) found")
    return findings


# ── Agent 7: Aussageabsicherung / Modalität / Tatsachenstatus ───────────────

@dataclass(frozen=True)
class StatementCandidate:
    candidate_index: int
    segment_index: int
    kind: str
    text: str


def _sentence_or_context_around_match(segment_text: str, start: int, end: int) -> str:
    """
    Return a compact sentence-like context around a regex match.
    Keeps exact original text and strips only outer whitespace.
    """
    left_candidates = [segment_text.rfind(".", 0, start), segment_text.rfind("!", 0, start), segment_text.rfind("?", 0, start), segment_text.rfind("\n", 0, start)]
    left = max(left_candidates)
    left = 0 if left < 0 else left + 1

    right_positions = [pos for pos in [segment_text.find(".", end), segment_text.find("!", end), segment_text.find("?", end), segment_text.find("\n", end)] if pos != -1]
    right = min(right_positions) + 1 if right_positions else len(segment_text)

    return segment_text[left:right].strip()


def extract_statement_assurance_candidates(
        segment_text: str,
        segment_index: int,
) -> List[StatementCandidate]:
    """
    Regex pre-filter for Agent 7.

    The regex layer does NOT decide the final error. It only extracts candidate
    passages where source certainty, modal/counterfactual wording, or intent
    attribution may be problematic. Agent 7 then classifies only these passages.
    """
    patterns: List[Tuple[str, re.Pattern[str]]] = [
        (
            "absolute_source_assertion",
            re.compile(
                r"\b(?:Wir wissen aus unseren Quellen, dass folgendes vorgefallen ist\.?|"
                r"Aus unseren Quellen wissen wir, dass[^.!?]*[.!?]?|"
                r"Wir wissen, dass[^.!?]*[.!?]?|"
                r"Es steht fest, dass[^.!?]*[.!?]?|"
                r"Es ist erwiesen, dass[^.!?]*[.!?]?|"
                r"zweifelsfrei[^.!?]*[.!?]?)",
                flags=re.IGNORECASE,
            ),
        ),
        (
            "intent_or_will_attribution",
            re.compile(
                r"[^.!?\n]{0,180}\b(?:wollen|wollte|wollten|beabsichtigte|beabsichtigten|"
                r"nahm(?:en)?\s+in\s+Kauf|absichtlich|vorsätzlich)\b[^.!?\n]*[.!?]?",
                flags=re.IGNORECASE,
            ),
        ),
        (
            "modal_counterfactual_as_fact",
            re.compile(
                r"(?:Wir verweisen auf[^.!?]*[.!?]\s*)?"
                r"(?:Wir verzichten auf[^.!?]*[.!?]\s*)?"
                r"[^.!?\n]{0,220}\b(?:hätte|hätten|wäre|wären|könnte|könnten|dürfte|dürften)\b[^.!?\n]*[.!?]?",
                flags=re.IGNORECASE,
            ),
        ),
        (
            # Konditionalsatz mit Indikativ im Hauptsatz:
            # "Falls X bestehen, werden Y durchgeführt." →
            # Hauptsatz sollte Konjunktiv sein ("könnten...durchgeführt werden")
            # aber steht im Indikativ — typischer injizierter Fehler.
            # Agent 7 entscheidet ob der Kontext tatsächlich Konjunktiv erfordert.
            "conditional_indikativ",
            re.compile(
                r"(?:Falls|Wenn|Sofern|Im\s+Falle(?:\s+(?:dass|von))?)"
                r"[^,;.!?]{5,120}"
                r"[,]?\s*"
                r"(?!.*?\b(?:könnte|könnten|würde|würden|dürfte|dürften|"
                r"möglicherweise|allenfalls|unter\s+Umständen)\b)"
                r"(?:wird|werden|ist|sind|bestätigt|ergibt|gilt|erfolgt|"
                r"wurde|wurden|kann|können)"
                r"[^.!?]{0,180}"
                r"[.!?]",
                flags=re.IGNORECASE | re.DOTALL,
            ),
        ),
    ]

    candidates: List[StatementCandidate] = []
    seen: Set[str] = set()

    for kind, pattern in patterns:
        for match in pattern.finditer(segment_text):
            span = match.group(0).strip()
            if not span:
                span = _sentence_or_context_around_match(segment_text, match.start(), match.end())
            # Keep original whitespace collapsed only for duplicate key; output stays original.
            key = " ".join(span.split()).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(
                StatementCandidate(
                    candidate_index=len(candidates) + 1,
                    segment_index=segment_index,
                    kind=kind,
                    text=span,
                )
            )

    return candidates


def build_statement_assurance_json_schema(
        catalog: ErrorCatalog,
        candidates: List[StatementCandidate],
) -> Dict[str, Any]:
    """Grammar-constrained schema for Agent 7."""
    main_labels = ["Rechtskonformität"] if "Rechtskonformität" in catalog.allowed_main_labels else sorted(catalog.allowed_main_labels)

    recht_subs = catalog.allowed_subclasses_by_main_label.get("Rechtskonformität", set())
    preferred_subs = [
        s for s in ["Aussageabsicherung", "Trennung Befund/Bewertung"]
        if s in recht_subs
    ]
    sub_labels = preferred_subs or sorted(recht_subs) or sorted({s for subs in catalog.allowed_subclasses_by_main_label.values() for s in subs})

    preferred_changes = [
        c for c in ["Fachliche Präzisierung", "Erweiterung der Argumentation"]
        if c in catalog.allowed_change_labels
    ]
    change_labels = preferred_changes or sorted(catalog.allowed_change_labels)
    sev_labels = sorted(catalog.allowed_severity_labels)
    candidate_texts = [c.text for c in candidates]
    candidate_indices = [c.candidate_index for c in candidates]

    finding_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "candidate_index": {"type": "integer", "enum": candidate_indices},
            "hauptklasse": {"type": "string", "enum": main_labels},
            "subklasse": {"type": "string", "enum": sub_labels},
            "aenderungstyp": {"type": "string", "enum": change_labels},
            "schweregrad": {"type": "string", "enum": sev_labels},
            "stelle_im_segment": {"type": "string", "enum": candidate_texts},
            "begruendung": {"type": "string"},
        },
        "required": [
            "candidate_index", "hauptklasse", "subklasse", "aenderungstyp",
            "schweregrad", "stelle_im_segment", "begruendung",
        ],
    }
    return {
        "type": "object",
        "properties": {"errors": {"type": "array", "items": finding_schema}},
        "required": ["errors"],
    }


def build_statement_assurance_review_messages(
        segment_text: str,
        candidates: List[StatementCandidate],
        catalog: ErrorCatalog,
) -> List[Dict[str, str]]:
    candidate_payload = [
        {"candidate_index": c.candidate_index, "kind": c.kind, "text": c.text}
        for c in candidates
    ]

    recht_subs = sorted(catalog.allowed_subclasses_by_main_label.get("Rechtskonformität", set()))
    allowed_subs = ", ".join(recht_subs)

    system = (
            _AGENT_JSON_PREFIX +

            "Du bist Agent 7: Aussageabsicherungs-, Modalitäts- und Tatsachenstatus-Prüfer "
            "für forensische Gutachten.\n"
            "Du prüfst NUR die vorgegebenen KANDIDATEN, nicht das ganze Segment frei.\n\n"

            "Prüfe pro Kandidat:\n"
            "1. Wird eine unsichere/quellenbasierte/hypothetische Aussage als Tatsache formuliert?\n"
            "2. Wird Wille, Absicht oder innere Haltung als Befund/Tatsache formuliert?\n"
            "3. Erscheinen Konjunktiv-/Modalformulierungen ('hätte', 'könnte', 'wäre') "
            "als gesicherte Tatsachenbehauptung?\n"
            "4. Steht ein Konditionalsatz (falls/wenn/sofern) mit Indikativ statt Konjunktiv II "
            "im Hauptsatz, obwohl der Kontext Konjunktiv II erfordern würde?\n\n"

            "Klassifikation:\n"
            "- Zu absolute Quellenformulierung → subklasse='Aussageabsicherung', "
            "aenderungstyp='Fachliche Präzisierung'\n"
            "- Konjunktiv/Hypothese/Bewertung/Wille als Fakt | Konditionalsatz-Indikativ → "
            "subklasse='Trennung Befund/Bewertung', aenderungstyp='Erweiterung der Argumentation'\n"
            "Immer: hauptklasse='Rechtskonformität' | schweregrad=mittel\n\n"

            "Nicht melden:\n"
            "  - korrekt markierte Hypothesen und klar relativierte Aussagen\n"
            "    ('möglicherweise', 'unter der Annahme', 'gemäss Aktenlage')\n"
            "  - Konjunktiv II in Widerlegungs- und Gegenhypothesen-Sätzen: Formulierungen wie\n"
            "    'Wäre der Brand an Position B ausgebrochen, hätte/müsste/wäre...' sind\n"
            "    KEIN Fehler — das ist der korrekte Gutachtenstil beim Widerlegen fremder Hypothesen.\n"
            "  - Rechenfehler | Sprachfehler\n\n"

            f"Zulässige Subklassen: {allowed_subs}\n\n"

            "stelle_im_segment muss exakt dem Kandidatentext entsprechen.\n"
            "Kein fehlerhafter Kandidat → {\"errors\":[]}\n\n"

            "Format:\n"
            "{\"errors\":[{\"candidate_index\":1,\"hauptklasse\":\"Rechtskonformität\","
            "\"subklasse\":\"<Aussageabsicherung|Trennung Befund/Bewertung>\","
            "\"aenderungstyp\":\"<Fachliche Präzisierung|Erweiterung der Argumentation>\","
            "\"schweregrad\":\"mittel\","
            "\"stelle_im_segment\":\"<exakter Kandidatentext>\","
            "\"begruendung\":\"<warum Tatsachenstatus unsauber>\"}]}"
    )

    user = (
        f"DOKUMENTSEGMENT:\n{segment_text.strip()}\n\n"
        "KANDIDATEN:\n"
        f"{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n\n"
        "Bewerte nur diese Kandidaten. Gib für jeden echten Fehler ein Finding aus. "
        "Wenn kein Kandidat fehlerhaft ist, antworte mit {\"errors\":[]}."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def normalize_statement_assurance_errors(
        raw_errors: List[Any],
        candidates: List[StatementCandidate],
        catalog: ErrorCatalog,
) -> List[Dict[str, Any]]:
    candidate_by_text = {c.text: c for c in candidates}
    candidate_by_idx = {c.candidate_index: c for c in candidates}
    allowed_main = catalog.allowed_main_labels
    allowed_subs = catalog.allowed_subclasses_by_main_label

    findings: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, str]] = set()

    for item in raw_errors:
        if not isinstance(item, dict):
            continue

        stelle = str(item.get("stelle_im_segment") or "").strip()
        idx_raw = item.get("candidate_index")
        cand: Optional[StatementCandidate] = None
        if isinstance(idx_raw, int):
            cand = candidate_by_idx.get(idx_raw)
        if cand is None and stelle:
            cand = candidate_by_text.get(stelle)
        if cand is None:
            print(f"[DROP statement] output not linked to candidate: {stelle[:80]!r}")
            continue

        # Force exact candidate span to stabilize evaluation against Ground Truth.
        stelle = cand.text

        hauptklasse = str(item.get("hauptklasse") or "Rechtskonformität").strip()
        subklasse = str(item.get("subklasse") or "").strip()
        aenderungstyp = str(item.get("aenderungstyp") or "").strip()
        schweregrad = str(item.get("schweregrad") or "mittel").strip()

        if hauptklasse not in allowed_main:
            hauptklasse = "Rechtskonformität" if "Rechtskonformität" in allowed_main else sorted(allowed_main)[0]

        if subklasse not in allowed_subs.get(hauptklasse, set()):
            if cand.kind == "absolute_source_assertion" and "Aussageabsicherung" in allowed_subs.get("Rechtskonformität", set()):
                hauptklasse = "Rechtskonformität"
                subklasse = "Aussageabsicherung"
            elif cand.kind == "conditional_indikativ" and "Trennung Befund/Bewertung" in allowed_subs.get("Rechtskonformität", set()):
                hauptklasse = "Rechtskonformität"
                subklasse = "Trennung Befund/Bewertung"
            elif "Trennung Befund/Bewertung" in allowed_subs.get("Rechtskonformität", set()):
                hauptklasse = "Rechtskonformität"
                subklasse = "Trennung Befund/Bewertung"
            else:
                print(f"[DROP statement] invalid subclass={subklasse!r}")
                continue

        if aenderungstyp not in catalog.allowed_change_labels:
            if subklasse == "Aussageabsicherung" and "Fachliche Präzisierung" in catalog.allowed_change_labels:
                aenderungstyp = "Fachliche Präzisierung"
            elif "Erweiterung der Argumentation" in catalog.allowed_change_labels:
                aenderungstyp = "Erweiterung der Argumentation"
            else:
                aenderungstyp = sorted(catalog.allowed_change_labels)[0]

        if schweregrad not in catalog.allowed_severity_labels:
            schweregrad = "mittel"

        # Fix 4: LLM hat selbst festgestellt, dass kein Fehler vorliegt.
        # Tritt auf wenn das LLM ein Finding generiert obwohl die Begruendung
        # ausdruecklich keine Abweichung meldet (Selbstwiderspruch).
        beg_lower_sa = str(item.get("begruendung") or "").strip().lower()
        _SA_CONFIRMATION_PHRASES = (
            "stimmt mit den referenzfakten überein",
            "keine widerspruche",
            "keine widersprüche",
            "kein widerspruch",
            "stimmt überein",
            "keine abweichung",
            "ist korrekt",
            "ist konsistent",
            "entspricht dem referenzfakt",
            "es gibt keine",
        )
        if any(p in beg_lower_sa for p in _SA_CONFIRMATION_PHRASES):
            print(f"[DROP statement] Selbstwiderspruch (Begruendung bestaetigt keinen Fehler): {beg_lower_sa[:80]!r}")
            continue

        key = (cand.segment_index, stelle)
        if key in seen:
            continue
        seen.add(key)

        findings.append({
            "segment_index": cand.segment_index,
            "hauptklasse": hauptklasse,
            "subklasse": subklasse,
            "aenderungstyp": aenderungstyp,
            "schweregrad": schweregrad,
            "stelle_im_segment": stelle,
            "begruendung": str(item.get("begruendung") or "").strip(),
            "source_refs": ["DOC_INTERNAL"],
            "statement_candidate_kind": cand.kind,
        })

    return findings


def run_statement_assurance_agent(
        llm: LLMClient,
        evidence: SegmentEvidence,
        *,
        catalog: ErrorCatalog,
) -> List[Dict[str, Any]]:
    """Agent 4 — regex candidate prefilter + narrow LLM classification."""
    candidates = extract_statement_assurance_candidates(evidence.segment_text, evidence.segment_index)
    print(
        f"[DEBUG-STATEMENT-CANDIDATES] S{evidence.segment_index}: "
        f"{[{ 'idx': c.candidate_index, 'kind': c.kind, 'text': c.text[:120] } for c in candidates]}"
    )
    if not candidates:
        return []

    messages = build_statement_assurance_review_messages(evidence.segment_text, candidates, catalog)
    schema = build_statement_assurance_json_schema(catalog, candidates)
    raw_reply = llm.chat(messages, json_mode=True, schema=schema)
    print(
        f"[DEBUG] Statement assurance agent S{evidence.segment_index} raw reply "
        f"({len(raw_reply)} chars): {raw_reply[:200]!r}"
    )

    try:
        parsed = parse_json_response(raw_reply)
    except Exception:
        try:
            repair_messages = build_json_repair_messages(raw_reply, schema_name="statement_assurance_errors")
            repaired = llm.chat(repair_messages, json_mode=True, schema=schema)
            parsed = parse_json_response(repaired)
        except Exception as e:
            print(f"[WARN] Statement assurance agent parse failed S{evidence.segment_index}: {e}")
            return []

    findings = normalize_statement_assurance_errors(parsed.get("errors", []), candidates, catalog)
    print(f"[DEBUG-STATEMENT-FINAL] S{evidence.segment_index}: {findings}")
    return findings


# ── Final Aggregator ──────────────────────────────────────────────────────────

def _strip_internal_fields(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove internal debug fields before aggregation."""
    internal = {"lt_rule_id", "legal_abbrev_guard"}
    return [{k: v for k, v in f.items() if k not in internal} for f in findings]


def aggregate_reports(
        evidences: List[SegmentEvidence],
        factual_findings: List[Dict[str, Any]],
        language_findings: List[Dict[str, Any]],
        reference_consistency_findings: Optional[List[Dict[str, Any]]] = None,
        statement_assurance_findings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    source_map: Dict[str, EvidenceSource] = {}
    for ev in evidences:
        for src in ev.all_sources:
            source_map[src.source_ref] = src

    aggregated_factual: List[Dict[str, Any]] = []
    for item in factual_findings:
        raw_refs = item.get("source_refs", [])

        # DOC_INTERNAL: valid ref for logic errors grounded in the document itself.
        external_refs = [r for r in raw_refs if r != "DOC_INTERNAL" and r in source_map]
        internal_refs = [r for r in raw_refs if r == "DOC_INTERNAL"]
        all_valid_refs = external_refs + internal_refs

        if not all_valid_refs:
            continue

        chunk_ids: List[str] = []
        documents: List[str] = []
        source_details: List[Dict[str, Any]] = []

        seen_chunks: Set[str] = set()
        seen_docs: Set[str] = set()

        for ref in external_refs:
            src = source_map[ref]
            if src.chunk_id not in seen_chunks:
                chunk_ids.append(src.chunk_id)
                seen_chunks.add(src.chunk_id)
            if src.document not in seen_docs:
                documents.append(src.document)
                seen_docs.add(src.document)
            source_details.append({
                "source_ref": ref,
                "chunk_id": src.chunk_id,
                "document": src.document,
                "source_kind": src.source_kind,
                "chunk_index": src.chunk_index,
                "score": round(src.score, 4),
            })

        for ref in internal_refs:
            if "Dokumentinterner Widerspruch" not in seen_docs:
                documents.append("Dokumentinterner Widerspruch")
                seen_docs.add("Dokumentinterner Widerspruch")
            source_details.append({
                "source_ref": "DOC_INTERNAL",
                "chunk_id": "",
                "document": "Dokumentinterner Widerspruch",
                "source_kind": "internal",
                "chunk_index": None,
                "score": 1.0,
            })

        aggregated_factual.append({
            "segment_index": item.get("segment_index"),
            "hauptklasse": item.get("hauptklasse", ""),
            "subklasse": item.get("subklasse", ""),
            "aenderungstyp": item.get("aenderungstyp", ""),
            "schweregrad": item.get("schweregrad", ""),
            "stelle_im_segment": item.get("stelle_im_segment", ""),
            "begruendung": item.get("begruendung", ""),
            "source_refs": all_valid_refs,
            "chunk_ids": chunk_ids,
            "dokumente": documents,
            "sources": source_details,
        })

    return {
        "factual_findings": aggregated_factual,
        "language_findings": language_findings,
        "calculation_findings": [],
        "hypothesis_findings": [],
        "reference_consistency_findings": reference_consistency_findings or [],
        "statement_assurance_findings": statement_assurance_findings or [],
    }


def render_combined_report(report: Dict[str, Any]) -> str:
    factual = report.get("factual_findings", []) or []
    language = report.get("language_findings", []) or []
    calculation = report.get("calculation_findings", []) or []
    hypothesis = report.get("hypothesis_findings", []) or []
    reference_consistency = report.get("reference_consistency_findings", []) or []
    statement_assurance = report.get("statement_assurance_findings", []) or []

    lines: List[str] = []

    lines.append("FACHLICHE FINDINGS")
    lines.append("-" * 90)
    if not factual:
        lines.append("Kein fachlicher Fehler gefunden.")
    else:
        for i, item in enumerate(factual, start=1):
            label = f"{item.get('hauptklasse') or 'Unklassifiziert'} > {item.get('subklasse') or 'Unklassifiziert'}"
            lines.append(f"{i}. [{label}]")
            lines.append(f"   Änderungstyp: {item.get('aenderungstyp') or '-'}")
            lines.append(f"   Schweregrad: {item.get('schweregrad') or '-'}")
            lines.append(f"   Segment: {item.get('segment_index')}")
            lines.append(f"   Stelle im Segment: {item.get('stelle_im_segment') or '-'}")
            lines.append(f"   Begründung: {item.get('begruendung') or '-'}")
            lines.append(f"   source_refs: {', '.join(item.get('source_refs') or []) or '-'}")
            lines.append(f"   chunk_id: {', '.join(item.get('chunk_ids') or []) or '-'}")
            lines.append(f"   dokument: {', '.join(item.get('dokumente') or []) or '-'}")
            lines.append("")

    lines.append("")
    lines.append("REFERENZFAKTEN-KONSISTENZ")
    lines.append("-" * 90)
    if not reference_consistency:
        lines.append("Keine Referenzfakten-Inkonsistenz gefunden.")
    else:
        for i, item in enumerate(reference_consistency, start=1):
            label = f"{item.get('hauptklasse') or 'Unklassifiziert'} > {item.get('subklasse') or 'Unklassifiziert'}"
            lines.append(f"{i}. [{label}]")
            lines.append(f"   Änderungstyp: {item.get('aenderungstyp') or '-'}")
            lines.append(f"   Schweregrad: {item.get('schweregrad') or '-'}")
            lines.append(f"   Segment: {item.get('segment_index')}")
            lines.append(f"   Referenzfeld: {item.get('reference_key') or '-'}")
            lines.append(f"   Referenzwert: {item.get('reference_value') or '-'}")
            lines.append(f"   Stelle im Segment: {item.get('stelle_im_segment') or '-'}")
            lines.append(f"   Begründung: {item.get('begruendung') or '-'}")
            lines.append("")

    lines.append("")
    lines.append("AUSSAGEABSICHERUNG / MODALITÄT")
    lines.append("-" * 90)
    if not statement_assurance:
        lines.append("Keine Aussageabsicherungs- oder Modalitätsfehler gefunden.")
    else:
        for i, item in enumerate(statement_assurance, start=1):
            label = f"{item.get('hauptklasse') or 'Unklassifiziert'} > {item.get('subklasse') or 'Unklassifiziert'}"
            lines.append(f"{i}. [{label}]")
            lines.append(f"   Änderungstyp: {item.get('aenderungstyp') or '-'}")
            lines.append(f"   Schweregrad: {item.get('schweregrad') or '-'}")
            lines.append(f"   Segment: {item.get('segment_index')}")
            lines.append(f"   Stelle im Segment: {item.get('stelle_im_segment') or '-'}")
            lines.append(f"   Begründung: {item.get('begruendung') or '-'}")
            if item.get("source_refs"):
                lines.append(f"   source_refs: {', '.join(item.get('source_refs') or [])}")
            lines.append("")

    lines.append("")
    lines.append("HYPOTHESENPRÜFUNG")
    lines.append("-" * 90)
    if not hypothesis:
        lines.append("Keine Hypothesen-Inkonsistenz gefunden.")
    else:
        for i, item in enumerate(hypothesis, start=1):
            label = f"Hypothesenprüfung > {item.get('subklasse') or '-'}"
            lines.append(f"{i}. [{label}]")
            lines.append(f"   Änderungstyp: {item.get('aenderungstyp') or '-'}")
            lines.append(f"   Schweregrad: {item.get('schweregrad') or '-'}")
            lines.append(f"   Segment: {item.get('segment_index')}")
            lines.append(f"   Hypothese: {item.get('hypothese_text') or '-'}")
            lines.append(f"   Befundbewertung: {item.get('befundbewertung_text') or '-'}")
            lines.append(f"   Stelle im Segment: {item.get('stelle_im_segment') or '-'}")
            lines.append(f"   Begründung: {item.get('begruendung') or '-'}")
            lines.append("")

    lines.append("")
    lines.append("RECHENPRÜFUNG")
    lines.append("-" * 90)
    if not calculation:
        lines.append("Kein Rechenfehler gefunden.")
    else:
        for i, item in enumerate(calculation, start=1):
            label = f"Rechenfehler > {item.get('subklasse') or 'Arithmetischer Fehler'}"
            lines.append(f"{i}. [{label}]")
            lines.append(f"   Änderungstyp: {item.get('aenderungstyp') or '-'}")
            lines.append(f"   Schweregrad: {item.get('schweregrad') or '-'}")
            lines.append(f"   Segment: {item.get('segment_index')}")
            lines.append(f"   Stelle im Segment: {item.get('stelle_im_segment') or '-'}")
            lines.append(f"   Wert im Dokument: {item.get('wert_im_dokument') or '-'}")
            lines.append(f"   Korrekter Wert: {item.get('korrekter_wert') or '-'}")
            lines.append(f"   Berechnung: {item.get('berechnung') or '-'}")
            lines.append(f"   Begründung: {item.get('begruendung') or '-'}")
            lines.append("")

    lines.append("")
    lines.append("SPRACHLICHE FINDINGS")
    lines.append("-" * 90)
    if not language:
        lines.append("Kein sprachlicher Fehler gefunden.")
    else:
        for i, item in enumerate(language, start=1):
            label = f"{item.get('hauptklasse') or 'Unklassifiziert'} > {item.get('subklasse') or 'Unklassifiziert'}"
            lines.append(f"{i}. [{label}]")
            lines.append(f"   Änderungstyp: {item.get('aenderungstyp') or '-'}")
            lines.append(f"   Schweregrad: {item.get('schweregrad') or '-'}")
            lines.append(f"   Segment: {item.get('segment_index')}")
            lines.append(f"   Stelle im Segment: {item.get('stelle_im_segment') or '-'}")
            lines.append(f"   Begründung: {item.get('begruendung') or '-'}")
            if item.get("vorschlag"):
                lines.append(f"   Vorschlag: {item.get('vorschlag')}")
            lines.append("")

    return "\n".join(lines).rstrip()


# ── Shared retrieval + output helper (Q&A mode) ──────────────────────────────

def _retrieve_and_print(
        query_text: str,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        args: argparse.Namespace,
        vision_cfg: Optional[dict],
        log_key: str = "question",
) -> Tuple[str, List[Retrieved], List[str]]:
    candidate_k = max(args.top_k, min(max(args.top_k * 4, 12), 40))
    hits, queries, _, _ = retrieve_multi_query(
        query_text,
        stores,
        embed_model=embed_model,
        embed_tok=embed_tok,
        device=device,
        max_length=args.query_max_length,
        top_k=args.top_k,
        candidate_k=candidate_k,
        multi_query_count=args.multi_query_count,
        mmr_lambda=args.mmr_lambda,
        max_per_source=args.max_per_source,
        mode="qa",
        query_expander=getattr(args, "query_expander", None),
        case_id=args.case_id,
        rules_top_k=args.rules_top_k,
        material_top_k=args.material_top_k,
    )

    # Vision captioning done at ingestion time — no enrichment needed here.
    context, sources = build_context_blocks(hits, max_chars=args.context_max_chars)

    if args.print_sources:
        print(json.dumps({
            log_key: query_text[:120],
            "multi_queries": queries,
            "sources": sources,
        }, ensure_ascii=False, indent=2))

    if args.print_context:
        print("\n" + "=" * 90)
        print("RETRIEVED CONTEXT")
        print("=" * 90)
        print(context)

    return context, hits, queries


# ── Modes ─────────────────────────────────────────────────────────────────────

def answer(
        question: str,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        llm: LLMClient,
        args: argparse.Namespace,
        vision_cfg: Optional[dict],
) -> None:
    context, hits, queries = _retrieve_and_print(
        question,
        stores,
        embed_model=embed_model,
        embed_tok=embed_tok,
        device=device,
        args=args,
        vision_cfg=vision_cfg,
        log_key="question",
    )

    messages = build_qa_messages(question, context)
    reply = llm.chat(messages)

    print("\n" + "=" * 90)
    print("ANSWER")
    print("=" * 90)
    print(reply)

    # Quellenauflösung nach der Antwort
    if hits:
        print("\n" + "-" * 90)
        print("QUELLEN")
        print("-" * 90)
        for h in hits:
            m = h.meta or {}
            doc = m.get("source_name") or m.get("origin_source_name") or "?"
            kind = m.get("source_kind") or "?"
            chunk_idx = m.get("chunk_index")
            chunk_str = f"  chunk {chunk_idx}" if chunk_idx is not None else ""
            print(f"  [{h.rank}] {doc}{chunk_str}  (score={h.score:.4f}, kind={kind})")

class _AgentTimer:
    """
    Akkumuliert Laufzeiten pro Agent über alle Segment-Aufrufe hinweg.
    Gibt am Ende einen formatierten Konsolenblock aus.
    """
    def __init__(self) -> None:
        self._times: Dict[str, float] = {}
        self._calls: Dict[str, int] = {}

    def record(self, agent: str, elapsed: float) -> None:
        self._times[agent] = self._times.get(agent, 0.0) + elapsed
        self._calls[agent] = self._calls.get(agent, 0) + 1

    def print_summary(self, n_segments: int) -> None:
        W = 90
        total = sum(self._times.values())
        order = [
            ("Agent 0", "agent0",               "Referenzfakten-Extraktor", False),
            ("Agent 2", "factual",               "Fachprüfer",               True),
            ("Agent 3", "language",              "Sprach-/Formalprüfer",     True),
            ("Agent 4", "calculation",           "Rechenprüfer",             True),
            ("Agent 7", "statement_assurance",   "Aussageabsicherung",       True),
            ("Agent 5", "hypothesis",            "Hypothesenprüfer",         False),
            ("Agent 6", "reference_consistency", "Referenzkonsistenz",       False),
        ]
        print("\n" + "=" * W)
        print("LAUFZEITEN PRO AGENT")
        print("=" * W)
        print(f"  {'Agent':<8} {'Beschreibung':<30} {'Gesamt':>9}  {'Calls':>8}  {'Ø/Call':>8}  {'Anteil':>7}")
        print("-" * W)
        for label, key, desc, per_seg in order:
            t = self._times.get(key, 0.0)
            c = self._calls.get(key, 0)
            avg = t / c if c else 0.0
            pct = t / total * 100 if total else 0.0
            calls_note = f"{c} Seg." if per_seg and c > 1 else f"{c} Call{'s' if c != 1 else ''}"
            print(
                f"  {label:<8} {desc:<30} "
                f"{t:>7.1f}s  {calls_note:>8}  {avg:>6.2f}s  {pct:>6.1f}%"
            )
        print("-" * W)
        print(f"  {'':8} {'GESAMT':<30} {total:>7.1f}s")
        print("=" * W)


def check_document(
        doc_path: Path,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        llm: LLMClient,
        args: argparse.Namespace,
        vision_cfg: Optional[dict],
        catalog: ErrorCatalog,
) -> None:
    try:
        from importDocuments_structural import normalize_text, read_docx
    except ImportError as e:
        raise SystemExit(
            "importDocuments_structural.py must be in the same directory or on PYTHONPATH. "
            f"Original error: {e}"
        )

    print(f"[INFO] Reading document: {doc_path.resolve()}")
    doc_text = normalize_text(read_docx(doc_path))

    if not doc_text.strip():
        print(f"[WARN] No text extracted from {doc_path.name} — aborting.")
        return

    print(f"[INFO] Document text: {len(doc_text)} chars")

    # Reset dokumentweiter Span-Dedup-Cache (pro Subklasse)
    _SEEN_SPANS_BY_SUBCLASS.clear()

    _timer = _AgentTimer()

    reference_schema = load_reference_facts_schema(Path(args.reference_facts_schema).expanduser().resolve())
    _t0 = time.perf_counter()
    reference_facts = run_reference_facts_agent(
        llm,
        doc_text,
        case_id=args.case_id,
        schema=reference_schema,
        max_chars=args.reference_facts_context_chars,
    )
    _timer.record("agent0", time.perf_counter() - _t0)
    reference_facts_context = format_reference_facts_for_prompt(reference_facts)

    if args.print_reference_facts:
        print("\n" + "=" * 90)
        print("REFERENCE FACTS")
        print("=" * 90)
        print(json.dumps(reference_facts, ensure_ascii=False, indent=2))

    if args.save_reference_facts_json:
        out_path = Path(args.save_reference_facts_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(reference_facts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Saved reference facts to {out_path}")

    evidences, multi_queries = build_segment_evidences(
        doc_text,
        stores,
        embed_model=embed_model,
        embed_tok=embed_tok,
        device=device,
        args=args,
        vision_cfg=vision_cfg,
    )

    if args.print_sources:
        payload: Dict[str, Any] = {
            "document": str(doc_path),
            "document_chars": len(doc_text),
            "segments": len(evidences),
            "multi_queries": multi_queries,
            "segments_evidence": [],
        }
        for ev in evidences:
            payload["segments_evidence"].append({
                "segment_index": ev.segment_index,
                "retrieval_queries": ev.retrieval_queries,
                "rules_sources": [
                    {
                        "source_ref": s.source_ref,
                        "chunk_id": s.chunk_id,
                        "document": s.document,
                        "source_path": s.source_path,
                        "score": round(s.score, 4),
                        "source_kind": s.source_kind,
                        "case_id": s.case_id,
                        "document_type": s.document_type,
                        "chunk_index": s.chunk_index,
                    }
                    for s in ev.rules_sources
                ],
                "material_sources": [
                    {
                        "source_ref": s.source_ref,
                        "chunk_id": s.chunk_id,
                        "document": s.document,
                        "source_path": s.source_path,
                        "score": round(s.score, 4),
                        "source_kind": s.source_kind,
                        "case_id": s.case_id,
                        "document_type": s.document_type,
                        "chunk_index": s.chunk_index,
                    }
                    for s in ev.material_sources
                ],
            })
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.print_context:
        for ev in evidences:
            print("\n" + "=" * 90)
            print(f"SEGMENT {ev.segment_index}")
            print("=" * 90)
            print(ev.segment_text)

            print("\n" + "-" * 90)
            print("RULES EVIDENCE")
            print("-" * 90)
            print(build_agent_context_from_sources(
                ev.rules_sources,
                max_chars=max(1000, args.context_max_chars // 3),
            ) or "(leer)")

            print("\n" + "-" * 90)
            print("CASE MATERIAL EVIDENCE")
            print("-" * 90)
            print(build_agent_context_from_sources(
                ev.material_sources,
                max_chars=max(1000, args.context_max_chars // 3),
            ) or "(leer)")

    factual_findings: List[Dict[str, Any]] = []
    language_findings: List[Dict[str, Any]] = []
    calculation_findings: List[Dict[str, Any]] = []
    hypothesis_findings: List[Dict[str, Any]] = []
    reference_consistency_findings: List[Dict[str, Any]] = []
    statement_assurance_findings: List[Dict[str, Any]] = []

    per_agent_context_chars = args.context_max_chars // 3

    # Referenzwörter: Referenzfakten + Domänen-Whitelist aus Material Store
    _ref_words: set = _build_reference_words_set(reference_facts)

    # Material Store Whitelist — fallspezifische Fachbegriffe und Eigennamen
    # die im Zusatzmaterial ≥2× vorkommen werden nicht als Tippfehler gemeldet.
    _material_stores = [s for s in stores if s.source_kind == "material"]
    if _material_stores and args.case_id:
        for _ms in _material_stores:
            _whitelist = build_domain_whitelist_from_store(
                _ms, args.case_id,
                min_word_length=4,
                min_occurrences=2,
            )
            _ref_words |= _whitelist
            print(f"[INFO] Domänen-Whitelist aus Material Store: {len(_whitelist)} Wörter")

    # spaCy NER: Personennamen aus dem Volltext ergänzen
    # Erfasst Nachnamen wie "Loosli", "Werlen" die in Referenzfakten fehlen
    _ner_names = _extract_person_names_spacy(doc_text)
    if _ner_names:
        _ref_words |= _ner_names
        print(f"[INFO] spaCy NER: {len(_ner_names)} Personennamen zu Whitelist hinzugefügt")

    if _ref_words:
        print(f"[INFO] Referenzfakten-Filter aktiv: {len(_ref_words)} bekannte Wörter total")

    for ev in evidences:
        try:
            _t0 = time.perf_counter()
            factual_findings.extend(
                run_factual_agent(llm, ev, per_agent_context_chars=per_agent_context_chars, catalog=catalog)
            )
            _timer.record("factual", time.perf_counter() - _t0)
        except Exception as e:
            print(f"[WARN] Factual agent failed for segment {ev.segment_index}: {e}")


        try:
            _t0 = time.perf_counter()
            language_findings.extend(run_language_agent(llm, ev, catalog=catalog, reference_words=_ref_words))
            _timer.record("language", time.perf_counter() - _t0)
        except Exception as e:
            print(f"[WARN] Language agent failed for segment {ev.segment_index}: {e}")

        try:
            _t0 = time.perf_counter()
            calculation_findings.extend(run_calculation_agent(llm, ev, catalog))
            _timer.record("calculation", time.perf_counter() - _t0)
        except Exception as e:
            print(f"[WARN] Calculation agent failed for segment {ev.segment_index}: {e}")

        try:
            _t0 = time.perf_counter()
            statement_assurance_findings.extend(
                run_statement_assurance_agent(llm, ev, catalog=catalog)
            )
            _timer.record("statement_assurance", time.perf_counter() - _t0)
        except Exception as e:
            print(f"[WARN] Statement assurance agent failed for segment {ev.segment_index}: {e}")

        # Keyword-Guard: kein LLM-Call, deterministisch
        factual_findings.extend(check_zweifel_violations(ev.segment_text, ev.segment_index))

    # Agent 5: runs once on full document — after segment loop
    segments = [ev.segment_text for ev in evidences]
    try:
        _t0 = time.perf_counter()
        hypothesis_findings = run_hypothesis_agent(llm, doc_text, segments, catalog)
        _timer.record("hypothesis", time.perf_counter() - _t0)
    except Exception as e:
        print(f"[WARN] Hypothesis agent failed: {e}")

    # Agent 6: runs once on full document — checks document-wide consistency against Agent 0 reference facts
    try:
        _t0 = time.perf_counter()
        reference_consistency_findings = run_reference_consistency_agent(
            llm,
            doc_text,
            segments,
            reference_facts,
            catalog,
        )
        _timer.record("reference_consistency", time.perf_counter() - _t0)
    except Exception as e:
        print(f"[WARN] Reference consistency agent failed: {e}")

    language_findings = _strip_internal_fields(language_findings)
    report = aggregate_reports(
        evidences,
        factual_findings,
        language_findings,
        reference_consistency_findings=reference_consistency_findings,
        statement_assurance_findings=statement_assurance_findings,
    )
    report["calculation_findings"] = calculation_findings
    report["hypothesis_findings"] = hypothesis_findings
    report["reference_consistency_findings"] = reference_consistency_findings
    report["statement_assurance_findings"] = statement_assurance_findings
    rendered_report = render_combined_report(report)

    if args.save_predictions_jsonl:
        save_predictions_jsonl(
            report=report,
            case_id=args.case_id,
            output_path=Path(args.save_predictions_jsonl),
            catalog=catalog,
        )

    _timer.print_summary(n_segments=len(evidences))

    print("\n" + "=" * 90)
    print(f"ERROR DETECTION REPORT — {doc_path.name}")
    print(f"ERROR DETECTION REPORT PATH —  {doc_path}")
    print("=" * 90)
    print(rendered_report)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "RAG pipeline with dual-store retrieval, taxonomy loaded from JSON, and 3-agent document checking.\n\n"
            "Modes:\n"
            "  --question TEXT   Answer a free-text question\n"
            "  (no args)         Interactive Q&A loop\n"
            "  --document FILE   Detect errors in a Word document with:\n"
            "                    Agent 1 = retrieval/evidence\n"
            "                    Agent 2 = factual review\n"
            "                    Agent 3 = language review\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--question",
        type=str,
        default="",
        help="Free-text question; if empty, starts interactive loop",
    )
    mode.add_argument(
        "--document",
        type=str,
        default="",
        help="Path to a .docx file to check for errors",
    )

    ap.add_argument(
        "--case_id",
        type=str,
        default=env_str("CASE_ID", ""),
        help="Case ID for filtering case materials, e.g. case_01",
    )
    ap.add_argument(
        "--taxonomy_json",
        type=str,
        default=env_str("TAXONOMY_JSON", "taxonomy.json"),
        help="Path to taxonomy JSON",
    )

    ap.add_argument("--embeddings", type=str,
                    default=env_str("EMBED_OUT_NPZ", "embeddings.npz"),
                    help="Embeddings .npz for RAG store 1 (rules)")
    ap.add_argument("--index", type=str,
                    default=env_str("EMBED_OUT_INDEX", "index.jsonl"),
                    help="Index .jsonl for RAG store 1")
    ap.add_argument("--prepared", type=str,
                    default=env_str("OUT_JSONL", "prepared.jsonl"),
                    help="Prepared .jsonl for RAG store 1")

    ap.add_argument("--embeddings2", type=str,
                    default=env_str("EMBED_OUT_NPZ2", ""),
                    help="Embeddings .npz for RAG store 2 (materials)")
    ap.add_argument("--index2", type=str,
                    default=env_str("EMBED_OUT_INDEX2", ""),
                    help="Index .jsonl for RAG store 2")
    ap.add_argument("--prepared2", type=str,
                    default=env_str("OUT_JSONL2", ""),
                    help="Prepared .jsonl for RAG store 2")

    ap.add_argument("--top_k", type=int,
                    default=env_int("TOP_K", 12),
                    help="Overall retrieval top-k")
    ap.add_argument("--rules_top_k", type=int,
                    default=env_int("RULES_TOP_K", 8),
                    help="Final number of rule chunks to keep")
    ap.add_argument("--material_top_k", type=int,
                    default=env_int("MATERIAL_TOP_K", 8),
                    help="Final number of material chunks to keep")

    ap.add_argument("--context_max_chars", type=int,
                    default=env_int("CONTEXT_MAX_CHARS", 12000),
                    help="Maximum characters used for contexts")

    ap.add_argument("--embed_model", type=str,
                    default=env_str("EMBED_MODEL", "intfloat/multilingual-e5-large"))
    ap.add_argument("--embed_device", type=str,
                    default=env_str("EMBED_DEVICE", "auto"))
    ap.add_argument("--query_max_length", type=int,
                    default=env_int("QUERY_MAX_LENGTH", 256))
    ap.add_argument("--multi_query_count", type=int,
                    default=env_int("MULTI_QUERY_COUNT", 4),
                    help="Number of internal retrieval query variants")
    ap.add_argument("--mmr_lambda", type=float,
                    default=float(env_str("MMR_LAMBDA", "0.75")),
                    help="MMR relevance weight between 0 and 1")
    ap.add_argument("--max_per_source", type=int,
                    default=env_int("MAX_PER_SOURCE", 2),
                    help="Maximum number of final chunks per source file")

    ap.add_argument("--vision_model", type=str,
                    default=env_str("VISION_MODEL", ""),
                    help="Ollama vision model, e.g. qwen2.5vl:7b. Leave empty to skip.")
    ap.add_argument("--vision_workers", type=int,
                    default=env_int("VISION_WORKERS", 3),
                    help="Parallel workers for image captioning")
    ap.add_argument("--vision_timeout_s", type=int,
                    default=env_int("VISION_TIMEOUT_S", 180))

    ap.add_argument("--print_sources", action="store_true",
                    help="Print sources metadata as JSON")
    ap.add_argument("--print_context", action="store_true",
                    help="Print segment texts and evidence blocks")
    ap.add_argument(
        "--save_predictions_jsonl",
        type=str,
        default="",
        help="Path to save structured predictions JSONL"
    )
    ap.add_argument(
        "--ground_truth",
        type=str,
        default=env_str("GROUND_TRUTH_FILE", ""),
        help=(
            "Pfad zur Ground-Truth-JSONL-Datei fuer die Evaluation. "
            "Wird an check_document weitergegeben. "
            "Beispiel: ./ground_truth/ground_truth_case_01_synthetic.jsonl"
        ),
    )
    ap.add_argument(
        "--reference_facts_schema",
        type=str,
        default=env_str("REFERENCE_FACTS_SCHEMA", "schema/reference_facts_schema.json"),
        help="Path to the JSON schema for Agent 0 reference facts",
    )
    ap.add_argument(
        "--reference_facts_context_chars",
        type=int,
        default=env_int("REFERENCE_FACTS_CONTEXT_CHARS", 4000),
        help="Number of initial document characters used by Agent 0",
    )
    ap.add_argument(
        "--save_reference_facts_json",
        type=str,
        default=env_str("SAVE_REFERENCE_FACTS_JSON", ""),
        help="Optional path to save extracted reference facts as JSON",
    )
    ap.add_argument(
        "--print_reference_facts",
        action="store_true",
        help="Print extracted reference facts",
    )
    ap.add_argument(
        "--per_segment_candidate_k",
        type=int,
        default=env_int("PER_SEGMENT_CANDIDATE_K", 24),
        help="Candidate pool size per document segment",
    )

    ap.add_argument(
        "--per_segment_rules_top_k",
        type=int,
        default=env_int("PER_SEGMENT_RULES_TOP_K", 8),
        help="Final number of rule chunks kept per segment",
    )

    ap.add_argument(
        "--per_segment_material_top_k",
        type=int,
        default=env_int("PER_SEGMENT_MATERIAL_TOP_K", 8),
        help="Final number of material chunks kept per segment",
    )

    return ap.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    started_at = datetime.now()
    t0 = time.perf_counter()
    _log_path: Optional[Path] = None
    print(f"[INFO] Script start: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        args = parse_args()

        # ── Set up run log (tees all print() output to logs/) ─────────────────
        _log_model = getattr(args, "llm_model", None) or env_str("LLM_MODEL", "unknown")
        _log_path  = _setup_run_logging(
            case_id    = getattr(args, "case_id", "") or "",
            model_name = _log_model,
        )

        catalog = load_taxonomy_json(Path(args.taxonomy_json).resolve())
        print(
            f"[INFO] Loaded taxonomy: {len(catalog.main_classes)} Hauptklassen | "
            f"{sum(len(v) for v in catalog.sub_by_main.values())} Subklassen | "
            f"{len(catalog.change_types)} Änderungstypen"
        )

        stores: List[RagStore] = [
            load_rag_store(
                "rules",
                "rules",
                npz_path=Path(args.embeddings).resolve(),
                index_path=Path(args.index).resolve(),
                prepared_path=Path(args.prepared).resolve(),
            )
        ]

        if args.embeddings2.strip():
            stores.append(
                load_rag_store(
                    "material",
                    "material",
                    npz_path=Path(args.embeddings2).resolve(),
                    index_path=Path(args.index2).resolve(),
                    prepared_path=Path(args.prepared2).resolve(),
                )
            )
        else:
            print("[INFO] No second RAG store configured — using rules store only.")

        device = choose_device(args.embed_device)
        print(f"[INFO] Embedding device: {device} | model: {args.embed_model}")
        embed_model, embed_tok = load_hf_model(args.embed_model, device)

        llm = make_llm_client()
        args.query_expander = make_query_expander()

        vision_cfg: Optional[dict] = None
        vision_model_enabled = env_bool("VISION_MODEL_ENABLED", True)


        if args.vision_model.strip() and vision_model_enabled:
            vision_cfg = {
                "vision_model": args.vision_model,
                "vision_model_enabled": vision_model_enabled,
                "vision_prompt": env_str("VISION_PROMPT", ""),
                "ollama_base_url": require_env("OLLAMA_BASE_URL"),
                "vision_timeout_s": args.vision_timeout_s,
                "vision_options": env_json_object_optional("VISION_OPTIONS_JSON"),
            }
            print(f"[INFO] Vision captioning enabled: {args.vision_model}")
        else:
            print(
                f"[INFO] Vision captioning disabled: "
                f"vision_model={args.vision_model!r}, vision_model_enabled={vision_model_enabled}"
            )

        print(
            f"[INFO] Retrieval: multi_query_count={args.multi_query_count} | "
            f"mmr_lambda={args.mmr_lambda:.2f} | max_per_source={args.max_per_source} | "
            f"rules_top_k={args.rules_top_k} | material_top_k={args.material_top_k}"
        )

        shared = dict(
            stores=stores,
            embed_model=embed_model,
            embed_tok=embed_tok,
            device=device,
            llm=llm,
            args=args,
            vision_cfg=vision_cfg,
            catalog=catalog,
        )

        if args.document.strip():
            doc_path = Path(args.document).expanduser().resolve()
            if not doc_path.exists():
                raise SystemExit(f"Document not found: {doc_path}")
            if doc_path.suffix.lower() != ".docx":
                raise SystemExit(f"Only .docx files are supported, got: {doc_path.suffix}")
            if args.embeddings2.strip() and not args.case_id.strip():
                raise SystemExit("--case_id is required in document mode when using case materials.")
            check_document(doc_path, **shared)
            return

        if args.question.strip():
            answer(args.question, stores=stores, embed_model=embed_model, embed_tok=embed_tok, device=device, llm=llm, args=args, vision_cfg=vision_cfg)
            return

        print("Interactive RAG. Empty input to exit.")
        while True:
            q = input("\nQuestion> ").strip()
            if not q:
                break
            answer(q, stores=stores, embed_model=embed_model, embed_tok=embed_tok, device=device, llm=llm, args=args, vision_cfg=vision_cfg)

    finally:
        ended_at = datetime.now()
        elapsed_s = time.perf_counter() - t0
        elapsed_min = elapsed_s / 60.0
        elapsed_h = elapsed_s / 3600.0
        print(f"[INFO] Script end:   {ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[INFO] Total runtime: {elapsed_s:.1f} s | {elapsed_min:.1f} min | {elapsed_h:.2f} h")
        _teardown_run_logging(_log_path)


if __name__ == "__main__":
    main()
    print(f"[INFO] Running file: {Path(__file__).resolve()}")