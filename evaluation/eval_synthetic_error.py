#!/usr/bin/env python3
"""
eval_synthetic_error.py
=======================
Auswertung der synthetischen Fehlergenerierung:
Wie viele Proposals wurden im zweiten Call (Plausibilitätsprüfung)
als "accepted" bzw. "rejected" bewertet?

Auswertung pro Fall und fallübergreifend (Fälle 1,2,3,4,5,7,8 — ohne Fall 6),
aufgeschlüsselt nach subclass_id und change_type_id.

Eingabe:  injection/proposals_case_<xy>_latest.json
Ausgabe:  evaluation/eval_synthetic_error.txt  (menschenlesbarer Report)
          evaluation/eval_synthetic_error.json (maschinenlesbarer Report)

Usage:
    python3 evaluation/eval_synthetic_error.py
    python3 evaluation/eval_synthetic_error.py --base_dir /pfad/zum/projekt
    python3 evaluation/eval_synthetic_error.py --cases 1 2 3 4 5 7 8
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Konfiguration ─────────────────────────────────────────────────────────────

DEFAULT_CASES = [1, 2, 3, 4, 5, 7, 8]   # Fall 6 explizit ausgeschlossen


# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(char: str = "─", width: int = 90) -> str:
    return char * width


def pct(n: int, total: int) -> str:
    if total == 0:
        return "  —  "
    return f"{n / total * 100:5.1f}%"


def load_proposals(path: Path) -> List[Dict[str, Any]]:
    """Lädt eine proposals_case_xy_latest.json Datei."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Erwartet JSON-Array in {path}, gefunden: {type(data)}")
    return data


def find_proposals_file(base_dir: Path, case_num: int) -> Optional[Path]:
    """Sucht die proposals-Datei für einen Fall (json oder jsonl)."""
    case_id = f"case_{case_num:02d}"
    for suffix in (".json", ".jsonl"):
        p = base_dir / "injection" / f"proposals_{case_id}_latest{suffix}"
        if p.exists():
            return p
    return None


# ── Kerndatenstruktur ─────────────────────────────────────────────────────────

def count_proposals(proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Zählt accepted/rejected Proposals aus einer Liste.

    Gibt zurück:
    {
        "total":      int,
        "accepted":   int,
        "rejected":   int,
        "by_subclass": {
            subclass_id: {"accepted": int, "rejected": int, "total": int}
        },
        "by_change_type": {
            change_type_id: {"accepted": int, "rejected": int, "total": int}
        },
        "by_subclass_change": {
            (subclass_id, change_type_id): {"accepted": int, "rejected": int, "total": int}
        }
    }
    """
    total    = 0
    accepted = 0
    rejected = 0
    by_sub:    Dict[str, Dict[str, int]] = defaultdict(lambda: {"accepted": 0, "rejected": 0, "total": 0})
    by_ct:     Dict[str, Dict[str, int]] = defaultdict(lambda: {"accepted": 0, "rejected": 0, "total": 0})
    by_sub_ct: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"accepted": 0, "rejected": 0, "total": 0})

    for p in proposals:
        status = str(p.get("status") or "").strip().lower()
        if status not in ("accepted", "rejected"):
            continue  # pending oder unbekannt → nicht zählen

        sub = str(p.get("subclass_id") or "UNKNOWN").strip()
        ct  = str(p.get("change_type_id") or "UNKNOWN").strip()

        total += 1
        if status == "accepted":
            accepted += 1
            by_sub[sub]["accepted"]      += 1
            by_ct[ct]["accepted"]        += 1
            by_sub_ct[(sub, ct)]["accepted"] += 1
        else:
            rejected += 1
            by_sub[sub]["rejected"]      += 1
            by_ct[ct]["rejected"]        += 1
            by_sub_ct[(sub, ct)]["rejected"] += 1

        by_sub[sub]["total"]         += 1
        by_ct[ct]["total"]           += 1
        by_sub_ct[(sub, ct)]["total"] += 1

    return {
        "total":             total,
        "accepted":          accepted,
        "rejected":          rejected,
        "by_subclass":       dict(by_sub),
        "by_change_type":    dict(by_ct),
        "by_subclass_change": {f"{k[0]} | {k[1]}": v for k, v in by_sub_ct.items()},
    }


def merge_counts(*counts_list: Dict[str, Any]) -> Dict[str, Any]:
    """Addiert mehrere count_proposals()-Ergebnisse zusammen."""
    merged: Dict[str, Any] = {
        "total":    0,
        "accepted": 0,
        "rejected": 0,
        "by_subclass":        defaultdict(lambda: {"accepted": 0, "rejected": 0, "total": 0}),
        "by_change_type":     defaultdict(lambda: {"accepted": 0, "rejected": 0, "total": 0}),
        "by_subclass_change": defaultdict(lambda: {"accepted": 0, "rejected": 0, "total": 0}),
    }
    for c in counts_list:
        merged["total"]    += c["total"]
        merged["accepted"] += c["accepted"]
        merged["rejected"] += c["rejected"]
        for sub, vals in c["by_subclass"].items():
            for k, v in vals.items():
                merged["by_subclass"][sub][k] += v
        for ct, vals in c["by_change_type"].items():
            for k, v in vals.items():
                merged["by_change_type"][ct][k] += v
        for key, vals in c["by_subclass_change"].items():
            for k, v in vals.items():
                merged["by_subclass_change"][key][k] += v

    merged["by_subclass"]        = dict(merged["by_subclass"])
    merged["by_change_type"]     = dict(merged["by_change_type"])
    merged["by_subclass_change"] = dict(merged["by_subclass_change"])
    return merged


# ── Report-Renderer ───────────────────────────────────────────────────────────

def render_counts_table(
        counts: Dict[str, Any],
        title: str,
        breakdown_key: str = "by_subclass_change",
) -> str:
    """Rendert eine Auswertungstabelle für einen Fall oder die Gesamtübersicht."""
    lines: List[str] = []
    total    = counts["total"]
    accepted = counts["accepted"]
    rejected = counts["rejected"]

    lines.append(sep("═"))
    lines.append(title)
    lines.append(sep("═"))
    lines.append(f"  Proposals gesamt:   {total:>4}")
    lines.append(f"  Accepted:           {accepted:>4}  ({pct(accepted, total)})")
    lines.append(f"  Rejected:           {rejected:>4}  ({pct(rejected, total)})")

    # ── Aufschlüsselung nach Subklasse ────────────────────────────────────────
    by_sub = counts.get("by_subclass", {})
    if by_sub:
        lines.append("")
        lines.append(sep("─"))
        lines.append("  Nach Subklasse:")
        lines.append(sep("─"))
        lines.append(f"  {'Subklasse':<40} {'Total':>6} {'Accept':>7} {'Reject':>7}  {'Accept%':>8} {'Reject%':>8}")
        lines.append(f"  {sep('·', 86)}")
        for sub in sorted(by_sub):
            v = by_sub[sub]
            t = v["total"]
            a = v["accepted"]
            r = v["rejected"]
            lines.append(
                f"  {sub:<40} {t:>6} {a:>7} {r:>7}  {pct(a, t):>8} {pct(r, t):>8}"
            )

    # ── Aufschlüsselung nach Change-Type ─────────────────────────────────────
    by_ct = counts.get("by_change_type", {})
    if by_ct:
        lines.append("")
        lines.append(sep("─"))
        lines.append("  Nach Change-Type:")
        lines.append(sep("─"))
        lines.append(f"  {'Change-Type':<35} {'Total':>6} {'Accept':>7} {'Reject':>7}  {'Accept%':>8} {'Reject%':>8}")
        lines.append(f"  {sep('·', 81)}")
        for ct in sorted(by_ct):
            v = by_ct[ct]
            t = v["total"]
            a = v["accepted"]
            r = v["rejected"]
            lines.append(
                f"  {ct:<35} {t:>6} {a:>7} {r:>7}  {pct(a, t):>8} {pct(r, t):>8}"
            )

    # ── Aufschlüsselung nach Subklasse × Change-Type ─────────────────────────
    by_sc = counts.get("by_subclass_change", {})
    if by_sc:
        lines.append("")
        lines.append(sep("─"))
        lines.append("  Nach Subklasse × Change-Type:")
        lines.append(sep("─"))
        lines.append(f"  {'Subklasse | Change-Type':<60} {'Total':>6} {'Accept':>7} {'Reject':>7}  {'Accept%':>8} {'Reject%':>8}")
        lines.append(f"  {sep('·', 102)}")
        for key in sorted(by_sc):
            v = by_sc[key]
            t = v["total"]
            a = v["accepted"]
            r = v["rejected"]
            lines.append(
                f"  {key:<60} {t:>6} {a:>7} {r:>7}  {pct(a, t):>8} {pct(r, t):>8}"
            )

    return "\n".join(lines)


def render_cross_case_summary(
        per_case: Dict[int, Dict[str, Any]],
        overall: Dict[str, Any],
        case_nums: List[int],
) -> str:
    """Rendert die fallübergreifende Vergleichstabelle."""
    lines: List[str] = []
    lines.append("")
    lines.append(sep("═"))
    lines.append("FALLÜBERGREIFENDER VERGLEICH")
    lines.append(sep("═"))
    lines.append(f"  {'Fall':<10} {'Total':>6} {'Accept':>7} {'Reject':>7}  {'Accept%':>8} {'Reject%':>8}")
    lines.append(f"  {sep('─', 54)}")

    for cn in sorted(case_nums):
        if cn not in per_case:
            lines.append(f"  case_{cn:02d}     {'—':>6}")
            continue
        c = per_case[cn]
        t = c["total"]
        a = c["accepted"]
        r = c["rejected"]
        lines.append(
            f"  case_{cn:02d}    {t:>6} {a:>7} {r:>7}  {pct(a, t):>8} {pct(r, t):>8}"
        )

    lines.append(f"  {sep('─', 54)}")
    t = overall["total"]
    a = overall["accepted"]
    r = overall["rejected"]
    lines.append(
        f"  {'GESAMT':<10} {t:>6} {a:>7} {r:>7}  {pct(a, t):>8} {pct(r, t):>8}"
    )

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Auswertung accepted/rejected Proposals pro synthetischem Fall."
    )
    ap.add_argument(
        "--base_dir", type=str, default=".",
        help="Projektverzeichnis mit injection/-Ordner (default: aktuelles Verzeichnis)",
    )
    ap.add_argument(
        "--cases", type=int, nargs="+", default=DEFAULT_CASES,
        help=f"Fallnummern (default: {DEFAULT_CASES})",
    )
    ap.add_argument(
        "--out_dir", type=str, default="evaluation",
        help="Ausgabeverzeichnis (default: evaluation)",
    )
    return ap.parse_args()


def main() -> None:
    args   = parse_args()
    base   = Path(args.base_dir).resolve()
    out_dir = (base / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_nums: List[int] = sorted(args.cases)
    per_case: Dict[int, Dict[str, Any]] = {}
    all_counts: List[Dict[str, Any]] = []
    missing: List[int] = []

    print(f"[INFO] Basis-Verzeichnis: {base}")
    print(f"[INFO] Fälle: {case_nums}")
    print()

    for cn in case_nums:
        path = find_proposals_file(base, cn)
        if path is None:
            print(f"  [WARN] Kein proposals-File für case_{cn:02d} gefunden — übersprungen")
            missing.append(cn)
            continue
        proposals = load_proposals(path)
        counts    = count_proposals(proposals)
        per_case[cn] = counts
        all_counts.append(counts)
        print(
            f"  case_{cn:02d}: {counts['total']:>3} proposals | "
            f"{counts['accepted']:>3} accepted | {counts['rejected']:>3} rejected"
            f"  ({path.name})"
        )

    if not all_counts:
        print("[ERROR] Keine Proposals geladen — Report kann nicht erstellt werden.")
        return

    overall = merge_counts(*all_counts)

    # ── Textreport ────────────────────────────────────────────────────────────
    sections: List[str] = []

    header = [
        sep("═"),
        "AUSWERTUNG SYNTHETISCHE FEHLERGENERIERUNG",
        "Accepted vs. Rejected Proposals (Plausibilitätsprüfung, 2. Call)",
        sep("═"),
        f"  Fälle ausgewertet: {sorted(per_case.keys())}",
        f"  Fälle nicht gefunden: {missing if missing else '—'}",
    ]
    sections.append("\n".join(header))

    # Fallübergreifender Vergleich zuerst
    sections.append(render_cross_case_summary(per_case, overall, case_nums))

    # Gesamtauswertung
    sections.append(render_counts_table(overall, "GESAMT (alle Fälle)"))

    # Pro Fall
    for cn in sorted(per_case.keys()):
        sections.append(
            render_counts_table(per_case[cn], f"FALL {cn:02d}  (case_{cn:02d})")
        )

    sections.append("")
    sections.append(sep("═"))
    sections.append("ENDE DES REPORTS")
    sections.append(sep("═"))

    report = "\n".join(sections)

    txt_path = out_dir / "eval_synthetic_error.txt"
    txt_path.write_text(report, encoding="utf-8")
    print(f"\n[OK] Textreport: {txt_path}")
    print(f"     {len(report.splitlines())} Zeilen, {len(report)} Zeichen")

    # ── JSON-Export ───────────────────────────────────────────────────────────
    json_out = {
        "cases_evaluated": sorted(per_case.keys()),
        "cases_missing":   missing,
        "overall":         overall,
        "per_case":        {f"case_{cn:02d}": v for cn, v in sorted(per_case.items())},
    }
    json_path = out_dir / "eval_synthetic_error.json"
    json_path.write_text(
        json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] JSON-Export:  {json_path}")


if __name__ == "__main__":
    main()
