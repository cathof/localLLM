#!/usr/bin/env python3
"""
Liest false_positives.jsonl und gibt alle FPs tabellarisch aus.
Ignoriert das Debug-Feld best_unmatched_gold_in_case.

Verwendung:
    python read_false_positives.py --fp false_positives.jsonl
    python read_false_positives.py --fp false_positives.jsonl --subclass STRUKT_EVIDENZ
    python read_false_positives.py --fp false_positives.jsonl --csv output.csv
"""

import argparse
import json
import csv
import sys
from pathlib import Path


def load_fps(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Zeile übersprungen: {e}", file=sys.stderr)
    return records


def truncate(text: str, max_len: int = 80) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def print_table(records: list[dict]) -> None:
    col_w = {"nr": 5, "id": 22, "subclass": 30, "span": 50, "correction": 30, "rationale": 55}

    header = (
        f"{'Nr':>{col_w['nr']}}  "
        f"{'finding_id':<{col_w['id']}}  "
        f"{'subclass_id':<{col_w['subclass']}}  "
        f"{'span_text':<{col_w['span']}}  "
        f"{'correction':<{col_w['correction']}}  "
        f"{'rationale':<{col_w['rationale']}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for i, r in enumerate(records, 1):
        print(
            f"{i:>{col_w['nr']}}  "
            f"{truncate(r.get('finding_id',''), col_w['id']):<{col_w['id']}}  "
            f"{truncate(r.get('subclass_id',''), col_w['subclass']):<{col_w['subclass']}}  "
            f"{truncate(r.get('span_text',''), col_w['span']):<{col_w['span']}}  "
            f"{truncate(r.get('correction',''), col_w['correction']):<{col_w['correction']}}  "
            f"{truncate(r.get('rationale',''), col_w['rationale']):<{col_w['rationale']}}"
        )

    print(sep)
    print(f"Total: {len(records)} False Positives")


def write_csv(records: list[dict], out_path: Path) -> None:
    fields = ["finding_id", "main_class_id", "subclass_id", "change_type_id",
              "severity_id", "span_text", "correction", "rationale", "agent_scope"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[INFO] CSV geschrieben: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", required=True, help="Pfad zur false_positives.jsonl")
    ap.add_argument("--subclass", default="", help="Optional: Filter auf eine Subklasse")
    ap.add_argument("--csv", default="", help="Optional: Ausgabe als CSV")
    args = ap.parse_args()

    path = Path(args.fp).resolve()
    if not path.exists():
        sys.exit(f"[ERROR] Datei nicht gefunden: {path}")

    records = load_fps(path)

    # Debug-Feld entfernen — spielt keine Rolle für die Analyse
    for r in records:
        r.pop("best_unmatched_gold_in_case", None)

    if args.subclass:
        records = [r for r in records if r.get("subclass_id") == args.subclass]
        print(f"[Filter] subclass_id = {args.subclass}  →  {len(records)} Einträge\n")

    print_table(records)

    if args.csv:
        write_csv(records, Path(args.csv))


if __name__ == "__main__":
    main()
