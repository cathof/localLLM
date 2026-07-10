#!/usr/bin/env python3
"""
analyze_fp_by_agent.py

Wertet aus, WELCHER AGENT die meisten False Positives erzeugt — pro Modell.

Datengrundlage: eval_results/aggregated_<modell>/false_positives.jsonl
Jede Zeile ist ein False-Positive-Record aus evaluate_predictions.py und enthält
u.a. die Felder: agent_scope, main_class_id, subclass_id, span_text.

Zuordnung Finding -> Agent (zweistufig):
  1. agent_scope (vom Evaluator gesetzt) trennt drei Agenten eindeutig:
       document_hypothesis  -> Agent 5 (Hypothesenkonsistenz)
       segment_calculation  -> Agent 4 (Rechenprüfer)
       segment_language     -> Agent 3 (Sprach-/Formalprüfer)
     Alle übrigen Findings landen im Sammelwert "segment_or_document_factual",
     weil die Pipeline Agent 2 / 6 / 7 vor dem Schreiben zusammenführt.
  2. Für diesen Sammelwert wird über die Hauptklasse (main_class_id) verfeinert.
     Dieses Mapping ist APPROXIMATIV — bitte gegen die eigenen Agenten-Definitionen
     prüfen und bei Bedarf anpassen (AGENT_FROM_MAIN unten).

Aufruf:
    python analyze_fp_by_agent.py --root eval_results
    python analyze_fp_by_agent.py --root eval_results --csv fp_by_agent.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

# ── Zuordnung agent_scope -> Agent (eindeutig) ────────────────────────────────
AGENT_FROM_SCOPE: Dict[str, str] = {
    "document_hypothesis": "Agent 5 – Hypothesenkonsistenz",
    "segment_calculation": "Agent 4 – Rechenprüfer",
    "segment_language":    "Agent 3 – Sprach-/Formalprüfer",
}

# ── Verfeinerung des factual-Sammelwerts über die Hauptklasse ─────────────────
# APPROXIMATIV — anpassen, falls die eigene Taxonomie/Agentenzuteilung abweicht.
# Schlüssel = main_class_id, Wert = Agent.
AGENT_FROM_MAIN: Dict[str, str] = {
    "STRUKT": "Agent 2 – Fachprüfer (Struktur/Evidenz)",
    "QMQS":   "Agent 2 – Fachprüfer (QM/QS)",
    "RECHT":  "Agent 7 – Aussageabsicherung/Modalität",
    # Hinweis: Agent 6 (Referenzfakten-Konsistenz) lässt sich nicht eindeutig
    # über die Hauptklasse trennen und kann in STRUKT/RECHT mit enthalten sein.
}

FACTUAL_SCOPE = "segment_or_document_factual"
UNRESOLVED = "Agent 2/6/7 – factual (unaufgeschlüsselt)"


def attribute_agent(record: dict) -> str:
    """Ordnet einen FP-Record einem Agenten zu."""
    scope = (record.get("agent_scope") or "").strip()
    if scope in AGENT_FROM_SCOPE:
        return AGENT_FROM_SCOPE[scope]

    if scope == FACTUAL_SCOPE or not scope:
        main = (record.get("main_class_id") or "").strip()
        if main in AGENT_FROM_MAIN:
            return AGENT_FROM_MAIN[main]
        return f"{UNRESOLVED}: {main or 'unbekannt'}"

    # Unerwarteter scope-Wert -> transparent ausweisen statt verschlucken
    return f"unbekannter scope: {scope}"


def load_false_positives(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {path.name} Zeile {line_no}: ungültiges JSON ({exc})")
    return records


def model_name_from_dir(d: Path) -> str:
    name = d.name
    return name[len("aggregated_"):] if name.startswith("aggregated_") else name


def analyze(root: Path) -> Tuple[Dict[str, Counter], Dict[str, Counter]]:
    """Liefert (agent_counts_per_model, mainclass_counts_per_model)."""
    agent_counts: Dict[str, Counter] = {}
    main_counts: Dict[str, Counter] = {}

    fp_files = sorted(root.glob("aggregated_*/false_positives.jsonl"))
    if not fp_files:
        # Fallback: beliebige false_positives.jsonl unter root
        fp_files = sorted(root.glob("*/false_positives.jsonl"))

    if not fp_files:
        raise SystemExit(f"Keine false_positives.jsonl unter {root} gefunden.")

    for fp_path in fp_files:
        model = model_name_from_dir(fp_path.parent)
        records = load_false_positives(fp_path)
        agent_counts[model] = Counter(attribute_agent(r) for r in records)
        main_counts[model] = Counter((r.get("main_class_id") or "?") for r in records)

    return agent_counts, main_counts


def print_per_model(agent_counts: Dict[str, Counter]) -> None:
    for model, counter in agent_counts.items():
        total = sum(counter.values())
        print(f"\n=== {model}  (False Positives gesamt: {total}) ===")
        if total == 0:
            print("  (keine False Positives)")
            continue
        for agent, n in counter.most_common():
            share = n / total * 100
            print(f"  {n:5d}  ({share:5.1f} %)  {agent}")


def print_matrix(agent_counts: Dict[str, Counter]) -> None:
    """Agent × Modell als kompakte Matrix."""
    all_agents = sorted({a for c in agent_counts.values() for a in c})
    models = list(agent_counts.keys())
    width = max((len(a) for a in all_agents), default=10)

    print("\n\n=== FP pro Agent über alle Modelle ===\n")
    header = f"{'Agent'.ljust(width)} | " + " | ".join(m[:14].rjust(14) for m in models) + " |   Summe"
    print(header)
    print("-" * len(header))
    for agent in all_agents:
        row_total = sum(agent_counts[m].get(agent, 0) for m in models)
        cells = " | ".join(str(agent_counts[m].get(agent, 0)).rjust(14) for m in models)
        print(f"{agent.ljust(width)} | {cells} | {row_total:7d}")
    print("-" * len(header))
    totals = " | ".join(str(sum(agent_counts[m].values())).rjust(14) for m in models)
    grand = sum(sum(c.values()) for c in agent_counts.values())
    print(f"{'Summe'.ljust(width)} | {totals} | {grand:7d}")


def write_csv(agent_counts: Dict[str, Counter], out: Path) -> None:
    all_agents = sorted({a for c in agent_counts.values() for a in c})
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "agent", "false_positives"])
        for model, counter in agent_counts.items():
            for agent in all_agents:
                writer.writerow([model, agent, counter.get(agent, 0)])
    print(f"\n[INFO] CSV geschrieben: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FP-Auswertung pro Agent und Modell.")
    ap.add_argument("--root", default="eval_results",
                    help="Wurzelverzeichnis mit aggregated_<modell>/false_positives.jsonl")
    ap.add_argument("--csv", default="", help="Optionaler Pfad für eine CSV-Ausgabe.")
    ap.add_argument("--show-mainclass", action="store_true",
                    help="Zusätzlich die rohe Hauptklassen-Verteilung ausgeben (zur Prüfung des Mappings).")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    agent_counts, main_counts = analyze(root)

    print_per_model(agent_counts)
    print_matrix(agent_counts)

    if args.show_mainclass:
        print("\n\n=== Rohe Hauptklassen-Verteilung der FP (zur Mapping-Prüfung) ===")
        for model, counter in main_counts.items():
            print(f"\n  {model}:")
            for main, n in counter.most_common():
                print(f"    {n:5d}  {main}")

    if args.csv:
        write_csv(agent_counts, Path(args.csv).expanduser().resolve())


if __name__ == "__main__":
    main()
