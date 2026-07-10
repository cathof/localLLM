#!/usr/bin/env python3
"""
Vergleicht false_positives.jsonl aller drei Modelle pro Case.
Schreibt FPs die in ALLEN drei Modellen vorkommen in ein CSV.

Verwendung:
    python compare_false_positives.py --eval_dir eval_results
    python compare_false_positives.py --eval_dir eval_results --similarity 0.72
    python compare_false_positives.py --eval_dir eval_results --out consensus_fps.csv

Struktur erwartet:
    eval_results/
        case_01_synthetic_mistral-nemo-latest/false_positives.jsonl
        case_01_synthetic_qwen2.5-72b-instruct-q4_K_M/false_positives.jsonl
        case_01_synthetic_gemma3-12b-it-q4_K_M/false_positives.jsonl
        case_02_synthetic_.../false_positives.jsonl
        ...
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Cases und Modelle ─────────────────────────────────────────────────────────

CASES = ["01", "02", "03", "04", "05", "07", "08"]  # 06 ausgeschlossen

MODELS = [
    "mistral-nemo-latest",
    "qwen2.5-72b-instruct-q4_K_M",
    "gemma3-12b-it-q4_K_M",
]

CSV_FIELDS = [
    "case_id",
    "models_agreed",
    "subclass_id",
    "main_class_id",
    "change_type_id",
    "severity_id",
    "span_text",
    "correction",
    "rationale",
    "agent_scope",
]


# ── Text-Normalisierung und Ähnlichkeit ───────────────────────────────────────

def _normalize(text: str) -> str:
    s = re.sub(r"[^\wäöüÄÖÜß]+", " ", (text or "").casefold())
    return re.sub(r"\s+", " ", s).strip()


def _token_set(text: str) -> set:
    return {t for t in _normalize(text).split() if t}


def jaccard(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── JSONL laden ───────────────────────────────────────────────────────────────

def load_fps(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec.pop("best_unmatched_gold_in_case", None)
                records.append(rec)
            except json.JSONDecodeError:
                pass
    return records


# ── Verzeichnis-Lookup ────────────────────────────────────────────────────────

def find_fp_path(eval_dir: Path, case: str, model: str) -> Optional[Path]:
    """Sucht den passenden Unterordner für case + model."""
    pattern = f"case_{case}_synthetic_{model}"
    # Exakter Match zuerst
    exact = eval_dir / pattern / "false_positives.jsonl"
    if exact.exists():
        return exact
    # Fallback: Unterordner der case und model im Namen hat
    for d in eval_dir.iterdir():
        if not d.is_dir():
            continue
        if f"case_{case}" in d.name and model in d.name:
            fp = d / "false_positives.jsonl"
            if fp.exists():
                return fp
    return None


# ── Konsens-Matching ──────────────────────────────────────────────────────────

def find_consensus_fps(
        fps_per_model: Dict[str, List[Dict]],
        similarity_threshold: float,
) -> List[Dict]:
    """
    Findet FPs die in ALLEN Modellen vorkommen (Jaccard-Ähnlichkeit auf span_text).

    Algorithmus:
    - Nimm das erste Modell als Basis.
    - Für jedes Basis-FP: prüfe ob alle anderen Modelle ein ähnliches FP haben.
    - "Ähnlich" = Jaccard(span_text_a, span_text_b) >= similarity_threshold.
    - Ausgabe: deduplizierte Liste mit dem Basis-FP + Liste der übereinstimmenden Modelle.
    """
    model_names = list(fps_per_model.keys())
    if len(model_names) < 2:
        return []

    base_model = model_names[0]
    other_models = model_names[1:]
    base_fps = fps_per_model[base_model]

    consensus = []

    for base_fp in base_fps:
        base_span = base_fp.get("span_text", "")
        if not base_span:
            continue

        matched_models = [base_model]

        for other in other_models:
            other_fps = fps_per_model[other]
            best_score = 0.0
            for other_fp in other_fps:
                score = jaccard(base_span, other_fp.get("span_text", ""))
                if score > best_score:
                    best_score = score
            if best_score >= similarity_threshold:
                matched_models.append(other)

        if len(matched_models) == len(model_names):
            row = dict(base_fp)
            row["models_agreed"] = " | ".join(matched_models)
            consensus.append(row)

    return consensus


# ── CSV schreiben ─────────────────────────────────────────────────────────────

def write_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


# ── Hauptlogik ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Findet FPs die in allen drei Modellen übereinstimmen."
    )
    ap.add_argument(
        "--eval_dir", required=True,
        help="Pfad zum eval_results-Ordner"
    )
    ap.add_argument(
        "--similarity", type=float, default=0.72,
        help="Jaccard-Schwellwert für span_text-Ähnlichkeit (default: 0.72)"
    )
    ap.add_argument(
        "--out", default="consensus_false_positives.csv",
        help="Ausgabe-CSV (default: consensus_false_positives.csv)"
    )
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    if not eval_dir.exists():
        sys.exit(f"[ERROR] Verzeichnis nicht gefunden: {eval_dir}")

    all_consensus: List[Dict] = []
    total_per_case: Dict[str, int] = {}

    print(f"{'Case':<8}  {'mistral':>8}  {'qwen72b':>8}  {'gemma':>8}  {'Konsens':>8}")
    print("-" * 52)

    for case in CASES:
        fps_per_model: Dict[str, List[Dict]] = {}

        for model in MODELS:
            fp_path = find_fp_path(eval_dir, case, model)
            if fp_path is None:
                print(f"[WARN] case_{case} / {model}: false_positives.jsonl nicht gefunden")
                fps_per_model[model] = []
            else:
                fps_per_model[model] = load_fps(fp_path)

        consensus = find_consensus_fps(fps_per_model, args.similarity)

        # case_id eintragen
        for row in consensus:
            row["case_id"] = f"case_{case}"

        all_consensus.extend(consensus)
        total_per_case[case] = len(consensus)

        counts = [len(fps_per_model[m]) for m in MODELS]
        print(
            f"case_{case}  "
            f"{counts[0]:>8}  {counts[1]:>8}  {counts[2]:>8}  {len(consensus):>8}"
        )

    print("-" * 52)
    print(f"{'Total':>8}  {'':>8}  {'':>8}  {'':>8}  {len(all_consensus):>8}")
    print()

    if not all_consensus:
        print("[INFO] Keine konsensus-FPs gefunden.")
        return

    out_path = Path(args.out).resolve()
    write_csv(all_consensus, out_path)
    print(f"[INFO] {len(all_consensus)} Konsensus-FPs geschrieben nach: {out_path}")
    print(f"[INFO] Ähnlichkeitsschwellwert: {args.similarity}")
    print()
    print("Tipp: Öffne das CSV und füge eine Spalte 'gt_aufnehmen' (ja/nein/prüfen) hinzu.")


if __name__ == "__main__":
    main()
