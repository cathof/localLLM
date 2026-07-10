#!/bin/bash
# pipeline/04_evaluate_synthetic.sh
# ─────────────────────────────────────────────────────────────────────────────
# Evaluiert die RAG-Predictions gegen die synthetischen Ground Truths.
# Cases 01–08, überspringt Case 06 (manuelle Ground Truth).
#
# Voraussetzung:
#   - pipeline/03_run_rag_synthetic.sh wurde ausgeführt
#   - predictions/predictions_case_XX_synthetic.jsonl existiert pro Case
#   - ground_truth/ground_truth_case_XX_synthetic.jsonl existiert pro Case
#
# Aufruf: bash pipeline/04_evaluate_synthetic.sh
#
# Output pro Case:
#   → eval_results/case_XX_synthetic/  (Metriken, CSV, Details)
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EVAL="$ROOT_DIR/evaluate_predictions.py"
GT_DIR="$ROOT_DIR/ground_truth"
PRED_DIR="$ROOT_DIR/predictions"
EVAL_DIR="$ROOT_DIR/eval_results"
TAX="$ROOT_DIR/tax/taxonomy.json"

MIN_SPAN_SCORE=0.20

# Cases 01–08 ohne 06
CASES=(01 02 03 04 05 07 08)

echo "========================================"
echo "  Phase 4: Evaluation (synthetisch)"
echo "  Cases: ${CASES[*]}"
echo "  Min-Span-Score: $MIN_SPAN_SCORE"
echo "  Root:  $ROOT_DIR"
echo "========================================"
echo ""

cd "$ROOT_DIR"

SKIPPED=()
DONE=()

for CASE in "${CASES[@]}"; do
    GT="$GT_DIR/ground_truth_case_${CASE}_synthetic.jsonl"
    PRED="$PRED_DIR/predictions_case_${CASE}_synthetic.jsonl"
    OUT="$EVAL_DIR/case_${CASE}_synthetic"

    echo "────────────────────────────────────────"
    echo "  Case ${CASE}"
    echo "────────────────────────────────────────"

    if [ ! -f "$GT" ]; then
        echo "[SKIP] $GT nicht gefunden"
        echo "       → pipeline/02_inject_validate.sh zuerst ausführen"
        SKIPPED+=("$CASE")
        echo ""
        continue
    fi

    if [ ! -f "$PRED" ]; then
        echo "[SKIP] $PRED nicht gefunden"
        echo "       → pipeline/03_run_rag_synthetic.sh zuerst ausführen"
        SKIPPED+=("$CASE")
        echo ""
        continue
    fi

    mkdir -p "$OUT"

    echo "[INFO] Ground Truth: $GT"
    echo "[INFO] Predictions:  $PRED"
    echo "[INFO] Output:       $OUT"
    echo ""

    python "$EVAL" \
        --ground_truth_jsonl "$GT" \
        --predictions_jsonl "$PRED" \
        --case_id "case_${CASE}" \
        --taxonomy_json "$TAX" \
        --output_dir "$OUT" \
        --min_span_score "$MIN_SPAN_SCORE"

    echo ""
    echo "[OK] Case ${CASE} abgeschlossen."
    echo ""
    DONE+=("$CASE")
done

echo "========================================"
echo "  Phase 4 abgeschlossen."
echo ""
if [ ${#DONE[@]} -gt 0 ]; then
    echo "  Evaluiert: ${DONE[*]}"
fi
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "  Übersprungen: ${SKIPPED[*]}"
fi
echo ""
echo "  Output: eval_results/case_XX_synthetic/"
echo "========================================"
