#!/bin/bash
# pipeline/04_evaluate_synthetic.sh
# ─────────────────────────────────────────────────────────────────────────────
# Evaluiert RAG-Predictions gegen synthetische Ground Truths.
# Cases 01–08, überspringt Case 06 (manuelle Ground Truth).
#
# Aufruf:
#   bash pipeline/04_evaluate_synthetic.sh                       # Modell aus .env
#   bash pipeline/04_evaluate_synthetic.sh qwen2.5:14b-instruct  # Modell explizit
#
# Output pro Case:
#   → eval_results/case_XX_synthetic_<model_tag>/
#
# Vergleichs-Summary aller bisher evaluierten Modelle:
#   → eval_results/model_comparison.txt
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EVAL="$ROOT_DIR/evaluate_predictions.py"
GT_DIR="$ROOT_DIR/ground_truth"
PRED_DIR="$ROOT_DIR/predictions"
EVAL_DIR="$ROOT_DIR/eval_results"
TAX="$ROOT_DIR/tax/taxonomy.json"
ENV_FILE="$ROOT_DIR/.env"

MIN_SPAN_SCORE=0.20

# ── Modell bestimmen ──────────────────────────────────────────────────────────
if [ -n "$1" ]; then
    MODEL="$1"
elif [ -f "$ENV_FILE" ]; then
    MODEL=$(grep -E "^LLM_MODEL=" "$ENV_FILE" | head -1 | cut -d'=' -f2 | tr -d '"' | tr -d "'")
fi

if [ -z "$MODEL" ]; then
    echo "[FATAL] Kein Modell angegeben."
    echo "        LLM_MODEL in .env setzen oder als Argument übergeben:"
    echo "        bash pipeline/04_evaluate_synthetic.sh qwen2.5:14b-instruct"
    exit 1
fi

MODEL_TAG=$(echo "$MODEL" | tr ':' '-' | tr '/' '-')

# Cases 01–08 ohne 06
CASES=(01 02 03 04 05 07 08)

echo "========================================"
echo "  Phase 4: Evaluation (synthetisch)"
echo "  Modell:         $MODEL"
echo "  Tag:            $MODEL_TAG"
echo "  Cases:          ${CASES[*]}"
echo "  Min-Span-Score: $MIN_SPAN_SCORE"
echo "  Root:           $ROOT_DIR"
echo "========================================"
echo ""

cd "$ROOT_DIR"
mkdir -p "$EVAL_DIR"

SKIPPED=()
DONE=()

# Aggregierte Metriken über alle Cases
TOTAL_TP=0
TOTAL_FP=0
TOTAL_FN=0
TOTAL_GOLD=0
TOTAL_PRED=0

for CASE in "${CASES[@]}"; do
    GT="$GT_DIR/ground_truth_case_${CASE}_synthetic.jsonl"
    PRED="$PRED_DIR/predictions_case_${CASE}_synthetic_${MODEL_TAG}.jsonl"
    OUT="$EVAL_DIR/case_${CASE}_synthetic_${MODEL_TAG}"

    echo "────────────────────────────────────────"
    echo "  Case ${CASE} | $MODEL_TAG"
    echo "────────────────────────────────────────"

    if [ ! -f "$GT" ]; then
        echo "[SKIP] $GT nicht gefunden"
        SKIPPED+=("$CASE")
        echo ""
        continue
    fi

    if [ ! -f "$PRED" ]; then
        echo "[SKIP] $PRED nicht gefunden"
        echo "       → pipeline/03_run_rag_synthetic.sh $MODEL zuerst ausführen"
        SKIPPED+=("$CASE")
        echo ""
        continue
    fi

    mkdir -p "$OUT"

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

    # Metriken aus summary.json aggregieren
    SUMMARY="$OUT/summary.json"
    if [ -f "$SUMMARY" ]; then
        TP=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('finding_level_span_only',{}).get('tp',0))" 2>/dev/null || echo 0)
        FP=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('finding_level_span_only',{}).get('fp',0))" 2>/dev/null || echo 0)
        FN=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('finding_level_span_only',{}).get('fn',0))" 2>/dev/null || echo 0)
        GOLD=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('gold_findings', $TP+$FN))" 2>/dev/null || echo 0)
        PRED_N=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('pred_findings', $TP+$FP))" 2>/dev/null || echo 0)
        TOTAL_TP=$((TOTAL_TP + TP))
        TOTAL_FP=$((TOTAL_FP + FP))
        TOTAL_FN=$((TOTAL_FN + FN))
        TOTAL_GOLD=$((TOTAL_GOLD + GOLD))
        TOTAL_PRED=$((TOTAL_PRED + PRED_N))
        echo "[OK] Case ${CASE}: TP=$TP FP=$FP FN=$FN"
    fi

    echo ""
    DONE+=("$CASE")
done

# ── Aggregierte Metriken berechnen ────────────────────────────────────────────
PREC=$(python3 -c "tp=$TOTAL_TP; fp=$TOTAL_FP; print(f'{tp/(tp+fp):.4f}' if tp+fp>0 else '0.0000')")
REC=$(python3  -c "tp=$TOTAL_TP; fn=$TOTAL_FN; print(f'{tp/(tp+fn):.4f}' if tp+fn>0 else '0.0000')")
F1=$(python3   -c "tp=$TOTAL_TP; fp=$TOTAL_FP; fn=$TOTAL_FN; p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0; print(f'{2*p*r/(p+r):.4f}' if p+r>0 else '0.0000')")

echo "========================================"
echo "  Gesamtergebnis: $MODEL"
echo "========================================"
echo "  Cases evaluiert: ${DONE[*]}"
echo "  Gold Findings:   $TOTAL_GOLD"
echo "  Pred Findings:   $TOTAL_PRED"
echo "  TP: $TOTAL_TP  FP: $TOTAL_FP  FN: $TOTAL_FN"
echo "  Precision: $PREC"
echo "  Recall:    $REC"
echo "  F1:        $F1"
echo ""

# ── Vergleichs-Summary schreiben ──────────────────────────────────────────────
COMPARISON="$EVAL_DIR/model_comparison.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Header beim ersten Eintrag
if [ ! -f "$COMPARISON" ]; then
    printf "%-50s  %6s  %6s  %6s  %6s  %7s  %7s  %7s  %s\n" \
        "Modell" "Gold" "Pred" "TP" "FP" "Prec" "Rec" "F1" "Timestamp" \
        > "$COMPARISON"
    printf "%-50s  %6s  %6s  %6s  %6s  %7s  %7s  %7s  %s\n" \
        "$(printf '%0.s-' {1..50})" "------" "------" "------" "------" "-------" "-------" "-------" "-------------------" \
        >> "$COMPARISON"
fi

# Eintrag für dieses Modell (bestehenden überschreiben falls vorhanden)
# Temporäre Datei ohne dieses Modell + neuen Eintrag
TMPFILE=$(mktemp)
grep -v "^${MODEL_TAG}" "$COMPARISON" > "$TMPFILE" || true
cat "$TMPFILE" > "$COMPARISON"
rm "$TMPFILE"

printf "%-50s  %6d  %6d  %6d  %6d  %7s  %7s  %7s  %s\n" \
    "$MODEL_TAG" "$TOTAL_GOLD" "$TOTAL_PRED" "$TOTAL_TP" "$TOTAL_FP" \
    "$PREC" "$REC" "$F1" "$TIMESTAMP" \
    >> "$COMPARISON"

echo "  Vergleichs-Summary aktualisiert: $COMPARISON"
echo ""
cat "$COMPARISON"
echo "========================================"