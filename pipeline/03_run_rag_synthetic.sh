#!/bin/bash
# pipeline/03_run_rag_synthetic.sh
# ─────────────────────────────────────────────────────────────────────────────
# Führt die RAG-Fehlererkennungspipeline auf allen synthetischen Dokumenten aus.
# Cases 01–08, überspringt Case 06 (manuelle Ground Truth).
#
# Aufruf:
#   bash pipeline/03_run_rag_synthetic.sh                       # Modell aus .env
#   bash pipeline/03_run_rag_synthetic.sh qwen2.5:14b-instruct  # Modell explizit
#
# Output pro Case (Modell-Tag im Dateinamen):
#   → predictions/predictions_case_XX_synthetic_qwen2.5-14b-instruct.jsonl
#   → reference_facts/reference_facts_case_XX_synthetic_qwen2.5-14b-instruct.json
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RAG="$ROOT_DIR/rag_answer_reference_facts.py"
CASES_DIR="$ROOT_DIR/case_documents"
GT_DIR="$ROOT_DIR/ground_truth"
PRED_DIR="$ROOT_DIR/predictions"
REF_DIR="$ROOT_DIR/reference_facts"
TAX="$ROOT_DIR/tax/taxonomy.json"
ENV_FILE="$ROOT_DIR/.env"

# ── Modell bestimmen ──────────────────────────────────────────────────────────
# Priorität: 1. CLI-Argument, 2. LLM_MODEL aus .env, 3. Fehler
if [ -n "$1" ]; then
    MODEL="$1"
elif [ -f "$ENV_FILE" ]; then
    MODEL=$(grep -E "^LLM_MODEL=" "$ENV_FILE" | head -1 | cut -d'=' -f2 | tr -d '"' | tr -d "'")
fi

if [ -z "$MODEL" ]; then
    echo "[FATAL] Kein Modell angegeben."
    echo "        LLM_MODEL in .env setzen oder als Argument übergeben:"
    echo "        bash pipeline/03_run_rag_synthetic.sh qwen2.5:14b-instruct"
    exit 1
fi

# ── Modell-Tag für Dateinamen ─────────────────────────────────────────────────
# qwen2.5:32b-instruct-q4_K_M  →  qwen2.5-32b-instruct-q4_K_M
MODEL_TAG=$(echo "$MODEL" | tr ':' '-' | tr '/' '-')

# Cases 01–08 ohne 06
CASES=(01 02 03 04 05 07 08)

echo "========================================"
echo "  Phase 3: RAG-Fehlererkennung (synthetisch)"
echo "  Modell:  $MODEL"
echo "  Tag:     $MODEL_TAG"
echo "  Cases:   ${CASES[*]}"
echo "  Root:    $ROOT_DIR"
echo "========================================"
echo ""

cd "$ROOT_DIR"

mkdir -p "$PRED_DIR" "$REF_DIR"

SKIPPED=()
DONE=()

for CASE in "${CASES[@]}"; do
    DOC="$CASES_DIR/case_${CASE}_modified.docx"
    GT="$GT_DIR/ground_truth_case_${CASE}_synthetic.jsonl"
    PRED="$PRED_DIR/predictions_case_${CASE}_synthetic_${MODEL_TAG}.jsonl"
    REF="$REF_DIR/reference_facts_case_${CASE}_synthetic_${MODEL_TAG}.json"

    echo "────────────────────────────────────────"
    echo "  Case ${CASE}"
    echo "────────────────────────────────────────"

    if [ ! -f "$DOC" ]; then
        echo "[SKIP] $DOC nicht gefunden"
        echo "       → pipeline/02_inject_validate.sh zuerst ausführen"
        SKIPPED+=("$CASE")
        echo ""
        continue
    fi

    if [ ! -f "$GT" ]; then
        echo "[SKIP] $GT nicht gefunden"
        echo "       → pipeline/02_inject_validate.sh zuerst ausführen"
        SKIPPED+=("$CASE")
        echo ""
        continue
    fi

    echo "[INFO] Dokument:     $DOC"
    echo "[INFO] Ground Truth: $GT"
    echo "[INFO] Predictions:  $PRED"
    echo "[INFO] Ref-Facts:    $REF"
    echo ""

    python "$RAG" \
        --document "$DOC" \
        --case_id "case_${CASE}" \
        --taxonomy_json "$TAX" \
        --embeddings artefacts/embeddings_rules.npz \
        --index artefacts/index_rules.jsonl \
        --prepared artefacts/prepared_rules.jsonl \
        --embeddings2 artefacts/embeddings_materials.npz \
        --index2 artefacts/index_materials.jsonl \
        --prepared2 artefacts/prepared_materials.jsonl \
        --save_predictions_jsonl "$PRED" \
        --print_sources \
        --print_context \
        --print_reference_facts \
        --save_reference_facts_json "$REF" \
        --ground_truth "$GT"

    echo ""
    echo "[OK] Case ${CASE} → $PRED"
    echo ""
    DONE+=("$CASE")
done

echo "========================================"
echo "  Phase 3 abgeschlossen."
echo "  Modell:      $MODEL"
echo ""
if [ ${#DONE[@]} -gt 0 ]; then
    echo "  Verarbeitet: ${DONE[*]}"
fi
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "  Übersprungen: ${SKIPPED[*]}"
fi
echo ""
echo "  Output:"
echo "    predictions/predictions_case_XX_synthetic_${MODEL_TAG}.jsonl"
echo "    reference_facts/reference_facts_case_XX_synthetic_${MODEL_TAG}.json"
echo ""
echo "  NÄCHSTER SCHRITT:"
echo "    bash pipeline/04_evaluate_synthetic.sh $MODEL"
echo "========================================"