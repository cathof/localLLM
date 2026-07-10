#!/bin/bash
# pipeline/03_run_rag_synthetic.sh
# ─────────────────────────────────────────────────────────────────────────────
# Führt die RAG-Fehlererkennungspipeline auf allen synthetischen Dokumenten aus.
# Cases 01–08, überspringt Case 06 (manuelle Ground Truth).
#
# Voraussetzung:
#   - pipeline/01_generate.sh + manueller Review + pipeline/02_inject_validate.sh
#     wurden für alle Cases ausgeführt
#   - case_documents/case_XX_modified.docx existiert pro Case
#   - ground_truth/ground_truth_case_XX_synthetic.jsonl existiert pro Case
#
# Aufruf: bash pipeline/03_run_rag_synthetic.sh
#
# Output pro Case:
#   → predictions/predictions_case_XX_synthetic.jsonl
#   → reference_facts/reference_facts_case_XX_synthetic.json
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

# Cases 01–08 ohne 06
CASES=(01 02 03 04 05 07 08)

echo "========================================"
echo "  Phase 3: RAG-Fehlererkennung (synthetisch)"
echo "  Cases: ${CASES[*]}"
echo "  Root:  $ROOT_DIR"
echo "========================================"
echo ""

cd "$ROOT_DIR"

# Output-Verzeichnisse anlegen
mkdir -p "$PRED_DIR" "$REF_DIR"

SKIPPED=()
DONE=()

for CASE in "${CASES[@]}"; do
    DOC="$CASES_DIR/case_${CASE}_modified.docx"
    GT="$GT_DIR/ground_truth_case_${CASE}_synthetic.jsonl"
    PRED="$PRED_DIR/predictions_case_${CASE}_synthetic.jsonl"
    REF="$REF_DIR/reference_facts_case_${CASE}_synthetic.json"

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

    echo "[INFO] Dokument:    $DOC"
    echo "[INFO] Ground Truth: $GT"
    echo "[INFO] Predictions:  $PRED"
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
    echo "[OK] Case ${CASE} abgeschlossen."
    echo ""
    DONE+=("$CASE")
done

echo "========================================"
echo "  Phase 3 abgeschlossen."
echo ""
if [ ${#DONE[@]} -gt 0 ]; then
    echo "  Verarbeitet: ${DONE[*]}"
fi
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "  Übersprungen: ${SKIPPED[*]}"
fi
echo ""
echo "  Output:"
echo "    predictions/predictions_case_XX_synthetic.jsonl"
echo "    reference_facts/reference_facts_case_XX_synthetic.json"
echo "========================================"
