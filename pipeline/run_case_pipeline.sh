#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_case_pipeline.sh --case_id CASE_ID [options]

Runs document checking and evaluation using defaults from .env.
Works reliably when stored inside a pipeline/ subdirectory.

Options:
  --case_id ID           Required, e.g. case_06
  --document PATH        Optional override for case document path
  --ground_truth PATH    Optional override for ground-truth jsonl
  --predictions PATH     Optional override for predictions jsonl
  --eval_output_dir PATH Optional override for evaluation output dir
  --project_dir PATH     Optional explicit project root. Default: parent of script dir
  --python BIN           Python executable. Default: python
  --no_print_sources     Do not pass --print_sources
  --no_print_context     Do not pass --print_context
  -h, --help             Show this help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CASE_ID=""
DOCUMENT_OVERRIDE=""
GT_OVERRIDE=""
PRED_OVERRIDE=""
EVAL_OVERRIDE=""
PRINT_SOURCES=1
PRINT_CONTEXT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case_id)
      CASE_ID="${2:-}"; shift 2 ;;
    --document)
      DOCUMENT_OVERRIDE="${2:-}"; shift 2 ;;
    --ground_truth)
      GT_OVERRIDE="${2:-}"; shift 2 ;;
    --predictions)
      PRED_OVERRIDE="${2:-}"; shift 2 ;;
    --eval_output_dir)
      EVAL_OVERRIDE="${2:-}"; shift 2 ;;
    --project_dir)
      PROJECT_DIR="${2:-}"; shift 2 ;;
    --python)
      PYTHON_BIN="${2:-}"; shift 2 ;;
    --no_print_sources)
      PRINT_SOURCES=0; shift ;;
    --no_print_context)
      PRINT_CONTEXT=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$CASE_ID" ]]; then
  echo "Error: --case_id is required." >&2
  usage
  exit 2
fi

PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

CASE_DOC_DIR="${CASE_DOC_DIR:-./case_documents}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-./predictions}"
EVAL_OUTPUT_BASE="${EVAL_OUTPUT_BASE:-./eval_results}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-./ground_truth}"
TAXONOMY_JSON="${TAXONOMY_JSON:-./tax/taxonomy.json}"
EMBED_OUT_NPZ="${EMBED_OUT_NPZ:-embeddings_rules.npz}"
EMBED_OUT_INDEX="${EMBED_OUT_INDEX:-index_rules.jsonl}"
OUT_JSONL="${OUT_JSONL:-prepared_rules.jsonl}"
EMBED_OUT_NPZ2="${EMBED_OUT_NPZ2:-embeddings_materials.npz}"
EMBED_OUT_INDEX2="${EMBED_OUT_INDEX2:-index_materials.jsonl}"
OUT_JSONL2="${OUT_JSONL2:-prepared_materials.jsonl}"

RAG_SCRIPT="${RAG_SCRIPT:-rag_answer_multi_query_diverse_rewritten.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-evaluate_predictions.py}"

DOCUMENT_PATH="${DOCUMENT_OVERRIDE:-${CASE_DOC_DIR%/}/${CASE_ID}.docx}"
GROUND_TRUTH_PATH="${GT_OVERRIDE:-${GROUND_TRUTH_DIR%/}/ground_truth_${CASE_ID}.jsonl}"
PREDICTIONS_PATH="${PRED_OVERRIDE:-${PREDICTIONS_DIR%/}/predictions_${CASE_ID}.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OVERRIDE:-${EVAL_OUTPUT_BASE%/}/${CASE_ID}}"

mkdir -p "$(dirname "$PREDICTIONS_PATH")" "$EVAL_OUTPUT_DIR"

RAG_CMD=(
  "$PYTHON_BIN" "$RAG_SCRIPT"
  --document "$DOCUMENT_PATH"
  --case_id "$CASE_ID"
  --taxonomy_json "$TAXONOMY_JSON"
  --embeddings "$EMBED_OUT_NPZ"
  --index "$EMBED_OUT_INDEX"
  --prepared "$OUT_JSONL"
  --embeddings2 "$EMBED_OUT_NPZ2"
  --index2 "$EMBED_OUT_INDEX2"
  --prepared2 "$OUT_JSONL2"
  --save_predictions_jsonl "$PREDICTIONS_PATH"
)
if [[ "$PRINT_SOURCES" -eq 1 ]]; then
  RAG_CMD+=(--print_sources)
fi
if [[ "$PRINT_CONTEXT" -eq 1 ]]; then
  RAG_CMD+=(--print_context)
fi

EVAL_CMD=(
  "$PYTHON_BIN" "$EVAL_SCRIPT"
  --ground_truth_jsonl "$GROUND_TRUTH_PATH"
  --predictions_jsonl "$PREDICTIONS_PATH"
  --taxonomy_json "$TAXONOMY_JSON"
  --output_dir "$EVAL_OUTPUT_DIR"
)

echo "Project dir      : $PROJECT_DIR"
echo "Script dir       : $SCRIPT_DIR"
echo "Python           : $PYTHON_BIN"
echo "Case ID          : $CASE_ID"
echo "Document         : $DOCUMENT_PATH"
echo "Ground truth     : $GROUND_TRUTH_PATH"
echo "Predictions      : $PREDICTIONS_PATH"
echo "Eval output dir  : $EVAL_OUTPUT_DIR"
echo "Taxonomy         : $TAXONOMY_JSON"
echo "Rules embeddings : $EMBED_OUT_NPZ"
echo "Rules index      : $EMBED_OUT_INDEX"
echo "Rules prepared   : $OUT_JSONL"
echo "Materials emb    : $EMBED_OUT_NPZ2"
echo "Materials index  : $EMBED_OUT_INDEX2"
echo "Materials prep   : $OUT_JSONL2"
echo
printf 'Running: '; printf '%q ' "${RAG_CMD[@]}"; echo
"${RAG_CMD[@]}"
echo
printf 'Running: '; printf '%q ' "${EVAL_CMD[@]}"; echo
"${EVAL_CMD[@]}"
echo
echo "Done."
