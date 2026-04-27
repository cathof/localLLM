#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_case_pipeline.sh --case_id CASE_ID [options]

Runs document checking and evaluation using defaults from .env.
Works reliably when stored inside a pipeline/ subdirectory.
Ground truth is per-case: ground_truth_<case_id>.jsonl in GROUND_TRUTH_DIR.

Options:
  --case_id ID           Required, e.g. case_06
  --document PATH        Optional override for case document path
  --ground_truth PATH    Optional override for shared ground-truth jsonl
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
PRINT_SOURCES=0
PRINT_CONTEXT=0

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

LOGS_DIR="${LOGS_DIR:-./logs}"
mkdir -p "$LOGS_DIR"
LOG_FILE="${LOGS_DIR}/case_pipeline_${CASE_ID}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[INFO] Log: $LOG_FILE"

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

# Defaults from .env, with safe fallbacks matching the project structure.
CASE_DOC_DIR="${CASE_DOC_DIR:-./case_documents}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-./predictions}"
EVAL_OUTPUT_BASE="${EVAL_OUTPUT_BASE:-./eval_results}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-./ground_truth}"
GROUND_TRUTH_JSONL="${GROUND_TRUTH_JSONL:-${GROUND_TRUTH_DIR%/}/ground_truth_${CASE_ID}.jsonl}"
TAXONOMY_JSON="${TAXONOMY_JSON:-./tax/taxonomy.json}"
EMBED_OUT_NPZ="${EMBED_OUT_NPZ:-embeddings_rules.npz}"
EMBED_OUT_INDEX="${EMBED_OUT_INDEX:-index_rules.jsonl}"
OUT_JSONL="${OUT_JSONL:-prepared_rules.jsonl}"
EMBED_OUT_NPZ2="${EMBED_OUT_NPZ2:-embeddings_materials.npz}"
EMBED_OUT_INDEX2="${EMBED_OUT_INDEX2:-index_materials.jsonl}"
OUT_JSONL2="${OUT_JSONL2:-prepared_materials.jsonl}"

DOCUMENT_PATH="${DOCUMENT_OVERRIDE:-${CASE_DOC_DIR%/}/${CASE_ID}.docx}"
GROUND_TRUTH_PATH="${GT_OVERRIDE:-$GROUND_TRUTH_JSONL}"
PREDICTIONS_PATH="${PRED_OVERRIDE:-${PREDICTIONS_DIR%/}/predictions_${CASE_ID}.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OVERRIDE:-${EVAL_OUTPUT_BASE%/}/${CASE_ID}}"

mkdir -p "$(dirname "$PREDICTIONS_PATH")" "$EVAL_OUTPUT_DIR"

for required in "$DOCUMENT_PATH" "$GROUND_TRUTH_PATH" "$TAXONOMY_JSON" "$EMBED_OUT_NPZ" "$EMBED_OUT_INDEX" "$OUT_JSONL" "$EMBED_OUT_NPZ2" "$EMBED_OUT_INDEX2" "$OUT_JSONL2"; do
  if [[ ! -f "$required" ]]; then
    echo "Error: required file not found: $required" >&2
    exit 1
  fi
done

RAG_CMD=(
  "$PYTHON_BIN" rag_answer_multi_query_diverse_rewritten.py
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
  "$PYTHON_BIN" evaluate_predictions.py
  --ground_truth_jsonl "$GROUND_TRUTH_PATH"
  --predictions_jsonl "$PREDICTIONS_PATH"
  --case_id "$CASE_ID"
  --taxonomy_json "$TAXONOMY_JSON"
  --output_dir "$EVAL_OUTPUT_DIR"
)

echo "========================================================================"
echo "CASE PIPELINE"
echo "========================================================================"
echo "project_dir      : $PROJECT_DIR"
echo "case_id          : $CASE_ID"
echo "document         : $DOCUMENT_PATH"
echo "ground_truth     : $GROUND_TRUTH_PATH"
echo "predictions_jsonl: $PREDICTIONS_PATH"
echo "eval_output_dir  : $EVAL_OUTPUT_DIR"
echo "taxonomy         : $TAXONOMY_JSON"
echo "========================================================================"

echo
printf '>>> Running detection: '; printf '%q ' "${RAG_CMD[@]}"; echo
"${RAG_CMD[@]}"

echo
printf '>>> Running evaluation: '; printf '%q ' "${EVAL_CMD[@]}"; echo
"${EVAL_CMD[@]}"

echo
echo "Done."
echo "Predictions: $PREDICTIONS_PATH"
echo "Evaluation : $EVAL_OUTPUT_DIR"