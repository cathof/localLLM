#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_load_pipeline.sh [options]

Rebuilds prepared JSONL files, embeddings, and indices for:
  1) rules / norm basis
  2) case materials

The script is robust when stored in a pipeline/ subdirectory and loads defaults from the
project root .env file. CLI options override .env values.

Options:
  --project_dir PATH   Optional project root. Default: parent of script directory.
  --python BIN         Python executable. Default: python
  -h, --help           Show this help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project_dir)
      PROJECT_DIR="${2:-}"; shift 2 ;;
    --python)
      PYTHON_BIN="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

cd "$PROJECT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

DATA_DIR="${DATA_DIR:-./data}"
CASES_DIR="${CASES_DIR:-./cases}"
IMAGE_CACHE_DIR="${IMAGE_CACHE_DIR:-./image_cache}"

OUT_JSONL="${OUT_JSONL:-prepared_rules.jsonl}"
OUT_JSONL2="${OUT_JSONL2:-prepared_materials.jsonl}"

EMBED_OUT_NPZ="${EMBED_OUT_NPZ:-embeddings_rules.npz}"
EMBED_OUT_INDEX="${EMBED_OUT_INDEX:-index_rules.jsonl}"
EMBED_OUT_NPZ2="${EMBED_OUT_NPZ2:-embeddings_materials.npz}"
EMBED_OUT_INDEX2="${EMBED_OUT_INDEX2:-index_materials.jsonl}"

echo "==> Projektverzeichnis        : $PROJECT_DIR"
echo "==> Python                    : $PYTHON_BIN"
echo "==> Data dir (rules)          : $DATA_DIR"
echo "==> Cases dir (materials)     : $CASES_DIR"
echo "==> Image cache dir           : $IMAGE_CACHE_DIR"
echo "==> Prepared rules            : $OUT_JSONL"
echo "==> Prepared materials        : $OUT_JSONL2"
echo "==> Embeddings rules          : $EMBED_OUT_NPZ"
echo "==> Index rules               : $EMBED_OUT_INDEX"
echo "==> Embeddings materials      : $EMBED_OUT_NPZ2"
echo "==> Index materials           : $EMBED_OUT_INDEX2"

echo "==> Lösche alte Artefakte"
rm -rf "$IMAGE_CACHE_DIR"
rm -f "$OUT_JSONL"
rm -f "$OUT_JSONL2"
rm -f "$EMBED_OUT_INDEX"
rm -f "$EMBED_OUT_INDEX2"
rm -f "$EMBED_OUT_NPZ"
rm -f "$EMBED_OUT_NPZ2"

echo "==> Erzeuge prepared rules (nur Normbasis)"
"$PYTHON_BIN" importDocuments_structural.py \
  --data_dir "$DATA_DIR" \
  --cases_dir "" \
  --out "$OUT_JSONL"

echo "==> Erzeuge prepared materials (nur Case-Materials)"
"$PYTHON_BIN" importDocuments_structural.py \
  --data_dir "" \
  --cases_dir "$CASES_DIR" \
  --out "$OUT_JSONL2"

echo "==> Erzeuge Embeddings + Index für rules"
"$PYTHON_BIN" embed_e5.py \
  --in "$OUT_JSONL" \
  --out_npz "$EMBED_OUT_NPZ" \
  --out_index "$EMBED_OUT_INDEX"

echo "==> Erzeuge Embeddings + Index für materials"
"$PYTHON_BIN" embed_e5.py \
  --in "$OUT_JSONL2" \
  --out_npz "$EMBED_OUT_NPZ2" \
  --out_index "$EMBED_OUT_INDEX2"

echo "==> Fertig"
