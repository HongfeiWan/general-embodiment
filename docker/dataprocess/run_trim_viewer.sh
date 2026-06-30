#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-dataprocess:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-dataprocess-trim-viewer}"
PORT="${PORT:-8501}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

exec docker run --rm -it \
  --name "$CONTAINER_NAME" \
  -p "${PORT}:8501" \
  -v "${REPO_DIR}:/workspace/general-embodiment" \
  -w /workspace/general-embodiment \
  -e STREAMLIT_PORT=8501 \
  -e TRIM_DATASET_DIR="${TRIM_DATASET_DIR:-missions/nero/mission2/lerobot_v2}" \
  -e TRIM_OUTPUT_ROOT="${TRIM_OUTPUT_ROOT:-missions}" \
  -e TRIM_OUTPUT_DATASET_NAME="${TRIM_OUTPUT_DATASET_NAME:-trimmed}" \
  "$IMAGE" \
  trim "$@"
