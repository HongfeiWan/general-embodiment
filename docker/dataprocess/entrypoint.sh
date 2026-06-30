#!/usr/bin/env bash
set -euo pipefail

export CONDA_PREFIX="${CONDA_PREFIX:-/opt/conda/envs/dataprocess}"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_HEADLESS="${STREAMLIT_SERVER_HEADLESS:-true}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}"

cd /workspace/general-embodiment

if [[ "${1:-trim}" == "trim" ]]; then
  shift || true
  exec streamlit run tools/data_chain/trim_lerobot_episode_viewer.py \
    --server.address=0.0.0.0 \
    --server.port="${STREAMLIT_PORT:-8501}" \
    -- \
    --dataset-dir "${TRIM_DATASET_DIR:-missions/nero/mission2/lerobot_v2}" \
    --output-root "${TRIM_OUTPUT_ROOT:-missions}" \
    --output-dataset-name "${TRIM_OUTPUT_DATASET_NAME:-trimmed}" \
    "$@"
fi

exec "$@"
