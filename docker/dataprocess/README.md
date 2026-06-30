# dataprocess Docker Environment

This Docker package freezes the local data-processing conda environment into an image
that can launch the mission2 Streamlit trim viewer directly.

## Build

Run from the repository root:

```bash
docker build -t dataprocess:latest docker/dataprocess
```

The build context expects:

```text
docker/dataprocess/dataprocess-conda-pack.tar.gz
```

This archive is generated from the local conda env and is intentionally ignored
by Git.

## Start The Trim Viewer

From the repository root:

```bash
bash docker/dataprocess/run_trim_viewer.sh
```

Then open:

```text
http://<host-ip>:8501
```

The container mounts the repository into:

```text
/workspace/general-embodiment
```

By default it runs:

```bash
streamlit run tools/data_chain/trim_lerobot_episode_viewer.py -- \
  --dataset-dir missions/nero/mission2/lerobot_v2 \
  --output-root missions \
  --output-dataset-name trimmed
```

Environment overrides:

```bash
PORT=8502 bash docker/dataprocess/run_trim_viewer.sh
TRIM_DATASET_DIR=missions/nero/mission2/lerobot_v2 bash docker/dataprocess/run_trim_viewer.sh
```

## Shell

```bash
docker run --rm -it \
  -v "$PWD:/workspace/general-embodiment" \
  -w /workspace/general-embodiment \
  dataprocess:latest \
  bash
```
