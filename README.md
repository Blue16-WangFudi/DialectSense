# DialectSense

End-to-end, fully automated dialect embedding + coarse-label classification pipeline:

```
Audio QC + preprocessing
 → WavLM-Large embeddings (chunked aggregation)
 → speaker-disjoint split (by uploader_id)
 → label coarsening (train-only label-centroid KMeans)
 → coarse-label training (StandardScaler + Linear SVM + calibration)
 → evaluation + visualizations + report artifacts
```

## Quickstart

```bash
make smoke
```

Artifacts are written to `artifacts/<run_name>/` (for smoke: `artifacts/smoke/`).

## Web UI

```bash
make ui CONFIG=configs/smoke.json
```

UI 内包含 `Realtime` 页：从麦克风流式截取固定长度 chunk，逐 chunk 输出所有候选粗类（cluster）的置信度折线图，便于实时展示。

## Dependencies

- Python 3.10+
- `make` (GNU Make). If you don't have it, either install it (Linux: `sudo apt-get install make`) or run the CLI commands directly (see below / `RUN_WINDOWS.md`).
- `ffmpeg` is required for `.ogg` decoding + silence trimming. The Makefile bootstraps a local static `ffmpeg` into `.cache/ffmpeg/` if you don't have a system `ffmpeg`.
- Python deps: `make deps` (handled automatically by `make smoke` / `make ui`)

## Makefile Usage

The Makefile is the recommended “one-command” runner on Linux/macOS/WSL:

```bash
make smoke
make ui CONFIG=configs/smoke.json
make clean CONFIG=configs/smoke.json
```

You can switch configs via `CONFIG=...`:

```bash
make smoke CONFIG=configs/smoke.json
make preprocess embed split coarsen train eval report CONFIG=configs/full.json
```

若目标是尽可能提高 Accuracy，可使用 `configs/full_accuracy.json`（以验证集 Accuracy 为目标做超参搜索并在 train+val 上重训）：

```bash
make train eval report CONFIG=configs/full_accuracy.json
```

If your environment does not have `make` (common on Windows), follow `RUN_WINDOWS.md` and run the Python CLI commands instead.

## Outputs

After `make smoke`, look at:

- `artifacts/smoke/audio_qc.csv` (per-clip preprocessing/QC decisions)
- `artifacts/smoke/splits.csv` (speaker-disjoint train/val/test)
- `artifacts/smoke/label_to_cluster.json` + `artifacts/smoke/cluster_summary.md` (coarse mapping)
- `artifacts/smoke/report_coarse.json` + `artifacts/smoke/top_confusions.csv` (metrics + confusions)
- `artifacts/smoke/figures/` (PNG plots: UMAP/t-SNE, confusion matrix, QC plots, etc.)

## CLI

Each stage is runnable independently (and reuses cached artifacts when present):

```bash
.venv/bin/python -m dialectsense.cli preprocess --config configs/smoke.json
.venv/bin/python -m dialectsense.cli embed      --config configs/smoke.json
.venv/bin/python -m dialectsense.cli split      --config configs/smoke.json
.venv/bin/python -m dialectsense.cli coarsen    --config configs/smoke.json
.venv/bin/python -m dialectsense.cli train      --config configs/smoke.json
.venv/bin/python -m dialectsense.cli eval       --config configs/smoke.json
.venv/bin/python -m dialectsense.cli report     --config configs/smoke.json
.venv/bin/python -m dialectsense.cli ui         --config configs/smoke.json
```

See `DESIGN.md` for the pipeline rationale and configuration keys.
