# DialectSense

End-to-end, fully automated dialect embedding + coarse-label classification pipeline:

```
Audio QC + preprocessing
 → WavLM-Large embeddings (chunked aggregation)
 → speaker-disjoint split (by uploader_id)
 → label coarsening (train-only label-centroid KMeans)
 → coarse-label training (stacked: Linear SVM + MLP → meta LogisticRegression)
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
The UI includes a Realtime page: streaming capture of fixed-length chunks from the microphone, progressively outputting confidence line charts for all candidate clusters, facilitating real-time visualization.

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

`configs/full.json` 默认使用 stacked coarse classifier（SVM+MLP→meta LR），用于提高 Accuracy；如只想重训模型与评测：

```bash
make train eval report CONFIG=configs/full.json
```

If your environment does not have `make` (common on Windows), follow `RUN_WINDOWS.md` and run the Python CLI commands instead.

## Outputs

After `make smoke`, look at:

- `artifacts/smoke/audio_qc.csv` (per-clip preprocessing/QC decisions)
- `artifacts/smoke/splits.csv` (speaker-disjoint train/val/test)
- `artifacts/smoke/label_to_cluster.json` + `artifacts/smoke/cluster_summary.md` (coarse mapping)
- `artifacts/smoke/models/coarse_model.joblib` (trained coarse classifier)
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
