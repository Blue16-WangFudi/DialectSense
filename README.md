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

## Dependencies

- Python 3.10+
- `ffmpeg` is required for `.ogg` decoding + silence trimming. `make smoke` bootstraps a local static `ffmpeg` into `.cache/ffmpeg/` if you don't have a system `ffmpeg`.
- Python deps: `make deps` (handled automatically by `make smoke`)

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
```

See `DESIGN.md` for the pipeline rationale and configuration keys.
