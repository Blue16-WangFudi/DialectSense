# DESIGN

This repo is data-centric (`datasets/`) and produces all derived artifacts under `artifacts/<run_name>/`.

Upstream dataset reference: `datasets/README.txt`.

## Pipeline (end-to-end)

1) **Audio QC + preprocessing** (`dialectsense.cli preprocess`)

- Robust decode via `ffmpeg` (do not rely on `torchaudio` backends).
- Convert to mono, resample to 16kHz.
- Trim silence via `ffmpeg` `silenceremove`.
- Optional RMS normalization (default enabled).
- Apply duration rules **after trimming**:
  - Drop if `effective_dur_sec < audio.min_sec` (default 1.0s).
  - Long utterances are **kept**; chunking/truncation happens at embedding time.

Outputs:

- `artifacts/<run_name>/audio_preprocessed/<clip_id>.wav`
- `artifacts/<run_name>/audio_qc.csv`
- `artifacts/<run_name>/figures/effective_duration_hist_kept.png`
- `artifacts/<run_name>/figures/dropped_reasons.png`

2) **WavLM-Large embeddings** (`dialectsense.cli embed`)

- Model: `microsoft/wavlm-large`
- Download to local cache via ModelScope `snapshot_download`.
- HF inference with `AutoFeatureExtractor` + `WavLMModel`.
- `output_hidden_states=True`; average mid layers by default (layers 6–12, inclusive).
- For long utterances (`embed.chunk.max_sec` threshold):
  - default: sliding-window chunking (`chunk_sec`, `hop_sec`), embed each chunk then aggregate
  - optional: truncate to `max_sec`
- L2-normalize the final utterance embedding.

Outputs:

- `artifacts/<run_name>/embeddings/<clip_id>.npy`
- `artifacts/<run_name>/embeddings/meta.json`

3) **Speaker-disjoint split** (`dialectsense.cli split`)

- Create train/val/test splits that are speaker-disjoint using `uploader_id`.
- Enforce per-label minimums by retrying the group split; if impossible, iteratively drop rare labels.

Outputs:

- `artifacts/<run_name>/splits.csv`

4) **Coarse label mapping (train-only)** (`dialectsense.cli coarsen`)

Goal: group original labels into K coarse clusters and evaluate at cluster level.

Rules:

- **Train-only mapping**: the mapping is computed strictly from the training split.
- Cluster **label centroids**, not raw samples:
  - compute a centroid embedding per original label (from training samples only)
  - L2-normalize centroids
  - run `KMeans(n_clusters=K, n_init>=20)`

Outputs:

- `artifacts/<run_name>/label_to_cluster.json`
- `artifacts/<run_name>/cluster_to_labels.json`
- `artifacts/<run_name>/cluster_centroids.npy`
- `artifacts/<run_name>/cluster_summary.md`

5) **Coarse-label training + calibration** (`dialectsense.cli train`)

- Default (configs): two-stage stacked classifier for better accuracy:
  - base: `StandardScaler → LinearSVC(class_weight="balanced")` + `StandardScaler → MLPClassifier`
  - meta: `LogisticRegression` trained on validation outputs (stacking)

Outputs:

- `artifacts/<run_name>/models/coarse_model.joblib` (and a compatibility copy `coarse_svm.joblib`)

6) **Evaluation** (`dialectsense.cli eval`)

- Evaluate on the test split using coarse labels.
- Metrics: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, top confusions.

Outputs:

- `artifacts/<run_name>/report_coarse.json`
- `artifacts/<run_name>/figures/confusion_matrix_coarse.png`
- `artifacts/<run_name>/top_confusions.csv`

7) **Reporting / visualization** (`dialectsense.cli report`)

Writes a compact but comprehensive figure set to `artifacts/<run_name>/figures/`:

- Data distribution (labels, clusters, speakers per split)
- Duration histograms (orig/effective)
- Embedding 2D projections (UMAP preferred, else t-SNE)
- Cluster centroid cosine distance heatmap
- Per-cluster F1 bar plot and top confusion pairs

## Config keys (high-level)

- `data.*`: metadata paths + fast smoke subsampling (`top_k_labels`, `max_per_label`, `max_total`)
- `audio.*`: preprocessing/QC parameters (ffmpeg trim + RMS normalization + min duration)
- `embed.*`: WavLM model + hidden-state aggregation + chunking strategy
- `split.*`: speaker-disjoint split ratios + per-label minimums
- `coarse.*`: K + centroid aggregation + KMeans settings
- `model.*`: Linear SVM hyperparams + calibration
- `report.*`: plotting limits + 2D projection settings
