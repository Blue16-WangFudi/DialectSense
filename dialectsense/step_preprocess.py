from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import soundfile as sf

from .audio_preprocess import preprocess_audio
from .audio_qc import AudioQCRow, audio_qc_stats, read_audio_qc_csv, write_audio_qc_csv
from .config import require
from .data import counts_by_label, filter_rows_by_votes, read_metadata_rows, subset_rows
from .util import ensure_dir, write_json
from .viz import plot_drop_reasons, plot_duration_hist


log = logging.getLogger("dialectsense")


def _try_soundfile_duration_sec(path: str) -> float | None:
    try:
        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def run_preprocess(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    data_cfg = require(cfg, "data")
    audio_cfg = require(cfg, "audio")

    rows_raw, stats_read = read_metadata_rows(
        metadata_csv=data_cfg["metadata_csv"],
        audio_dir=data_cfg["audio_dir"],
        id_col=data_cfg.get("id_col", "id"),
        label_col=data_cfg.get("label_col", "dialect.province"),
        uploader_col=data_cfg.get("uploader_col", "uploader_id"),
    )

    rows_votes, stats_votes = filter_rows_by_votes(rows_raw, min_votes=data_cfg.get("min_votes", None))

    rows_sub, stats_subset = subset_rows(
        rows_votes,
        seed=int(cfg.get("seed", 42)),
        top_k_labels=data_cfg.get("top_k_labels", None),
        max_per_label=data_cfg.get("max_per_label", None),
        max_total=data_cfg.get("max_total", None),
    )

    out_dir = ensure_dir(artifact_dir / "audio_preprocessed")
    figs_dir = ensure_dir(artifact_dir / "figures")
    qc_csv = artifact_dir / "audio_qc.csv"

    existing: dict[str, AudioQCRow] = {}
    if qc_csv.exists():
        for r in read_audio_qc_csv(qc_csv):
            existing[r.clip_id] = r

    min_sec = float(audio_cfg.get("min_sec", 1.0))
    qc_rows: list[AudioQCRow] = []
    for i, r in enumerate(rows_sub):
        out_wav = out_dir / f"{r.clip_id}.wav"
        if r.clip_id in existing:
            prev = existing[r.clip_id]
            if prev.preprocessed_wav and Path(prev.preprocessed_wav).exists():
                qc_rows.append(prev)
                continue

        orig_dur = r.sound_length_sec
        orig_source = "metadata_sound_length"
        if orig_dur is None:
            orig_dur = _try_soundfile_duration_sec(r.audio_path)
            orig_source = "soundfile_info" if orig_dur is not None else "unknown"

        try:
            res = preprocess_audio(in_path=r.audio_path, out_wav_path=out_wav, audio_cfg=audio_cfg)
            eff_dur = float(res.y.size) / float(res.sr) if res.y.size else 0.0
            if eff_dur < min_sec:
                try:
                    out_wav.unlink(missing_ok=True)
                except Exception:
                    pass
                qc_rows.append(
                    AudioQCRow(
                        clip_id=r.clip_id,
                        label=r.label,
                        uploader_id=r.uploader_id,
                        audio_path=r.audio_path,
                        preprocessed_wav=str(out_wav),
                        orig_dur_sec=orig_dur,
                        orig_dur_source=orig_source,
                        effective_dur_sec=eff_dur,
                        kept=0,
                        drop_reason="too_short_after_trim",
                        sample_rate=int(res.sr),
                        n_samples=int(res.y.size),
                        rms_dbfs=float(res.rms_dbfs),
                        peak=float(res.peak),
                        clipping_flag=int(res.clipping_flag),
                        norm_enabled=int(bool(audio_cfg.get("normalize", {}).get("enabled", True))),
                        norm_target_rms_dbfs=float(audio_cfg.get("normalize", {}).get("target_rms_dbfs", -20.0)),
                        norm_gain_db=float(res.norm_gain_db),
                    )
                )
            else:
                qc_rows.append(
                    AudioQCRow(
                        clip_id=r.clip_id,
                        label=r.label,
                        uploader_id=r.uploader_id,
                        audio_path=r.audio_path,
                        preprocessed_wav=str(out_wav),
                        orig_dur_sec=orig_dur,
                        orig_dur_source=orig_source,
                        effective_dur_sec=eff_dur,
                        kept=1,
                        drop_reason="",
                        sample_rate=int(res.sr),
                        n_samples=int(res.y.size),
                        rms_dbfs=float(res.rms_dbfs),
                        peak=float(res.peak),
                        clipping_flag=int(res.clipping_flag),
                        norm_enabled=int(bool(audio_cfg.get("normalize", {}).get("enabled", True))),
                        norm_target_rms_dbfs=float(audio_cfg.get("normalize", {}).get("target_rms_dbfs", -20.0)),
                        norm_gain_db=float(res.norm_gain_db),
                    )
                )
        except Exception as e:
            try:
                out_wav.unlink(missing_ok=True)
            except Exception:
                pass
            qc_rows.append(
                AudioQCRow(
                    clip_id=r.clip_id,
                    label=r.label,
                    uploader_id=r.uploader_id,
                    audio_path=r.audio_path,
                    preprocessed_wav=str(out_wav),
                    orig_dur_sec=orig_dur,
                    orig_dur_source=orig_source,
                    effective_dur_sec=None,
                    kept=0,
                    drop_reason=f"decode_failed:{type(e).__name__}",
                    sample_rate=int(audio_cfg.get("target_sr", 16000)),
                    n_samples=None,
                    rms_dbfs=None,
                    peak=None,
                    clipping_flag=0,
                    norm_enabled=int(bool(audio_cfg.get("normalize", {}).get("enabled", True))),
                    norm_target_rms_dbfs=float(audio_cfg.get("normalize", {}).get("target_rms_dbfs", -20.0)),
                    norm_gain_db=None,
                )
            )

        if (i + 1) % 25 == 0 or (i + 1) == len(rows_sub):
            log.info("preprocess: %d/%d", i + 1, len(rows_sub))

    write_audio_qc_csv(qc_csv, qc_rows)

    qc_stats = audio_qc_stats(qc_rows)
    write_json(
        artifact_dir / "preprocess_stats.json",
        {
            "read": stats_read,
            "votes_filter": stats_votes,
            "subset": stats_subset,
            "label_counts_selected": counts_by_label(rows_sub),
            "audio_qc": qc_stats,
        },
    )

    kept_durs = [r.effective_dur_sec for r in qc_rows if r.kept == 1 and r.effective_dur_sec is not None]
    plot_duration_hist(
        out_path=figs_dir / "effective_duration_hist_kept.png",
        durations_sec=kept_durs,
        title="Effective duration histogram (kept, after trim)",
    )
    plot_drop_reasons(
        out_path=figs_dir / "dropped_reasons.png",
        reason_counts=qc_stats["drop_reasons"],
        title="Drop reasons",
    )

    log.info("preprocess: wrote %s", qc_csv)
    return qc_csv

