import os
import tempfile
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torchaudio

from modelscope import snapshot_download
from transformers import WavLMModel, AutoFeatureExtractor

import subprocess
import soundfile as sf


def _download_file(url: str, dst_path: str, timeout: int = 30) -> None:
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _load_audio_any_format(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Use ffmpeg to decode any audio into 16k mono PCM wav, then read with soundfile.
    Returns mono waveform at target_sr, shape: [num_samples]
    """
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "decoded.wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", path,
            "-ac", "1",             # mono
            "-ar", str(target_sr),  # resample
            "-f", "wav",
            wav_path
        ]
        # run ffmpeg quietly, but keep stderr for debugging if fails
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed:\n{proc.stderr}")

        audio, sr = sf.read(wav_path, dtype="float32")  # audio: [samples]
        if sr != target_sr:
            # should not happen because ffmpeg forces it
            raise RuntimeError(
                f"Unexpected sample rate {sr}, expected {target_sr}")
        return torch.from_numpy(audio)


@torch.inference_mode()
def audio_url_to_wavlm_embedding(
    audio_url: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    target_sr: int = 16000,
    pooling: str = "mean",  # "mean" or "cls"
) -> np.ndarray:
    """
    Download audio from a public URL and convert it to a WavLM-Large utterance embedding.
    Returns: np.ndarray shape [hidden_dim]
    """
    # 1) Download model (or reuse cached)
    model_dir = snapshot_download("microsoft/wavlm-large")

    # 2) Load feature extractor & model from local dir
    fe = AutoFeatureExtractor.from_pretrained(model_dir)

    model = WavLMModel.from_pretrained(model_dir).to(device)
    model.eval()

    # 3) Download audio to a temp file
    suffix = os.path.splitext(urlparse(audio_url).path)[1] or ".audio"
    with tempfile.TemporaryDirectory() as td:
        audio_path = os.path.join(td, f"input{suffix}")
        _download_file(audio_url, audio_path)

        # 4) Load & resample audio
        wav = _load_audio_any_format(
            audio_path, target_sr=target_sr)  # [samples]

        # 5) Feature extraction (normalization inside)
        inputs = fe(
            wav.numpy(),
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device)  # [1, T]

        # 6) Forward to get frame-level embeddings
        outputs = model(input_values=input_values)
        hidden = outputs.last_hidden_state  # [1, frames, hidden_dim]

        # 7) Pooling to utterance embedding
        if pooling == "cls":
            emb = hidden[:, 0, :]  # [1, hidden_dim]
        elif pooling == "mean":
            emb = hidden.mean(dim=1)  # [1, hidden_dim]
        else:
            raise ValueError("pooling must be 'mean' or 'cls'")

        emb = torch.nn.functional.normalize(
            emb, p=2, dim=-1)  # optional but common
        return emb.squeeze(0).detach().cpu().numpy()


if __name__ == "__main__":
    url = '''https://fudiwang.site/downloads/xixiaomiao/%E4%B8%80%E5%B0%81%E4%BF%A1_%E6%94%80%E6%9E%9D%E8%8A%B1.mp3'''
    emb = audio_url_to_wavlm_embedding(url, pooling="mean")
    print(emb.shape, emb[:10])
