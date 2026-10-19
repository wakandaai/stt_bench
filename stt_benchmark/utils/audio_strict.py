# stt_benchmark/utils/audio_strict.py

"""Strict audio preparation for the leaderboard adapters.

Unlike utils/audio.py this module has no librosa dependency, never substitutes
anything for a bad sample, and raises on any problem: empty audio, non-finite
values, missing sampling rate. Resampling uses torchaudio (sinc, deterministic).
"""

from typing import Tuple

import numpy as np
import soundfile as sf


def prepare_audio(sample, target_sr: int = 16_000) -> np.ndarray:
    """Return mono float32 audio at `target_sr` for an AudioSample, or raise."""
    array = getattr(sample, "audio_array", None)
    sr = getattr(sample, "sampling_rate", None)
    path = getattr(sample, "audio_path", None)
    if array is not None:
        audio = np.asarray(array, dtype=np.float32)
        if sr is None:
            raise ValueError(f"{sample.sample_id}: audio_array without sampling_rate")
    elif path is not None:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        raise ValueError(f"{getattr(sample, 'sample_id', '?')}: neither audio_array nor audio_path")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError(f"{sample.sample_id}: empty audio")
    if not np.isfinite(audio).all():
        raise ValueError(f"{sample.sample_id}: non-finite audio values")
    if int(sr) != target_sr:
        import torch
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio), int(sr), target_sr).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def duration_s(audio: np.ndarray, sr: int = 16_000) -> float:
    return float(audio.shape[0]) / sr
