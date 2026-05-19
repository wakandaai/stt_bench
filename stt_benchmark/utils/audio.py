# stt_benchmark/utils/audio.py

"""Audio loading and preprocessing utilities."""

import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from stt_benchmark.datasets.base import AudioSample, ParallelAudioSample


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sampling rate.

    Args:
        path: Path to audio file (wav, flac, mp3, etc.)
        target_sr: Target sampling rate (default 16000 for most STT models)

    Returns:
        Tuple of (audio_array, sampling_rate)
    """
    audio, sr = sf.read(path, dtype='float32')

    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def resolve_audio(
    sample: Union["AudioSample", "ParallelAudioSample"],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Return (audio_array, sampling_rate) for an AudioSample or ParallelAudioSample.

    Handles both backends transparently:
      - If the sample carries a pre-decoded array (HF backend), resample if needed.
      - If the sample carries a path, load from disk via load_audio.

    For ParallelAudioSample, returns the *source* audio.
    """
    # ParallelAudioSample has source_* fields; AudioSample has plain fields.
    array = getattr(sample, "audio_array", None)
    if array is None:
        array = getattr(sample, "source_audio_array", None)

    path = getattr(sample, "audio_path", None)
    if path is None:
        path = getattr(sample, "source_audio_path", None)

    sr = getattr(sample, "sampling_rate", None)
    if sr is None:
        sr = getattr(sample, "source_sampling_rate", None)

    if array is not None:
        # Ensure float32 mono
        audio = np.asarray(array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr is None:
            raise ValueError(
                "Sample has audio_array but no sampling_rate; cannot resample."
            )
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio, sr

    if path is not None:
        return load_audio(path, target_sr)

    raise ValueError(
        f"Sample {getattr(sample, 'sample_id', '?')} has neither audio_path "
        f"nor audio_array set."
    )


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def get_audio_duration(audio: np.ndarray, sampling_rate: int) -> float:
    """Get audio duration in seconds."""
    return len(audio) / sampling_rate