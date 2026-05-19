# stt_benchmark/datasets/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Set
from dataclasses import dataclass
import numpy as np


@dataclass
class AudioSample:
    """A single audio sample with transcription and metadata.

    Exactly one of (audio_path, audio_array) must be set:
      - audio_path: path on disk; sampling_rate may be None (loader resamples).
      - audio_array: pre-decoded waveform; sampling_rate must be set.
    """
    transcription: str
    language: str                          # FLEURS code (e.g., 'sw_ke')
    sample_id: str
    audio_path: Optional[str] = None
    audio_array: Optional[np.ndarray] = None
    sampling_rate: Optional[int] = None


@dataclass
class ParallelAudioSample:
    """A parallel sample for AST evaluation: source audio + target text.

    Source audio follows the same path-or-array contract as AudioSample.
    """
    source_transcription: str
    source_language: str                   # FLEURS code
    target_transcription: str
    target_language: str                   # FLEURS code
    sample_id: str
    source_audio_path: Optional[str] = None
    source_audio_array: Optional[np.ndarray] = None
    source_sampling_rate: Optional[int] = None


class BaseASRDataset(ABC):
    """Abstract base class for ASR datasets (monolingual audio + transcription)."""

    @abstractmethod
    def get_language_samples(self, language: str) -> List[AudioSample]:
        pass

    @abstractmethod
    def list_languages(self) -> Set[str]:
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        pass

    def has_language(self, language: str) -> bool:
        return language in self.list_languages()

    def get_language_batch(self, language: str,
                           batch_size: int = 1) -> Iterator[List[AudioSample]]:
        samples = self.get_language_samples(language)
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]


class BaseASTDataset(ABC):
    """Abstract base class for AST datasets (parallel audio + cross-lingual text)."""

    @abstractmethod
    def get_parallel_samples(self, source_lang: str,
                             target_lang: str) -> List[ParallelAudioSample]:
        pass

    @abstractmethod
    def list_language_pairs(self) -> Set[tuple]:
        pass

    def has_language_pair(self, source_lang: str, target_lang: str) -> bool:
        return (source_lang, target_lang) in self.list_language_pairs()

    def get_parallel_batch(self, source_lang: str, target_lang: str,
                           batch_size: int = 1) -> Iterator[List[ParallelAudioSample]]:
        samples = self.get_parallel_samples(source_lang, target_lang)
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]