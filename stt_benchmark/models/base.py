# stt_benchmark/models/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from stt_benchmark.datasets.base import AudioSample, ParallelAudioSample


class BaseASRModel(ABC):
    """Abstract base class for Automatic Speech Recognition models."""

    @abstractmethod
    def transcribe(self,
                   samples: List["AudioSample"],
                   language: str) -> List[str]:
        """Transcribe audio samples to text in the same language.

        Args:
            samples: List of AudioSample objects. Each sample carries either
                     an `audio_path` or an `audio_array` (+ `sampling_rate`).
                     Models read audio via `utils.audio.resolve_audio(sample)`.
            language: FLEURS language code (e.g., 'sw_ke')

        Returns:
            List of transcribed strings
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get model metadata."""
        pass

    @abstractmethod
    def get_supported_languages(self) -> Set[str]:
        """Get FLEURS language codes supported for ASR.

        Returns:
            Set of FLEURS language codes (e.g., {'en_us', 'sw_ke', ...})
        """
        pass

    def supports_language(self, language: str) -> bool:
        """Check if ASR is supported for a language."""
        return language in self.get_supported_languages()

    @property
    def supports_batch(self) -> bool:
        """Whether the model supports batch processing."""
        return True


class BaseASTModel(ABC):
    """Abstract base class for Automatic Speech Translation models."""

    @abstractmethod
    def translate(self,
                  samples: List["ParallelAudioSample"],
                  source_lang: str,
                  target_lang: str) -> List[str]:
        """Translate speech from source language to text in target language.

        Args:
            samples: List of ParallelAudioSample objects. Each sample carries
                     the source audio as either `source_audio_path` or
                     `source_audio_array` (+ `source_sampling_rate`).
                     Models read audio via `utils.audio.resolve_audio(sample)`.
            source_lang: Source FLEURS language code
            target_lang: Target FLEURS language code

        Returns:
            List of translated strings in target language
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get model metadata."""
        pass

    @abstractmethod
    def get_supported_pairs(self) -> Set[tuple]:
        """Get supported (source, target) FLEURS language pairs for AST."""
        pass

    def supports_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if AST is supported for this language pair."""
        return (source_lang, target_lang) in self.get_supported_pairs()


class BaseSTTModel(BaseASRModel, BaseASTModel):
    """Combined interface for models that support both ASR and AST."""
    pass