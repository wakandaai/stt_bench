# stt_benchmark/models/hf/omni_asr.py

"""Meta Omnilingual ASR model.

Wraps `omnilingual_asr.models.inference.pipeline.ASRInferencePipeline`.
ASR-only — omni does not do speech translation.

The pipeline accepts either:
  - List[str | Path]: audio file paths (loaded internally)
  - List[Dict]: pre-decoded {"waveform": tensor, "sample_rate": int} dicts

Samples with `audio_path` go through as paths; samples with in-memory arrays
(e.g. FLEURS HF backend) go through as dicts. The pipeline handles resampling
to 16kHz, mono conversion, normalization, and real batching internally.
"""

import torch
from typing import List, Dict, Set, Any

from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.config.language_support.omni_asr import (
    fleurs_to_omni,
    omni_supports_asr,
    get_omni_asr_languages,
)


class OmniAsrModel(BaseASRModel):
    """Meta Omnilingual ASR model."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """Initialize omni ASR model.

        Args:
            model_name: Omni model card name (e.g. 'omniASR_LLM_1B').
                Passed to ASRInferencePipeline as `model_card`.
            model_config: Configuration dict from models.yaml.
        """
        # Lazy import so this file is harmless when the package isn't installed.
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        from omnilingual_asr.models.inference import pipeline as _omni_pipeline

        # Disable the hardcoded 40s clip cap. assert_max_length is mapped into
        # the data pipeline at build time, so we replace it with a passthrough
        # before constructing the pipeline.
        _omni_pipeline.assert_max_length = lambda audio_data, target_sample_rate=16000: audio_data

        self.model_name = model_name
        self.config = model_config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_str = model_config.get("torch_dtype", "bfloat16")
        self.torch_dtype = getattr(torch, dtype_str)

        print(f"  Loading omni ASR model: {model_name}")
        self.pipeline = ASRInferencePipeline(
            model_card=model_name,
            device=self.device,
            dtype=self.torch_dtype,
        )

        self._asr_languages = get_omni_asr_languages()

    def _sample_to_pipeline_input(self, sample: AudioSample):
        """Convert one AudioSample to the form the omni pipeline expects.

        Returns a str (path) or a dict ({"waveform", "sample_rate"}).
        """
        if sample.audio_path is not None:
            return sample.audio_path

        if sample.audio_array is not None:
            if sample.sampling_rate is None:
                raise ValueError(
                    f"Sample {sample.sample_id} has audio_array but no sampling_rate"
                )
            # Pass numpy array (not a torch tensor): omni's pipeline does
            # `torch.tensor(x["waveform"])` internally, which warns when given
            # a tensor. Numpy input avoids the warning.
            import numpy as np
            return {
                "waveform": np.asarray(sample.audio_array, dtype=np.float32),
                "sample_rate": int(sample.sampling_rate),
            }

        raise ValueError(
            f"Sample {sample.sample_id} has neither audio_path nor audio_array"
        )

    def transcribe(self,
                   samples: List[AudioSample],
                   language: str) -> List[str]:
        """Transcribe audio samples via the omni pipeline.

        Args:
            samples: List of AudioSample objects.
            language: FLEURS language code (e.g. 'sw_ke').

        Returns:
            List of transcription strings (empty string per sample on failure).
        """
        omni_lang = fleurs_to_omni(language)
        if omni_lang is None or not omni_supports_asr(language):
            print(f"Omni ASR does not support language {language}")
            return [""] * len(samples)

        # Build inputs; track which samples failed audio resolution so we can
        # slot empty strings back in at the right indices.
        inputs = []
        valid_indices = []
        for i, sample in enumerate(samples):
            try:
                inputs.append(self._sample_to_pipeline_input(sample))
                valid_indices.append(i)
            except Exception as e:
                print(f"    Omni ASR input prep error for {sample.sample_id}: {e}")

        if not inputs:
            return [""] * len(samples)

        langs = [omni_lang] * len(inputs)

        try:
            transcriptions = self.pipeline.transcribe(
                inputs, lang=langs, batch_size=len(inputs),
            )
        except Exception as e:
            print(f"    Omni ASR transcription error: {e}")
            return [""] * len(samples)

        # Slot results back into the original order, filling failures with "".
        results = [""] * len(samples)
        for idx, hyp in zip(valid_indices, transcriptions):
            results[idx] = (hyp or "").strip()
        return results

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_type": "omni_asr",
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "tasks": "asr",
        }

    def get_supported_languages(self) -> Set[str]:
        return self._asr_languages

    @property
    def supports_batch(self) -> bool:
        return True