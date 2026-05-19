# stt_benchmark/models/hf/mms.py

"""Meta MMS (Massively Multilingual Speech) ASR model.

MMS supports ASR only (no speech translation).
- mms-1b-fl102: 102 FLEURS languages
- mms-1b-all: 1162 languages via switchable adapters

Uses Wav2Vec2 architecture with CTC decoding.
"""

import torch
from typing import List, Dict, Set, Any
from transformers import Wav2Vec2ForCTC, AutoProcessor

from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.config.language_support.mms import (
    fleurs_to_mms,
    get_mms_asr_languages,
)
from stt_benchmark.utils.audio import resolve_audio


class MMSModel(BaseASRModel):
    """Meta MMS ASR model using Wav2Vec2 + CTC."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """Initialize MMS model.

        Args:
            model_name: HuggingFace model id (e.g. 'facebook/mms-1b-all')
            model_config: Configuration dict from models.yaml
        """
        self.model_name = model_name
        self.config = model_config
        self.target_sr = 16_000  # MMS expects 16 kHz

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, model_config.get("torch_dtype", "float16"))
        if self.device == "cpu":
            self.torch_dtype = torch.float32

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        # Track current adapter language to avoid redundant loads
        self._current_adapter_lang = None

        # Cache supported languages
        self._asr_languages = get_mms_asr_languages()

    def _set_language(self, fleurs_code: str) -> bool:
        """Load the adapter for a given FLEURS language.

        Returns True if successful, False otherwise.
        """
        mms_code = fleurs_to_mms(fleurs_code)
        if mms_code is None:
            return False

        if mms_code == self._current_adapter_lang:
            return True

        try:
            self.processor.tokenizer.set_target_lang(mms_code)
            self.model.load_adapter(mms_code)
            self._current_adapter_lang = mms_code
            return True
        except Exception as e:
            print(f"    Failed to load MMS adapter for {mms_code}: {e}")
            return False

    def transcribe(self,
                   samples: List[AudioSample],
                   language: str) -> List[str]:
        """Transcribe audio samples using MMS CTC decoding.

        Args:
            samples: List of AudioSample objects
            language: FLEURS language code (e.g. 'sw_ke')

        Returns:
            List of transcription strings
        """
        if not self._set_language(language):
            print(f"MMS does not support language {language}")
            return [""] * len(samples)

        transcriptions = []
        for sample in samples:
            try:
                audio, sr = resolve_audio(sample, self.target_sr)

                inputs = self.processor(
                    audio,
                    sampling_rate=self.target_sr,
                    return_tensors="pt",
                )
                inputs = {
                            k: v.to(device=self.device, dtype=self.torch_dtype) if v.is_floating_point() else v.to(self.device)
                            for k, v in inputs.items()
                        }

                with torch.no_grad():
                    outputs = self.model(**inputs)

                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                text = self.processor.batch_decode(predicted_ids)[0]
                transcriptions.append(text.strip())

            except Exception as e:
                print(f"    MMS transcription error for {sample.sample_id}: {e}")
                transcriptions.append("")

        return transcriptions

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_type": "mms",
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "tasks": "asr",
        }

    def get_supported_languages(self) -> Set[str]:
        """FLEURS codes supported for ASR."""
        return self._asr_languages