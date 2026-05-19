# stt_benchmark/models/hf/whisper.py

"""Hugging Face implementation of OpenAI Whisper STT models.

Supports:
- ASR: transcribe audio in ~99 languages
- AST: translate audio from ~99 languages to English only
"""

import torch
from typing import List, Dict, Set, Any
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from stt_benchmark.models.base import BaseSTTModel
from stt_benchmark.datasets.base import AudioSample, ParallelAudioSample
from stt_benchmark.config.language_support.whisper import (
    fleurs_to_whisper,
    get_whisper_asr_languages,
    get_whisper_ast_pairs,
    WHISPER_AST_TARGET,
)
from stt_benchmark.utils.audio import resolve_audio


class WhisperModel(BaseSTTModel):
    """Hugging Face implementation of OpenAI Whisper models."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.config = model_config
        self.target_sr = 16_000

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, model_config.get("torch_dtype", "float16"))
        if self.device == "cpu":
            self.torch_dtype = torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        gen_config = model_config.get("generation_config", {})
        self.generate_kwargs = {
            "max_new_tokens": gen_config.get("max_new_tokens", 384),
            "num_beams": gen_config.get("num_beams", 1),
            "return_timestamps": gen_config.get("return_timestamps", False),
            "temperature": gen_config.get("temperature", 0.0),
        }

        self._asr_pipe = None
        self._ast_pipe = None

        self._asr_languages = get_whisper_asr_languages()
        self._ast_pairs = get_whisper_ast_pairs()

    def _get_asr_pipeline(self):
        if self._asr_pipe is None:
            self._asr_pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
        return self._asr_pipe

    def _samples_to_pipe_inputs(self, samples) -> List[dict]:
        """Convert samples to the {raw, sampling_rate} dicts the HF pipeline expects."""
        results = []
        for sample in samples:
            try:
                audio, sr = resolve_audio(sample, self.target_sr)
                results.append({"raw": audio, "sampling_rate": sr})
            except Exception as e:
                print(f"    Audio resolve error for {sample.sample_id}: {e}")
                results.append(None)
        return results

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def transcribe(self,
                   samples: List[AudioSample],
                   language: str) -> List[str]:
        whisper_lang = fleurs_to_whisper(language)
        if whisper_lang is None:
            print(f"Whisper does not support language {language}")
            return [""] * len(samples)

        pipe = self._get_asr_pipeline()

        gen_kwargs = {
            **self.generate_kwargs,
            "language": whisper_lang,
            "task": "transcribe",
        }

        audio_batch = self._samples_to_pipe_inputs(samples)

        transcriptions = []
        for audio_input in audio_batch:
            if audio_input is None:
                transcriptions.append("")
                continue
            try:
                result = pipe(audio_input, generate_kwargs=gen_kwargs)
                transcriptions.append(result["text"].strip())
            except Exception as e:
                print(f"    Transcription error: {e}")
                transcriptions.append("")

        return transcriptions

    # ------------------------------------------------------------------
    # AST
    # ------------------------------------------------------------------

    def translate(self,
                  samples: List[ParallelAudioSample],
                  source_lang: str,
                  target_lang: str) -> List[str]:
        if target_lang != "en_us":
            print(f"Whisper AST only translates to English, not {target_lang}")
            return [""] * len(samples)

        whisper_lang = fleurs_to_whisper(source_lang)
        if whisper_lang is None:
            print(f"Whisper does not support source language {source_lang}")
            return [""] * len(samples)

        pipe = self._get_asr_pipeline()

        gen_kwargs = {
            **self.generate_kwargs,
            "language": whisper_lang,
            "task": "translate",
        }

        audio_batch = self._samples_to_pipe_inputs(samples)

        translations = []
        for audio_input in audio_batch:
            if audio_input is None:
                translations.append("")
                continue
            try:
                result = pipe(audio_input, generate_kwargs=gen_kwargs)
                translations.append(result["text"].strip())
            except Exception as e:
                print(f"    Translation error: {e}")
                translations.append("")

        return translations

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_type": "whisper",
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "tasks": "asr,ast",
        }

    def get_supported_languages(self) -> Set[str]:
        return self._asr_languages

    def get_supported_pairs(self) -> Set[tuple]:
        return self._ast_pairs