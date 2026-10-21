# stt_benchmark/models/hf/strict_omni.py

"""Strict adapter for Meta's Omnilingual ASR (omniASR_LLM_*, omniASR_CTC_*) through its own inference pipeline.

Decoding is the library's documented path, unchanged: ASRInferencePipeline.transcribe (16 kHz mono, per-utterance
normalisation, bf16 by default, CTC greedy / LLM beam search with nbest=1 and no length normalisation).
Differences from the older omni_asr.py adapter, all about trust:
  * the language is FORCED to the audio's language for LLM models (a `<iso3>_<script>` code); CTC models take no
    language and ignore it;
  * a language without a code in the library's `supported_langs` is unsupported (recorded by the runner), not
    guessed from a neighbour; a code that maps to several scripts raises at construction;
  * audio over the library's 40 s cap raises (no silent truncation), as does any decode error or a hypothesis
    count different from the sample count;
  * audio goes through utils.audio_strict.prepare_audio and is handed to the pipeline as decoded waveforms.
"""

from typing import Any, Dict, List, Set

import numpy as np
import torch

from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.models.hf.strict_hf import LEADERBOARD_CODES
from stt_benchmark.utils.audio_strict import prepare_audio

SR = 16_000
MAX_SECONDS = 40.0  # omnilingual_asr.models.inference.pipeline.MAX_ALLOWED_AUDIO_SEC


class StrictOmniAsrModel(BaseASRModel):
    model_type = "strict_omni"

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

        self.model_name = model_name
        self.config = model_config
        self.card = model_config["model_card"]
        self.is_llm = "_LLM_" in self.card
        self.batch_size = int(model_config.get("batch_size", 1))
        self.dtype = getattr(torch, model_config.get("torch_dtype", "bfloat16"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            raise RuntimeError(f"{model_name}: a GPU is required (the library's bf16 path is not run on CPU here)")
        self.lang_log: List[Dict[str, Any]] = []

        by_iso: Dict[str, List[str]] = {}
        for code in supported_langs:
            by_iso.setdefault(code.split("_")[0], []).append(code)
        self.omni_code: Dict[str, str] = {}
        for c in LEADERBOARD_CODES:
            iso = FLEURS_LANGUAGES[c]["iso639_3"]
            cands = by_iso.get(iso, [])
            if len(cands) > 1:
                raise ValueError(f"{model_name}: {iso} maps to several scripts {cands}; set the code explicitly")
            if cands:
                self.omni_code[c] = cands[0]
        self.languages: Set[str] = set(self.omni_code)

        self.pipeline = ASRInferencePipeline(self.card, device=self.device, dtype=self.dtype)

    def get_supported_languages(self) -> Set[str]:
        return set(self.languages)

    def get_model_info(self) -> Dict[str, str]:
        return {"model_name": self.model_name, "model_type": self.model_type, "model_card": self.card,
                "device": self.device, "dtype": str(self.dtype), "batch_size": str(self.batch_size),
                "decoding": "LLM beam search nbest=1" if self.is_llm else "CTC greedy", "tasks": "asr"}

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        if language not in self.languages:
            raise ValueError(f"{self.model_name}: language '{language}' has no Omnilingual code")
        inp = []
        for s in samples:
            a = prepare_audio(s, SR)
            if a.shape[0] / SR > MAX_SECONDS:
                raise ValueError(f"{s.sample_id}: {a.shape[0]/SR:.1f}s > {MAX_SECONDS}s (no silent truncation)")
            inp.append({"waveform": np.ascontiguousarray(a), "sample_rate": SR})
        code = self.omni_code[language]
        lang = [code] * len(inp) if self.is_llm else None
        out = self.pipeline.transcribe(inp, lang=lang, batch_size=self.batch_size)
        if len(out) != len(samples):
            raise RuntimeError(f"{len(out)} hypotheses for {len(samples)} samples")
        for s in samples:
            self.lang_log.append({"sample_id": s.sample_id, "language": language,
                                  "lang_code": code if self.is_llm else None})
        return [t.strip() for t in out]
