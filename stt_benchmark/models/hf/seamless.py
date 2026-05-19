# stt_benchmark/models/hf/seamless.py

"""Meta SeamlessM4T v2 model for ASR and AST.

Supports:
- ASR: transcribe audio in ~100 languages
- AST: translate audio from ~100 source languages to ~100 target languages
"""

import torch
from typing import List, Dict, Set, Any, Tuple
from transformers import SeamlessM4Tv2Model as HFSeamlessM4Tv2Model, AutoProcessor

from stt_benchmark.models.base import BaseSTTModel
from stt_benchmark.datasets.base import AudioSample, ParallelAudioSample
from stt_benchmark.config.language_support.seamless import (
    fleurs_to_seamless,
    get_seamless_asr_languages,
    get_seamless_ast_pairs,
    seamless_supports_asr,
    seamless_supports_ast,
)
from stt_benchmark.utils.audio import resolve_audio


class SeamlessModel(BaseSTTModel):
    """SeamlessM4T v2 model for ASR and AST."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """Initialize SeamlessM4T model."""
        self.model_name = model_name
        self.config = model_config
        self.target_sr = 16_000  # SeamlessM4T expects 16 kHz

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, model_config.get("torch_dtype", "float16"))
        if self.device == "cpu":
            self.torch_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = HFSeamlessM4Tv2Model.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        gen_config = model_config.get("generation_config", {})
        self.gen_kwargs = {
            "num_beams": gen_config.get("num_beams", 1),
            "max_new_tokens": gen_config.get("max_new_tokens", 384),
        }

        self._asr_languages = get_seamless_asr_languages()
        self._ast_pairs = get_seamless_ast_pairs(anchor_targets=None)

    def _build_all_ast_pairs(self) -> Set[Tuple[str, str]]:
        """Build the full set of supported AST pairs from the language support module."""
        from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES
        pairs = set()
        for src_fleurs in FLEURS_LANGUAGES:
            for tgt_fleurs in FLEURS_LANGUAGES:
                if src_fleurs != tgt_fleurs and seamless_supports_ast(src_fleurs, tgt_fleurs):
                    pairs.add((src_fleurs, tgt_fleurs))
        return pairs

    def _process_single(self, sample, src_lang: str, tgt_lang: str) -> str:
        """Process a single sample for either ASR or AST."""
        audio, sr = resolve_audio(sample, self.target_sr)

        inputs = self.processor(
            audios=audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                tgt_lang=tgt_lang,
                generate_speech=False,
                **self.gen_kwargs,
            )

        if isinstance(output_tokens, tuple):
            token_ids = output_tokens[0]
        else:
            token_ids = output_tokens

        text = self.processor.decode(
            token_ids[0].tolist()[0],
            skip_special_tokens=True,
        )
        return text.strip()

    # ------------------------------------------------------------------
    # ASR interface
    # ------------------------------------------------------------------

    def transcribe(self,
                   samples: List[AudioSample],
                   language: str) -> List[str]:
        iso3 = fleurs_to_seamless(language)
        if iso3 is None or not seamless_supports_asr(language):
            print(f"SeamlessM4T does not support ASR for {language}")
            return [""] * len(samples)

        transcriptions = []
        for sample in samples:
            try:
                text = self._process_single(sample, src_lang=iso3, tgt_lang=iso3)
                transcriptions.append(text)
            except Exception as e:
                print(f"    SeamlessM4T ASR error for {sample.sample_id}: {e}")
                transcriptions.append("")

        return transcriptions

    # ------------------------------------------------------------------
    # AST interface
    # ------------------------------------------------------------------

    def translate(self,
                  samples: List[ParallelAudioSample],
                  source_lang: str,
                  target_lang: str) -> List[str]:
        src_iso3 = fleurs_to_seamless(source_lang)
        tgt_iso3 = fleurs_to_seamless(target_lang)

        if src_iso3 is None or tgt_iso3 is None:
            print(f"SeamlessM4T cannot map {source_lang}→{target_lang}")
            return [""] * len(samples)

        if not seamless_supports_ast(source_lang, target_lang):
            print(f"SeamlessM4T does not support AST {source_lang}→{target_lang}")
            return [""] * len(samples)

        translations = []
        for sample in samples:
            try:
                text = self._process_single(sample, src_lang=src_iso3, tgt_lang=tgt_iso3)
                translations.append(text)
            except Exception as e:
                print(f"    SeamlessM4T AST error for {sample.sample_id}: {e}")
                translations.append("")

        return translations

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_type": "seamless",
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "tasks": "asr,ast",
        }

    def get_supported_languages(self) -> Set[str]:
        return self._asr_languages

    def get_supported_pairs(self) -> Set[tuple]:
        return self._ast_pairs