# stt_benchmark/models/hf/cascaded.py

"""
Cascaded ASR→MT baseline for speech translation.

Pipeline: MMS (ASR) → NLLB (MT)
  1. MMS transcribes source audio to source-language text
  2. NLLB translates the transcript to the target language
"""

import torch
from typing import List, Dict, Set, Any, Tuple, Optional
from transformers import AutoTokenizer, M2M100ForConditionalGeneration

from stt_benchmark.models.base import BaseSTTModel
from stt_benchmark.models.hf.mms import MMSModel
from stt_benchmark.datasets.base import AudioSample, ParallelAudioSample
from stt_benchmark.config.language_support.mms import (
    get_mms_asr_languages,
    mms_supports_asr,
)
from stt_benchmark.config.language_support.nllb import (
    fleurs_to_nllb,
    nllb_supports_language,
    nllb_supports_pair,
    get_nllb_supported_languages,
)


def _parallel_to_asr_sample(p: ParallelAudioSample) -> AudioSample:
    """Adapt a ParallelAudioSample's source side to an AudioSample for the ASR stage."""
    return AudioSample(
        transcription=p.source_transcription,
        language=p.source_language,
        sample_id=p.sample_id,
        audio_path=p.source_audio_path,
        audio_array=p.source_audio_array,
        sampling_rate=p.source_sampling_rate,
    )


class CascadedMmsNllbModel(BaseSTTModel):
    """Cascaded MMS (ASR) + NLLB (MT) model for ASR and AST."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.config = model_config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, model_config.get("torch_dtype", "float16"))
        if self.device == "cpu":
            self.torch_dtype = torch.float32

        # ── Load MMS (ASR stage) ─────────────────────────────────────────
        asr_model_name = model_config.get("asr_model", "facebook/mms-1b-all")
        asr_config = {
            "model_name": asr_model_name,
            "model_type": "mms",
            "torch_dtype": model_config.get("torch_dtype", "float16"),
            "device_map": model_config.get("device_map", "auto"),
            "generation_config": {},
        }
        print(f"  Loading ASR model: {asr_model_name}")
        self.asr_model = MMSModel(asr_model_name, asr_config)

        # ── Load NLLB (MT stage) ─────────────────────────────────────────
        mt_model_name = model_config.get("mt_model", "facebook/nllb-200-3.3B")
        print(f"  Loading MT model: {mt_model_name}")
        self.mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_name)
        self.mt_model = M2M100ForConditionalGeneration.from_pretrained(
            mt_model_name,
            torch_dtype=self.torch_dtype,
        )
        self.mt_model.to(self.device)
        self.mt_model.eval()

        gen_config = model_config.get("generation_config", {})
        self.mt_gen_kwargs = {
            "max_new_tokens": gen_config.get("max_new_tokens", 384),
            "num_beams": gen_config.get("num_beams", 5),
        }

        self._asr_languages = get_mms_asr_languages()
        self._nllb_languages = get_nllb_supported_languages()
        self._ast_pairs = self._build_ast_pairs()

        self._last_intermediate_transcripts: List[str] = []

    def _build_ast_pairs(self) -> Set[Tuple[str, str]]:
        pairs = set()
        for src in self._asr_languages:
            if not nllb_supports_language(src):
                continue
            for tgt in self._nllb_languages:
                if src != tgt:
                    pairs.add((src, tgt))
        return pairs

    def _translate_text(
        self,
        texts: List[str],
        source_fleurs: str,
        target_fleurs: str,
    ) -> List[str]:
        src_nllb = fleurs_to_nllb(source_fleurs)
        tgt_nllb = fleurs_to_nllb(target_fleurs)

        if src_nllb is None or tgt_nllb is None:
            print(f"NLLB cannot map {source_fleurs}→{target_fleurs}")
            return [""] * len(texts)

        self.mt_tokenizer.src_lang = src_nllb

        inputs = self.mt_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        forced_bos_token_id = self.mt_tokenizer.convert_tokens_to_ids(tgt_nllb)

        with torch.no_grad():
            generated_ids = self.mt_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                **self.mt_gen_kwargs,
            )

        translations = self.mt_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return [t.strip() for t in translations]

    # ------------------------------------------------------------------
    # ASR interface (delegates to MMS)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        samples: List[AudioSample],
        language: str,
    ) -> List[str]:
        return self.asr_model.transcribe(samples, language)

    # ------------------------------------------------------------------
    # AST interface (MMS → NLLB cascade)
    # ------------------------------------------------------------------

    def translate(
        self,
        samples: List[ParallelAudioSample],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        # Stage 1: ASR — adapt parallel samples to the ASR model's interface
        asr_samples = [_parallel_to_asr_sample(s) for s in samples]
        transcripts = self.asr_model.transcribe(asr_samples, source_lang)

        self._last_intermediate_transcripts = list(transcripts)

        # Stage 2: MT — translate only the non-empty transcripts
        non_empty_indices = [i for i, t in enumerate(transcripts) if t.strip()]

        if not non_empty_indices:
            return [""] * len(samples)

        non_empty_texts = [transcripts[i] for i in non_empty_indices]

        try:
            translated = self._translate_text(
                non_empty_texts, source_lang, target_lang
            )
        except Exception as e:
            print(f"    NLLB translation error: {e}")
            translated = [""] * len(non_empty_texts)

        results = [""] * len(samples)
        for idx, trans in zip(non_empty_indices, translated):
            results[idx] = trans

        return results

    def get_last_intermediate_transcripts(self) -> List[str]:
        return list(self._last_intermediate_transcripts)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, str]:
        asr_name = self.config.get("asr_model", "facebook/mms-1b-all")
        mt_name = self.config.get("mt_model", "facebook/nllb-200-3.3B")
        return {
            "model_name": self.model_name,
            "model_type": "cascaded",
            "asr_model": asr_name,
            "mt_model": mt_name,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "tasks": "asr,ast",
        }

    def get_supported_languages(self) -> Set[str]:
        return self._asr_languages

    def get_supported_pairs(self) -> Set[Tuple[str, str]]:
        return self._ast_pairs