# stt_benchmark/models/hf/strict_hf.py

"""Strict adapters for the stock public models (Whisper large-v3, SeamlessM4T-v2, MMS-1b-all).

They replace the older whisper.py / seamless.py / mms.py for leaderboard runs. Differences, all about trust:
  * any error raises (the old adapters return "" on internal errors, which is indistinguishable from silence);
  * a language the model cannot do is reported as unsupported by `supports_language` (the runner records it) and
    raises if forced through `transcribe` - there is no fallback language;
  * the language is FORCED to the audio's language where the model supports it (Whisper language token, Seamless
    tgt_lang, MMS adapter) so a model is never scored on a language it was not asked to produce;
  * Whisper audio over 30 s raises (no silent truncation); fp16 batch effects are avoided with batch_size 1 by default;
  * weights are pinned to a revision; audio goes through utils.audio_strict.prepare_audio (no librosa).
Whisper and Seamless reuse the (already tested) strict decode paths in simba.py; only the language tables differ.
"""

from typing import Any, Dict, List, Set

import torch

from stt_benchmark.config.language_support.mms import fleurs_to_mms
from stt_benchmark.config.language_support.seamless import fleurs_to_seamless, seamless_supports_asr
from stt_benchmark.config.language_support.whisper import fleurs_to_whisper
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.models.hf.simba import SR, SimbaSeamlessModel, SimbaWhisperModel, _SimbaBase
from stt_benchmark.utils.audio_strict import prepare_audio

# The 25 leaderboard languages (FLEURS-style codes used by the cleaned-eval loader).
LEADERBOARD_CODES = ["af_za", "am_et", "ar_eg", "bem", "en_us", "fon_bj", "fr_fr", "ha_ng", "ig_ng", "rw_rw", "ln_cd",
                     "lg_ug", "mg_mg", "ny_mw", "pt_br", "sn_zw", "so_so", "st_za", "sw_ke", "ti_et", "tn_za", "wo_sn",
                     "xh_za", "yo_ng", "zu_za"]


class StrictWhisperModel(SimbaWhisperModel):
    """Stock Whisper with the language token forced to the audio's language (only where Whisper has that language)."""
    model_type = "strict_whisper"

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        cfg = dict(model_config)
        force = {c: fleurs_to_whisper(c) for c in LEADERBOARD_CODES if fleurs_to_whisper(c)}
        cfg["languages"] = sorted(force)
        cfg["force_language"] = force
        super().__init__(model_name, cfg)


class StrictSeamlessModel(SimbaSeamlessModel):
    """Stock SeamlessM4T-v2 speech-to-text with tgt_lang = the audio's language (ASR), where Seamless has it for both
    speech input and text output."""
    model_type = "strict_seamless"

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        cfg = dict(model_config)
        tgt = {c: fleurs_to_seamless(c) for c in LEADERBOARD_CODES if seamless_supports_asr(c)}
        cfg["languages"] = sorted(tgt)
        cfg["tgt_lang_map"] = tgt
        super().__init__(model_name, cfg)


class StrictMmsModel(_SimbaBase):
    """facebook/mms-1b-all: wav2vec2 CTC with one language adapter per language (loaded on demand, cached)."""
    model_type = "strict_mms"

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        cfg = dict(model_config)
        cfg.setdefault("languages", [])
        super().__init__(model_name, cfg)
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.processor = AutoProcessor.from_pretrained(model_name, revision=self.revision, cache_dir=self.cache_dir)
        vocabs = set(self.processor.tokenizer.vocab.keys())            # language -> sub-vocabulary for mms-1b-all
        self.mms_code = {c: fleurs_to_mms(c) for c in LEADERBOARD_CODES if fleurs_to_mms(c) in vocabs}
        self.languages = set(self.mms_code)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, revision=self.revision, cache_dir=self.cache_dir,
                                                    torch_dtype=self.torch_dtype, ignore_mismatched_sizes=False)
        self.model.to(self.device).eval()
        self._current = None

    @property
    def supports_batch(self) -> bool:
        return False

    def _set_adapter(self, code: str):
        if code == self._current:
            return
        self.processor.tokenizer.set_target_lang(code)
        self.model.load_adapter(code)                                   # raises if the adapter is missing
        self.model.to(self.device)
        self._current = code

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        self._check_language(language)
        self._set_adapter(self.mms_code[language])
        out = []
        for s in samples:
            audio = prepare_audio(s, SR)
            inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt")
            inputs = {k: (v.to(self.device, dtype=self.torch_dtype) if v.is_floating_point() else v.to(self.device))
                      for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits
            out.append(self.processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip())
        if len(out) != len(samples):
            raise RuntimeError(f"{len(out)} hypotheses for {len(samples)} samples")
        return out
