# stt_benchmark/models/hf/sunflower_whisper.py

"""SunflowerASR (Sunbird AI): Whisper large-v3 adapted to 51 African languages.

Sunbird/SunflowerASR-51-african-languages. The language is NOT detected: the caller must supply it, and
for the 40 languages Whisper never had, the model reuses UNUSED Whisper language-token ids. The id table below is
copied from the model card; at load time every id is checked against the tokenizer and the table is printed
(so a changed repo revision cannot silently shift a language onto another token).

Strictness (leaderboard use):
  * a language without a Sunbird token raises (no fallback language, no empty hypothesis);
  * audio longer than 30 s raises (no silent truncation; chunking is not implemented here);
  * greedy decoding only (num_beams=1, no sampling), as in the model card.

Test sets from Sunbird's own corpora (the Sunbird Speech Benchmark) are in-domain for this model.
"""

from typing import Any, Dict, List, Set

import numpy as np
import torch

from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.utils.audio_strict import prepare_audio

SR = 16_000
MAX_SECONDS = 30.0

# iso639-3 -> language-token id (verbatim from the model card).
SUNBIRD_LANG_TOKENS: Dict[str, int] = {
    "eng": 50259, "fra": 50265, "swa": 50318, "sna": 50324, "yor": 50325, "som": 50326,
    "afr": 50327, "amh": 50334, "mlg": 50349, "lin": 50353, "hau": 50354,
    "ach": 50357, "aka": 50356, "bam": 50355, "bem": 50352, "ber": 50351,
    "cgg": 50350, "dag": 50348, "dga": 50347, "ewe": 50346, "ful": 50345,
    "ibo": 50344, "kab": 50343, "kau": 50342, "kik": 50341, "kin": 50340,
    "kln": 50339, "koo": 50338, "kpo": 50337, "led": 50336, "lgg": 50335,
    "lth": 50333, "lug": 50332, "luo": 50331, "luy": 50330, "myx": 50329,
    "nbl": 50328, "nya": 50323, "nyn": 50322, "orm": 50321, "pcm": 50320,
    "ruc": 50319, "rwm": 50317, "sot": 50316, "teo": 50315, "tsn": 50314,
    "ttj": 50313, "wol": 50312, "xho": 50311, "xog": 50310, "zul": 50309,
}

# Repo (FLEURS-style) language code -> Sunbird iso3. Only languages of our set that the model covers.
REPO_TO_SUNBIRD: Dict[str, str] = {
    "en_us": "eng", "fr_fr": "fra", "sw_ke": "swa", "sn_zw": "sna", "yo_ng": "yor", "so_so": "som",
    "af_za": "afr", "am_et": "amh", "mg_mg": "mlg", "ln_cd": "lin", "ha_ng": "hau", "bem": "bem",
    "ig_ng": "ibo", "rw_rw": "kin", "lg_ug": "lug", "ny_mw": "nya", "st_za": "sot", "tn_za": "tsn",
    "wo_sn": "wol", "xh_za": "xho", "zu_za": "zul",
}


class SunflowerWhisperModel(BaseASRModel):
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_name = model_name
        self.config = model_config
        self.revision = model_config.get("revision")
        cache_dir = model_config.get("cache_dir")
        self.batch_size = int(model_config.get("batch_size", 8))
        self.max_new_tokens = int(model_config.get("generation_config", {}).get("max_new_tokens", 384))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, model_config.get("torch_dtype", "float16"))
        if self.device == "cpu":
            self.torch_dtype = torch.float32

        self.processor = WhisperProcessor.from_pretrained(model_name, revision=self.revision, cache_dir=cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, revision=self.revision, cache_dir=cache_dir, torch_dtype=self.torch_dtype)
        self.model.to(self.device).eval()

        tok = self.processor.tokenizer
        self._transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
        self._notimestamps = tok.convert_tokens_to_ids("<|notimestamps|>")
        for name, i in (("<|transcribe|>", self._transcribe), ("<|notimestamps|>", self._notimestamps)):
            if i is None or i == tok.unk_token_id:
                raise ValueError(f"tokenizer has no id for {name}")
        if len(set(SUNBIRD_LANG_TOKENS.values())) != len(SUNBIRD_LANG_TOKENS):
            raise ValueError("duplicate ids in SUNBIRD_LANG_TOKENS")
        self.token_report = {}
        for iso3, i in SUNBIRD_LANG_TOKENS.items():
            self.token_report[iso3] = (i, tok.convert_ids_to_tokens(i))
        bad = [k for k, (i, t) in self.token_report.items() if t is None]
        if bad:
            raise ValueError(f"language-token ids not in the tokenizer: {bad}")
        self._languages = set(REPO_TO_SUNBIRD)

    # ------------------------------------------------------------------
    def _forced_ids(self, language: str):
        if language not in REPO_TO_SUNBIRD:
            raise ValueError(f"{self.model_name} has no language token for '{language}'. "
                             f"Supported: {sorted(REPO_TO_SUNBIRD)}")
        lang_tok = SUNBIRD_LANG_TOKENS[REPO_TO_SUNBIRD[language]]
        return [(1, lang_tok), (2, self._transcribe), (3, self._notimestamps)]

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        forced = self._forced_ids(language)
        audios = []
        for s in samples:
            a = prepare_audio(s, SR)
            if a.shape[0] / SR > MAX_SECONDS:
                raise ValueError(f"{s.sample_id}: {a.shape[0]/SR:.1f}s > {MAX_SECONDS}s (no silent truncation)")
            audios.append(a)
        out: List[str] = []
        for i in range(0, len(audios), self.batch_size):
            chunk = audios[i:i + self.batch_size]
            feats = self.processor(chunk, sampling_rate=SR, do_normalize=True, return_tensors="pt").input_features
            feats = feats.to(self.device, dtype=self.torch_dtype)
            with torch.inference_mode():
                ids = self.model.generate(feats, forced_decoder_ids=forced, num_beams=1, do_sample=False,
                                          max_new_tokens=self.max_new_tokens)
            out += [t.strip() for t in self.processor.batch_decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)]
        if len(out) != len(samples):
            raise RuntimeError(f"got {len(out)} hypotheses for {len(samples)} samples")
        return out

    def get_model_info(self) -> Dict[str, str]:
        return {"model_name": self.model_name, "model_type": "sunflower_whisper", "revision": str(self.revision),
                "device": self.device, "dtype": str(self.torch_dtype), "decoding": "greedy (num_beams=1)",
                "tasks": "asr"}

    def get_supported_languages(self) -> Set[str]:
        return set(self._languages)
