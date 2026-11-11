# stt_benchmark/models/hf/strict_aura.py

"""Strict adapters for the two private WakandaAI checkpoints.

  strict_aura_asr      WakandaAI/Aura-ASR-v2 (stage6_v1_step95000): omniASR encoder -> CTC compressor -> projector -> Aura-1B,
                       autoregressive greedy decoding. Loaded with the repo's own `AuraASRv2.from_pretrained`; decoded with
                       the repo's own `SpeechAura.generate` (KV-cached greedy, max_new_tokens 256, bf16 autocast).
  strict_aura_encoder  WakandaAI/Aura-Wav2Vec-Encoder: the CTC-only encoder, greedy CTC + SentencePiece, no language input.

Differences from the repo wrappers, all about trust:
  * `generate` falls back to the ENGLISH language token for any language missing from LANG_MAP
    (`LANG_MAP.get(target_lang, LANG_MAP["eng"])`); here a language without an entry is unsupported and raises if forced;
  * `no_repeat_ngram_size` is explicit (the wrapper default is 3; 0 is plain greedy) and recorded in the model info;
  * a hypothesis that used the whole `max_new_tokens` budget is a failure, not a silently cut transcript;
  * audio goes through utils.audio_strict.prepare_audio, then the same per-utterance layer_norm the wrappers apply;
  * the repos are pinned to a commit and read with the `wakanda_download` read-only token profile (never printed).
The Aura source tree ships its own top-level `st` package, which collides with the Phase 1 `st` used by the Conformer
adapter: run one model per process (as the runner does for a single model id).
"""

import configparser
import os
import sys
from typing import Any, Dict, List, Set

import numpy as np
import torch

from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.models.hf.strict_hf import LEADERBOARD_CODES
from stt_benchmark.utils.audio_strict import prepare_audio

SR = 16_000


def _profile_token(profile: str) -> str:
    cp = configparser.ConfigParser()
    cp.read(os.path.expanduser("~/.cache/huggingface/stored_tokens"))
    return cp[profile]["hf_token"]


def _snapshot(repo: str, revision: str, cache_dir: str, profile: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision, token=_profile_token(profile), cache_dir=cache_dir)


def _layer_norm_wave(a: np.ndarray, device: str):
    w = torch.from_numpy(np.ascontiguousarray(a)).float()
    w = torch.nn.functional.layer_norm(w, w.shape).unsqueeze(0).to(device)
    return w, torch.tensor([w.size(1)], device=device)


class _AuraBase(BaseASRModel):
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.config = model_config
        self.repo = model_config["repo"]
        self.revision = model_config["revision"]
        if len(self.revision) != 40:
            raise ValueError(f"{model_name}: revision must be a full 40-char commit sha")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            raise RuntimeError(f"{model_name}: a GPU is required")
        self.lang_log: List[Dict[str, Any]] = []
        self.local_dir = _snapshot(self.repo, self.revision, model_config["cache_dir"],
                                   model_config.get("token_profile", "wakanda_download"))
        self.languages: Set[str] = set()

    def get_supported_languages(self) -> Set[str]:
        return set(self.languages)

    def _audio(self, s: AudioSample, max_seconds: float = None) -> np.ndarray:
        a = prepare_audio(s, SR)
        if max_seconds and a.shape[0] / SR > max_seconds:
            raise ValueError(f"{s.sample_id}: {a.shape[0]/SR:.1f}s > {max_seconds}s")
        return a

    def _info(self, **extra) -> Dict[str, str]:
        d = {"model_name": self.model_name, "model_type": self.model_type, "repo": self.repo,
             "revision": self.revision, "device": self.device, "tasks": "asr"}
        d.update({k: str(v) for k, v in extra.items()})
        return d


class StrictAuraAsrModel(_AuraBase):
    model_type = "strict_aura_asr"

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        super().__init__(model_name, model_config)
        self.max_new_tokens = int(model_config.get("max_new_tokens", 256))
        self.no_repeat_ngram_size = int(model_config["no_repeat_ngram_size"])  # explicit: 3 (wrapper default) or 0
        for p in (self.local_dir, os.path.join(self.local_dir, "modeling")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from aura_asr_v2 import AuraASRv2
        from st.models.aura import LANG_MAP

        self.lang_map = LANG_MAP
        self.omni_code = {}
        for c in LEADERBOARD_CODES:
            iso = FLEURS_LANGUAGES[c]["iso639_3"]
            if iso in LANG_MAP:
                self.omni_code[c] = iso
        self.languages = set(self.omni_code)
        self.wrapper = AuraASRv2.from_pretrained(self.local_dir, device=self.device)
        self.model = self.wrapper.model
        self.tokenizer = self.model.aura.tokenizer

    def get_model_info(self) -> Dict[str, str]:
        return self._info(decoding=f"greedy KV-cache, no_repeat_ngram_size={self.no_repeat_ngram_size}",
                          max_new_tokens=self.max_new_tokens)

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        if language not in self.languages:
            raise ValueError(f"{self.model_name}: language '{language}' is not in LANG_MAP (no English fallback)")
        code = self.omni_code[language]
        out: List[str] = []
        for s in samples:
            wave, length = _layer_norm_wave(self._audio(s), self.device)
            with torch.inference_mode():
                text = self.model.generate(wave, length, target_lang=code, max_new_tokens=self.max_new_tokens,
                                           no_repeat_ngram_size=self.no_repeat_ngram_size)
            n_tok = len(self.tokenizer.encode(text, add_special_tokens=False))
            if n_tok >= self.max_new_tokens:
                raise RuntimeError(f"{s.sample_id}: hypothesis used the whole {self.max_new_tokens}-token budget")
            self.lang_log.append({"sample_id": s.sample_id, "language": language, "lang_token": code,
                                  "n_tokens": n_tok})
            out.append(text.strip())
        return out


class StrictAuraEncoderModel(_AuraBase):
    model_type = "strict_aura_encoder"

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        super().__init__(model_name, model_config)
        if self.local_dir not in sys.path:
            sys.path.insert(0, self.local_dir)
        from aura_wav2vec_encoder import AuraWav2VecEncoder

        # one language-agnostic CTC model: asked to transcribe every leaderboard group
        self.languages = set(LEADERBOARD_CODES)
        self.enc = AuraWav2VecEncoder.from_pretrained(self.local_dir, device=self.device)
        if self.enc.sp is None:
            raise RuntimeError(f"{model_name}: tokenizer.model missing from the snapshot")

    def get_model_info(self) -> Dict[str, str]:
        return self._info(decoding="greedy CTC, blank 0, SentencePiece")

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        if language not in self.languages:
            raise ValueError(f"{self.model_name}: language '{language}' not configured")
        out: List[str] = []
        for s in samples:
            wave, length = _layer_norm_wave(self._audio(s), self.device)
            o = self.enc.encode(wave, length)
            out.append(self.enc.decode_ctc(o["ctc_logits"], int(o["lengths"][0].item())).strip())
        return out
