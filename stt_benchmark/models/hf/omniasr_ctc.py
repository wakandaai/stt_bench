# stt_benchmark/models/hf/omniasr_ctc.py
"""Our fine-tuned omniASR_CTC_1B encoder, wrapped for the STT benchmark.

ASR-only, greedy CTC decoding over the encoder's own 9812-piece SentencePiece
vocab (no LLM). Loaded directly from its real checkpoint + tokenizer paths —
no export-directory indirection (unlike ctc_encoder.py/speech_aura.py, which
load from a colleague-produced export bundle): we own both this adapter and
the aura-asr-v1 checkpoints, so there's nothing to decouple.

Requires the aura-asr-v1 repo's `src/` on sys.path, inserted here rather than
via `pip install -e` — aura-asr-v1 has its own top-level `st` package that
would collide with the colleague's speechaura `st` package (used by
ctc_encoder.py/speech_aura.py) if both were pip-installed into the same
environment. Since `evaluate.py` instantiates exactly one model per process
and this import is lazy (only runs when this model_type is actually built),
inserting the path here never collides with a colleague-model run in a
different process.

Also requires `fairseq2` (the encoder's underlying model library), which is
not in stt_bench's own pyproject.toml — install it into whatever environment
runs this (this project already maintains a working one: aura-asr-v1's
`.envs/omniasr_extract`, pinned to the same torch==2.8.0 stt_bench itself
requires).
"""

import sys
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple

import torch

from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.config.language_support.omniasr_ctc import (
    omniasr_ctc_supports_asr,
    get_omniasr_ctc_languages,
)
from stt_benchmark.utils.audio import resolve_audio

_AURA_ASR_V1_SRC = "/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src"
_CTC_BLANK_ID = 0


class OmniAsrCtcModel(BaseASRModel):
    """Fine-tuned omniASR_CTC_1B — greedy CTC decoding, no LLM."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        if _AURA_ASR_V1_SRC not in sys.path:
            sys.path.insert(0, _AURA_ASR_V1_SRC)
        # Lazy import — only triggered when this model_type is actually
        # instantiated (see module docstring re: the `st` package collision).
        import sentencepiece as spm
        from st.models.omniasr_encoder import build_omniasr_encoder_from_config

        self.model_name = model_name
        self.config = model_config
        self.target_sr = 16_000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = model_config["checkpoint"]
        tokenizer_path = model_config["tokenizer"]
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        if not Path(tokenizer_path).exists():
            raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

        enc_cfg = {
            "type": "omniasr_live",
            "checkpoint": checkpoint_path,
            "dropout_p": 0.0,
            "attn_dropout_p": 0.0,
            "ffn_inner_dropout_p": 0.0,
            "layer_drop_p": 0.0,
            "final_dropout_p": 0.0,
            "freeze_ctc_head": True,
        }
        self.encoder = build_omniasr_encoder_from_config(enc_cfg)
        self.encoder = self.encoder.to(self.device).eval()

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        self._asr_languages = get_omniasr_ctc_languages()

    # ------------------------------------------------------------------
    # Audio -> raw waveform (matches RawAudioDataset._load_sample exactly:
    # mono, resample to 16kHz, whole-tensor layer_norm — no mel step).
    # ------------------------------------------------------------------

    def _sample_to_waveform(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_np, _ = resolve_audio(sample, self.target_sr)
        waveform = torch.from_numpy(audio_np)
        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)
        waveform = waveform.unsqueeze(0).to(self.device)          # (1, T)
        length = torch.tensor([waveform.size(1)], device=self.device)
        return waveform, length

    # ------------------------------------------------------------------
    # Greedy CTC decode (matches pretrain_omniasr_ctc.py's
    # greedy_ctc_decode exactly: collapse repeats, drop blank, sp.decode)
    # ------------------------------------------------------------------

    def _decode_one(self, logits: torch.Tensor, length: int) -> str:
        pred_ids = logits.argmax(dim=-1)[0, :length].tolist()
        decoded, prev = [], -1
        for tid in pred_ids:
            if tid != _CTC_BLANK_ID and tid != prev:
                decoded.append(tid)
            prev = tid
        if not decoded:
            return ""
        return self.sp.decode(decoded)

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        if not omniasr_ctc_supports_asr(language):
            print(f"omniASR_CTC does not support ASR for {language}")
            return [""] * len(samples)

        results = []
        for sample in samples:
            try:
                waveform, length = self._sample_to_waveform(sample)
                with torch.inference_mode():
                    out = self.encoder(waveform, length)
                hyp = self._decode_one(out["ctc_logits"], int(out["lengths"][0].item()))
                results.append(hyp)
            except Exception as e:
                print(f"    omniASR_CTC ASR error for {sample.sample_id}: {e}")
                results.append("")
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_type": "omniasr_ctc",
            "device": self.device,
            "tasks": "asr",
            "checkpoint": str(self.config.get("checkpoint", "")),
        }

    def get_supported_languages(self) -> Set[str]:
        return self._asr_languages
