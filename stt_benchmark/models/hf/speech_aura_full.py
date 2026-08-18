# stt_benchmark/models/hf/speech_aura_full.py
"""Our own full pipeline (omniASR encoder -> projector -> Aura-1B LLM),
wrapped for the STT benchmark. Distinct from speech_aura.py, which wraps the
colleague's own separate speechaura repo/export-bundle convention and a
richer generate(task=, src_lang=, tgt_lang=) signature we don't have.

Loaded directly from our own training checkpoint directory (e.g.
runs/stage4_v1_interactive/checkpoint_step45000) via SpeechAura.load_checkpoint(),
the same method train_st.py itself uses to resume training — so this adapter
is guaranteed to reconstruct the exact model that was actually trained.

Requires the aura-asr-v1 repo's `src/` on sys.path (see omniasr_ctc.py's
docstring for why: `st` package name collision with the colleague's own
speechaura repo, avoided by lazy-importing only when this model_type is
actually instantiated). Also requires fairseq2, same as omniasr_ctc.py.
"""

import sys
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple

import torch

from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.config.language_support.omniasr_ctc import (
    omniasr_ctc_supports_asr as speech_aura_full_supports_asr,
    get_omniasr_ctc_languages as get_speech_aura_full_languages,
)
from stt_benchmark.config.language_support.omniasr_ctc import FLEURS_TO_OMNIASR_CTC
from stt_benchmark.utils.audio import resolve_audio

_AURA_ASR_V1_SRC = "/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src"


class SpeechAuraFullModel(BaseASRModel):
    """Our own encoder -> projector -> Aura-1B pipeline — full autoregressive
    generation, loaded from a real training checkpoint directory."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        if _AURA_ASR_V1_SRC not in sys.path:
            sys.path.insert(0, _AURA_ASR_V1_SRC)
        # Lazy imports — same reasoning as omniasr_ctc.py's adapter.
        from st.models import SpeechAura, AuraLLM, build_ctc_compressor
        from st.models.omniasr_encoder import build_omniasr_encoder_from_config

        self.model_name = model_name
        self.config = model_config
        self.target_sr = 16_000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = model_config.get("max_new_tokens", 128)

        checkpoint_dir = model_config["checkpoint_dir"]
        if not Path(checkpoint_dir).is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

        encoder_checkpoint = model_config["encoder_checkpoint"]
        aura_checkpoint = model_config["aura_checkpoint"]
        aura_tokenizer = model_config["aura_tokenizer"]

        enc_cfg = {
            "type": "omniasr_live",
            "checkpoint": encoder_checkpoint,
            "dropout_p": 0.0,
            "attn_dropout_p": 0.0,
            "ffn_inner_dropout_p": 0.0,
            "layer_drop_p": 0.0,
            "final_dropout_p": 0.0,
            "freeze_ctc_head": True,
        }
        encoder = build_omniasr_encoder_from_config(enc_cfg)

        aura = AuraLLM(
            ckpt_path=aura_checkpoint,
            tokenizer_path=aura_tokenizer,
            size=model_config.get("aura_size", "1b"),
            freeze=True,     # eval-only — no gradients needed either way
            lora_rank=0,
        )

        ctc_compress_cfg = {
            "enabled": True,
            "strategy": "avg",
            "remove_blanks": True,
            "blank_id": 0,
        }

        self.model = SpeechAura(
            encoder=encoder,
            aura=aura,
            projector_cfg={"type": "transformer", "num_layers": 2, "num_heads": 8, "dropout": 0.0},
            ctc_compress_cfg=ctc_compress_cfg,
            ctc_weight=0.0,
            aux_ctc_weight=0.0,   # inference-only — this head isn't used by generate()
            freeze_encoder=True,  # eval-only — no gradients needed either way
            freeze_llm=True,
        )
        self.model.load_checkpoint(checkpoint_dir)   # pulls in projector.pt + llm_full.pt
        self.model = self.model.to(self.device).eval()

        self._asr_languages = get_speech_aura_full_languages()

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
    # ASR
    # ------------------------------------------------------------------

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        if not speech_aura_full_supports_asr(language):
            print(f"SpeechAura (full) does not support ASR for {language}")
            return [""] * len(samples)

        target_lang = FLEURS_TO_OMNIASR_CTC[language]  # -> our LANG_MAP key, e.g. "amharic"

        results = []
        for sample in samples:
            try:
                waveform, length = self._sample_to_waveform(sample)
                with torch.inference_mode():
                    hyp = self.model.generate(
                        waveform, length,
                        target_lang=target_lang,
                        max_new_tokens=self.max_new_tokens,
                    )
                results.append(hyp.strip())
            except Exception as e:
                print(f"    SpeechAura (full) ASR error for {sample.sample_id}: {e}")
                results.append("")
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_type": "speech_aura_full",
            "device": self.device,
            "tasks": "asr",
            "checkpoint_dir": str(self.config.get("checkpoint_dir", "")),
        }

    def get_supported_languages(self) -> Set[str]:
        return self._asr_languages
