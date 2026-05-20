# stt_benchmark/models/hf/ctc_encoder.py
"""CTC encoder wrapper for the STT benchmark.

ASR-only — wraps a pretrained Conformer + CTC encoder exported by
`st.export.export_checkpoint encoder`. Greedy CTC decoding (no LLM).

Export directory layout:
    encoder.pt            (state_dict)
    encoder_config.yaml   (architecture)
    vocab.json            (CTC vocab, index 0 = blank)

Requires `pip install -e .` of the iwslt2026 repo so `import st` works.
"""

import json
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple

import torch
import yaml

from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.config.language_support.speech_aura import (
    speech_aura_supports_asr,
    get_speech_aura_asr_languages,
)
from stt_benchmark.utils.audio import resolve_audio


class CTCEncoderModel(BaseASRModel):
    """Pretrained Conformer + CTC encoder with greedy decoding."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        # Lazy import — only triggered when this model is actually instantiated.
        from st.models.encoder import SpeechEncoder

        self.model_name = model_name
        self.config = model_config
        self.target_sr = 16_000

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        export_dir = Path(model_config["export_dir"]).resolve()
        if not export_dir.is_dir():
            raise FileNotFoundError(f"export_dir not found: {export_dir}")

        cfg_path   = export_dir / "encoder_config.yaml"
        vocab_path = export_dir / "vocab.json"
        wt_path    = export_dir / "encoder.pt"
        for p in (cfg_path, vocab_path, wt_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {p.name} in {export_dir} — was this directory "
                    f"produced by `st.export.export_checkpoint encoder`?"
                )

        with open(cfg_path) as f:
            enc_cfg = yaml.safe_load(f)
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.idx_to_char = {v: k for k, v in self.vocab.items()}

        # Drop the `checkpoint` field if it leaked in — it's a path, not an arch arg.
        enc_cfg.pop("checkpoint", None)
        # vocab_size may be present from the SpeechAura export rewrite; the
        # SpeechEncoder constructor takes it as a separate kwarg.
        enc_cfg.pop("vocab_size", None)

        self.encoder = SpeechEncoder(**enc_cfg, vocab_size=len(self.vocab))
        state = torch.load(wt_path, map_location="cpu", weights_only=True)
        # Export may save either {"model_state_dict": ...} or a raw state_dict;
        # handle both for robustness.
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.encoder.load_state_dict(state)
        self.encoder = self.encoder.to(self.device).eval()

        self._asr_languages = get_speech_aura_asr_languages()

    # ------------------------------------------------------------------
    # Audio → mel (matches SpeechDataset / st.inference.generate exactly)
    # ------------------------------------------------------------------

    def _sample_to_mel(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        import torchaudio
        waveform_np, _ = resolve_audio(sample, self.target_sr)
        waveform = torch.from_numpy(waveform_np)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr, n_fft=400, hop_length=160, n_mels=80,
        )
        mel = mel_transform(waveform)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = mel.T.unsqueeze(0).to(self.device)        # (1, T, 80)
        mel_len = torch.tensor([mel.size(1)], device=self.device)
        return mel, mel_len

    # ------------------------------------------------------------------
    # Greedy CTC decode
    # ------------------------------------------------------------------

    def _decode_one(self, logits: torch.Tensor, length: int) -> str:
        """Argmax → collapse repeats → drop blanks. blank_id = 0."""
        pred_ids = logits.argmax(dim=-1)[0, :length].tolist()
        out, prev = [], -1
        for tid in pred_ids:
            if tid != 0 and tid != prev:
                out.append(self.idx_to_char.get(tid, ""))
            prev = tid
        return "".join(out).strip()

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        # The CTC encoder is language-agnostic at decode time — it just emits
        # characters. We still gate on the SpeechAura-supported set so the
        # benchmark only evaluates languages this model was trained on.
        if not speech_aura_supports_asr(language):
            print(f"CTC encoder does not support ASR for {language}")
            return [""] * len(samples)

        results = []
        for sample in samples:
            try:
                mel, mel_len = self._sample_to_mel(sample)
                with torch.inference_mode():
                    out = self.encoder(mel, mel_len)
                hyp = self._decode_one(out["ctc_logits"], int(out["lengths"][0].item()))
                results.append(hyp)
            except Exception as e:
                print(f"    CTC encoder ASR error for {sample.sample_id}: {e}")
                results.append("")
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_type": "ctc_encoder",
            "device": self.device,
            "tasks": "asr",
            "export_dir": str(self.config.get("export_dir", "")),
        }

    def get_supported_languages(self) -> Set[str]:
        return self._asr_languages