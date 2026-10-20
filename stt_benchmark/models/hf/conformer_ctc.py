# stt_benchmark/models/hf/conformer_ctc.py

"""Phase 1 Conformer CTC encoder (character-level CTC, 20-language pretraining), greedy decoding.

Replicates the Phase 1 decode path (phase1_asr/scripts/eval/eval_asr_decode.py, --system ctc) without change:
  waveform -> first channel, float32, 16 kHz, NO amplitude normalisation
           -> torchaudio MelSpectrogram(sr 16000, n_fft 400, hop 160, n_mels 80) -> clamp(1e-10).log10() -> (T, 80)
           -> st SpeechEncoder -> ctc_logits -> argmax -> collapse repeats -> drop blank (id 0) -> vocab.json characters.
The model is language-agnostic (one 925-symbol character vocabulary, no language conditioning): it is asked to
transcribe every group. Characters outside its vocabulary can never be produced, which is part of what is measured.
Strictness: no fallbacks - any load/decode error raises. The `st` package comes from the Phase 1 source tree
(`st_src`), the same code that trained the checkpoint.
"""

import json
import sys
from typing import Any, Dict, List, Set

import torch

from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.utils.audio_strict import prepare_audio

SR = 16_000


class ConformerCtcModel(BaseASRModel):
    model_type = "conformer_ctc"

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.config = model_config
        st_src = model_config["st_src"]
        if st_src not in sys.path:
            sys.path.insert(0, st_src)
        import torchaudio
        from st.models import load_encoder_from_checkpoint

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocab = json.load(open(model_config["vocab_path"], encoding="utf-8"))
        self.idx2c = {v: k for k, v in self.vocab.items()}
        enc_cfg = dict(model_config["encoder"])
        self.encoder = load_encoder_from_checkpoint(config=enc_cfg, checkpoint_path=model_config["checkpoint"],
                                                    vocab_size=len(self.vocab), strict=True)
        self.encoder = self.encoder.to(self.device).eval()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=400, hop_length=160, n_mels=80)
        self.languages: Set[str] = set(model_config["languages"])

    @property
    def supports_batch(self) -> bool:
        return False

    def get_model_info(self) -> Dict[str, str]:
        return {"model_name": self.model_name, "model_type": self.model_type, "device": self.device,
                "checkpoint": str(self.config["checkpoint"]), "vocab_size": str(len(self.vocab)),
                "note": "language-agnostic character CTC; trained on 20 languages (ASR_INDEX_V3)", "tasks": "asr"}

    def get_supported_languages(self) -> Set[str]:
        return set(self.languages)

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        if language not in self.languages:
            raise ValueError(f"{self.model_name}: language '{language}' not configured")
        out: List[str] = []
        for s in samples:
            wav = torch.from_numpy(prepare_audio(s, SR))
            mel = self.mel(wav)                                   # (80, T)
            mel = torch.clamp(mel, min=1e-10).log10().T           # (T, 80)
            mel_in = mel.unsqueeze(0).to(self.device)
            mel_len = torch.tensor([mel.size(0)], device=self.device)
            with torch.inference_mode():
                o = self.encoder(mel_in, mel_len)
            ids = o["ctc_logits"][0, : int(o["lengths"][0])].argmax(-1).tolist()
            txt, prev = [], -1
            for t in ids:
                if t != 0 and t != prev:
                    txt.append(self.idx2c[t])                    # KeyError (not "") if an id is unknown
                prev = t
            out.append("".join(txt).strip())
        if len(out) != len(samples):
            raise RuntimeError(f"{len(out)} hypotheses for {len(samples)} samples")
        return out
