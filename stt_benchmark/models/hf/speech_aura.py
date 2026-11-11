# stt_benchmark/models/hf/speech_aura.py
"""SpeechAura wrapper for the STT benchmark.

Loads from an export directory produced by
`st.export.export_checkpoint speech_aura`. The export directory contains a
self-contained config.yaml with paths rewritten to point at sibling files,
so we just `cd` into it conceptually (resolve paths relative to it) and
hand the rewritten config to st.training.train_st.build_model.

Two modes (set via `mode` in the YAML config, default 'transcribe'):
  - 'transcribe': ASR-only model. transcribe() uses task='asr',
                  translate() is unsupported (returns empty + warning).
  - 'translate':  AST-only model. translate() uses task='st' (direct
                  speech translation), transcribe() is unsupported.

This class also backs speech_nllb and causal_lm (see get_model_info) —
st.training.train_st.build_model dispatches on which backbone block the exported
config.yaml carries (aura: / nllb: / causal_lm:), and all three honour the same
generate()/transcribe()/translate() contract. The one thing that does NOT
generalize automatically is which language-support module to consult (FLEURS code
-> dataset language name differs across language sets) — that is selected below
via model_config["model_type"].

Requires `pip install -e .` of the speechaura repo so `import st` works.
"""

import os
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple

import torch

from stt_benchmark.models.base import BaseSTTModel
from stt_benchmark.datasets.base import AudioSample, ParallelAudioSample
from stt_benchmark.config.language_support.speech_aura import (
    fleurs_to_aura,
    speech_aura_supports_asr,
    speech_aura_supports_ast,
    get_speech_aura_asr_languages,
    get_speech_aura_ast_pairs,
)
from stt_benchmark.config.language_support.causal_lm import (
    fleurs_to_causal_lm,
    causal_lm_supports_asr,
    causal_lm_supports_ast,
    get_causal_lm_asr_languages,
    get_causal_lm_ast_pairs,
)
from stt_benchmark.utils.audio import resolve_audio

# model_type -> (fleurs_to_<backbone>, <backbone>_supports_asr, <backbone>_supports_ast,
#                get_<backbone>_asr_languages, get_<backbone>_ast_pairs)
# "speech_nllb" reuses speech_aura's module: both trained on the same 20-language AST_INDEX
# set with the same dataset-name convention, so there is nothing backbone-specific to swap.
_LANGUAGE_SUPPORT = {
    "speech_aura": (fleurs_to_aura, speech_aura_supports_asr, speech_aura_supports_ast,
                     get_speech_aura_asr_languages, get_speech_aura_ast_pairs),
    "speech_nllb": (fleurs_to_aura, speech_aura_supports_asr, speech_aura_supports_ast,
                     get_speech_aura_asr_languages, get_speech_aura_ast_pairs),
    "causal_lm":   (fleurs_to_causal_lm, causal_lm_supports_asr, causal_lm_supports_ast,
                     get_causal_lm_asr_languages, get_causal_lm_ast_pairs),
}


class SpeechAuraModel(BaseSTTModel):
    """SpeechAura (Conformer + CTC compressor + projector + Aura-1B) loaded from an export dir."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        # Lazy import — only loaded when this model is actually instantiated.
        from core.utils.config import load_config
        from st.training.train_st import build_model

        self.model_name = model_config.get("model_name", model_name)
        self.config = model_config
        self.target_sr = 16_000

        self.mode = model_config.get("mode", "transcribe")
        if self.mode not in ("transcribe", "translate"):
            raise ValueError(
                f"speech_aura mode must be 'transcribe' or 'translate', got '{self.mode}'"
            )

        backbone_type = model_config.get("model_type", "speech_aura")
        if backbone_type not in _LANGUAGE_SUPPORT:
            raise ValueError(
                f"Unknown model_type {backbone_type!r} for language support — "
                f"expected one of {list(_LANGUAGE_SUPPORT)}"
            )
        (
            self._fleurs_to_backbone,
            self._backbone_supports_asr,
            self._backbone_supports_ast,
            self._get_backbone_asr_languages,
            self._get_backbone_ast_pairs,
        ) = _LANGUAGE_SUPPORT[backbone_type]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve the export directory. Accept either:
        #   export_dir: /path/to/export   (preferred — uses config.yaml inside)
        #   st_config + checkpoint        (legacy, kept for backward compat)
        export_dir = model_config.get("export_dir")
        if export_dir:
            export_dir = Path(export_dir).resolve()
            if not export_dir.is_dir():
                raise FileNotFoundError(f"export_dir not found: {export_dir}")
            config_path = export_dir / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"No config.yaml in export_dir {export_dir}. "
                    f"Was this directory produced by st.export.export_checkpoint?"
                )
            # The exported config has paths like "encoder.pt", "aura/tokenizer.json".
            # build_model() opens them with the current working directory, so we
            # `cd` into the export dir for the duration of model build.
            st_cfg = load_config(str(config_path))
            checkpoint = str(export_dir)  # projector.pt / lora.pt / llm_full.pt live here
            build_cwd = export_dir
        else:
            # Legacy path: training YAML + separate checkpoint dir
            st_config_path = model_config["st_config"]
            st_cfg = load_config(st_config_path)
            checkpoint = model_config["checkpoint"]
            build_cwd = None

        prev_cwd = os.getcwd()
        try:
            if build_cwd is not None:
                os.chdir(build_cwd)
            self.st_model = build_model(st_cfg).to(self.device)
            self.st_model.load_checkpoint(checkpoint)
        finally:
            os.chdir(prev_cwd)

        self.st_model.eval()

        gen_cfg = model_config.get("generation_config", {})
        self.max_new_tokens_asr = gen_cfg.get("max_new_tokens_asr", 128)
        self.max_new_tokens_cot = gen_cfg.get("max_new_tokens_cot", 256)
        self.max_new_tokens_st  = gen_cfg.get("max_new_tokens_st", 256)

        # Capability sets depend on mode. Transcribe-only models advertise no
        # AST pairs; translate-only models advertise no ASR languages. The
        # benchmark's pre-flight validation will then skip the irrelevant
        # half of any eval config gracefully.
        if self.mode == "transcribe":
            self._asr_languages = self._get_backbone_asr_languages()
            self._ast_pairs: Set[Tuple[str, str]] = set()
        else:  # translate
            self._asr_languages: Set[str] = set()
            self._ast_pairs = self._get_backbone_ast_pairs()

    # ------------------------------------------------------------------
    # Audio → mel (matches st/inference/generate.py exactly)
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

    def _generate(self, sample, src_fleurs: str, task: str,
                  tgt_fleurs: str = "en_us") -> str:
        """Run inference.

        Args:
            sample: AudioSample or ParallelAudioSample
            src_fleurs: FLEURS code of the source audio
            task: 'asr' or 'st'
            tgt_fleurs: FLEURS code of the target language (only used for 'st')
        """
        mel, mel_len = self._sample_to_mel(sample)
        src_aura = self._fleurs_to_backbone(src_fleurs)
        tgt_aura = self._fleurs_to_backbone(tgt_fleurs) or "english"

        if task == "asr":
            max_new = self.max_new_tokens_asr
        elif task == "st":
            max_new = self.max_new_tokens_st
        else:
            max_new = self.max_new_tokens_cot

        with torch.inference_mode():
            output = self.st_model.generate(
                audio_features=mel,
                audio_lengths=mel_len,
                src_lang=src_aura,
                tgt_lang=tgt_aura,
                task=task,
                max_new_tokens=max_new,
            )

        # For both 'asr' and 'st', generate() returns the single output stream
        # (transcript or translation, respectively) with special tokens still
        # present so split_cot_output can work. We just strip them.
        return self.st_model._strip_special_tokens(output).strip()

    # ------------------------------------------------------------------
    # ASR — only meaningful in 'transcribe' mode
    # ------------------------------------------------------------------

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        if self.mode != "transcribe":
            print(
                f"SpeechAura model '{self.model_name}' is in mode='{self.mode}' "
                f"and does not support ASR"
            )
            return [""] * len(samples)

        if not self._backbone_supports_asr(language):
            print(f"SpeechAura does not support ASR for {language}")
            return [""] * len(samples)

        results = []
        for sample in samples:
            try:
                hyp = self._generate(sample, language, task="asr")
                results.append(hyp)
            except Exception as e:
                print(f"    SpeechAura ASR error for {sample.sample_id}: {e}")
                results.append("")
        return results

    # ------------------------------------------------------------------
    # AST — only meaningful in 'translate' mode (uses direct ST task)
    # ------------------------------------------------------------------

    def translate(self, samples: List[ParallelAudioSample], source_lang: str,
                  target_lang: str) -> List[str]:
        if self.mode != "translate":
            print(
                f"SpeechAura model '{self.model_name}' is in mode='{self.mode}' "
                f"and does not support AST"
            )
            return [""] * len(samples)

        if not self._backbone_supports_ast(source_lang, target_lang):
            print(f"SpeechAura does not support AST {source_lang}→{target_lang}")
            return [""] * len(samples)

        translations = []
        for sample in samples:
            try:
                hyp = self._generate(sample, source_lang, task="st",
                                     tgt_fleurs=target_lang)
                translations.append(hyp)
            except Exception as e:
                print(f"    SpeechAura AST error for {sample.sample_id}: {e}")
                translations.append("")
        return translations

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            # Report what was configured rather than a hardcoded name — this
            # class also backs speech_nllb and causal_lm (see the module docstring).
            "model_type": self.config.get("model_type", "speech_aura"),
            "mode": self.mode,
            "device": self.device,
            "tasks": "asr" if self.mode == "transcribe" else "ast",
            "export_dir": str(self.config.get("export_dir", "")),
        }

    def get_supported_languages(self) -> Set[str]:
        return self._asr_languages

    def get_supported_pairs(self) -> Set[tuple]:
        return self._ast_pairs