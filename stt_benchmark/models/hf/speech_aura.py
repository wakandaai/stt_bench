# stt_benchmark/models/hf/speech_aura.py
"""SpeechAura wrapper for the STT benchmark.

Loads from an export directory produced by
`st.export.export_checkpoint speech_aura`. The export directory contains a
self-contained config.yaml with paths rewritten to point at sibling files,
so we just `cd` into it conceptually (resolve paths relative to it) and
hand the rewritten config to st.training.train_st.build_model.

Requires `pip install -e .` of the iwslt2026 repo so `import st` works.
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
from stt_benchmark.utils.audio import resolve_audio


class SpeechAuraModel(BaseSTTModel):
    """SpeechAura (Conformer + CTC compressor + projector + Aura-1B) loaded from an export dir."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        # Lazy import — only loaded when this model is actually instantiated.
        from st.utils.config import load_config
        from st.training.train_st import build_model

        self.model_name = model_name
        self.config = model_config
        self.target_sr = 16_000

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

        self._asr_languages = get_speech_aura_asr_languages()
        self._ast_pairs = get_speech_aura_ast_pairs()
        self._last_intermediate_transcripts: List[str] = []

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

    def _generate(self, sample, src_fleurs: str, task: str) -> Dict[str, str]:
        """Run inference.

        Args:
            sample: AudioSample or ParallelAudioSample
            src_fleurs: FLEURS code of the source audio
            task: 'asr' or 'cot'
        """
        mel, mel_len = self._sample_to_mel(sample)
        src_aura = fleurs_to_aura(src_fleurs)

        with torch.inference_mode():
            output = self.st_model.generate(
                audio_features=mel,
                audio_lengths=mel_len,
                src_lang=src_aura,
                tgt_lang="english",   # only English target supported
                task=task,
                max_new_tokens=(self.max_new_tokens_cot if task == "cot"
                                else self.max_new_tokens_asr),
            )

        if task == "asr":
            return {
                "transcript": self.st_model._strip_special_tokens(output).strip(),
                "translation": "",
            }
        transcript, translation = self.st_model.split_cot_output(output)
        return {"transcript": transcript, "translation": translation}

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        if not speech_aura_supports_asr(language):
            print(f"SpeechAura does not support ASR for {language}")
            return [""] * len(samples)

        results = []
        for sample in samples:
            try:
                out = self._generate(sample, language, task="asr")
                results.append(out["transcript"])
            except Exception as e:
                print(f"    SpeechAura ASR error for {sample.sample_id}: {e}")
                results.append("")
        return results

    # ------------------------------------------------------------------
    # AST (CoT)
    # ------------------------------------------------------------------

    def translate(self, samples: List[ParallelAudioSample], source_lang: str,
                  target_lang: str) -> List[str]:
        if not speech_aura_supports_ast(source_lang, target_lang):
            print(f"SpeechAura does not support AST {source_lang}→{target_lang}")
            return [""] * len(samples)

        translations = []
        self._last_intermediate_transcripts = []
        for sample in samples:
            try:
                out = self._generate(sample, source_lang, task="cot")
                self._last_intermediate_transcripts.append(out["transcript"])
                translations.append(out["translation"])
            except Exception as e:
                print(f"    SpeechAura AST error for {sample.sample_id}: {e}")
                self._last_intermediate_transcripts.append("")
                translations.append("")
        return translations

    def get_last_intermediate_transcripts(self) -> List[str]:
        return list(self._last_intermediate_transcripts)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_type": "speech_aura",
            "device": self.device,
            "tasks": "asr,ast",
            "export_dir": str(self.config.get("export_dir", "")),
        }

    def get_supported_languages(self) -> Set[str]:
        return self._asr_languages

    def get_supported_pairs(self) -> Set[tuple]:
        return self._ast_pairs