# stt_benchmark/models/hf/speech_aura.py
"""SpeechAura wrapper for the STT benchmark.

Wraps the SpeechAura model as a BaseSTTModel.
Requires `pip install -e .` of the iwslt2026-st repo so `import st` works.
"""

import torch
from typing import List, Dict, Set, Any, Tuple

from stt_benchmark.models.base import BaseSTTModel
from stt_benchmark.config.language_support.speech_aura import (
    fleurs_to_aura,
    speech_aura_supports_asr,
    speech_aura_supports_ast,
    get_speech_aura_asr_languages,
    get_speech_aura_ast_pairs,
)
from stt_benchmark.utils.audio import load_audio


class SpeechAuraModel(BaseSTTModel):
    """SpeechAura (Conformer + CTC compressor + projector + Aura-1B).

    Loads once at init, runs inference per audio file. Designed for the
    BaseSTTModel interface used by the FLEURS evaluation pipeline.
    """

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """
        Args:
            model_name: Identifier for results (e.g. 'speech_aura_stage5_step50000').
            model_config: dict with keys:
                - st_config:  Path to iwslt2026-st experiment YAML
                - checkpoint: Path to checkpoint directory
                - max_new_tokens_asr: int (default 128)
                - max_new_tokens_cot: int (default 256)
        """
        # Lazy import — only loaded when this model is actually instantiated,
        # so users without the iwslt2026-st repo aren't forced to install it.
        from st.utils.config import load_config
        from st.training.train_st import build_model

        self.model_name = model_name
        self.config = model_config
        self.target_sr = 16_000

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        st_cfg = load_config(model_config["st_config"])
        self.st_model = build_model(st_cfg).to(self.device)
        self.st_model.load_checkpoint(model_config["checkpoint"])
        self.st_model.eval()

        gen_cfg = model_config.get("generation_config", {})
        self.max_new_tokens_asr = gen_cfg.get("max_new_tokens_asr", 128)
        self.max_new_tokens_cot = gen_cfg.get("max_new_tokens_cot", 256)

        self._asr_languages = get_speech_aura_asr_languages()
        self._ast_pairs = get_speech_aura_ast_pairs()

    # ------------------------------------------------------------------
    # Audio → mel (matches st/inference/generate.py exactly)
    # ------------------------------------------------------------------

    def _audio_to_mel(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        import torchaudio
        waveform, _ = load_audio(audio_path, self.target_sr)
        waveform = torch.from_numpy(waveform)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr, n_fft=400, hop_length=160, n_mels=80,
        )
        mel = mel_transform(waveform)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = mel.T.unsqueeze(0).to(self.device)        # (1, T, 80)
        mel_len = torch.tensor([mel.size(1)], device=self.device)
        return mel, mel_len

    def _generate(self, audio_path: str, fleurs_lang: str, task: str) -> Dict[str, str]:
        mel, mel_len = self._audio_to_mel(audio_path)

        # For ASR: target = source language (transcribe same-language)
        # For CoT: target = "english" (always — model was trained that way)
        if task == "asr":
            aura_lang = fleurs_to_aura(fleurs_lang)
        else:
            aura_lang = "english"

        with torch.inference_mode():
            output = self.st_model.generate(
                audio_features=mel,
                audio_lengths=mel_len,
                target_lang=aura_lang,
                max_new_tokens=self.max_new_tokens_cot if task == "cot" else self.max_new_tokens_asr,
            )

        if task == "asr":
            return {"transcript": self.st_model._strip_special_tokens(output).strip(),
                    "translation": ""}
        transcript, translation = self.st_model.split_cot_output(output)
        return {"transcript": transcript, "translation": translation}

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def transcribe(self, audio_paths: List[str], language: str) -> List[str]:
        if not speech_aura_supports_asr(language):
            print(f"SpeechAura does not support ASR for {language}")
            return [""] * len(audio_paths)

        results = []
        for path in audio_paths:
            try:
                out = self._generate(path, language, task="asr")
                results.append(out["transcript"])
            except Exception as e:
                print(f"    SpeechAura ASR error for {path}: {e}")
                results.append("")
        return results

    # ------------------------------------------------------------------
    # AST (CoT — uses transcript-then-translate chain)
    # ------------------------------------------------------------------

    def translate(self, audio_paths: List[str], source_lang: str,
                  target_lang: str) -> List[str]:
        if not speech_aura_supports_ast(source_lang, target_lang):
            print(f"SpeechAura does not support AST {source_lang}→{target_lang}")
            return [""] * len(audio_paths)

        translations = []
        self._last_intermediate_transcripts = []  # match cascaded model's contract
        for path in audio_paths:
            try:
                out = self._generate(path, source_lang, task="cot")
                self._last_intermediate_transcripts.append(out["transcript"])
                translations.append(out["translation"])
            except Exception as e:
                print(f"    SpeechAura AST error for {path}: {e}")
                self._last_intermediate_transcripts.append("")
                translations.append("")
        return translations

    def get_last_intermediate_transcripts(self) -> List[str]:
        """Expose CoT transcripts so the pipeline records them in the AST CSV.

        This mirrors CascadedMmsNllbModel — `_is_cascaded_model` in pipeline.py
        will pick this up and the AST predictions CSV will get an
        intermediate_transcript column. Very useful for ASR-vs-MT error
        attribution when comparing against the MMS+NLLB cascade.
        """
        return list(getattr(self, "_last_intermediate_transcripts", []))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_type": "speech_aura",
            "device": self.device,
            "tasks": "asr,ast",
            "checkpoint": self.config.get("checkpoint", ""),
        }

    def get_supported_languages(self) -> Set[str]:
        return self._asr_languages

    def get_supported_pairs(self) -> Set[tuple]:
        return self._ast_pairs