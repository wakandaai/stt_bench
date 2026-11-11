# stt_benchmark/models/hf/simba.py

"""UBC-NLP Simba ASR series (EMNLP 2025, "Voice of a Continent"), three families:

  simba_ctc      Simba-M (MMS-1b-all + adapter "multilingual_african"), Simba-X (XLS-R), Simba-H (AfriHuBERT):
                 CTC models with ONE shared 428-symbol multilingual vocabulary; no language conditioning.
  simba_whisper  Simba-W (Whisper large-v3 fine-tuned on 43 languages).
  simba_seamless Simba-S (SeamlessM4T-v2 speech-to-text fine-tuned on 43 languages).

The model cards and paper do NOT say how Simba-W / Simba-S choose the language (the card example passes none; Simba-S's
language table is the stock Seamless one and lacks kin/bem/fon/wol/hau/sot/tsn/xho). These adapters therefore decode
WITHOUT forcing a language by default (the documented usage) and RECORD what the model did per utterance
(`lang_log`: detected Whisper language token / requested tgt_lang), so a language that the model silently decodes as
another language stays measurable. An optional `tgt_lang_map` (repo code -> Seamless code) / `force_language` can
be set in the model config once the behaviour has been checked.

Strictness: a language outside the configured `languages` raises; any audio error raises; Whisper audio > 30 s raises.
The Simba models were fine-tuned on SimbaBench's train/dev split (5 h per language), so SimbaBench test sets are
in-domain for them.
"""

import re
import zlib
from typing import Any, Dict, List, Optional, Set

import torch

from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.utils.audio_strict import prepare_audio

SR = 16_000
MAX_WHISPER_SECONDS = 30.0
_LANG_TOKEN = re.compile(r"<\|([a-z]{2,3})\|>")


class _SimbaBase(BaseASRModel):
    model_type = "simba"

    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.config = model_config
        self.languages: Set[str] = set(model_config["languages"])
        self.revision = model_config.get("revision")
        self.cache_dir = model_config.get("cache_dir")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, model_config.get("torch_dtype", "float32"))
        if self.device == "cpu":
            self.torch_dtype = torch.float32
        self.batch_size = int(model_config.get("batch_size", 8))
        self.lang_log: List[Dict[str, Optional[str]]] = []

    def _check_language(self, language: str):
        if language not in self.languages:
            raise ValueError(f"{self.model_name}: language '{language}' not in the configured set {sorted(self.languages)}")

    def get_model_info(self) -> Dict[str, str]:
        return {"model_name": self.model_name, "model_type": self.model_type, "revision": str(self.revision),
                "device": self.device, "dtype": str(self.torch_dtype), "tasks": "asr"}

    def get_supported_languages(self) -> Set[str]:
        return set(self.languages)


class SimbaCtcModel(_SimbaBase):
    model_type = "simba_ctc"

    def __init__(self, model_name, model_config):
        super().__init__(model_name, model_config)
        from transformers import AutoModelForCTC, AutoProcessor
        self.adapter = model_config.get("adapter")             # e.g. "multilingual_african" (Simba-M only)
        self.processor = AutoProcessor.from_pretrained(model_name, revision=self.revision, cache_dir=self.cache_dir)
        self.model = AutoModelForCTC.from_pretrained(model_name, revision=self.revision, cache_dir=self.cache_dir,
                                                     torch_dtype=self.torch_dtype)
        if self.adapter:
            self.model.load_adapter(self.adapter)
            self.processor.tokenizer.set_target_lang(self.adapter)
        self.model.to(self.device).eval()

    @property
    def supports_batch(self) -> bool:
        return False

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        self._check_language(language)
        out = []
        for s in samples:
            audio = prepare_audio(s, SR)
            inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt")
            inputs = {k: (v.to(self.device, dtype=self.torch_dtype) if v.is_floating_point() else v.to(self.device))
                      for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits
            out.append(self.processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip())
        return out


class SimbaWhisperModel(_SimbaBase):
    model_type = "simba_whisper"

    def __init__(self, model_name, model_config):
        super().__init__(model_name, model_config)
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.max_new_tokens = int(model_config.get("generation_config", {}).get("max_new_tokens", 384))
        self.force_language = model_config.get("force_language", {})   # repo code -> whisper language code, optional
        self.detect_fraction = float(model_config.get("detect_language_fraction", 1.0))   # share of clips whose language is also detected (diagnostic only)
        if not 0.0 <= self.detect_fraction <= 1.0:
            raise ValueError("detect_language_fraction must be in [0, 1]")
        self.processor = WhisperProcessor.from_pretrained(model_name, revision=self.revision, cache_dir=self.cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, revision=self.revision, cache_dir=self.cache_dir, torch_dtype=self.torch_dtype)
        self.model.to(self.device).eval()

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        self._check_language(language)
        audios = []
        for s in samples:
            a = prepare_audio(s, SR)
            if a.shape[0] / SR > MAX_WHISPER_SECONDS:
                raise ValueError(f"{s.sample_id}: {a.shape[0]/SR:.1f}s > {MAX_WHISPER_SECONDS}s (no silent truncation)")
            audios.append(a)
        kw = {"task": "transcribe"}
        if language in self.force_language:
            kw["language"] = self.force_language[language]
        out: List[str] = []
        for i in range(0, len(audios), self.batch_size):
            feats = self.processor(audios[i:i + self.batch_size], sampling_rate=SR, return_tensors="pt").input_features
            feats = feats.to(self.device, dtype=self.torch_dtype)
            with torch.inference_mode():
                ids = self.model.generate(feats, num_beams=1, do_sample=False, max_new_tokens=self.max_new_tokens, **kw)
            # Language Whisper's own detector picks for each clip (what the free-running decode conditions on when
            # nothing is forced). Recorded, never used to override anything.
            # detect_language re-runs the whole encoder, so it can be limited to a deterministic subset of the
            # utterances (crc32 of the sample id); it only records, it never changes a transcript.
            pick = [j for j in range(feats.shape[0])
                    if zlib.crc32(samples[i + j].sample_id.encode()) % 1000 < int(self.detect_fraction * 1000)]
            det_by_j = {}
            if pick:
                with torch.inference_mode():
                    det = torch.as_tensor(self.model.detect_language(input_features=feats[pick])).flatten().tolist()
                det_by_j = dict(zip(pick, self.processor.tokenizer.convert_ids_to_tokens(det)))
            txt = self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for j, t in enumerate(txt):
                tok = det_by_j.get(j)
                m = _LANG_TOKEN.fullmatch(tok or "")
                self.lang_log.append({"sample_id": samples[i + j].sample_id, "language": language,
                                      "forced": kw.get("language"),
                                      "detected": (m.group(1) if m else tok) if j in det_by_j else None})
                out.append(t.strip())
        if len(out) != len(samples):
            raise RuntimeError(f"{len(out)} hypotheses for {len(samples)} samples")
        return out


class SimbaSeamlessModel(_SimbaBase):
    model_type = "simba_seamless"

    def __init__(self, model_name, model_config):
        super().__init__(model_name, model_config)
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
        self.max_new_tokens = int(model_config.get("generation_config", {}).get("max_new_tokens", 256))
        self.tgt_lang_map = model_config.get("tgt_lang_map", {})       # repo code -> Seamless code, optional
        self.processor = AutoProcessor.from_pretrained(model_name, revision=self.revision, cache_dir=self.cache_dir)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            model_name, revision=self.revision, cache_dir=self.cache_dir, torch_dtype=self.torch_dtype)
        self.model.to(self.device).eval()

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        self._check_language(language)
        tgt = self.tgt_lang_map.get(language)
        out: List[str] = []
        for i in range(0, len(samples), self.batch_size):
            audios = [prepare_audio(s, SR) for s in samples[i:i + self.batch_size]]
            inputs = self.processor(audio=audios, sampling_rate=SR, return_tensors="pt", padding=True)
            inputs = {k: (v.to(self.device, dtype=self.torch_dtype) if v.is_floating_point() else v.to(self.device))
                      for k, v in inputs.items()}
            kw = {"tgt_lang": tgt} if tgt else {}
            with torch.inference_mode():
                ids = self.model.generate(**inputs, num_beams=1, do_sample=False, max_new_tokens=self.max_new_tokens, **kw)
            ids = ids[0] if isinstance(ids, (tuple, list)) else getattr(ids, "sequences", ids)
            for j, t in enumerate(self.processor.batch_decode(ids, skip_special_tokens=True)):
                self.lang_log.append({"sample_id": samples[i + j].sample_id, "language": language, "tgt_lang": tgt})
                out.append(t.strip())
        if len(out) != len(samples):
            raise RuntimeError(f"{len(out)} hypotheses for {len(samples)} samples")
        return out
