# stt_benchmark/models/hf/w2v_bert_ctc.py

"""Wav2Vec2-BERT 2.0 CTC ASR (single language, greedy CTC decoding).

First use: badrex/w2v-bert-2.0-kinyarwanda-asr (Kinyarwanda only; fine-tuned on the
~1000 h Kaggle "Track B" corpus from Digital Umuganda: health, government, finance,
education, agriculture). The model is in-domain for that corpus, so test sets drawn
from the same source (e.g. AfriVoice Kinyarwanda) are NOT zero-shot for it.

Strictness: any language other than the model's own raises; any audio problem raises
(the leaderboard layer counts errors, it never sees silently empty hypotheses).
"""

import re
from typing import Any, Dict, List, Optional, Set

import torch

from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.datasets.base import AudioSample
from stt_benchmark.utils.audio_strict import prepare_audio

SR = 16_000


class W2vBertCtcModel(BaseASRModel):
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

        self.model_name = model_name
        self.config = model_config
        self.languages: Set[str] = set(model_config["languages"])      # repo codes, e.g. ["rw_rw"]
        # Joint ASR+LID models (e.g. Ethio-ASR) emit a leading language tag token such as "[AMH]". The caller cannot force
        # the language; the tag is stripped from the hypothesis and recorded so language-ID mismatches stay measurable.
        self.language_tags: Dict[str, str] = dict(model_config.get("language_tags") or {})
        self.tag_log: List[Dict[str, Optional[str]]] = []
        if self.language_tags and set(self.language_tags) != self.languages:
            raise ValueError("language_tags keys must equal languages")
        self.revision = model_config.get("revision")
        cache_dir = model_config.get("cache_dir")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, model_config.get("torch_dtype", "float32"))
        if self.device == "cpu":
            self.torch_dtype = torch.float32

        self.processor = Wav2Vec2BertProcessor.from_pretrained(
            model_name, revision=self.revision, cache_dir=cache_dir)
        self.model = Wav2Vec2BertForCTC.from_pretrained(
            model_name, revision=self.revision, cache_dir=cache_dir, torch_dtype=self.torch_dtype)
        self.model.to(self.device).eval()

    def transcribe(self, samples: List[AudioSample], language: str) -> List[str]:
        if language not in self.languages:
            raise ValueError(
                f"{self.model_name} supports only {sorted(self.languages)}; got '{language}'")
        out: List[str] = []
        for s in samples:
            audio = prepare_audio(s, SR)
            inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt")
            inputs = {k: (v.to(self.device, dtype=self.torch_dtype) if v.is_floating_point() else v.to(self.device))
                      for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(ids)[0].strip()
            if self.language_tags:
                mt = re.match(r"^\s*(\[[A-Z]{3}\])\s*", text)
                predicted = mt.group(1) if mt else None
                if mt:
                    text = text[mt.end():].strip()
                self.tag_log.append({"sample_id": s.sample_id, "language": language,
                                     "expected_tag": self.language_tags[language], "predicted_tag": predicted})
            out.append(text)
        return out

    @property
    def supports_batch(self) -> bool:
        return False   # one utterance per forward pass: no padding effects

    def get_model_info(self) -> Dict[str, str]:
        return {"model_name": self.model_name, "model_type": "w2v_bert_ctc", "revision": str(self.revision),
                "device": self.device, "dtype": str(self.torch_dtype), "decoding": "greedy CTC", "tasks": "asr"}

    def get_supported_languages(self) -> Set[str]:
        return set(self.languages)
