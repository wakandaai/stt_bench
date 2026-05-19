# stt_benchmark/datasets/fleurs.py

"""
FLEURS dataset loader backed by HuggingFace `google/fleurs`.

Loads one config per language on demand. For AST, pairs source rows to target
text by joining on the FLEURS `id` field (the canonical sentence ID — same
sentence across all languages). Policy: every source utterance is paired
with one reference text from the target language (first speaker per id).

Per-row decode errors (e.g. torchcodec/FFmpeg failing on a malformed audio
blob) are caught and the row is skipped, so a single bad sample doesn't
abort a multi-language run.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
import numpy as np
from datasets import load_dataset

from stt_benchmark.datasets.base import (
    BaseASRDataset, BaseASTDataset, AudioSample, ParallelAudioSample,
)
from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES

HF_DATASET_NAME = "google/fleurs"


class FleursDataset(BaseASRDataset, BaseASTDataset):
    """HuggingFace FLEURS loader for ASR (monolingual) and AST (parallel via id-join)."""

    def __init__(
        self,
        split: str = "test",
        languages: Optional[List[str]] = None,
        ast_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """Initialize FLEURS dataset.

        Args:
            split: HF split name ('train', 'validation', or 'test').
            languages: Optional list of FLEURS language codes to allow for ASR.
                       If None, all FLEURS_LANGUAGES are advertised as available
                       (configs are loaded lazily on first request).
            ast_pairs: Optional list of (source, target) pairs to allow for AST.
                       If None, any pair of supported languages is allowed.
        """
        self.split = split

        all_codes = set(FLEURS_LANGUAGES.keys())
        self._allowed_languages: Set[str] = (
            set(languages) & all_codes if languages else all_codes
        )
        self._allowed_pairs: Optional[Set[Tuple[str, str]]] = (
            set(ast_pairs) if ast_pairs is not None else None
        )

        # Caches
        self._mono_cache: Dict[str, List[AudioSample]] = {}
        self._parallel_cache: Dict[str, List[ParallelAudioSample]] = {}
        # Per-language target-text lookup: lang -> {id: transcription}
        self._target_text_index: Dict[str, Dict[int, str]] = {}

    # ------------------------------------------------------------------
    # HF loading
    # ------------------------------------------------------------------

    def _load_hf_split(self, language: str):
        """Load one HF config for one split. Returns a `datasets.Dataset`."""
        return load_dataset(HF_DATASET_NAME, language, split=self.split)

    def _build_target_text_index(self, language: str) -> Dict[int, str]:
        """Build {id: transcription} for one language. First speaker wins.

        Note: we deliberately don't touch row["audio"] here, so corrupt
        audio in the *target* language doesn't break the AST pair load.
        We only need text from the target side.
        """
        if language in self._target_text_index:
            return self._target_text_index[language]

        ds = self._load_hf_split(language)
        index: Dict[int, str] = {}
        for row in ds:
            sid = row["id"]
            if sid not in index:
                index[sid] = row["transcription"]

        self._target_text_index[language] = index
        return index

    # ------------------------------------------------------------------
    # ASR (monolingual)
    # ------------------------------------------------------------------

    def _load_mono(self, language: str) -> List[AudioSample]:
        if language in self._mono_cache:
            return self._mono_cache[language]

        ds = self._load_hf_split(language)

        samples: List[AudioSample] = []
        skipped = 0
        for row in ds:
            try:
                audio = row["audio"]
                samples.append(AudioSample(
                    transcription=row["transcription"],
                    language=language,
                    sample_id=f"{row['id']}_{row.get('num_samples', len(samples))}",
                    audio_array=np.asarray(audio["array"], dtype=np.float32),
                    sampling_rate=int(audio["sampling_rate"]),
                ))
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(
                        f"  [{language}] skipping row id={row.get('id', '?')}: {e}"
                    )

        if skipped:
            print(
                f"  [{language}] loaded {len(samples)} samples, "
                f"skipped {skipped} undecodable"
            )

        self._mono_cache[language] = samples
        return samples

    def get_language_samples(self, language: str) -> List[AudioSample]:
        if language not in self._allowed_languages:
            raise ValueError(
                f"Language '{language}' not in allowed set. "
                f"Available: {sorted(self._allowed_languages)}"
            )
        return self._load_mono(language)

    def list_languages(self) -> Set[str]:
        return set(self._allowed_languages)

    def get_dataset_info(self) -> Dict[str, Any]:
        cached_counts = {lang: len(s) for lang, s in self._mono_cache.items()}
        return {
            "dataset_name": "google/fleurs (HF)",
            "split": self.split,
            "num_allowed_languages": len(self._allowed_languages),
            "allowed_languages": sorted(self._allowed_languages),
            "allowed_pairs": (
                sorted(self._allowed_pairs)
                if self._allowed_pairs is not None
                else None
            ),
            "cached_mono_counts": cached_counts,
        }

    # ------------------------------------------------------------------
    # AST (parallel via id-join)
    # ------------------------------------------------------------------

    def _load_parallel(
        self, source_lang: str, target_lang: str
    ) -> List[ParallelAudioSample]:
        """Pair source audio with target text by FLEURS `id`.

        Policy (a): every source utterance becomes one pair, with the
        first-seen target transcription for that id as reference.
        Source rows whose id has no matching target are skipped.
        Source rows whose audio fails to decode are also skipped.
        """
        pair_key = f"{source_lang}-{target_lang}"
        if pair_key in self._parallel_cache:
            return self._parallel_cache[pair_key]

        src_ds = self._load_hf_split(source_lang)
        tgt_index = self._build_target_text_index(target_lang)

        samples: List[ParallelAudioSample] = []
        skipped_no_target = 0
        skipped_decode = 0
        for row in src_ds:
            sid = row["id"]
            target_text = tgt_index.get(sid)
            if target_text is None:
                skipped_no_target += 1
                continue

            try:
                audio = row["audio"]
                samples.append(ParallelAudioSample(
                    source_transcription=row["transcription"],
                    source_language=source_lang,
                    target_transcription=target_text,
                    target_language=target_lang,
                    sample_id=f"{sid}_{row.get('num_samples', len(samples))}",
                    source_audio_array=np.asarray(audio["array"], dtype=np.float32),
                    source_sampling_rate=int(audio["sampling_rate"]),
                ))
            except Exception as e:
                skipped_decode += 1
                if skipped_decode <= 3:
                    print(f"  [{pair_key}] skipping row id={sid}: {e}")

        if skipped_no_target or skipped_decode:
            print(
                f"  [{pair_key}] joined {len(samples)} pairs, "
                f"skipped {skipped_no_target} (no target id), "
                f"{skipped_decode} (undecodable audio)"
            )

        self._parallel_cache[pair_key] = samples
        return samples

    def get_parallel_samples(
        self, source_lang: str, target_lang: str
    ) -> List[ParallelAudioSample]:
        if source_lang not in FLEURS_LANGUAGES or target_lang not in FLEURS_LANGUAGES:
            raise ValueError(
                f"Unknown FLEURS language in pair {source_lang}->{target_lang}"
            )
        return self._load_parallel(source_lang, target_lang)

    def list_language_pairs(self) -> Set[tuple]:
        if self._allowed_pairs is not None:
            return set(self._allowed_pairs)
        langs = self._allowed_languages
        return {(s, t) for s in langs for t in langs if s != t}

    def get_languages(self) -> Set[str]:
        """All language codes seen in mono + parallel scope."""
        langs = set(self._allowed_languages)
        for src, tgt in self.list_language_pairs():
            langs.add(src)
            langs.add(tgt)
        return langs