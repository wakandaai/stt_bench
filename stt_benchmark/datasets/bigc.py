# stt_benchmark/datasets/bigc.py

"""
BIG-C loader (local-first, ASR + native bem→en_us AST).

Expected on-disk layout: `local_path` points at the `bem/` directory under
`bigc/data/`:

    <local_path>/
    ├── audio/
    │   └── <audio_id>.wav
    └── splits/
        ├── train.tsv
        ├── valid.tsv
        ├── test.tsv
        ├── ...
        └── unaligned.tsv   (ignored)

TSV format (with header row, tab-separated, 14 columns):
    pair_id, image, sentence_id, image_id, audio_id, sentence, translation,
    speaker_id, gender, duration, sample_rate, audio_format, recording_lang,
    native_lang

For our purposes only these columns matter:
    audio_id      — wav filename
    sentence      — Bemba transcription
    translation   — English translation (provides native AST capability)

AST direction policy: bem -> en_us only. BIG-C has Bemba audio + English text
in each row, so the forward direction is native. The reverse direction
(en_us -> bem) is NOT supported by BIG-C — there is no English audio anywhere
in this corpus. Configs that request reverse pairs will get them auto-filtered.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

from stt_benchmark.datasets.base import (
    BaseASRDataset, BaseASTDataset, AudioSample, ParallelAudioSample,
)


SOURCE_LANG = "bem"
ANCHOR_LANG = "en_us"

# Map FLEURS-style split names to BIG-C's TSV filenames.
SPLIT_FILENAMES = {
    "train": "train",
    "validation": "valid",   # FLEURS calls it 'validation', BIG-C calls it 'valid'
    "valid": "valid",
    "dev": "valid",
    "test": "test",
}


def _resolve_split(split: str) -> str:
    if split not in SPLIT_FILENAMES:
        raise ValueError(
            f"Unknown split '{split}'. Valid: {sorted(SPLIT_FILENAMES.keys())}"
        )
    return SPLIT_FILENAMES[split]


class BigCDataset(BaseASRDataset, BaseASTDataset):
    """BIG-C loader (ASR + native bem→en_us AST)."""

    def __init__(
        self,
        local_path: str,
        split: str = "test",
        languages: Optional[List[str]] = None,
        ast_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """Initialize the dataset.

        Args:
            local_path: Filesystem path to BIG-C's `bem/` directory.
            split: FLEURS-style split name. 'valid', 'dev', and 'validation'
                are all aliases for the validation split.
            languages: Allowed ASR languages. Only 'bem' is meaningful.
            ast_pairs: Allowed AST pairs. Only (bem, en_us) is supported.
        """
        self.local_path = Path(local_path).resolve()
        if not self.local_path.is_dir():
            raise FileNotFoundError(
                f"local_path does not exist: {self.local_path}\n"
                f"Point this at BIG-C's `bem/` directory (under bigc/data/bem/)."
            )

        self.split = split
        self._split_filename = _resolve_split(split)

        self._tsv_path = self.local_path / "splits" / f"{self._split_filename}.tsv"
        if not self._tsv_path.exists():
            raise FileNotFoundError(
                f"TSV not found: {self._tsv_path}\n"
                f"Expected at {self.local_path}/splits/{self._split_filename}.tsv"
            )

        self._audio_dir = self.local_path / "audio"
        if not self._audio_dir.is_dir():
            raise FileNotFoundError(
                f"Audio directory not found: {self._audio_dir}\n"
                f"Expected at {self.local_path}/audio/"
            )

        # Scope filters
        if languages is not None:
            self._allowed_asr = {l for l in languages if l == SOURCE_LANG}
        else:
            self._allowed_asr = {SOURCE_LANG}

        # Only bem -> en_us is supportable. Reverse is filtered out.
        supportable_pairs = {(SOURCE_LANG, ANCHOR_LANG)}
        if ast_pairs is not None:
            self._allowed_pairs = set(ast_pairs) & supportable_pairs
        else:
            self._allowed_pairs = None  # sentinel: bem -> en_us allowed by default

        # Caches
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._mono_cache: Dict[str, List[AudioSample]] = {}
        self._parallel_cache: Dict[str, List[ParallelAudioSample]] = {}

    def _load_tsv(self) -> List[Dict[str, Any]]:
        """Parse the TSV (with header) into a list of dicts. Cached."""
        if self._rows is not None:
            return self._rows

        rows: List[Dict[str, Any]] = []
        with open(self._tsv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for lineno, row in enumerate(reader, start=2):
                audio_id = (row.get("audio_id") or "").strip()
                sentence = (row.get("sentence") or "").strip()
                translation = (row.get("translation") or "").strip()
                if not audio_id or not sentence:
                    print(
                        f"  [bigc] skipping malformed line {lineno} "
                        f"in {self._tsv_path.name} (missing audio_id or sentence)"
                    )
                    continue
                rows.append({
                    "audio_filename": audio_id,
                    "transcription": sentence,
                    "translation": translation,   # may be empty for some rows
                })

        self._rows = rows
        return rows

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def _load_mono(self) -> List[AudioSample]:
        if SOURCE_LANG in self._mono_cache:
            return self._mono_cache[SOURCE_LANG]

        rows = self._load_tsv()
        samples: List[AudioSample] = []
        skipped_missing = 0

        for i, row in enumerate(rows):
            audio_path = self._audio_dir / row["audio_filename"]
            if not audio_path.exists():
                skipped_missing += 1
                if skipped_missing <= 3:
                    print(f"  [bigc] missing audio: {audio_path}")
                continue
            samples.append(AudioSample(
                transcription=row["transcription"],
                language=SOURCE_LANG,
                sample_id=f"bem_{i}_{Path(row['audio_filename']).stem}",
                audio_path=str(audio_path),
            ))

        if skipped_missing:
            print(
                f"  [bigc] loaded {len(samples)} samples, "
                f"skipped {skipped_missing} missing audio"
            )

        self._mono_cache[SOURCE_LANG] = samples
        return samples

    def get_language_samples(self, language: str) -> List[AudioSample]:
        if language != SOURCE_LANG:
            raise ValueError(
                f"bigc only supports ASR for {SOURCE_LANG}, got '{language}'"
            )
        if language not in self._allowed_asr:
            raise ValueError(
                f"Language '{language}' not in allowed set: {sorted(self._allowed_asr)}"
            )
        return self._load_mono()

    def list_languages(self) -> Set[str]:
        return set(self._allowed_asr)

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "dataset_name": "BIG-C",
            "local_path": str(self.local_path),
            "split": self.split,
            "asr_languages": sorted(self._allowed_asr),
            "ast_pairs": (
                sorted(self._allowed_pairs)
                if self._allowed_pairs is not None
                else "[(bem, en_us)]"
            ),
        }

    # ------------------------------------------------------------------
    # AST — bem -> en_us only (native parallel)
    # ------------------------------------------------------------------

    def _load_bem_to_en(self) -> List[ParallelAudioSample]:
        key = f"{SOURCE_LANG}-{ANCHOR_LANG}"
        if key in self._parallel_cache:
            return self._parallel_cache[key]

        rows = self._load_tsv()
        samples: List[ParallelAudioSample] = []
        skipped_no_translation = 0
        skipped_missing_audio = 0

        for i, row in enumerate(rows):
            if not row["translation"]:
                skipped_no_translation += 1
                continue

            audio_path = self._audio_dir / row["audio_filename"]
            if not audio_path.exists():
                skipped_missing_audio += 1
                if skipped_missing_audio <= 3:
                    print(f"  [bigc/{key}] missing audio: {audio_path}")
                continue

            samples.append(ParallelAudioSample(
                source_transcription=row["transcription"],
                source_language=SOURCE_LANG,
                target_transcription=row["translation"],
                target_language=ANCHOR_LANG,
                sample_id=f"bem_{i}_{Path(row['audio_filename']).stem}",
                source_audio_path=str(audio_path),
            ))

        if skipped_no_translation or skipped_missing_audio:
            print(
                f"  [bigc/{key}] loaded {len(samples)} pairs, "
                f"skipped {skipped_no_translation} (empty translation), "
                f"{skipped_missing_audio} (missing audio)"
            )

        self._parallel_cache[key] = samples
        return samples

    def get_parallel_samples(
        self, source_lang: str, target_lang: str
    ) -> List[ParallelAudioSample]:
        pair = (source_lang, target_lang)
        if pair == (SOURCE_LANG, ANCHOR_LANG):
            return self._load_bem_to_en()
        raise ValueError(
            f"bigc only supports AST pair (bem, en_us); got {source_lang}->{target_lang}. "
            f"BIG-C has no English audio so reverse direction is impossible."
        )

    def list_language_pairs(self) -> Set[tuple]:
        if self._allowed_pairs is not None:
            return set(self._allowed_pairs)
        return {(SOURCE_LANG, ANCHOR_LANG)}