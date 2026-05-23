# stt_benchmark/datasets/bembaspeech.py

"""
BembaSpeech loader (local-first, ASR-only).

Expected on-disk layout: `local_path` points at the `bem/` directory inside
the BembaSpeech repo:

    <local_path>/
    ├── train.tsv
    ├── dev.tsv
    ├── test.tsv
    └── audio/
        └── <filename>.wav

TSV format (with header row, tab-separated, 2 columns):
    audio    sentence

ASR-only. No translation column, no parallel data.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

from stt_benchmark.datasets.base import (
    BaseASRDataset, BaseASTDataset, AudioSample, ParallelAudioSample,
)


SOURCE_LANG = "bem"

# Map FLEURS-style split names to BembaSpeech's TSV filenames.
SPLIT_FILENAMES = {
    "train": "train",
    "validation": "dev",   # FLEURS calls it 'validation', BembaSpeech calls it 'dev'
    "dev": "dev",
    "test": "test",
}


def _resolve_split(split: str) -> str:
    if split not in SPLIT_FILENAMES:
        raise ValueError(
            f"Unknown split '{split}'. Valid: {sorted(SPLIT_FILENAMES.keys())}"
        )
    return SPLIT_FILENAMES[split]


class BembaSpeechDataset(BaseASRDataset, BaseASTDataset):
    """BembaSpeech loader (ASR-only)."""

    def __init__(
        self,
        local_path: str,
        split: str = "test",
        languages: Optional[List[str]] = None,
        ast_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """Initialize the dataset.

        Args:
            local_path: Filesystem path to the BembaSpeech `bem/` directory.
            split: FLEURS-style split name ('train', 'validation', 'test').
                'dev' is accepted as an alias for 'validation'.
            languages: Allowed ASR languages. Only 'bem' is meaningful.
            ast_pairs: Always filtered to the empty set — BembaSpeech is ASR-only.
        """
        self.local_path = Path(local_path).resolve()
        if not self.local_path.is_dir():
            raise FileNotFoundError(
                f"local_path does not exist: {self.local_path}\n"
                f"Point this at the BembaSpeech `bem/` directory."
            )

        self.split = split
        self._split_filename = _resolve_split(split)

        self._tsv_path = self.local_path / f"{self._split_filename}.tsv"
        if not self._tsv_path.exists():
            raise FileNotFoundError(
                f"TSV not found: {self._tsv_path}\n"
                f"Expected at {self.local_path}/{self._split_filename}.tsv"
            )

        self._audio_dir = self.local_path / "audio"
        # Audio dir is required for ASR; check now and fail fast.
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

        # BembaSpeech is ASR-only. If the config asks for any AST pair, filter
        # it down to nothing — pre-flight validation will report the skip with
        # the dataset-doesn't-support-it reason.
        self._allowed_pairs: Set[Tuple[str, str]] = set()

        # Caches
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._mono_cache: Dict[str, List[AudioSample]] = {}

    def _load_tsv(self) -> List[Dict[str, Any]]:
        """Parse the TSV (with header) into a list of dicts. Cached."""
        if self._rows is not None:
            return self._rows

        rows: List[Dict[str, Any]] = []
        with open(self._tsv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for lineno, row in enumerate(reader, start=2):  # start=2 since header is line 1
                if not row.get("audio") or not row.get("sentence"):
                    print(
                        f"  [bembaspeech] skipping malformed line {lineno} "
                        f"in {self._tsv_path.name}"
                    )
                    continue
                rows.append({
                    "audio_filename": row["audio"].strip(),
                    "transcription": row["sentence"].strip(),
                })

        self._rows = rows
        return rows

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
                    print(f"  [bembaspeech] missing audio: {audio_path}")
                continue
            samples.append(AudioSample(
                transcription=row["transcription"],
                language=SOURCE_LANG,
                sample_id=f"bem_{i}_{Path(row['audio_filename']).stem}",
                audio_path=str(audio_path),
            ))

        if skipped_missing:
            print(
                f"  [bembaspeech] loaded {len(samples)} samples, "
                f"skipped {skipped_missing} missing audio"
            )

        self._mono_cache[SOURCE_LANG] = samples
        return samples

    def get_language_samples(self, language: str) -> List[AudioSample]:
        if language != SOURCE_LANG:
            raise ValueError(
                f"bembaspeech only supports ASR for {SOURCE_LANG}, got '{language}'"
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
            "dataset_name": "BembaSpeech",
            "local_path": str(self.local_path),
            "split": self.split,
            "asr_languages": sorted(self._allowed_asr),
            "ast_pairs": "(ASR-only; no AST support)",
        }

    # AST — always empty for BembaSpeech.
    def get_parallel_samples(
        self, source_lang: str, target_lang: str
    ) -> List[ParallelAudioSample]:
        raise ValueError(
            f"bembaspeech does not support AST (pure ASR corpus); "
            f"requested {source_lang}->{target_lang}"
        )

    def list_language_pairs(self) -> Set[tuple]:
        return set()