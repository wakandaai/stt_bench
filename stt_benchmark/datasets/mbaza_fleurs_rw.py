# stt_benchmark/datasets/mbaza_fleurs_rw.py

"""
mbazaNLP/fleurs-kinyarwanda loader (local-first).

Expected on-disk layout (populated by scripts/fetch_mbaza_fleurs_rw.py):

    <local_path>/
    ├── train.tsv
    ├── dev.tsv
    ├── test.tsv
    └── audio/
        ├── train/<split_subdir>/<filename>.wav
        ├── dev/<split_subdir>/<filename>.wav
        └── test/<split_subdir>/<filename>.wav

The tarballs published by mbazaNLP wrap their contents in a per-split subdir
(e.g. test.tar.xz extracts into a `test_data/` directory), so this loader globs
one level under `audio/<split>/` to find the actual wav directory rather than
hardcoding that name.

TSV format (no header, tab-separated, 7 columns):
    col 0: sentence id (int) — aligns with Google FLEURS sentence ids
    col 1: audio filename (e.g. "669040034415576949.wav")
    col 2: raw transcription (with punctuation/casing)
    col 3: normalized transcription (lowercased, simplified)
    col 4: char-tokenized with `|` word separators (ignored)
    col 5: speaker id (ignored)
    col 6: gender (ignored)

AST: rw_rw <-> en_us, joining mbaza rows to `google/fleurs:en_us` rows by id.
mbaza preserved Google FLEURS sentence ids when translating to Kinyarwanda,
so the cross-join produces real parallel pairs. The id-overlap rate is logged
on first AST load as a sanity check.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

import numpy as np

from stt_benchmark.datasets.base import (
    BaseASRDataset, BaseASTDataset, AudioSample, ParallelAudioSample,
)


GOOGLE_FLEURS_HF_NAME = "google/fleurs"

SOURCE_LANG = "rw_rw"
ANCHOR_LANG = "en_us"
ANCHOR_HF_CONFIG = "en_us"

# Map FLEURS-style split names to mbaza's TSV filenames.
SPLIT_FILENAMES = {
    "train": "train",
    "validation": "dev",   # FLEURS calls it 'validation', mbaza calls it 'dev'
    "dev": "dev",
    "test": "test",
}

# Which column in the TSV to use as the reference transcription.
# Column 3 (normalized) matches what `google/fleurs` exposes as `transcription`,
# so we use it for consistency.
TRANSCRIPTION_COL = 3

ID_OVERLAP_WARN_THRESHOLD = 0.5


def _resolve_split(split: str) -> str:
    """Normalize a split name to mbaza's filename convention."""
    if split not in SPLIT_FILENAMES:
        raise ValueError(
            f"Unknown split '{split}'. Valid: {sorted(SPLIT_FILENAMES.keys())}"
        )
    return SPLIT_FILENAMES[split]


def _find_audio_subdir(audio_split_dir: Path) -> Path:
    """Find the single subdirectory under audio/<split>/ holding the wavs.

    mbaza's tarballs extract into e.g. `test_data/`, so we glob rather than
    hardcode. Errors if zero or multiple subdirs are found.
    """
    subdirs = [p for p in audio_split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(subdirs) == 0:
        raise FileNotFoundError(
            f"No subdirectory under {audio_split_dir}. "
            f"Did you run scripts/fetch_mbaza_fleurs_rw.py for this split?"
        )
    if len(subdirs) > 1:
        raise RuntimeError(
            f"Expected one subdirectory under {audio_split_dir}, found {len(subdirs)}: "
            f"{[p.name for p in subdirs]}"
        )
    return subdirs[0]


class MbazaFleursRwDataset(BaseASRDataset, BaseASTDataset):
    """mbazaNLP/fleurs-kinyarwanda loader (ASR + AST via id-join with Google FLEURS)."""

    def __init__(
        self,
        local_path: str,
        split: str = "test",
        languages: Optional[List[str]] = None,
        ast_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """Initialize the dataset.

        Args:
            local_path: Filesystem path to the prepared dataset directory
                (see module docstring for expected layout).
            split: FLEURS-style split name ('train', 'validation', 'test').
                'dev' is accepted as an alias for 'validation'.
            languages: Allowed ASR languages. Only 'rw_rw' is meaningful.
            ast_pairs: Allowed AST pairs. Only (rw_rw, en_us) and (en_us, rw_rw)
                are supported.
        """
        self.local_path = Path(local_path).resolve()
        if not self.local_path.is_dir():
            raise FileNotFoundError(
                f"local_path does not exist: {self.local_path}\n"
                f"Run scripts/fetch_mbaza_fleurs_rw.py --output-dir {self.local_path} first."
            )

        self.split = split
        self._split_filename = _resolve_split(split)

        # Locate TSV
        self._tsv_path = self.local_path / f"{self._split_filename}.tsv"
        if not self._tsv_path.exists():
            raise FileNotFoundError(
                f"TSV not found: {self._tsv_path}\n"
                f"Run scripts/fetch_mbaza_fleurs_rw.py --output-dir {self.local_path} "
                f"--splits {self._split_filename}"
            )

        # Locate audio directory (we only resolve it when ASR is actually requested,
        # so AST-only configs don't fail if audio isn't extracted yet)
        self._audio_split_dir = self.local_path / "audio" / self._split_filename
        self._audio_data_dir: Optional[Path] = None  # set lazily

        # Scope filters
        if languages is not None:
            self._allowed_asr = {l for l in languages if l == SOURCE_LANG}
        else:
            self._allowed_asr = {SOURCE_LANG}

        supportable_pairs = {(SOURCE_LANG, ANCHOR_LANG), (ANCHOR_LANG, SOURCE_LANG)}
        if ast_pairs is not None:
            self._allowed_pairs = set(ast_pairs) & supportable_pairs
        else:
            self._allowed_pairs = None  # sentinel: any supportable pair allowed

        # Caches
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._mbaza_text_by_id: Optional[Dict[int, str]] = None
        self._anchor_rows = None
        self._anchor_text_by_id: Optional[Dict[int, str]] = None
        self._mono_cache: Dict[str, List[AudioSample]] = {}
        self._parallel_cache: Dict[str, List[ParallelAudioSample]] = {}

    # ------------------------------------------------------------------
    # TSV + audio resolution
    # ------------------------------------------------------------------

    def _load_tsv(self) -> List[Dict[str, Any]]:
        """Parse the TSV into a list of dicts. Cached."""
        if self._rows is not None:
            return self._rows

        rows: List[Dict[str, Any]] = []
        with open(self._tsv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for lineno, parts in enumerate(reader, start=1):
                if len(parts) < TRANSCRIPTION_COL + 1:
                    print(
                        f"  [mbaza_fleurs_rw] skipping malformed line {lineno} "
                        f"in {self._tsv_path.name}: {len(parts)} columns"
                    )
                    continue
                try:
                    rows.append({
                        "id": int(parts[0]),
                        "audio_filename": parts[1],
                        "transcription": parts[TRANSCRIPTION_COL],
                    })
                except ValueError as e:
                    print(f"  [mbaza_fleurs_rw] skipping line {lineno}: {e}")

        self._rows = rows
        return rows

    def _resolve_audio_path(self, filename: str) -> Path:
        """Return absolute path to a wav file referenced by the TSV."""
        if self._audio_data_dir is None:
            if not self._audio_split_dir.is_dir():
                raise FileNotFoundError(
                    f"Audio split dir does not exist: {self._audio_split_dir}\n"
                    f"Run scripts/fetch_mbaza_fleurs_rw.py --output-dir {self.local_path} "
                    f"--splits {self._split_filename}"
                )
            self._audio_data_dir = _find_audio_subdir(self._audio_split_dir)
        return self._audio_data_dir / filename

    # ------------------------------------------------------------------
    # ASR (Kinyarwanda only)
    # ------------------------------------------------------------------

    def _load_mono_rw(self) -> List[AudioSample]:
        if SOURCE_LANG in self._mono_cache:
            return self._mono_cache[SOURCE_LANG]

        rows = self._load_tsv()
        samples: List[AudioSample] = []
        skipped_missing = 0

        for i, row in enumerate(rows):
            audio_path = self._resolve_audio_path(row["audio_filename"])
            if not audio_path.exists():
                skipped_missing += 1
                if skipped_missing <= 3:
                    print(f"  [mbaza_fleurs_rw] missing audio: {audio_path}")
                continue
            samples.append(AudioSample(
                transcription=row["transcription"],
                language=SOURCE_LANG,
                sample_id=f"{row['id']}_{i}",
                audio_path=str(audio_path),
            ))

        if skipped_missing:
            print(
                f"  [mbaza_fleurs_rw] loaded {len(samples)} samples, "
                f"skipped {skipped_missing} missing audio"
            )

        self._mono_cache[SOURCE_LANG] = samples
        return samples

    def get_language_samples(self, language: str) -> List[AudioSample]:
        if language != SOURCE_LANG:
            raise ValueError(
                f"mbaza_fleurs_rw only supports ASR for {SOURCE_LANG}, got '{language}'"
            )
        if language not in self._allowed_asr:
            raise ValueError(
                f"Language '{language}' not in allowed set: {sorted(self._allowed_asr)}"
            )
        return self._load_mono_rw()

    def list_languages(self) -> Set[str]:
        return set(self._allowed_asr)

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "dataset_name": "mbazaNLP/fleurs-kinyarwanda",
            "local_path": str(self.local_path),
            "split": self.split,
            "asr_languages": sorted(self._allowed_asr),
            "ast_pairs": (
                sorted(self._allowed_pairs)
                if self._allowed_pairs is not None
                else "any of {(rw_rw, en_us), (en_us, rw_rw)}"
            ),
        }

    # ------------------------------------------------------------------
    # AST cross-join with google/fleurs en_us by id
    # ------------------------------------------------------------------

    def _build_mbaza_text_index(self) -> Dict[int, str]:
        """{id: rw_rw transcription}. First-seen recording per id wins."""
        if self._mbaza_text_by_id is not None:
            return self._mbaza_text_by_id
        rows = self._load_tsv()
        index: Dict[int, str] = {}
        for row in rows:
            sid = row["id"]
            if sid not in index:
                index[sid] = row["transcription"]
        self._mbaza_text_by_id = index
        return index

    def _load_anchor(self):
        if self._anchor_rows is None:
            from datasets import load_dataset
            self._anchor_rows = load_dataset(
                GOOGLE_FLEURS_HF_NAME, ANCHOR_HF_CONFIG, split=self.split
            )
        return self._anchor_rows

    def _build_anchor_text_index(self) -> Dict[int, str]:
        """{id: en_us transcription}. First speaker per id wins."""
        if self._anchor_text_by_id is not None:
            return self._anchor_text_by_id
        ds = self._load_anchor()
        index: Dict[int, str] = {}
        for row in ds:
            sid = row["id"]
            if sid not in index:
                index[sid] = row["transcription"]
        self._anchor_text_by_id = index
        return index

    def _check_id_overlap(self, side: str):
        """Log id-overlap rate between mbaza and Google FLEURS en_us for the requested split."""
        mbaza_ids = set(self._build_mbaza_text_index().keys())
        anchor_ids = set(self._build_anchor_text_index().keys())

        if not mbaza_ids or not anchor_ids:
            print(
                f"  [mbaza_fleurs_rw/{side}] id overlap check: empty index "
                f"(mbaza={len(mbaza_ids)}, en_us={len(anchor_ids)})"
            )
            return

        overlap = mbaza_ids & anchor_ids
        rate_mbaza = len(overlap) / len(mbaza_ids)
        rate_anchor = len(overlap) / len(anchor_ids)
        print(
            f"  [mbaza_fleurs_rw/{side}] id overlap: {len(overlap)} shared ids "
            f"({rate_mbaza:.0%} of mbaza, {rate_anchor:.0%} of Google FLEURS en_us)"
        )
        if rate_mbaza < ID_OVERLAP_WARN_THRESHOLD:
            print(
                f"  ⚠️  [mbaza_fleurs_rw/{side}] WARNING: low id-overlap "
                f"({rate_mbaza:.0%}). AST results will be sparse."
            )

    def _load_rw_to_en(self) -> List[ParallelAudioSample]:
        """(rw_rw audio from local wavs, en_us text from Google FLEURS)."""
        key = f"{SOURCE_LANG}-{ANCHOR_LANG}"
        if key in self._parallel_cache:
            return self._parallel_cache[key]

        self._check_id_overlap("rw->en")
        anchor_text = self._build_anchor_text_index()
        rows = self._load_tsv()

        samples: List[ParallelAudioSample] = []
        skipped_no_target = 0
        skipped_missing_audio = 0

        for i, row in enumerate(rows):
            sid = row["id"]
            target_text = anchor_text.get(sid)
            if target_text is None:
                skipped_no_target += 1
                continue

            audio_path = self._resolve_audio_path(row["audio_filename"])
            if not audio_path.exists():
                skipped_missing_audio += 1
                if skipped_missing_audio <= 3:
                    print(f"  [mbaza_fleurs_rw/{key}] missing audio: {audio_path}")
                continue

            samples.append(ParallelAudioSample(
                source_transcription=row["transcription"],
                source_language=SOURCE_LANG,
                target_transcription=target_text,
                target_language=ANCHOR_LANG,
                sample_id=f"{sid}_{i}",
                source_audio_path=str(audio_path),
            ))

        if skipped_no_target or skipped_missing_audio:
            print(
                f"  [mbaza_fleurs_rw/{key}] joined {len(samples)} pairs, "
                f"skipped {skipped_no_target} (no en_us id match), "
                f"{skipped_missing_audio} (missing audio)"
            )

        self._parallel_cache[key] = samples
        return samples

    def _load_en_to_rw(self) -> List[ParallelAudioSample]:
        """(en_us audio from Google FLEURS, rw_rw text from mbaza TSV)."""
        key = f"{ANCHOR_LANG}-{SOURCE_LANG}"
        if key in self._parallel_cache:
            return self._parallel_cache[key]

        self._check_id_overlap("en->rw")
        rw_text = self._build_mbaza_text_index()
        src_ds = self._load_anchor()

        samples: List[ParallelAudioSample] = []
        skipped_no_target = 0
        skipped_decode = 0
        for row in src_ds:
            sid = row["id"]
            target_text = rw_text.get(sid)
            if target_text is None:
                skipped_no_target += 1
                continue
            try:
                audio = row["audio"]
                samples.append(ParallelAudioSample(
                    source_transcription=row["transcription"],
                    source_language=ANCHOR_LANG,
                    target_transcription=target_text,
                    target_language=SOURCE_LANG,
                    sample_id=f"{sid}_{row.get('num_samples', len(samples))}",
                    source_audio_array=np.asarray(audio["array"], dtype=np.float32),
                    source_sampling_rate=int(audio["sampling_rate"]),
                ))
            except Exception as e:
                skipped_decode += 1
                if skipped_decode <= 3:
                    print(f"  [mbaza_fleurs_rw/{key}] skipping row id={sid}: {e}")

        if skipped_no_target or skipped_decode:
            print(
                f"  [mbaza_fleurs_rw/{key}] joined {len(samples)} pairs, "
                f"skipped {skipped_no_target} (no rw_rw id match), "
                f"{skipped_decode} (undecodable audio)"
            )

        self._parallel_cache[key] = samples
        return samples

    def get_parallel_samples(
        self, source_lang: str, target_lang: str
    ) -> List[ParallelAudioSample]:
        pair = (source_lang, target_lang)
        if pair == (SOURCE_LANG, ANCHOR_LANG):
            return self._load_rw_to_en()
        if pair == (ANCHOR_LANG, SOURCE_LANG):
            return self._load_en_to_rw()
        raise ValueError(
            f"mbaza_fleurs_rw only supports AST pairs (rw_rw, en_us) and "
            f"(en_us, rw_rw); got {source_lang}->{target_lang}"
        )

    def list_language_pairs(self) -> Set[tuple]:
        if self._allowed_pairs is not None:
            return set(self._allowed_pairs)
        return {(SOURCE_LANG, ANCHOR_LANG), (ANCHOR_LANG, SOURCE_LANG)}