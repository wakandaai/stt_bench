# stt_benchmark/datasets/simbabench.py

"""
SimbaBench ASR test loader (UBC-NLP/SimbaBench_dataset, asr_test/) — ASR only.

Reads the per-language test parquet files from a LOCAL copy, so no network is
needed at evaluation time:

    <local_path>/asr_test/HF_test-<iso3>*<source>.parquet

Columns per row: split, benchmark_id, audio {bytes, path} (16 kHz), text,
duration_s, lang_iso3, lang_name.

Design rules (this loader feeds a leaderboard, so nothing may be silently lost):
  * unknown / unmapped languages raise (no fallback language);
  * a language whose files are missing raises;
  * rows with an empty transcript or undecodable audio are NOT dropped quietly:
    they are counted, reported in get_dataset_info(), and the load raises if
    more than MAX_BAD_FRACTION of a language's rows are affected;
  * sample ids come from `benchmark_id` and must be unique;
  * the transcript is returned as released (whitespace-stripped only): all text
    normalisation belongs to the scorer, never to the loader.

Language codes follow the repo convention (FLEURS-style). Swahili is split over
two files in the release (`swa` and `swh`); both are read and concatenated.

Known size limits of the release (rows): ibo 5, tir 7, twi 12, mlq 182, fuf 129 ...
Languages with fewer than SMALL_N rows are flagged in get_dataset_info(); the
loader does not filter them (the reporting layer decides what is reportable).
"""

import glob
import io
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from stt_benchmark.datasets.base import BaseASRDataset, AudioSample

# Repo language code -> iso639-3 prefix(es) of the SimbaBench file names.
# Only languages of the Aura 25-language set that SimbaBench covers.
FLEURS_TO_SIMBA_ISO3: Dict[str, Tuple[str, ...]] = {
    "af_za": ("afr",),
    "am_et": ("amh",),
    "bem": ("bem",),
    "fon_bj": ("fon",),
    "ha_ng": ("hau",),
    "ig_ng": ("ibo",),     # 5 test rows in the release: not reportable
    "lg_ug": ("lug",),
    "ny_mw": ("nya",),
    "rw_rw": ("kin",),     # source is Common Voice 19 (contamination risk, flag downstream)
    "st_za": ("sot",),
    "sw_ke": ("swa", "swh"),
    "ti_et": ("tir",),     # 7 test rows in the release: not reportable
    "tn_za": ("tsn",),
    "wo_sn": ("wol",),
    "xh_za": ("xho",),
    "yo_ng": ("yor",),
    "zu_za": ("zul",),
}

MAX_BAD_FRACTION = 0.002   # same abort threshold as the project scorer
SMALL_N = 300              # flag only; not a filter
SIMBA_SR = 16_000


class SimbaBenchDataset(BaseASRDataset):
    """SimbaBench ASR test, local parquet copy, ASR-only."""

    def __init__(
        self,
        local_path: str,
        split: str = "test",
        languages: Optional[List[str]] = None,
    ):
        if split != "test":
            raise ValueError(
                f"simbabench ASR data released here is test-only; got split='{split}'"
            )
        self.root = os.path.join(local_path, "asr_test")
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"SimbaBench asr_test directory not found: {self.root}")
        self.split = split

        supported = set(FLEURS_TO_SIMBA_ISO3)
        if languages is not None:
            unknown = set(languages) - supported
            if unknown:
                raise ValueError(
                    f"simbabench has no mapping for {sorted(unknown)}. "
                    f"Supported: {sorted(supported)}"
                )
            self._allowed = set(languages)
        else:
            self._allowed = supported

        self._cache: Dict[str, List[AudioSample]] = {}
        self._report: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def _files_for(self, code: str) -> List[str]:
        files: List[str] = []
        for iso3 in FLEURS_TO_SIMBA_ISO3[code]:
            files += sorted(glob.glob(os.path.join(self.root, f"HF_test-{iso3}*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"no SimbaBench test parquet for {code} "
                f"({FLEURS_TO_SIMBA_ISO3[code]}) under {self.root}"
            )
        return files

    @staticmethod
    def _decode(audio: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.ascontiguousarray(data), int(sr)

    def _load(self, code: str) -> List[AudioSample]:
        if code in self._cache:
            return self._cache[code]
        import pyarrow.parquet as pq

        files = self._files_for(code)
        samples: List[AudioSample] = []
        seen: Set[str] = set()
        n_rows = n_empty = n_bad_audio = 0
        for f in files:
            table = pq.read_table(f)
            for row in table.to_pylist():
                n_rows += 1
                text = (row.get("text") or "").strip()
                if not text:
                    n_empty += 1
                    continue
                try:
                    arr, sr = self._decode(row["audio"])
                    if arr.size == 0:
                        raise ValueError("empty audio")
                except Exception:
                    n_bad_audio += 1
                    continue
                sid = f"simbabench_{row['benchmark_id']}"
                if sid in seen:
                    raise ValueError(f"duplicate benchmark_id in {code}: {sid}")
                seen.add(sid)
                samples.append(AudioSample(
                    transcription=text, language=code, sample_id=sid,
                    audio_array=arr, sampling_rate=sr,
                ))

        bad = n_empty + n_bad_audio
        if n_rows == 0 or bad / n_rows > MAX_BAD_FRACTION:
            raise ValueError(
                f"[simbabench/{code}] {bad}/{n_rows} rows unusable "
                f"(empty text {n_empty}, undecodable audio {n_bad_audio}); "
                f"limit {MAX_BAD_FRACTION:.1%}"
            )
        self._report[code] = {
            "files": [os.path.basename(f) for f in files],
            "rows_in_files": n_rows, "rows_used": len(samples),
            "rows_empty_text": n_empty, "rows_bad_audio": n_bad_audio,
            "small_n": len(samples) < SMALL_N,
        }
        self._cache[code] = samples
        return samples

    # ------------------------------------------------------------------
    def get_language_samples(self, language: str) -> List[AudioSample]:
        if language not in FLEURS_TO_SIMBA_ISO3:
            raise ValueError(
                f"simbabench does not support language '{language}'. "
                f"Supported: {sorted(FLEURS_TO_SIMBA_ISO3)}"
            )
        if language not in self._allowed:
            raise ValueError(
                f"Language '{language}' not in allowed set: {sorted(self._allowed)}"
            )
        return self._load(language)

    def list_languages(self) -> Set[str]:
        return set(self._allowed)

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "dataset_name": "UBC-NLP/SimbaBench_dataset (asr_test, local parquet)",
            "split": self.split,
            "local_root": self.root,
            "asr_languages": sorted(self._allowed),
            "max_bad_fraction": MAX_BAD_FRACTION,
            "small_n_threshold": SMALL_N,
            "per_language": dict(self._report),
        }
