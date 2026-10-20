# stt_benchmark/datasets/cleaned_eval.py

"""
Loader for the audited + cleaned evaluation groups (scripts/assemble_eval_manifests.py).

One dataset name = ONE group (one source x one language), e.g. `cleaned__simbabench__afr__Lwazi`, so results
of different sources are never mixed. The name is resolved by `registry.create_dataset` (prefix `cleaned__`).

What this loader guarantees (it raises, it never skips, substitutes or falls back):
  * the group exists in groups.csv and its manifest has exactly the approved number of rows;
  * sample ids are unique and every transcript is non-empty;
  * every audio file exists and is 16 kHz, mono, frames > 0 (wav: header; parquet: decoded);
  * for SimbaBench (audio embedded in parquet) every manifest id is found in the parquet, and the stored raw
    transcript equals the manifest's raw transcript (guards the id -> text join);
  * the requested language equals the group's language.
References are the CLEANED transcripts (NFC, annotation tags removed, whitespace collapsed). Scoring-time text
normalisation is the scorer's job. The raw transcript stays in the manifest (`transcript_raw`).
"""

import csv
import io
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np

from stt_benchmark.datasets.base import BaseASRDataset, AudioSample

SR = 16_000


def _read_tsv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


class CleanedEvalDataset(BaseASRDataset):
    """One audited, cleaned (source, language) group. ASR only, test split only."""

    def __init__(
        self,
        manifest_dir: str,
        group: str,
        split: str = "test",
        languages: Optional[List[str]] = None,
    ):
        if split != "test":
            raise ValueError(f"cleaned eval groups are test-only; got split='{split}'")
        groups = {g["group_key"]: g for g in _read_tsv_csv(os.path.join(manifest_dir, "groups.csv"))}
        if group not in groups:
            raise ValueError(f"unknown group '{group}'. {len(groups)} groups available; e.g. {sorted(groups)[:3]}")
        self.manifest_dir = manifest_dir
        self.group = group
        self.info = groups[group]
        self.code = self.info["language_code"]
        if languages is not None and set(languages) != {self.code}:
            raise ValueError(f"group {group} is language '{self.code}', config asked for {languages}")
        self.split = split
        self._cache: Optional[List[AudioSample]] = None

    # ------------------------------------------------------------------
    def _load(self) -> List[AudioSample]:
        if self._cache is not None:
            return self._cache
        rows = _read_tsv(os.path.join(self.manifest_dir, self.info["manifest_file"]))
        want = int(self.info["n_clean"])
        if len(rows) != want:
            raise ValueError(f"{self.group}: manifest has {len(rows)} rows, expected {want}")
        ids = [r["sample_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.group}: duplicate sample ids")
        if any(not r["transcript"].strip() for r in rows):
            raise ValueError(f"{self.group}: empty transcript in manifest")
        kind = {r["audio_kind"] for r in rows}
        if len(kind) != 1:
            raise ValueError(f"{self.group}: mixed audio kinds {kind}")
        samples = self._load_parquet(rows) if kind == {"parquet"} else self._load_wav(rows)
        self._cache = samples
        return samples

    def _load_wav(self, rows: List[Dict[str, str]]) -> List[AudioSample]:
        import soundfile as sf
        out = []
        for r in rows:
            p = r["audio_path"]
            if not os.path.isfile(p):
                raise FileNotFoundError(f"{self.group}: missing audio {p}")
            info = sf.info(p)
            if info.samplerate != SR or info.channels != 1 or info.frames <= 0:
                raise ValueError(f"{self.group}: {p} is {info.samplerate} Hz / {info.channels} ch / {info.frames} frames")
            out.append(AudioSample(transcription=r["transcript"], language=self.code, sample_id=r["sample_id"],
                                   audio_path=p, sampling_rate=SR))
        return out

    def _load_parquet(self, rows: List[Dict[str, str]]) -> List[AudioSample]:
        import pyarrow.parquet as pq
        import soundfile as sf
        pq_paths = {r["audio_path"] for r in rows}
        if len(pq_paths) != 1:
            raise ValueError(f"{self.group}: expected one parquet file, got {pq_paths}")
        path = pq_paths.pop()
        need: Dict[str, Dict[str, str]] = {r["parquet_id"]: r for r in rows}
        found: Dict[str, AudioSample] = {}
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                bid = str(row["benchmark_id"])
                if bid not in need:
                    continue
                m = need[bid]
                stored = (row.get("text") or "").replace("\t", " ").replace("\n", "\\n")
                if stored != m["transcript_raw"]:
                    raise ValueError(f"{self.group}: id {bid} raw transcript differs from manifest (join broken)")
                data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
                if data.ndim != 1 or int(sr) != SR or data.size == 0 or not np.isfinite(data).all():
                    raise ValueError(f"{self.group}: id {bid} audio is not finite 16 kHz mono")
                found[bid] = AudioSample(transcription=m["transcript"], language=self.code,
                                         sample_id=m["sample_id"], audio_array=np.ascontiguousarray(data),
                                         sampling_rate=SR)
        missing = set(need) - set(found)
        if missing:
            raise ValueError(f"{self.group}: {len(missing)} manifest ids not found in {os.path.basename(path)}")
        return [found[r["parquet_id"]] for r in rows]          # manifest order

    # ------------------------------------------------------------------
    def get_language_samples(self, language: str) -> List[AudioSample]:
        if language != self.code:
            raise ValueError(f"group {self.group} has only language '{self.code}', requested '{language}'")
        return self._load()

    def list_languages(self) -> Set[str]:
        return {self.code}

    def get_dataset_info(self) -> Dict[str, Any]:
        return {"dataset_name": self.group, "config_name": self.info["config_name"], "split": self.split,
                "language": self.code, "n_clean": int(self.info["n_clean"]),
                "hours_clean": float(self.info["hours_clean"]), "audio_kind": self.info["audio_kind"],
                "manifest_dir": self.manifest_dir}


def _read_tsv_csv(path: str) -> List[Dict[str, str]]:
    """groups.csv is a comma-separated file (unlike the per-group tab-separated manifests)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
