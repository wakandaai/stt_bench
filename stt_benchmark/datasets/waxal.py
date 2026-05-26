# stt_benchmark/datasets/waxal.py

"""
Waxal (google/WaxalNLP) loader — ASR only.

HuggingFace-backed multi-language ASR corpus. Each language is exposed as a
separate HF config (`amh_asr`, `lin_asr`, `mlg_asr`, …), but loading via the
config path resolves ALL four splits (train, validation, test, unlabeled)
and downloads every shard regardless of what `split=` is passed —
prohibitively expensive when you only want the test shards.

We work around this by loading the parquet files directly via the `parquet`
builder + an `hf://datasets/google/WaxalNLP/...` glob pointed at just the
requested split's shards:

    hf://datasets/google/WaxalNLP/data/ASR/<iso639_3>/<iso639_3>-<split>-*.parquet

This path layout is declared in the dataset README's `configs:` block. HF
lists the matching files and downloads only those.

Per-row schema:
    id            : str
    speaker_id    : str
    audio         : {array: np.ndarray, sampling_rate: int}
    transcription : str
    language      : str (iso639-3)
    gender        : "Male" | "Female"

ASR-only. No translation column, no parallel data. Six languages supported in
our benchmark — the four that already have FLEURS coverage (amh, lin, lug,
sna) plus the two FLEURS doesn't cover (mlg, tir).
"""

from typing import List, Dict, Any, Set, Optional, Tuple
import numpy as np
from datasets import load_dataset

from stt_benchmark.datasets.base import (
    BaseASRDataset, BaseASTDataset, AudioSample, ParallelAudioSample,
)

HF_DATASET_NAME = "google/WaxalNLP"

# FLEURS-style code -> Waxal iso639-3 prefix (used to build parquet paths).
# - amh, lin, lug, sna: already in FLEURS; Waxal is a second ASR source.
# - mlg: FLEURS `mg_mg` uses iso639-3 `plt` (Plateau Malagasy); Waxal uses
#   the macrolanguage code `mlg`. Same standard written language.
# - tir: not in official FLEURS; we add `ti_et` to the FLEURS catalog as a
#   community entry so it can be used here and in language_support mappings.
FLEURS_TO_WAXAL_ISO3 = {
    "am_et": "amh",
    "ln_cd": "lin",
    "lg_ug": "lug",
    "mg_mg": "mlg",
    "sn_zw": "sna",
    "ti_et": "tir",
}


def _parquet_glob(iso3: str, split: str) -> str:
    """Build the hf:// parquet file glob for one (language, split) pair.

    Matches the path declared in Waxal's README configs block:
        data/ASR/<iso3>/<iso3>-<split>-*.parquet

    Using an `hf://datasets/<repo>/...` URL bypasses the dataset config path
    entirely, so HF resolves just this glob against the repo's file listing
    and downloads only matching shards (not all four splits' shards).
    """
    return f"hf://datasets/{HF_DATASET_NAME}/data/ASR/{iso3}/{iso3}-{split}-*.parquet"


class WaxalDataset(BaseASRDataset, BaseASTDataset):
    """Waxal (google/WaxalNLP) loader — ASR-only, HF-backed via direct parquet load."""

    def __init__(
        self,
        split: str = "test",
        languages: Optional[List[str]] = None,
        ast_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """Initialize the loader.

        Args:
            split: 'train', 'validation', 'test', or 'unlabeled'. Defaults to
                'test' for benchmarking.
            languages: Allowed ASR languages (FLEURS codes). Filtered against
                the set Waxal supports.
            ast_pairs: Always filtered to the empty set — Waxal is ASR-only.
        """
        self.split = split

        supported = set(FLEURS_TO_WAXAL_ISO3.keys())
        if languages is not None:
            self._allowed_asr = set(languages) & supported
        else:
            self._allowed_asr = supported

        self._allowed_pairs: Set[Tuple[str, str]] = set()
        self._mono_cache: Dict[str, List[AudioSample]] = {}

    # ------------------------------------------------------------------
    # HF loading — direct parquet path to avoid pulling all splits
    # ------------------------------------------------------------------

    def _load_hf_split(self, fleurs_code: str):
        """Load one Waxal language's split via direct parquet glob.

        Uses the `parquet` builder + an `hf://datasets/...` URL pointed at
        just this language's split shards. HF lists the matching files and
        downloads only those — no train/validation/unlabeled shards touched.

        The parquet builder yields the audio column as raw {path, bytes}
        rather than as a decoded {array, sampling_rate} dict, so we cast
        it to the `Audio` feature to trigger HF's standard audio decoding.
        """
        from datasets import Audio
        iso3 = FLEURS_TO_WAXAL_ISO3[fleurs_code]
        pattern = _parquet_glob(iso3, self.split)
        ds = load_dataset(
            "parquet",
            data_files={self.split: pattern},
            split=self.split,
        )
        # Decode audio to {array, sampling_rate} at 16 kHz (the rate Waxal
        # ships at and what all downstream ASR models expect).
        return ds.cast_column("audio", Audio(sampling_rate=16_000))

    def _load_mono(self, fleurs_code: str) -> List[AudioSample]:
        if fleurs_code in self._mono_cache:
            return self._mono_cache[fleurs_code]

        ds = self._load_hf_split(fleurs_code)

        samples: List[AudioSample] = []
        skipped = 0
        for i, row in enumerate(ds):
            try:
                audio = row["audio"]
                transcription = (row.get("transcription") or "").strip()
                if not transcription:
                    skipped += 1
                    continue
                row_id = row.get("id") or f"{fleurs_code}_{i}"
                samples.append(AudioSample(
                    transcription=transcription,
                    language=fleurs_code,
                    sample_id=f"waxal_{fleurs_code}_{row_id}",
                    audio_array=np.asarray(audio["array"], dtype=np.float32),
                    sampling_rate=int(audio["sampling_rate"]),
                ))
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(
                        f"  [waxal/{fleurs_code}] skipping row "
                        f"id={row.get('id', '?')}: {e}"
                    )

        if skipped:
            print(
                f"  [waxal/{fleurs_code}] loaded {len(samples)} samples, "
                f"skipped {skipped} (empty transcription or undecodable)"
            )

        self._mono_cache[fleurs_code] = samples
        return samples

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def get_language_samples(self, language: str) -> List[AudioSample]:
        if language not in FLEURS_TO_WAXAL_ISO3:
            raise ValueError(
                f"waxal does not support language '{language}'. "
                f"Supported: {sorted(FLEURS_TO_WAXAL_ISO3.keys())}"
            )
        if language not in self._allowed_asr:
            raise ValueError(
                f"Language '{language}' not in allowed set: "
                f"{sorted(self._allowed_asr)}"
            )
        return self._load_mono(language)

    def list_languages(self) -> Set[str]:
        return set(self._allowed_asr)

    def get_dataset_info(self) -> Dict[str, Any]:
        cached_counts = {lang: len(s) for lang, s in self._mono_cache.items()}
        return {
            "dataset_name": "google/WaxalNLP (HF, direct parquet)",
            "split": self.split,
            "asr_languages": sorted(self._allowed_asr),
            "ast_pairs": "(ASR-only; no AST support)",
            "cached_mono_counts": cached_counts,
        }

    # ------------------------------------------------------------------
    # AST — always empty for Waxal
    # ------------------------------------------------------------------

    def get_parallel_samples(
        self, source_lang: str, target_lang: str
    ) -> List[ParallelAudioSample]:
        raise ValueError(
            f"waxal does not support AST (pure ASR corpus); "
            f"requested {source_lang}->{target_lang}"
        )

    def list_language_pairs(self) -> Set[tuple]:
        return set()