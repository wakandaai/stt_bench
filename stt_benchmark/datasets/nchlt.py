# stt_benchmark/datasets/nchlt.py

"""
NCHLT (Lwazi II / South African Centre for Digital Language Resources) loader.

Read-speech corpus covering 5 South African languages: Afrikaans, Sesotho,
Tswana, Xhosa, Zulu. Test-set only — this is a benchmarking corpus.

Expected on-disk layout: `local_path` points at the NCHLT root containing
per-language subdirectories:

    <local_path>/
    ├── nchlt_afr/
    │   ├── audio/<speaker_id>/<utt>.wav
    │   └── transcriptions/
    │       └── nchlt_afr.tst.xml
    ├── nchlt_sot/...
    ├── nchlt_tsn/...
    ├── nchlt_xho/...
    └── nchlt_zul/...

XML schema:
    <corpus name="nchlt sot">
      <speaker id="500" age="18" gender="female" location="...">
        <recording audio="nchlt_sot/audio/500/..." duration="..." ...>
          <orth>boholo ba masimo a hao [s]</orth>
        </recording>
        ...

The `audio` attribute is relative to `local_path` (the NCHLT root).
The `<orth>` text contains `[s]` markers for non-speech regions; we strip
these so they don't pollute WER scores.

ASR-only. No translation data.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
from xml.etree import ElementTree as ET

from stt_benchmark.datasets.base import (
    BaseASRDataset, BaseASTDataset, AudioSample, ParallelAudioSample,
)


# Map our FLEURS-style codes <-> NCHLT's ISO-639-3 directory codes.
FLEURS_TO_NCHLT_ISO3 = {
    "af_za": "afr",
    "st_za": "sot",
    "tn_za": "tsn",
    "xh_za": "xho",
    "zu_za": "zul",
}
NCHLT_ISO3_TO_FLEURS = {v: k for k, v in FLEURS_TO_NCHLT_ISO3.items()}

# Strip `[s]` silence markers from references. Only bracketed token in the
# corpus per `grep -ohE '\[[^]]*\]' *.tst.xml`.
SILENCE_MARKER_RE = re.compile(r"\[s\]")
WHITESPACE_RE = re.compile(r"\s+")


def _clean_orth(text: str) -> str:
    """Remove `[s]` markers and normalize whitespace."""
    if text is None:
        return ""
    text = SILENCE_MARKER_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


class NchltDataset(BaseASRDataset, BaseASTDataset):
    """NCHLT loader — multi-language, ASR-only, test-set only."""

    def __init__(
        self,
        local_path: str,
        split: str = "test",
        languages: Optional[List[str]] = None,
        ast_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """Initialize the dataset.

        Args:
            local_path: Filesystem path to the NCHLT root directory.
            split: Only 'test' is supported. NCHLT has train+test on disk but
                this is a benchmarking corpus so we only expose test.
            languages: Allowed ASR languages. Filtered against the 5 NCHLT
                languages and against what's actually present on disk.
            ast_pairs: Filtered to empty set — NCHLT is ASR-only.
        """
        if split != "test":
            raise ValueError(
                f"NCHLT loader only supports split='test', got '{split}'. "
                f"This is a benchmarking corpus; train data is not exposed."
            )

        self.local_path = Path(local_path).resolve()
        if not self.local_path.is_dir():
            raise FileNotFoundError(
                f"local_path does not exist: {self.local_path}\n"
                f"Point this at the NCHLT root (containing nchlt_afr/, nchlt_sot/, etc.)."
            )

        self.split = split

        # Discover which languages are actually present on disk.
        present_iso3: Set[str] = set()
        for iso3 in FLEURS_TO_NCHLT_ISO3.values():
            lang_dir = self.local_path / f"nchlt_{iso3}"
            xml_path = lang_dir / "transcriptions" / f"nchlt_{iso3}.tst.xml"
            if lang_dir.is_dir() and xml_path.exists():
                present_iso3.add(iso3)

        if not present_iso3:
            raise FileNotFoundError(
                f"No NCHLT language directories found under {self.local_path}.\n"
                f"Expected one or more of: {sorted(FLEURS_TO_NCHLT_ISO3.values())}"
            )

        present_fleurs = {NCHLT_ISO3_TO_FLEURS[i] for i in present_iso3}

        # Scope ASR to (requested ∩ present). Anything requested but absent
        # gets reported at pre-flight validation as "not in dataset".
        if languages is not None:
            self._allowed_asr = set(languages) & present_fleurs
        else:
            self._allowed_asr = present_fleurs

        # NCHLT is ASR-only. AST pairs filter to empty.
        self._allowed_pairs: Set[Tuple[str, str]] = set()

        # Caches
        self._mono_cache: Dict[str, List[AudioSample]] = {}

    # ------------------------------------------------------------------
    # XML parsing
    # ------------------------------------------------------------------

    def _xml_path(self, fleurs_code: str) -> Path:
        iso3 = FLEURS_TO_NCHLT_ISO3[fleurs_code]
        return self.local_path / f"nchlt_{iso3}" / "transcriptions" / f"nchlt_{iso3}.tst.xml"

    def _parse_xml(self, fleurs_code: str) -> List[Dict[str, str]]:
        """Parse one language's test XML into a list of {audio_path, transcription}."""
        xml_path = self._xml_path(fleurs_code)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        rows: List[Dict[str, str]] = []
        for recording in root.iter("recording"):
            audio_rel = recording.get("audio")
            if not audio_rel:
                continue
            orth_elem = recording.find("orth")
            if orth_elem is None:
                continue
            transcription = _clean_orth(orth_elem.text)
            if not transcription:
                # Could happen if the orth was entirely `[s]` — skip.
                continue
            rows.append({
                "audio_rel": audio_rel,
                "transcription": transcription,
            })
        return rows

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def _load_mono(self, fleurs_code: str) -> List[AudioSample]:
        if fleurs_code in self._mono_cache:
            return self._mono_cache[fleurs_code]

        rows = self._parse_xml(fleurs_code)
        samples: List[AudioSample] = []
        skipped_missing = 0

        for i, row in enumerate(rows):
            audio_path = self.local_path / row["audio_rel"]
            if not audio_path.exists():
                skipped_missing += 1
                if skipped_missing <= 3:
                    print(f"  [nchlt/{fleurs_code}] missing audio: {audio_path}")
                continue
            samples.append(AudioSample(
                transcription=row["transcription"],
                language=fleurs_code,
                sample_id=f"nchlt_{fleurs_code}_{i}_{Path(row['audio_rel']).stem}",
                audio_path=str(audio_path),
            ))

        if skipped_missing:
            print(
                f"  [nchlt/{fleurs_code}] loaded {len(samples)} samples, "
                f"skipped {skipped_missing} missing audio"
            )

        self._mono_cache[fleurs_code] = samples
        return samples

    def get_language_samples(self, language: str) -> List[AudioSample]:
        if language not in FLEURS_TO_NCHLT_ISO3:
            raise ValueError(
                f"nchlt does not support language '{language}'. "
                f"Supported: {sorted(FLEURS_TO_NCHLT_ISO3.keys())}"
            )
        if language not in self._allowed_asr:
            raise ValueError(
                f"Language '{language}' not in allowed set "
                f"(either filtered by config or absent on disk): "
                f"{sorted(self._allowed_asr)}"
            )
        return self._load_mono(language)

    def list_languages(self) -> Set[str]:
        return set(self._allowed_asr)

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "dataset_name": "NCHLT",
            "local_path": str(self.local_path),
            "split": self.split,
            "asr_languages": sorted(self._allowed_asr),
            "ast_pairs": "(ASR-only; no AST support)",
        }

    # ------------------------------------------------------------------
    # AST — always empty for NCHLT
    # ------------------------------------------------------------------

    def get_parallel_samples(
        self, source_lang: str, target_lang: str
    ) -> List[ParallelAudioSample]:
        raise ValueError(
            f"nchlt does not support AST (pure ASR corpus); "
            f"requested {source_lang}->{target_lang}"
        )

    def list_language_pairs(self) -> Set[tuple]:
        return set()