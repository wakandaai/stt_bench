#!/usr/bin/env python3
"""
Shared helpers for the Aura-ASR error-analysis pipeline (see analysis/README).

This module is the single source of truth for:
  * which prediction CSVs enter the analysis (ASR-only include/exclude filter);
  * how (model, language, benchmark) is parsed from a file path;
  * the ONE normalization scheme applied identically to every system;
  * Tier A provenance/pathology detection (A1 script / A2 loop / A3 truncation);
  * the language -> typology mapping used for the grouped rollups.

Nothing here does inference. Everything is deterministic. Thresholds are module
level constants so they can be swept and reported (guardrail: state thresholds).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from stt_benchmark.utils.text_normalize import TextNormalizer


# =========================================================================
# The one normalization scheme (applied identically to ALL systems).
# NFC -> lowercase -> strip punctuation (keep intra-word ' and -) -> collapse ws.
# Tone marks / diacritics are PRESERVED here (NFC composes, never strips).
# =========================================================================

NORMALIZER = TextNormalizer(
    lowercase=True,
    remove_punctuation=True,
    unicode_form="NFC",
    collapse_whitespace=True,
)


# =========================================================================
# Tunable thresholds (heuristics; report them, validate against §6).
# =========================================================================

LOOP_LEN_RATIO = 3.0     # A2: hyp at least ~3x longer than ref ...
LOOP_REP_RATE = 0.50     # ... AND >=50% of its 5-grams are duplicates
LOOP_NGRAM = 5
TRUNC_LEN_RATIO = 0.50   # A3: hyp <= half the ref length (near-empty/collapsed)
SEG_CER_MAX = 10.0       # pct_seg_dominated: characters ~right (CER < 10) ...
SEG_WER_MIN = 30.0       # ... but words wrong (WER > 30)


# =========================================================================
# File inclusion / parsing
# =========================================================================

# A file is ASR iff its stem carries the transcription task marker.
_ASR_MARKER = "_asr_"
# Exclude anything that is a translation output. All AST/translation files in
# this tree carry one of these tokens; ASR of English/French/Portuguese (the
# Indo-European anchor points for the WER/CER ratio) is intentionally KEPT.
_EXCLUDE_TOKENS = ("ast", "translate", "cascaded")

# Raw model id (left of _asr_) -> short display name used in every output.
MODEL_DISPLAY: Dict[str, str] = {
    "ctc_encoder": "Conformer-CTC",
    "facebook_mms-1b-all": "MMS-1B",
    "facebook_mms-1b-fl102": "MMS-1B-fl102",
    "omniASR_LLM_1B": "OmniASR-LLM-1B",
    "speech_aura_transcribe": "Aura-ASR",
    "facebook_seamless-m4t-v2-large": "SeamlessM4T-v2",
}

# Whether a model uses an LM decoder (drives the B4->B5 contrast reading).
MODEL_HAS_LM: Dict[str, bool] = {
    "Conformer-CTC": False,   # no LM  -> expected B4 (phonetic) dominant
    "MMS-1B": False,          # CTC head, no strong LM
    "MMS-1B-fl102": False,
    "OmniASR-LLM-1B": True,   # English-centric LLM decoder
    "Aura-ASR": True,         # African-text LLM decoder (ours)
    "SeamlessM4T-v2": True,   # seq2seq w/ implicit LM
}


def is_asr_prediction(path: Path) -> bool:
    """True iff `path` is an ASR-mode transcription CSV we should include."""
    stem = path.stem
    if _ASR_MARKER not in stem:
        return False
    return not any(tok in stem for tok in _EXCLUDE_TOKENS)


def parse_path(path: Path) -> Tuple[str, str, str]:
    """Return (benchmark, model_display, lang) for a predictions CSV.

    Layout: .../<benchmark>/predictions/<model_id>_asr_<lang>.csv
    """
    benchmark = path.parent.parent.name
    stem = path.stem
    idx = stem.rfind(_ASR_MARKER)
    model_id = stem[:idx]
    lang = stem[idx + len(_ASR_MARKER):]
    model = MODEL_DISPLAY.get(model_id, model_id)
    return benchmark, model, lang


def collect_prediction_files(root: Path) -> Tuple[List[Path], List[Path]]:
    """Return (included, excluded) prediction CSVs under <root>/**/predictions/."""
    all_csv = sorted(root.rglob("predictions/*.csv"))
    included = [p for p in all_csv if is_asr_prediction(p)]
    excluded = [p for p in all_csv if not is_asr_prediction(p)]
    return included, excluded


# =========================================================================
# Typology mapping (grouped rollup: Indo-European / tonal / Bantu-agglutinative
# / own-script / other). `ortho` captures the Bantu conjunctive-vs-disjunctive
# split that drives the boundary (B1) story. NOTE: several of these are judgment
# calls (Hausa, Malagasy, Shona tone) -- documented and adjustable.
# =========================================================================

@dataclass(frozen=True)
class LangInfo:
    name: str
    typology: str          # indo_european | tonal | bantu_agglutinative | own_script | other
    ortho: str             # conjunctive | disjunctive | agglutinative | none
    note: str = ""


LANG_INFO: Dict[str, LangInfo] = {
    # Indo-European anchors (expected WER/CER ratio ~2.0)
    "en_us": LangInfo("English", "indo_european", "none"),
    "fr_fr": LangInfo("French", "indo_european", "none"),
    "pt_br": LangInfo("Portuguese", "indo_european", "none"),
    # Own-script Semitic
    "am_et": LangInfo("Amharic", "own_script", "none", "Ethiopic script"),
    "ti_et": LangInfo("Tigrinya", "own_script", "none", "Ethiopic script"),
    "ar_eg": LangInfo("Arabic", "own_script", "none", "known routing-bug cell"),
    # West-African tonal (Latin + diacritics)
    "yo_ng": LangInfo("Yoruba", "tonal", "agglutinative", "tone + sub-dot diacritics"),
    "ig_ng": LangInfo("Igbo", "tonal", "agglutinative", "tone diacritics"),
    # Bantu agglutinative
    "bem":   LangInfo("Bemba", "bantu_agglutinative", "agglutinative"),
    "lg_ug": LangInfo("Luganda", "bantu_agglutinative", "agglutinative"),
    "ln_cd": LangInfo("Lingala", "bantu_agglutinative", "agglutinative"),
    "sn_zw": LangInfo("Shona", "bantu_agglutinative", "agglutinative", "tonal, tone usually unmarked"),
    "sw_ke": LangInfo("Swahili", "bantu_agglutinative", "agglutinative"),
    "rw_rw": LangInfo("Kinyarwanda", "bantu_agglutinative", "agglutinative", "hallucination failure case"),
    "xh_za": LangInfo("Xhosa", "bantu_agglutinative", "conjunctive", "Nguni conjunctive"),
    "zu_za": LangInfo("Zulu", "bantu_agglutinative", "conjunctive", "Nguni conjunctive"),
    "st_za": LangInfo("Sesotho", "bantu_agglutinative", "disjunctive", "Sotho-Tswana disjunctive"),
    "tn_za": LangInfo("Setswana", "bantu_agglutinative", "disjunctive", "Sotho-Tswana disjunctive"),
    # Other
    "ha_ng": LangInfo("Hausa", "other", "none", "Chadic/Afro-Asiatic, Latin, tone unmarked"),
    "mg_mg": LangInfo("Malagasy", "other", "agglutinative", "Austronesian, not Bantu"),
}


def lang_info(lang: str) -> LangInfo:
    return LANG_INFO.get(lang, LangInfo(lang, "unknown", "none", "unmapped language"))


# =========================================================================
# Tier A provenance / pathology detection (per utterance)
# =========================================================================

def _script_of_char(ch: str) -> Optional[str]:
    """Coarse Unicode script name from the character's Unicode name prefix."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    return name.split(" ")[0]  # LATIN / ARABIC / ETHIOPIC / CYRILLIC / ...


def dominant_script(s: str) -> Optional[str]:
    """Most common script among alphabetic characters, or None if none."""
    counts: Dict[str, int] = {}
    for ch in s:
        if not ch.isalpha():
            continue
        sc = _script_of_char(ch)
        if sc:
            counts[sc] = counts.get(sc, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def ngram_repeat_rate(tokens: List[str], n: int = LOOP_NGRAM) -> float:
    """Fraction of n-grams that are duplicates (0 = all unique, ->1 = looping)."""
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def tier_a_label(norm_ref: str, norm_hyp: str) -> str:
    """Assign the first-matching Tier A provenance label (A1 > A2 > A3 > CLEAN).

    Rows labelled A1-A3 are pathologies and are EXCLUDED from Tier B linguistic
    stats so they cannot contaminate category proportions.
    """
    ref_words = norm_ref.split()
    hyp_words = norm_hyp.split()
    n_ref = len(ref_words)
    len_ratio = (len(hyp_words) / n_ref) if n_ref else 0.0

    # A1 - wrong script/language: dominant script of hyp differs from ref.
    sr = dominant_script(norm_ref)
    sh = dominant_script(norm_hyp)
    if sr and sh and sr != sh:
        return "A1_script"

    # A2 - degeneration loop: much longer AND highly repetitive.
    if len_ratio >= LOOP_LEN_RATIO and ngram_repeat_rate(hyp_words) >= LOOP_REP_RATE:
        return "A2_loop"

    # A3 - near-empty / collapsed output.
    if len_ratio <= TRUNC_LEN_RATIO:
        return "A3_truncation"

    return "CLEAN"


# =========================================================================
# Diacritic / tone handling (B2 proxy support)
# =========================================================================

def strip_diacritics(s: str) -> str:
    """Remove combining marks (tone marks, accents, sub-dots) via NFD decomposition.

    Yoruba/Igbo tone + sub-dot diacritics and Latin accents are dropped; the base
    segmental string is kept. Used to estimate the tone/diacritic error share
    (B2) as the WER drop when both sides are stripped.
    """
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in nfd if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped)


_WS = re.compile(r"\s+")


def remove_spaces(s: str) -> str:
    """Collapse all whitespace out (for spaceless CER / boundary proxy)."""
    return _WS.sub("", s)
