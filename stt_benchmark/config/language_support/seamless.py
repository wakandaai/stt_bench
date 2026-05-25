# stt_benchmark/config/language_support/seamless.py

"""
SeamlessM4T v2 language support for speech tasks.

SeamlessM4T supports:
- ASR (S2T same language): ~100 languages
- AST (S2T cross language): speech input from ~100 languages, text output to ~100 languages

Uses ISO 639-3 codes for language identification.
"""

from typing import Optional, Set, Tuple
from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES

# SeamlessM4T speech input languages (languages it can transcribe/translate FROM)
SEAMLESS_SPEECH_INPUT = {
    "afr", "amh", "arb", "ary", "arz", "asm", "ast", "azj", "bel", "ben",
    "bos", "bul", "cat", "ceb", "ces", "ckb", "cmn", "cmn_Hant", "cym",
    "dan", "deu", "ell", "eng", "est", "eus", "fin", "fra", "fuv", "gaz",
    "gle", "glg", "guj", "hau", "heb", "hin", "hrv", "hun", "hye", "ibo",
    "ind", "isl", "ita", "jav", "jpn", "kam", "kan", "kat", "kaz", "kea",
    "khk", "khm", "kir", "kor", "lao", "lin", "lit", "ltz", "lug", "luo",
    "lvs", "mai", "mal", "mar", "mkd", "mlt", "mni", "mya", "nld", "nno",
    "nob", "npi", "nya", "oci", "ory", "pan", "pbt", "pes", "pol", "por",
    "ron", "rus", "slk", "slv", "sna", "snd", "som", "spa", "srp", "swe",
    "swh", "tam", "tel", "tgk", "tgl", "tha", "tur", "ukr", "urd", "uzn",
    "vie", "xho", "yor", "yue", "zlm", "zul",
}

SEAMLESS_TEXT_OUTPUT = {
    "afr", "amh", "arb", "ary", "arz", "asm", "azj", "bel", "ben", "bos",
    "bul", "cat", "ceb", "ces", "ckb", "cmn", "cmn_Hant", "cym", "dan",
    "deu", "ell", "eng", "est", "eus", "fin", "fra", "fuv", "gaz", "gle",
    "glg", "guj", "hau", "heb", "hin", "hrv", "hun", "hye", "ibo", "ind",
    "isl", "ita", "jav", "jpn", "kan", "kat", "kaz", "khk", "khm", "kir",
    "kor", "lao", "lin", "lit", "lug", "luo", "lvs", "mai", "mal", "mar",
    "mkd", "mlt", "mni", "mya", "nld", "nno", "nob", "npi", "nya", "ory",
    "pan", "pbt", "pes", "pol", "por", "ron", "rus", "slk", "slv", "sna",
    "snd", "som", "spa", "srp", "swe", "swh", "tam", "tel", "tgk", "tgl",
    "tha", "tur", "ukr", "urd", "uzn", "vie", "yor", "yue", "zsm", "zul",
}


def fleurs_to_seamless(fleurs_code: str) -> Optional[str]:
    """Convert FLEURS code to SeamlessM4T language code (ISO 639-3)."""
    info = FLEURS_LANGUAGES.get(fleurs_code)
    if not info:
        return None
    return info.get("iso639_3")


def seamless_supports_asr(fleurs_code: str) -> bool:
    """Check if SeamlessM4T supports ASR for a FLEURS language."""
    iso3 = fleurs_to_seamless(fleurs_code)
    if not iso3:
        return False
    return iso3 in SEAMLESS_SPEECH_INPUT and iso3 in SEAMLESS_TEXT_OUTPUT


def seamless_supports_ast(source_fleurs: str, target_fleurs: str) -> bool:
    """Check if SeamlessM4T supports AST for a language pair."""
    src_iso3 = fleurs_to_seamless(source_fleurs)
    tgt_iso3 = fleurs_to_seamless(target_fleurs)
    if not src_iso3 or not tgt_iso3:
        return False
    return src_iso3 in SEAMLESS_SPEECH_INPUT and tgt_iso3 in SEAMLESS_TEXT_OUTPUT


def get_seamless_asr_languages() -> Set[str]:
    """Get FLEURS codes for all languages SeamlessM4T supports for ASR."""
    supported = set()
    for fleurs_code in FLEURS_LANGUAGES:
        if seamless_supports_asr(fleurs_code):
            supported.add(fleurs_code)
    return supported


def get_seamless_ast_pairs(anchor_targets: Set[str] = None) -> Set[Tuple[str, str]]:
    """Get supported AST pairs.

    Args:
        anchor_targets: If provided, only return pairs targeting these languages.
                       Defaults to all FLEURS languages that Seamless can output.
    """
    if anchor_targets is None:
        anchor_targets = set(FLEURS_LANGUAGES.keys())

    pairs = set()
    for src_fleurs in FLEURS_LANGUAGES:
        for tgt_fleurs in anchor_targets:
            if src_fleurs != tgt_fleurs and seamless_supports_ast(src_fleurs, tgt_fleurs):
                pairs.add((src_fleurs, tgt_fleurs))
    return pairs