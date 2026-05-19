# stt_benchmark/config/language_support/speech_aura.py
"""SpeechAura language support — Aura-1B trained on 4 African languages."""

from typing import Optional, Set, Tuple
from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES

# FLEURS code -> SpeechAura's internal language name (matches LANG_MAP in aura.py)
FLEURS_TO_AURA = {
    "ig_ng": "igbo",
    "yo_ng": "yoruba",
    "ha_ng": "hausa",
    "en_us": "english",
}

AST_TARGETS = {"en_us"}  # Only English target supported


def fleurs_to_aura(fleurs_code: str) -> Optional[str]:
    return FLEURS_TO_AURA.get(fleurs_code)


def speech_aura_supports_asr(fleurs_code: str) -> bool:
    return fleurs_code in FLEURS_TO_AURA and fleurs_code != "en_us"


def speech_aura_supports_ast(source_fleurs: str, target_fleurs: str) -> bool:
    if target_fleurs not in AST_TARGETS:
        return False
    return speech_aura_supports_asr(source_fleurs)


def get_speech_aura_asr_languages() -> Set[str]:
    return {c for c in FLEURS_TO_AURA if speech_aura_supports_asr(c)}


def get_speech_aura_ast_pairs() -> Set[Tuple[str, str]]:
    return {(src, "en_us") for src in get_speech_aura_asr_languages()}