# stt_benchmark/config/language_support/speech_aura.py
"""SpeechAura language support — Aura-1B trained on 23 languages."""

from typing import Optional, Set, Tuple
from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES

# FLEURS code -> SpeechAura's internal language name (matches LANG_MAP in aura.py).
# Training set: 23 languages from configs/experiment/stage1.yaml (Bemba excluded —
# not in FLEURS). Tigrinya and Kinyarwanda are in the training set but not in FLEURS
# either; they're listed in comments below for reference but cannot be evaluated here.
FLEURS_TO_AURA = {
    "af_za": "afrikaans",
    "am_et": "amharic",
    "ar_eg": "arabic",
    # bemba         — trained, not in FLEURS
    "en_us": "english",
    "fr_fr": "french",
    "ha_ng": "hausa",
    "ig_ng": "igbo",
    # rw_rw         — Kinyarwanda trained, not in FLEURS
    "ln_cd": "lingala",
    "lg_ug": "luganda",
    "mg_mg": "malagasy",
    "pt_br": "portuguese",
    "sn_zw": "shona",
    "so_so": "somali",
    # "nso_za": "sotho",   # FLEURS has Northern Sotho (nso); training "sotho" maps here
    "sw_ke": "swahili",
    # tigrinya      — trained, not in FLEURS
    # tswana        — trained, not in FLEURS (no tn_za in FLEURS)
    "wo_sn": "wolof",
    "xh_za": "xhosa",
    "yo_ng": "yoruba",
    "zu_za": "zulu",
}

AST_TARGETS = {"en_us"}  # Only English target supported


def fleurs_to_aura(fleurs_code: str) -> Optional[str]:
    return FLEURS_TO_AURA.get(fleurs_code)


def speech_aura_supports_asr(fleurs_code: str) -> bool:
    return fleurs_code in FLEURS_TO_AURA


def speech_aura_supports_ast(source_fleurs: str, target_fleurs: str) -> bool:
    if target_fleurs not in AST_TARGETS:
        return False
    return speech_aura_supports_asr(source_fleurs)


def get_speech_aura_asr_languages() -> Set[str]:
    return {c for c in FLEURS_TO_AURA if speech_aura_supports_asr(c)}


def get_speech_aura_ast_pairs() -> Set[Tuple[str, str]]:
    return {(src, "en_us") for src in get_speech_aura_asr_languages()}