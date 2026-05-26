# stt_benchmark/config/language_support/speech_aura.py
"""SpeechAura language support — Aura-1B trained on 20 languages."""

from typing import Optional, Set, Tuple
from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES

# FLEURS code -> SpeechAura's internal language name (matches LANG_MAP in aura.py).
# Training set: 20 languages from configs/experiment/stage1.yaml.
FLEURS_TO_AURA = {
    "am_et": "amharic",
    "ar_eg": "arabic",
    "bem": "bemba",            # via bembaspeech and bigc (not official FLEURS)
    "en_us": "english",
    "fr_fr": "french",
    "ha_ng": "hausa",
    "ig_ng": "igbo",
    "rw_rw": "kinyarwanda",    # via mbaza_fleurs_rw (not official FLEURS)
    "ln_cd": "lingala",
    "lg_ug": "luganda",
    "mg_mg": "malagasy",       # via waxal (not official FLEURS)
    "pt_br": "portuguese",
    "sn_zw": "shona",
    "st_za": "sotho",          # Sesotho / Southern Sotho — via nchlt (not in FLEURS).
                               # Distinct from Northern Sotho (nso_za).
    "sw_ke": "swahili",
    "ti_et": "tigrinya",       # via waxal (not official FLEURS)
    "tn_za": "tswana",         # Setswana — via nchlt (not official FLEURS)
    "xh_za": "xhosa",
    "yo_ng": "yoruba",
    "zu_za": "zulu",
}

AST_TARGETS = {
    "am_et": "amharic",
    "ar_eg": "arabic",
    "bem": "bemba",            # via bembaspeech and bigc (not official FLEURS)
    "en_us": "english",
    "fr_fr": "french",
    "ha_ng": "hausa",
    "ig_ng": "igbo",
    "rw_rw": "kinyarwanda",    # via mbaza_fleurs_rw (not official FLEURS)
    "ln_cd": "lingala",
    "lg_ug": "luganda",
    "mg_mg": "malagasy",       # via waxal (not official FLEURS)
    "pt_br": "portuguese",
    "sn_zw": "shona",
    "st_za": "sotho",          # Sesotho / Southern Sotho — via nchlt (not in FLEURS).
                               # Distinct from Northern Sotho (nso_za).
    "sw_ke": "swahili",
    "ti_et": "tigrinya",       # via waxal (not official FLEURS)
    "tn_za": "tswana",         # Setswana — via nchlt (not official FLEURS)
    "xh_za": "xhosa",
    "yo_ng": "yoruba",
    "zu_za": "zulu",
}


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
    return {(c, t) for c in FLEURS_TO_AURA for t in AST_TARGETS if speech_aura_supports_ast(c, t)}