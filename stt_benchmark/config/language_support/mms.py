# stt_benchmark/config/language_support/mms.py

"""
Meta MMS (Massively Multilingual Speech) language support.

MMS-1B supports 1162 languages for ASR. It uses ISO 639-3 codes.
MMS does NOT support speech translation (AST).

Model variants:
- facebook/mms-1b-all (1162 languages, switchable adapters)
- facebook/mms-1b-fl102 (102 FLEURS languages)
- facebook/mms-1b-l1107 (1107 languages)
"""

from typing import Optional, Set
from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES

# MMS uses ISO 639-3 codes. For FLEURS languages, all are supported by mms-1b-fl102.
# The full mms-1b-all supports 1162 languages via adapter switching.
# We map FLEURS codes to the ISO 639-3 codes MMS expects.

# Special mappings where MMS code differs from FLEURS iso639_3
MMS_CODE_OVERRIDES = {
    "plt": "mlg",   # FLEURS uses iso639_3 'plt' (Plateau Malagasy); MMS uses 'mlg' (macrolanguage)
}


def fleurs_to_mms(fleurs_code: str) -> Optional[str]:
    """Convert FLEURS code to MMS language code (ISO 639-3)."""
    info = FLEURS_LANGUAGES.get(fleurs_code)
    if not info:
        return None
    iso3 = info.get("iso639_3")
    if not iso3:
        return None
    # Apply any overrides
    return MMS_CODE_OVERRIDES.get(iso3, iso3)


def mms_supports_asr(fleurs_code: str) -> bool:
    """Check if MMS supports ASR for a FLEURS language.
    
    mms-1b-fl102 covers all 102 FLEURS languages.
    """
    return fleurs_to_mms(fleurs_code) is not None


def mms_supports_ast(source_fleurs: str, target_fleurs: str) -> bool:
    """MMS does not support speech translation."""
    return False


def get_mms_asr_languages() -> Set[str]:
    """Get FLEURS codes for all languages MMS supports for ASR."""
    supported = set()
    for fleurs_code in FLEURS_LANGUAGES:
        if mms_supports_asr(fleurs_code):
            supported.add(fleurs_code)
    return supported