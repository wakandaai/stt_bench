# stt_benchmark/config/language_support/omni_asr.py

"""
Meta Omnilingual ASR language support.

Omni ASR supports 1600+ languages using {iso639_3}_{script} codes,
e.g. 'eng_Latn', 'swh_Latn', 'cmn_Hans'. The same shape as NLLB codes.

ASR-only. The package ships its authoritative language list in
`omnilingual_asr.models.wav2vec2_llama.lang_ids.supported_langs`, which we
intersect with our FLEURS catalog at import time. Unsupported FLEURS entries
silently drop out.

Model variants (HuggingFace model cards passed to ASRInferencePipeline):
- omniASR_LLM_1B / omniASR_LLM_3B / omniASR_LLM_7B
- omniASR_CTC_300M / omniASR_CTC_1B / omniASR_CTC_3B / omniASR_CTC_7B
"""

from typing import Optional, Set
from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES

# Manual overrides where the FLEURS iso639_3/script combo doesn't match what
# omni's supported_langs list uses.
OMNI_CODE_OVERRIDES = {
    # FLEURS uses "cmn" + "Hans"; omni uses "cmn_Hans" (matches automatically).
    # FLEURS uses "arb" + "Arab"; omni uses "arb_Arab" (matches automatically).
    # Add overrides here if any mismatches are discovered.
}


def _load_supported() -> Set[str]:
    """Load omni's supported language list. Returns empty set if package missing."""
    try:
        from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
        return set(supported_langs)
    except ImportError:
        return set()


_OMNI_SUPPORTED: Set[str] = _load_supported()


def fleurs_to_omni(fleurs_code: str) -> Optional[str]:
    """Convert FLEURS code to omni language code ({iso639_3}_{script})."""
    if fleurs_code in OMNI_CODE_OVERRIDES:
        candidate = OMNI_CODE_OVERRIDES[fleurs_code]
        return candidate if candidate in _OMNI_SUPPORTED else None

    info = FLEURS_LANGUAGES.get(fleurs_code)
    if not info:
        return None

    iso3 = info.get("iso639_3")
    script = info.get("script")
    if not iso3 or not script:
        return None

    candidate = f"{iso3}_{script}"
    if candidate in _OMNI_SUPPORTED:
        return candidate
    return None


def omni_supports_asr(fleurs_code: str) -> bool:
    """Check if omni ASR supports a FLEURS language."""
    return fleurs_to_omni(fleurs_code) is not None


def get_omni_asr_languages() -> Set[str]:
    """Get FLEURS codes for all languages omni ASR supports."""
    return {c for c in FLEURS_LANGUAGES if omni_supports_asr(c)}