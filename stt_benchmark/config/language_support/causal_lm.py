# stt_benchmark/config/language_support/causal_lm.py
"""SpeechCausalLM language support — african_mt's Qwen3.5-based MT backbone, 25 languages.

Mirrors speech_aura.py's structure exactly (same FLEURS->dataset-name / AST_TARGETS / four
helper functions), but with the fuller language set african_mt's LANG_CODE_TO_NAME covers
(src/african_mt/data/dataset.py) — Fon, Wolof, Somali, and Nyanja/Chichewa are not in
SpeechAura's 20-language set but are in this backbone's.

This backbone is speech-translation only (see st.models.speech_causal_lm.SpeechCausalLM's
module docstring: the MT model was never trained to transcribe), so the ASR-side functions
always report no support — kept only so the dispatch in models/hf/speech_aura.py has a uniform
5-function interface across all three backbones.
"""

from typing import Optional, Set, Tuple

# FLEURS code -> dataset language name, as used by causal_lm.lang_names in the experiment YAML
# (configs/stt/experiment/stage*_causal_lm_st.yaml) and by SpeechEncoder's shared training set.
FLEURS_TO_CAUSAL_LM = {
    "af_za": "afrikaans",
    "am_et": "amharic",
    "ar_eg": "arabic",
    "bem": "bemba",             # via bembaspeech and bigc (not official FLEURS)
    "en_us": "english",
    "fon_bj": "fon",            # via simbabench (not official FLEURS)
    "fr_fr": "french",
    "ha_ng": "hausa",
    "ig_ng": "igbo",
    "rw_rw": "kinyarwanda",     # via mbaza_fleurs_rw (not official FLEURS)
    "ln_cd": "lingala",
    "lg_ug": "luganda",
    "ny_mw": "chichewa",        # african_mt calls this "Nyanja"
    "mg_mg": "malagasy",        # via waxal (not official FLEURS)
    "pt_br": "portuguese",
    "sn_zw": "shona",
    "so_so": "somali",
    "st_za": "sotho",           # Sesotho / Southern Sotho — via nchlt (not in FLEURS)
    "sw_ke": "swahili",
    "ti_et": "tigrinya",        # via waxal (not official FLEURS)
    "tn_za": "tswana",          # via nchlt (not official FLEURS)
    "wo_sn": "wolof",
    "xh_za": "xhosa",
    "yo_ng": "yoruba",
    "zu_za": "zulu",
}

AST_TARGETS = dict(FLEURS_TO_CAUSAL_LM)


def fleurs_to_causal_lm(fleurs_code: str) -> Optional[str]:
    return FLEURS_TO_CAUSAL_LM.get(fleurs_code)


def causal_lm_supports_asr(fleurs_code: str) -> bool:
    # The MT backbone has no transcription capability — see module docstring.
    return False


def causal_lm_supports_ast(source_fleurs: str, target_fleurs: str) -> bool:
    if target_fleurs not in AST_TARGETS:
        return False
    return source_fleurs in FLEURS_TO_CAUSAL_LM


def get_causal_lm_asr_languages() -> Set[str]:
    return set()


def get_causal_lm_ast_pairs() -> Set[Tuple[str, str]]:
    return {
        (c, t) for c in FLEURS_TO_CAUSAL_LM for t in AST_TARGETS
        if causal_lm_supports_ast(c, t)
    }
