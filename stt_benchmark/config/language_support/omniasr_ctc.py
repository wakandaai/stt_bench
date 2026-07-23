# stt_benchmark/config/language_support/omniasr_ctc.py
"""Our fine-tuned omniASR_CTC_1B — language support.

Trained on 22 languages (see aura-asr-v1's ASR_INDEX_V4_16k.csv). 21 of the
22 have a confirmed FLEURS code and are wired up here; Tsonga is the
exception — it has no confirmed FLEURS code and no dataset entry in this
framework's current eval configs (african_asr_eval.yaml etc. don't reference
it either), so it's omitted rather than guessed. If a Tsonga-covering
dataset is ever added, add its FLEURS code here to pick it up.
"""

from typing import Set

# FLEURS code -> our model's internal language name (matches the `language`
# column in aura-asr-v1's training index / dataset.py).
FLEURS_TO_OMNIASR_CTC = {
    "af_za": "afrikaans",
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
    "st_za": "sotho",          # Sesotho / Southern Sotho — via nchlt (not in FLEURS)
    "sw_ke": "swahili",
    "ti_et": "tigrinya",       # via waxal (not official FLEURS)
    "tn_za": "tswana",         # Setswana — via nchlt (not official FLEURS)
    "xh_za": "xhosa",
    "yo_ng": "yoruba",
    "zu_za": "zulu",
}


def omniasr_ctc_supports_asr(fleurs_code: str) -> bool:
    return fleurs_code in FLEURS_TO_OMNIASR_CTC


def get_omniasr_ctc_languages() -> Set[str]:
    return set(FLEURS_TO_OMNIASR_CTC)
