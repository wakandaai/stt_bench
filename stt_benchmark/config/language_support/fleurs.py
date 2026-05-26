# stt_benchmark/config/language_support/fleurs.py

"""
FLEURS Language Support Configuration

Maps between FLEURS language codes (e.g., 'sw_ke'), ISO 639-3 codes (e.g., 'swh'),
and model-specific codes for Whisper, MMS, and SeamlessM4T.

FLEURS uses BCP-47-like codes: {iso639-1}_{country} (e.g., 'af_za', 'sw_ke').

Note: this catalog also includes a handful of languages that are NOT in
official Google FLEURS but that our benchmark supports via community-released
FLEURS-style datasets or local datasets. Those entries are flagged in comments.
"""

from typing import Dict, List, Optional, Set

# Complete FLEURS language catalog with mappings
# Format: fleurs_code -> {name, iso639_3, iso639_1, script}
FLEURS_LANGUAGES = {
    "af_za": {"name": "Afrikaans", "iso639_3": "afr", "iso639_1": "af", "script": "Latn"},
    "am_et": {"name": "Amharic", "iso639_3": "amh", "iso639_1": "am", "script": "Ethi"},
    "ar_eg": {"name": "Arabic", "iso639_3": "arb", "iso639_1": "ar", "script": "Arab"},
    "as_in": {"name": "Assamese", "iso639_3": "asm", "iso639_1": "as", "script": "Beng"},
    "ast_es": {"name": "Asturian", "iso639_3": "ast", "iso639_1": None, "script": "Latn"},
    "az_az": {"name": "Azerbaijani", "iso639_3": "azj", "iso639_1": "az", "script": "Latn"},
    "be_by": {"name": "Belarusian", "iso639_3": "bel", "iso639_1": "be", "script": "Cyrl"},
    # Bemba — NOT in official Google FLEURS. Listed here so the bembaspeech and
    # bigc datasets can use a first-class language code in our benchmark.
    "bem": {"name": "Bemba", "iso639_3": "bem", "iso639_1": None, "script": "Latn"},
    "bg_bg": {"name": "Bulgarian", "iso639_3": "bul", "iso639_1": "bg", "script": "Cyrl"},
    "bn_in": {"name": "Bengali", "iso639_3": "ben", "iso639_1": "bn", "script": "Beng"},
    "bs_ba": {"name": "Bosnian", "iso639_3": "bos", "iso639_1": "bs", "script": "Latn"},
    "ca_es": {"name": "Catalan", "iso639_3": "cat", "iso639_1": "ca", "script": "Latn"},
    "ceb_ph": {"name": "Cebuano", "iso639_3": "ceb", "iso639_1": None, "script": "Latn"},
    "ckb_iq": {"name": "Central Kurdish", "iso639_3": "ckb", "iso639_1": None, "script": "Arab"},
    "cmn_hans_cn": {"name": "Mandarin Chinese", "iso639_3": "cmn", "iso639_1": "zh", "script": "Hans"},
    "cs_cz": {"name": "Czech", "iso639_3": "ces", "iso639_1": "cs", "script": "Latn"},
    "cy_gb": {"name": "Welsh", "iso639_3": "cym", "iso639_1": "cy", "script": "Latn"},
    "da_dk": {"name": "Danish", "iso639_3": "dan", "iso639_1": "da", "script": "Latn"},
    "de_de": {"name": "German", "iso639_3": "deu", "iso639_1": "de", "script": "Latn"},
    "el_gr": {"name": "Greek", "iso639_3": "ell", "iso639_1": "el", "script": "Grek"},
    "en_us": {"name": "English", "iso639_3": "eng", "iso639_1": "en", "script": "Latn"},
    "es_419": {"name": "Spanish", "iso639_3": "spa", "iso639_1": "es", "script": "Latn"},
    "et_ee": {"name": "Estonian", "iso639_3": "ekk", "iso639_1": "et", "script": "Latn"},
    "eu_es": {"name": "Basque", "iso639_3": "eus", "iso639_1": "eu", "script": "Latn"},
    "fa_ir": {"name": "Persian", "iso639_3": "pes", "iso639_1": "fa", "script": "Arab"},
    "ff_sn": {"name": "Fulah", "iso639_3": "fuv", "iso639_1": "ff", "script": "Latn"},
    "fi_fi": {"name": "Finnish", "iso639_3": "fin", "iso639_1": "fi", "script": "Latn"},
    "fil_ph": {"name": "Filipino", "iso639_3": "fil", "iso639_1": None, "script": "Latn"},
    "fr_fr": {"name": "French", "iso639_3": "fra", "iso639_1": "fr", "script": "Latn"},
    "ga_ie": {"name": "Irish", "iso639_3": "gle", "iso639_1": "ga", "script": "Latn"},
    "gl_es": {"name": "Galician", "iso639_3": "glg", "iso639_1": "gl", "script": "Latn"},
    "gu_in": {"name": "Gujarati", "iso639_3": "guj", "iso639_1": "gu", "script": "Gujr"},
    "ha_ng": {"name": "Hausa", "iso639_3": "hau", "iso639_1": "ha", "script": "Latn"},
    "he_il": {"name": "Hebrew", "iso639_3": "heb", "iso639_1": "he", "script": "Hebr"},
    "hi_in": {"name": "Hindi", "iso639_3": "hin", "iso639_1": "hi", "script": "Deva"},
    "hr_hr": {"name": "Croatian", "iso639_3": "hrv", "iso639_1": "hr", "script": "Latn"},
    "hu_hu": {"name": "Hungarian", "iso639_3": "hun", "iso639_1": "hu", "script": "Latn"},
    "hy_am": {"name": "Armenian", "iso639_3": "hye", "iso639_1": "hy", "script": "Armn"},
    "id_id": {"name": "Indonesian", "iso639_3": "ind", "iso639_1": "id", "script": "Latn"},
    "ig_ng": {"name": "Igbo", "iso639_3": "ibo", "iso639_1": "ig", "script": "Latn"},
    "is_is": {"name": "Icelandic", "iso639_3": "isl", "iso639_1": "is", "script": "Latn"},
    "it_it": {"name": "Italian", "iso639_3": "ita", "iso639_1": "it", "script": "Latn"},
    "ja_jp": {"name": "Japanese", "iso639_3": "jpn", "iso639_1": "ja", "script": "Jpan"},
    "jv_id": {"name": "Javanese", "iso639_3": "jav", "iso639_1": "jv", "script": "Latn"},
    "ka_ge": {"name": "Georgian", "iso639_3": "kat", "iso639_1": "ka", "script": "Geor"},
    "kam_ke": {"name": "Kamba", "iso639_3": "kam", "iso639_1": None, "script": "Latn"},
    "kea_cv": {"name": "Kabuverdianu", "iso639_3": "kea", "iso639_1": None, "script": "Latn"},
    "kk_kz": {"name": "Kazakh", "iso639_3": "kaz", "iso639_1": "kk", "script": "Cyrl"},
    "km_kh": {"name": "Khmer", "iso639_3": "khm", "iso639_1": "km", "script": "Khmr"},
    "kn_in": {"name": "Kannada", "iso639_3": "kan", "iso639_1": "kn", "script": "Knda"},
    "ko_kr": {"name": "Korean", "iso639_3": "kor", "iso639_1": "ko", "script": "Hang"},
    "ky_kg": {"name": "Kyrgyz", "iso639_3": "kir", "iso639_1": "ky", "script": "Cyrl"},
    "lb_lu": {"name": "Luxembourgish", "iso639_3": "ltz", "iso639_1": "lb", "script": "Latn"},
    "lg_ug": {"name": "Luganda", "iso639_3": "lug", "iso639_1": "lg", "script": "Latn"},
    "ln_cd": {"name": "Lingala", "iso639_3": "lin", "iso639_1": "ln", "script": "Latn"},
    "lo_la": {"name": "Lao", "iso639_3": "lao", "iso639_1": "lo", "script": "Laoo"},
    "lt_lt": {"name": "Lithuanian", "iso639_3": "lit", "iso639_1": "lt", "script": "Latn"},
    "luo_ke": {"name": "Luo", "iso639_3": "luo", "iso639_1": None, "script": "Latn"},
    "lv_lv": {"name": "Latvian", "iso639_3": "lvs", "iso639_1": "lv", "script": "Latn"},
    # Malagasy — NOT in official Google FLEURS. Listed here so the waxal dataset
    # (mlg_asr) can use a first-class language code. iso639_3 'plt' (Plateau
    # Malagasy) is the FLORES/NLLB code; Waxal uses macrolanguage 'mlg' but
    # the same standard written variety.
    "mg_mg": {"name": "Malagasy", "iso639_3": "plt", "iso639_1": "mg", "script": "Latn"},
    "mi_nz": {"name": "Maori", "iso639_3": "mri", "iso639_1": "mi", "script": "Latn"},
    "mk_mk": {"name": "Macedonian", "iso639_3": "mkd", "iso639_1": "mk", "script": "Cyrl"},
    "ml_in": {"name": "Malayalam", "iso639_3": "mal", "iso639_1": "ml", "script": "Mlym"},
    "mn_mn": {"name": "Mongolian", "iso639_3": "khk", "iso639_1": "mn", "script": "Cyrl"},
    "mr_in": {"name": "Marathi", "iso639_3": "mar", "iso639_1": "mr", "script": "Deva"},
    "ms_my": {"name": "Malay", "iso639_3": "zsm", "iso639_1": "ms", "script": "Latn"},
    "mt_mt": {"name": "Maltese", "iso639_3": "mlt", "iso639_1": "mt", "script": "Latn"},
    "my_mm": {"name": "Burmese", "iso639_3": "mya", "iso639_1": "my", "script": "Mymr"},
    "nb_no": {"name": "Norwegian Bokmål", "iso639_3": "nob", "iso639_1": "nb", "script": "Latn"},
    "ne_np": {"name": "Nepali", "iso639_3": "npi", "iso639_1": "ne", "script": "Deva"},
    "nl_nl": {"name": "Dutch", "iso639_3": "nld", "iso639_1": "nl", "script": "Latn"},
    "nso_za": {"name": "Northern Sotho", "iso639_3": "nso", "iso639_1": None, "script": "Latn"},
    "ny_mw": {"name": "Chichewa", "iso639_3": "nya", "iso639_1": "ny", "script": "Latn"},
    "oc_fr": {"name": "Occitan", "iso639_3": "oci", "iso639_1": "oc", "script": "Latn"},
    "om_et": {"name": "Oromo", "iso639_3": "gaz", "iso639_1": "om", "script": "Latn"},
    "or_in": {"name": "Odia", "iso639_3": "ory", "iso639_1": "or", "script": "Orya"},
    "pa_in": {"name": "Punjabi", "iso639_3": "pan", "iso639_1": "pa", "script": "Guru"},
    "pl_pl": {"name": "Polish", "iso639_3": "pol", "iso639_1": "pl", "script": "Latn"},
    "ps_af": {"name": "Pashto", "iso639_3": "pbt", "iso639_1": "ps", "script": "Arab"},
    "pt_br": {"name": "Portuguese", "iso639_3": "por", "iso639_1": "pt", "script": "Latn"},
    "ro_ro": {"name": "Romanian", "iso639_3": "ron", "iso639_1": "ro", "script": "Latn"},
    "ru_ru": {"name": "Russian", "iso639_3": "rus", "iso639_1": "ru", "script": "Cyrl"},
    # Kinyarwanda — NOT in official Google FLEURS. Listed here so the
    # mbaza_fleurs_rw dataset (community-released FLEURS-style corpus from
    # mbazaNLP) can use a first-class language code in our benchmark.
    "rw_rw": {"name": "Kinyarwanda", "iso639_3": "kin", "iso639_1": "rw", "script": "Latn"},
    "sd_in": {"name": "Sindhi", "iso639_3": "snd", "iso639_1": "sd", "script": "Arab"},
    "sk_sk": {"name": "Slovak", "iso639_3": "slk", "iso639_1": "sk", "script": "Latn"},
    "sl_si": {"name": "Slovenian", "iso639_3": "slv", "iso639_1": "sl", "script": "Latn"},
    "sn_zw": {"name": "Shona", "iso639_3": "sna", "iso639_1": "sn", "script": "Latn"},
    "so_so": {"name": "Somali", "iso639_3": "som", "iso639_1": "so", "script": "Latn"},
    "sr_rs": {"name": "Serbian", "iso639_3": "srp", "iso639_1": "sr", "script": "Cyrl"},
    # Sesotho / Southern Sotho — NOT in official Google FLEURS. Listed here so
    # the nchlt dataset can use a first-class language code. Distinct from
    # Northern Sotho (nso_za), which IS in FLEURS.
    "st_za": {"name": "Sesotho", "iso639_3": "sot", "iso639_1": "st", "script": "Latn"},
    "sv_se": {"name": "Swedish", "iso639_3": "swe", "iso639_1": "sv", "script": "Latn"},
    "sw_ke": {"name": "Swahili", "iso639_3": "swh", "iso639_1": "sw", "script": "Latn"},
    "ta_in": {"name": "Tamil", "iso639_3": "tam", "iso639_1": "ta", "script": "Taml"},
    "te_in": {"name": "Telugu", "iso639_3": "tel", "iso639_1": "te", "script": "Telu"},
    "tg_tj": {"name": "Tajik", "iso639_3": "tgk", "iso639_1": "tg", "script": "Cyrl"},
    "th_th": {"name": "Thai", "iso639_3": "tha", "iso639_1": "th", "script": "Thai"},
    # Tigrinya — NOT in official Google FLEURS. Listed here so the waxal
    # dataset (tir_asr) can use a first-class language code. Tagged as ti_et
    # (Ethiopia) since that matches our training data; Tigrinya is also
    # spoken in Eritrea (ti_er) but Waxal does not specify a region.
    "ti_et": {"name": "Tigrinya", "iso639_3": "tir", "iso639_1": "ti", "script": "Ethi"},
    # Tswana / Setswana — NOT in official Google FLEURS. Listed here so the
    # nchlt dataset can use a first-class language code.
    "tn_za": {"name": "Tswana", "iso639_3": "tsn", "iso639_1": "tn", "script": "Latn"},
    "tr_tr": {"name": "Turkish", "iso639_3": "tur", "iso639_1": "tr", "script": "Latn"},
    "uk_ua": {"name": "Ukrainian", "iso639_3": "ukr", "iso639_1": "uk", "script": "Cyrl"},
    "umb_ao": {"name": "Umbundu", "iso639_3": "umb", "iso639_1": None, "script": "Latn"},
    "ur_pk": {"name": "Urdu", "iso639_3": "urd", "iso639_1": "ur", "script": "Arab"},
    "uz_uz": {"name": "Uzbek", "iso639_3": "uzn", "iso639_1": "uz", "script": "Latn"},
    "vi_vn": {"name": "Vietnamese", "iso639_3": "vie", "iso639_1": "vi", "script": "Latn"},
    "wo_sn": {"name": "Wolof", "iso639_3": "wol", "iso639_1": "wo", "script": "Latn"},
    "xh_za": {"name": "Xhosa", "iso639_3": "xho", "iso639_1": "xh", "script": "Latn"},
    "yo_ng": {"name": "Yoruba", "iso639_3": "yor", "iso639_1": "yo", "script": "Latn"},
    "yue_hant_hk": {"name": "Cantonese", "iso639_3": "yue", "iso639_1": None, "script": "Hant"},
    "zu_za": {"name": "Zulu", "iso639_3": "zul", "iso639_1": "zu", "script": "Latn"},
}

# Languages of interest for African-focused benchmarking
AFRICAN_LANGUAGES = {
    "af_za", "am_et", "ar_eg", "bem", "ff_sn", "ha_ng", "ig_ng", "kam_ke",
    "lg_ug", "ln_cd", "luo_ke", "mg_mg", "nso_za", "ny_mw", "om_et",
    "rw_rw", "sn_zw", "so_so", "st_za", "sw_ke", "ti_et", "tn_za", "umb_ao",
    "wo_sn", "xh_za", "yo_ng", "zu_za",
}

# Target languages of interest (user-specified)
TARGET_LANGUAGES = {
    "en_us",    # English (anchor)
    "fr_fr",    # French (anchor)
    "pt_br",    # Portuguese
    "ar_eg",    # Arabic
    "af_za",    # Afrikaans
    "sw_ke",    # Swahili
    "so_so",    # Somali
    "ha_ng",    # Hausa
    "am_et",    # Amharic
    "mg_mg",    # Malagasy (via waxal, not official FLEURS)
    "rw_rw",    # Kinyarwanda (via mbaza_fleurs_rw, not official FLEURS)
    "xh_za",    # Xhosa
    "zu_za",    # Zulu
    "ny_mw",    # Chichewa/Nyanja
    "st_za",    # Sesotho (via NCHLT, not official FLEURS)
    "sn_zw",    # Shona
    "ig_ng",    # Igbo
    "yo_ng",    # Yoruba
    "ti_et",    # Tigrinya (via waxal, not official FLEURS)
    "lg_ug",    # Luganda
    "ln_cd",    # Lingala
    "tn_za",    # Tswana (via NCHLT, not official FLEURS)
    "wo_sn",    # Wolof
    "bem",      # Bemba (via bembaspeech and bigc, not official FLEURS)
    # "fon",    # Fongbe — NOT in FLEURS
}


def get_fleurs_language_info(fleurs_code: str) -> Optional[Dict[str, str]]:
    """Get language info for a FLEURS language code."""
    return FLEURS_LANGUAGES.get(fleurs_code)


def fleurs_to_iso639_3(fleurs_code: str) -> Optional[str]:
    """Convert FLEURS code to ISO 639-3."""
    info = FLEURS_LANGUAGES.get(fleurs_code)
    return info["iso639_3"] if info else None


def fleurs_to_iso639_1(fleurs_code: str) -> Optional[str]:
    """Convert FLEURS code to ISO 639-1."""
    info = FLEURS_LANGUAGES.get(fleurs_code)
    return info["iso639_1"] if info else None


def iso639_1_to_fleurs(iso1_code: str) -> Optional[str]:
    """Convert ISO 639-1 code to FLEURS code. Returns first match."""
    for fleurs_code, info in FLEURS_LANGUAGES.items():
        if info["iso639_1"] == iso1_code:
            return fleurs_code
    return None


def iso639_3_to_fleurs(iso3_code: str) -> Optional[str]:
    """Convert ISO 639-3 code to FLEURS code. Returns first match."""
    for fleurs_code, info in FLEURS_LANGUAGES.items():
        if info["iso639_3"] == iso3_code:
            return fleurs_code
    return None


def get_all_fleurs_codes() -> Set[str]:
    """Get all FLEURS language codes."""
    return set(FLEURS_LANGUAGES.keys())


def get_target_languages() -> Set[str]:
    """Get the target languages of interest."""
    return TARGET_LANGUAGES.copy()


def get_african_languages() -> Set[str]:
    """Get African languages available in FLEURS."""
    return AFRICAN_LANGUAGES.copy()