# stt_benchmark/evaluation/strict_scoring.py

"""
Strict ASR scoring for the leaderboard (policy v1). Separate from metrics.py on purpose: old results stay
reproducible; nothing here is shared mutable state.

TEXT NORMALISATION (applied identically to references and hypotheses, per language code):
  1. NFC. Never NFKC/NFKD for the primary score (they would decompose tone marks).
  2. Lowercase.
  3. Apostrophes canonicalised: ' (U+2019) ' (U+2018) U+02BC U+00B4 ` -> '. An apostrophe is KEPT only between two
     word characters (letter / mark / digit); at a word edge it is treated as a quote mark and becomes a space.
     (Limitation: a word-initial glottal mark, e.g. Hausa 'yan, is therefore written without it; applied to both sides.)
  4. Hyphens (Unicode Pd) -> space. All other punctuation (P*) and symbols (S*) -> space. This covers Latin, Arabic,
     Ethiopic (Amharic, Tigrinya) punctuation. Invisible format characters (Cf: zero-width, bidi marks) are deleted;
     control and separator characters become spaces.
  5. Letters, ALL combining marks (tone marks, harakat) and DIGITS are kept as written. Digits are not converted.
  6. Whitespace collapsed.
  English (en_us) only: the pinned, vendored openai/whisper EnglishTextNormalizer instead of steps 2-6. This is the
  field standard (Whisper paper, Open ASR Leaderboard) and is the HEADLINE English score. Because it rewrites number
  words and spellings, the plain policy is ALSO reported for English as a disclosed sensitivity column
  (wer_plain / cer_plain); neither is hidden.
  The Whisper BasicTextNormalizer is deliberately NOT used: it splits words at combining marks.

METRICS: corpus WER and CER (jiwer) on the normalised text; 95% bootstrap CI by resampling utterances;
  an optional diagnostic variant with all combining marks removed (reported separately, never the headline).
ERROR ACCOUNTING: a hypothesis of None means the model failed on that utterance. Failures are counted and reported;
  they are scored as empty hypotheses; if more than `max_fail_fraction` fail the scoring RAISES (a model must not
  win by crashing quietly). A reference that is empty after normalisation RAISES (it is never silently dropped).
"""

import os
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

POLICY_VERSION = "v1"
MAX_FAIL_FRACTION = 0.002
DIAGNOSTIC_LANGS = {"yo_ng", "ig_ng", "ar_eg", "ha_ng"}
ENGLISH_CODES = {"en_us"}

_APOS_VARIANTS = "’‘ʼ´`"
_english = None


class ScoringAborted(RuntimeError):
    """Raised when failures or empty references make a score untrustworthy."""


def _english_normalizer():
    global _english
    if _english is None:
        from stt_benchmark.vendor.whisper_normalizers import EnglishTextNormalizer
        _english = EnglishTextNormalizer()
    return _english


def _is_word_char(c: str) -> bool:
    return unicodedata.category(c)[0] in "LMN"


def normalize(text: str, lang: str, plain: bool = False) -> str:
    """Policy-v1 normalisation. See module docstring. `plain=True` forces the language-neutral steps (also for
    English) - used only for the English sensitivity column."""
    t = unicodedata.normalize("NFC", text)
    if lang in ENGLISH_CODES and not plain:
        return " ".join(_english_normalizer()(t).split())
    t = t.lower()
    for a in _APOS_VARIANTS:
        t = t.replace(a, "'")
    out: List[str] = []
    n = len(t)
    for i, c in enumerate(t):
        cat = unicodedata.category(c)
        if c == "'":
            keep = 0 < i < n - 1 and _is_word_char(t[i - 1]) and _is_word_char(t[i + 1])
            out.append("'" if keep else " ")
        elif cat[0] in "PS":                 # includes Pd (hyphens) -> space
            out.append(" ")
        elif cat == "Cf":
            continue
        elif cat[0] in "CZ":
            out.append(" ")
        else:                                # letters, marks, digits, everything else kept
            out.append(c)
    return " ".join("".join(out).split())


def strip_marks(text: str) -> str:
    """Diagnostic only: remove every combining mark (tone marks, harakat) after NFD, then NFC."""
    d = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in d if unicodedata.category(c) != "Mn"))


def provenance() -> Dict[str, str]:
    """Everything needed to reproduce a score: policy version, normaliser file hashes, library versions."""
    import hashlib, platform
    from importlib.metadata import version
    base = os.path.join(os.path.dirname(__file__), "..", "vendor", "whisper_normalizers")
    h = {f: hashlib.sha256(open(os.path.join(base, f), "rb").read()).hexdigest() for f in ("english.py", "english.json", "basic.py")}
    return {"policy": POLICY_VERSION, "whisper_normalizers_commit": "86098128c0b4f24f0e2aa2994de830614b474227",
            "sha256": h, "jiwer": version("jiwer"), "numpy": np.__version__, "python": platform.python_version()}


# ----------------------------------------------------------------------------------------------------------
def _per_utterance_errors(refs: Sequence[str], hyps: Sequence[str], level: str) -> Tuple[np.ndarray, np.ndarray]:
    """(errors, ref_units) per utterance via jiwer alignments. level: 'word' | 'char'."""
    import jiwer
    out = jiwer.process_words(list(refs), list(hyps)) if level == "word" else jiwer.process_characters(list(refs), list(hyps))
    errs = np.zeros(len(refs), dtype=np.int64)
    nref = np.zeros(len(refs), dtype=np.int64)
    for i, chunks in enumerate(out.alignments):
        e = 0
        for ch in chunks:
            if ch.type in ("substitute", "delete"):
                e += ch.ref_end_idx - ch.ref_start_idx
            elif ch.type == "insert":
                e += ch.hyp_end_idx - ch.hyp_start_idx
        errs[i] = e
        nref[i] = len(out.references[i])
    return errs, nref


def _corpus_and_ci(errs: np.ndarray, nref: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    rate = 100.0 * errs.sum() / nref.sum()
    rng = np.random.default_rng(seed)
    n = len(errs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = 100.0 * errs[idx].sum(axis=1) / nref[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rate": float(rate), "ci_low": float(lo), "ci_high": float(hi)}


def score_group(
    hypotheses: Sequence[Optional[str]],
    references: Sequence[str],
    lang: str,
    n_boot: int = 1000,
    seed: int = 0,
    max_fail_fraction: float = MAX_FAIL_FRACTION,
    diagnostic: Optional[bool] = None,
) -> Dict[str, object]:
    """Score one (model, group). Returns a flat dict; see keys below."""
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} references")
    n = len(references)
    if n == 0:
        raise ScoringAborted("no utterances to score")
    failed = [i for i, h in enumerate(hypotheses) if h is None]
    if len(failed) / n > max_fail_fraction:
        raise ScoringAborted(f"{len(failed)}/{n} utterances failed (> {max_fail_fraction:.1%}); refusing to score")
    refs_n = [normalize(r, lang) for r in references]
    empty = [i for i, r in enumerate(refs_n) if not r]
    if empty:
        raise ScoringAborted(f"{len(empty)} references are empty after normalisation (e.g. index {empty[0]})")
    hyps_n = [normalize(h, lang) if h is not None else "" for h in hypotheses]

    res: Dict[str, object] = {"policy": POLICY_VERSION, "lang": lang, "n": n, "n_failed": len(failed),
                              "normalizer": "whisper-english (pinned, vendored)" if lang in ENGLISH_CODES
                              else "policy-v1 plain (NFC, tone marks kept)", "bootstrap": f"{n_boot} resamples, seed {seed}"}
    for level, key in (("word", "wer"), ("char", "cer")):
        errs, nref = _per_utterance_errors(refs_n, hyps_n, level)
        r = _corpus_and_ci(errs, nref, n_boot, seed)
        res[key], res[f"{key}_ci_low"], res[f"{key}_ci_high"] = r["rate"], r["ci_low"], r["ci_high"]
        res[f"{key}_ref_units"] = int(nref.sum())
        res[f"_{key}_errs"] = errs                       # per-utterance arrays, kept for paired comparisons
        res[f"_{key}_nref"] = nref
    if lang in ENGLISH_CODES:
        refs_p = [normalize(r, lang, plain=True) for r in references]
        if any(not r for r in refs_p):
            raise ScoringAborted("a reference is empty under the plain policy")
        hyps_p = [normalize(h, lang, plain=True) if h is not None else "" for h in hypotheses]
        for level, key in (("word", "wer_plain"), ("char", "cer_plain")):
            errs, nref = _per_utterance_errors(refs_p, hyps_p, level)
            r = _corpus_and_ci(errs, nref, n_boot, seed)
            res[key], res[f"{key}_ci_low"], res[f"{key}_ci_high"] = r["rate"], r["ci_low"], r["ci_high"]
    want_diag = (lang in DIAGNOSTIC_LANGS) if diagnostic is None else diagnostic
    if want_diag:
        refs_d = [strip_marks(r) for r in refs_n]
        hyps_d = [strip_marks(h) for h in hyps_n]
        if any(not r.strip() for r in refs_d):
            raise ScoringAborted("a reference is empty after mark stripping")
        for level, key in (("word", "wer_nomarks"), ("char", "cer_nomarks")):
            errs, nref = _per_utterance_errors(refs_d, hyps_d, level)
            r = _corpus_and_ci(errs, nref, n_boot, seed)
            res[key], res[f"{key}_ci_low"], res[f"{key}_ci_high"] = r["rate"], r["ci_low"], r["ci_high"]
    return res


def paired_bootstrap_delta(res_a: Dict[str, object], res_b: Dict[str, object], metric: str = "wer",
                           n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    """A - B difference in `metric` ('wer' or 'cer', percentage points) with a paired utterance-level bootstrap
    95% CI. Both results must come from score_group on the same utterances in the same order."""
    ea, eb = res_a[f"_{metric}_errs"], res_b[f"_{metric}_errs"]
    na, nb = res_a[f"_{metric}_nref"], res_b[f"_{metric}_nref"]
    if len(ea) != len(eb) or not np.array_equal(na, nb):
        raise ValueError("results are not over the same utterances")
    rng = np.random.default_rng(seed)
    n = len(ea)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = 100.0 * (ea[idx].sum(axis=1) - eb[idx].sum(axis=1)) / na[idx].sum(axis=1)
    delta = 100.0 * (ea.sum() - eb.sum()) / na.sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(delta), "ci_low": float(lo), "ci_high": float(hi)}
