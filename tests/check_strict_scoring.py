#!/usr/bin/env python3
"""Unit + data checks for stt_benchmark/evaluation/strict_scoring.py (policy v1). Light CPU, runs in seconds."""
import csv, os, sys, unicodedata

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stt_benchmark.evaluation import strict_scoring as S

passes, fails = 0, []


def check(name, cond, detail=""):
    global passes
    if cond:
        passes += 1
    else:
        fails.append(f"{name} {detail}")
        print("FAIL", name, detail)


def raises(name, fn, exc=S.ScoringAborted):
    try:
        fn()
    except exc:
        check(name, True); return
    except Exception as e:
        check(name, False, f"wrong exception {type(e).__name__}: {e}"); return
    check(name, False, "did not raise")


N = lambda t, l="yo_ng": S.normalize(t, l)

# ---- normalisation -------------------------------------------------------------------------------------
yo = "ọ́ọ̀ nǹkan lówà"
check("yoruba stacked tone marks preserved", N(yo) == unicodedata.normalize("NFC", yo), repr(N(yo)))
check("yoruba decomposed == composed", N(unicodedata.normalize("NFD", yo)) == N(unicodedata.normalize("NFC", yo)))
check("NOT split at combining marks", len(N(yo).split()) == 3)
check("arabic harakat preserved", N("سَيسمح ذلك", "ar_eg") == "سَيسمح ذلك")
check("arabic comma removed", N("الاجتماعية، التي", "ar_eg") == "الاجتماعية التي")
check("amharic punctuation removed", N("ሰላም። እንዴት ነህ፧", "am_et") == "ሰላም እንዴት ነህ")
check("ethiopic wordspace -> space", N("ሰላም፡እንዴት", "am_et") == "ሰላም እንዴት")
check("curly apostrophe == straight", N("n’ime", "ig_ng") == N("n'ime", "ig_ng") == "n'ime")
check("edge apostrophes (quotes) removed", N("'hello' world", "ha_ng") == "hello world")
check("inner apostrophe kept", N("don't", "sw_ke") == "don't")
check("hyphen -> space", N("kade-kade", "ha_ng") == "kade kade")
check("digits untouched", N("2023 na 5%", "sw_ke") == "2023 na 5")
check("zero-width format char deleted", N("a‍b", "sw_ke") == "ab")
check("lowercase + whitespace", N("  Habari   YA Asubuhi ", "sw_ke") == "habari ya asubuhi")
check("english uses whisper normaliser", N("Hello, World!", "en_us") == "hello world", repr(N("Hello, World!", "en_us")))
check("english: number word -> digit (whisper)", "5" in N("I have five apples", "en_us"), repr(N("I have five apples", "en_us")))
for s_ in [yo, "n’ime ọ̀rọ̀-ìyà 5%", "ሰላም። እንዴት", "'x' y-z"]:
    check(f"idempotent {s_!r}", N(N(s_)) == N(s_))
check("strip_marks removes tone+dot-below", S.strip_marks(N(yo)) == "oo nnkan lowa", repr(S.strip_marks(N(yo))))

# ---- metrics: hand-computed cases ----------------------------------------------------------------------
r = S.score_group(["a b c d"], ["a b c d"], "sw_ke", n_boot=50)
check("identical -> 0", r["wer"] == 0.0 and r["cer"] == 0.0)
r = S.score_group(["a b x d"], ["a b c d"], "sw_ke", n_boot=50)
check("1 sub in 4 words = 25%", abs(r["wer"] - 25.0) < 1e-9, r["wer"])
r = S.score_group([""], ["a b c d"], "sw_ke", n_boot=50)
check("empty hyp = 100%", abs(r["wer"] - 100.0) < 1e-9)
r = S.score_group(["a b c d e"], ["a b c d"], "sw_ke", n_boot=50)
check("1 insertion in 4 = 25%", abs(r["wer"] - 25.0) < 1e-9)
# CER: 'cat' vs 'cut' -> 1/3
r = S.score_group(["cut"], ["cat"], "sw_ke", n_boot=50)
check("CER 1/3", abs(r["cer"] - 100 / 3) < 1e-9, r["cer"])
# corpus aggregation equals jiwer on the same normalised text
import jiwer
refs = ["habari ya asubuhi rafiki", "karibu sana nyumbani", "asante kwa msaada wako mkubwa"]
hyps = ["habari asubuhi rafiki", "karibu sana nyumbani leo", "asante kwa msaada mkubwa"]
r = S.score_group(hyps, refs, "sw_ke", n_boot=200)
check("corpus WER == jiwer.wer", abs(r["wer"] - 100 * jiwer.wer(refs, hyps)) < 1e-9, f"{r['wer']} vs {100*jiwer.wer(refs, hyps)}")
check("corpus CER == jiwer.cer", abs(r["cer"] - 100 * jiwer.cer(refs, hyps)) < 1e-9)
check("CI brackets the estimate", r["wer_ci_low"] <= r["wer"] <= r["wer_ci_high"])
r2 = S.score_group(hyps, refs, "sw_ke", n_boot=200)
check("bootstrap deterministic (seed)", r["wer_ci_low"] == r2["wer_ci_low"] and r["wer_ci_high"] == r2["wer_ci_high"])

# ---- error accounting ---------------------------------------------------------------------------------
hyps_ok = ["x"] * 1000
refs_ok = ["x"] * 1000
r = S.score_group(hyps_ok[:999] + [None], refs_ok, "sw_ke", n_boot=50, max_fail_fraction=0.002)
check("1/1000 failed tolerated and counted", r["n_failed"] == 1 and r["n"] == 1000)
raises("3/1000 failed aborts", lambda: S.score_group(hyps_ok[:997] + [None] * 3, refs_ok, "sw_ke", n_boot=50))
raises("empty-after-normalisation reference raises", lambda: S.score_group(["x"], ["?!"], "sw_ke", n_boot=50))
raises("length mismatch raises", lambda: S.score_group(["x"], ["x", "y"], "sw_ke", n_boot=50), ValueError)
raises("zero utterances raises", lambda: S.score_group([], [], "sw_ke", n_boot=50))

# ---- diagnostic variant & paired comparison ------------------------------------------------------------
yo_ref = ["ó dára púpọ̀", "mo fẹ́ lọ sílé"]
yo_hyp = ["o dara pupọ", "mo fe lo sile"]          # right letters, tone marks lost
r = S.score_group(yo_hyp, yo_ref, "yo_ng", n_boot=100)
check("tone loss hurts primary WER", r["wer"] > 50.0, r["wer"])
check("diagnostic WER ignores tone marks (dot-below kept as base letter after NFD strip)", r["wer_nomarks"] < r["wer"], f"{r['wer_nomarks']} vs {r['wer']}")
r_sw = S.score_group(hyps, refs, "sw_ke", n_boot=50)
check("no diagnostic for non-tone language by default", "wer_nomarks" not in r_sw)
a = S.score_group(hyps, refs, "sw_ke", n_boot=100)
d0 = S.paired_bootstrap_delta(a, a, "wer", n_boot=200)
check("paired delta of a system with itself is 0", d0["delta"] == 0.0 and d0["ci_low"] == 0.0 == d0["ci_high"])
b = S.score_group(refs, refs, "sw_ke", n_boot=100)
d1 = S.paired_bootstrap_delta(a, b, "wer", n_boot=200)
check("paired delta = WER_a - WER_b", abs(d1["delta"] - (a["wer"] - b["wer"])) < 1e-9)

# ---- English: Whisper standard is the headline, plain is a disclosed sensitivity column -------------------
re_ = S.score_group(["i have 5 apples"], ["I have five apples"], "en_us", n_boot=50)
check("english headline uses Whisper normaliser (five == 5)", re_["wer"] == 0.0, re_["wer"])
check("english plain column present and stricter (five != 5)", re_["wer_plain"] > 0.0 and "cer_plain" in re_, re_.get("wer_plain"))
check("english result names its normaliser", "whisper" in re_["normalizer"])
check("non-English has no plain column", "wer_plain" not in S.score_group(["a"], ["a"], "sw_ke", n_boot=20))
pv = S.provenance()
check("provenance pins Whisper files (hashes match Phase 1 vendor)", pv["sha256"]["english.py"].startswith("d31a52685b5b") and pv["sha256"]["english.json"].startswith("6607f948be98"), pv["sha256"])
check("provenance records versions", all(k in pv for k in ("policy", "jiwer", "numpy", "python", "whisper_normalizers_commit")))

# ---- real data: FLEURS reference text -----------------------------------------------------------------
root = "/work/nvme/bfbf/shared/datasets_16k/FLEURS/google_fleurs_test/data"
if os.path.isdir(root):
    for code in ("yo_ng", "ig_ng", "ar_eg", "am_et", "ha_ng"):
        rows = list(csv.reader(open(f"{root}/{code}/test.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
        raw = [x[2] for x in rows]
        nrm = [S.normalize(t, code) for t in raw]
        check(f"{code}: idempotent on all {len(raw)} refs", all(S.normalize(t, code) == t for t in nrm))
        check(f"{code}: no empty refs", all(nrm))
        fl = [x[3] for x in rows]                      # FLEURS' own normalised column
        same = sum(1 for a_, b_ in zip(nrm, fl) if a_ == " ".join(S.normalize(b_, code).split()))
        check(f"{code}: oracle WER 0", S.score_group(raw, raw, code, n_boot=20)["wer"] == 0.0)
        # agreement with the dataset's own normalisation (informational, not a pass/fail claim)
        print(f"  info {code}: my normalised raw == my normalised FLEURS-normalised for {same}/{len(rows)} rows")
        # tone marks must survive: combining-mark count of normalised text equals that of NFC raw (ignoring removed punctuation)
        marks_raw = sum(1 for t in raw for c in unicodedata.normalize("NFC", t.lower()) if unicodedata.category(c) == "Mn")
        marks_n = sum(1 for t in nrm for c in t if unicodedata.category(c) == "Mn")
        check(f"{code}: no combining mark lost", marks_raw == marks_n, f"{marks_raw} vs {marks_n}")
else:
    print("  (FLEURS data not found; data checks skipped)")

print(f"\n{passes} PASS, {len(fails)} FAIL")
for f in fails:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
