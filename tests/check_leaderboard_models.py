"""Smoke/acceptance check for the two leaderboard model adapters (w2v-BERT Kinyarwanda CTC, SunflowerASR).
Run on a GPU node. Loads each model, decodes 16 SimbaBench clips, checks batch invariance, strictness, and a
rough WER (smoke only: lowercase, punctuation stripped; the real scorer is separate). Exit != 0 on any FAIL."""
import re, sys, time
sys.path.insert(0, ".")
import numpy as np, torch
from jiwer import wer, cer
from stt_benchmark.datasets.registry import create_dataset
from stt_benchmark.datasets.base import AudioSample

CACHE = "/work/nvme/bfbf/etanghang/stt_bench_data/hf_models"
FAILS = []
def check(n, ok, d=""):
    print(("PASS " if ok else "FAIL ") + n + (f"  [{d}]" if d else ""), flush=True)
    if not ok: FAILS.append(n)
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", "", t.lower().replace("’", "'"))).strip()
def rough(h, r): return wer([norm(x) for x in r], [norm(x) for x in h]) * 100, cer([norm(x) for x in r], [norm(x) for x in h]) * 100

simba = create_dataset("simbabench", local_path="/work/nvme/bfbf/shared/datasets_16k/SimbaBench",
                       languages=["rw_rw", "wo_sn", "yo_ng", "am_et"])
def take(code, n=16): return simba.get_language_samples(code)[:n]

# ---------- w2v-BERT Kinyarwanda
from stt_benchmark.models.hf.w2v_bert_ctc import W2vBertCtcModel
t = time.time()
m = W2vBertCtcModel("badrex/w2v-bert-2.0-kinyarwanda-asr", {"revision": "fca35cff2c827d02f2e87c29a15abd358cd69f25",
        "cache_dir": CACHE, "torch_dtype": "float32", "languages": ["rw_rw"]})
print("loaded w2v-bert in %.0fs" % (time.time() - t), m.get_model_info())
S = take("rw_rw"); H = m.transcribe(S, "rw_rw")
check("w2v-bert: one non-empty hypothesis per clip", len(H) == len(S) and sum(1 for h in H if h) >= len(S) - 1, f"{sum(1 for h in H if not h)} empty")
print("  rough WER/CER on 16 Kinyarwanda clips (Common Voice 19, smoke only): %.1f / %.1f" % rough(H, [s.transcription for s in S]))
for s, h in list(zip(S, H))[:3]: print("   REF:", s.transcription, "\n   HYP:", h)
check("w2v-bert: same hypothesis when decoded alone vs in a list", m.transcribe(S[:4], "rw_rw") == [m.transcribe([x], "rw_rw")[0] for x in S[:4]])
try: m.transcribe(take("am_et", 2), "am_et"); check("w2v-bert: unsupported language raises", False)
except ValueError: check("w2v-bert: unsupported language raises", True)
del m; torch.cuda.empty_cache()

# ---------- SunflowerASR
from stt_benchmark.models.hf.sunflower_whisper import SunflowerWhisperModel, REPO_TO_SUNBIRD
t = time.time()
m = SunflowerWhisperModel("Sunbird/SunflowerASR-51-african-languages", {"revision": "4583c9dd251c422a4325fb0a0a509a3828097be2",
        "cache_dir": CACHE, "torch_dtype": "float16", "batch_size": 8, "generation_config": {"max_new_tokens": 384}})
print("loaded sunflower in %.0fs" % (time.time() - t), m.get_model_info())
print("  language-token table check (iso3: id -> tokenizer string), covered languages:",
      {v: m.token_report[v] for v in sorted(set(REPO_TO_SUNBIRD.values()))})
for code in ["rw_rw", "wo_sn", "yo_ng", "am_et"]:
    S = take(code); H = m.transcribe(S, code)
    check(f"sunflower {code}: one non-empty hypothesis per clip", len(H) == len(S) and sum(1 for h in H if h) >= len(S) - 1, f"{sum(1 for h in H if not h)} empty")
    print("  rough WER/CER on 16 clips of %s (smoke only): %.1f / %.1f" % ((code,) + rough(H, [s.transcription for s in S])))
    if code == "rw_rw":
        for s, h in list(zip(S, H))[:3]: print("   REF:", s.transcription, "\n   HYP:", h)
S = take("rw_rw", 8)
check("sunflower: batch of 8 == singles (batch invariance, fp16)", m.transcribe(S, "rw_rw") == [m.transcribe([x], "rw_rw")[0] for x in S])
# the forced language token must actually steer decoding: the same Kinyarwanda audio forced to Wolof/Swahili must change
H_rw = m.transcribe(S, "rw_rw"); H_wo = m.transcribe(S, "wo_sn"); H_sw = m.transcribe(S, "sw_ke")
refs = [s_.transcription for s_ in S]
w_rw, w_wo, w_sw = rough(H_rw, refs)[0], rough(H_wo, refs)[0], rough(H_sw, refs)[0]
check("sunflower: forcing a different language token changes the output and hurts (token is really used)",
      H_rw != H_wo and H_rw != H_sw and w_wo > w_rw + 10 and w_sw > w_rw + 10, f"WER rw {w_rw:.1f} | forced wo {w_wo:.1f} | forced sw {w_sw:.1f}")
for lang in ["ti_et", "fon_bj"]:
    try: m.transcribe(S[:1], lang); check(f"sunflower: language {lang} without a token raises", False)
    except ValueError: check(f"sunflower: language {lang} without a token raises", True)
long = AudioSample(transcription="x", language="rw_rw", sample_id="long", audio_array=np.zeros(16000 * 31, dtype=np.float32) + 0.01, sampling_rate=16000)
try: m.transcribe([long], "rw_rw"); check("sunflower: audio > 30 s raises (no silent truncation)", False)
except ValueError: check("sunflower: audio > 30 s raises (no silent truncation)", True)
# ---------- Ethio-ASR (joint ASR + LID, language tag stripped and recorded)
del m; torch.cuda.empty_cache()
t = time.time()
m = W2vBertCtcModel("badrex/Ethio-ASR-multilingual-600M", {"revision": "5a099192a8a56f118d3515f70f5f44428fd29730", "cache_dir": CACHE,
        "torch_dtype": "float32", "languages": ["am_et", "ti_et", "om_et"], "language_tags": {"am_et": "[AMH]", "ti_et": "[TIR]", "om_et": "[ORM]"}})
print("loaded ethio-asr in %.0fs" % (time.time() - t))
S = take("am_et"); H = m.transcribe(S, "am_et")
check("ethio: non-empty hypotheses, no tag left in the text", len(H) == len(S) and all(h and not h.startswith("[") for h in H), f"{sum(1 for h in H if not h)} empty")
tags = [r["predicted_tag"] for r in m.tag_log]
check("ethio: predicted language tag recorded for every clip", len(tags) == len(S) and all(t_ is not None for t_ in tags), f"tags: {sorted(set(map(str, tags)))}")
print("  Amharic: LID match %d/%d | rough WER/CER on 16 SimbaBench Amharic clips (ALFFA/CV, out of domain, smoke only): %.1f / %.1f" % (
    sum(1 for r in m.tag_log if r["predicted_tag"] == r["expected_tag"]), len(m.tag_log), *rough(H, [x.transcription for x in S])))
for x, h in list(zip(S, H))[:2]: print("   REF:", x.transcription, "\n   HYP:", h)
check("ethio: batch == singles", m.transcribe(S[:4], "am_et") == [m.transcribe([x], "am_et")[0] for x in S[:4]])
try: m.transcribe(take("rw_rw", 2), "rw_rw"); check("ethio: unsupported language raises", False)
except ValueError: check("ethio: unsupported language raises", True)
print("\nLEADERBOARD MODELS:", "ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
