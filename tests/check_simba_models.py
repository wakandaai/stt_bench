"""Smoke/acceptance + language-conditioning probe for the Simba adapters (Simba-M CTC, Simba-S Seamless, Simba-W Whisper).
Run on a GPU node. Rough WER/CER here is a smoke signal only (16 clips, simple normaliser); the real scorer is separate."""
import collections, re, sys, time
sys.path.insert(0, ".")
import torch
from jiwer import wer, cer
from stt_benchmark.datasets.registry import create_dataset
from stt_benchmark.models.hf.simba import SimbaCtcModel, SimbaWhisperModel, SimbaSeamlessModel

CACHE = "/work/nvme/bfbf/etanghang/stt_bench_data/hf_models"
LANGS = ["af_za", "am_et", "ar_eg", "bem", "fon_bj", "fr_fr", "ha_ng", "ig_ng", "rw_rw", "ln_cd", "lg_ug", "mg_mg", "pt_br", "sn_zw",
         "so_so", "st_za", "sw_ke", "ti_et", "tn_za", "wo_sn", "xh_za", "yo_ng", "zu_za", "ny_mw"]
FAILS = []
def check(n, ok, d=""):
    print(("PASS " if ok else "FAIL ") + n + (f"  [{d}]" if d else ""), flush=True)
    if not ok: FAILS.append(n)
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", "", t.lower().replace("’", "'"))).strip()
def rough(h, r): return wer([norm(x) for x in r], [norm(x) for x in h]) * 100, cer([norm(x) for x in r], [norm(x) for x in h]) * 100

simba = create_dataset("simbabench", local_path="/work/nvme/bfbf/etanghang/stt_bench_data/simbabench",
                       languages=["rw_rw", "af_za", "wo_sn", "am_et", "sw_ke", "ha_ng"])
CLIPS = {c: simba.get_language_samples(c)[:16] for c in ["rw_rw", "af_za", "wo_sn", "am_et", "sw_ke", "ha_ng"]}
def run(m, tag):
    for c, S in CLIPS.items():
        t = time.time(); H = m.transcribe(S, c)
        check(f"{tag} {c}: one non-empty hypothesis per clip", len(H) == len(S) and sum(1 for h in H if h) >= len(S) - 1, f"{sum(1 for h in H if not h)} empty")
        print("   %s %s rough WER/CER %.1f / %.1f  (%.0fs)" % (tag, c, *rough(H, [x.transcription for x in S]), time.time() - t))
        if c == "af_za": print("   REF:", S[0].transcription, "\n   HYP:", H[0])
    try: m.transcribe(CLIPS["af_za"][:1], "en_us"); check(f"{tag}: language outside the set raises", False)
    except ValueError: check(f"{tag}: language outside the set raises", True)

CFG = dict(revision=None, cache_dir=CACHE, languages=LANGS)
# ---------------- Simba-M
t = time.time(); m = SimbaCtcModel("UBC-NLP/Simba-M", {**CFG, "revision": "f4010586dcdac8a2cf7cbde62bed512156843d7a", "adapter": "multilingual_african", "torch_dtype": "float32"})
print("loaded Simba-M in %.0fs" % (time.time() - t), m.get_model_info()); run(m, "Simba-M")
S = CLIPS["rw_rw"][:4]; check("Simba-M: list == singles", m.transcribe(S, "rw_rw") == [m.transcribe([x], "rw_rw")[0] for x in S])
del m; torch.cuda.empty_cache()

# ---------------- Simba-W
t = time.time(); m = SimbaWhisperModel("UBC-NLP/Simba-W", {**CFG, "revision": "2c9c6f54f9e362e5d8a7521575a64ef36df9b9b8", "torch_dtype": "float16", "batch_size": 8,
        "generation_config": {"max_new_tokens": 384}})
print("loaded Simba-W in %.0fs" % (time.time() - t), m.get_model_info()); run(m, "Simba-W")
by = collections.defaultdict(collections.Counter)
for r in m.lang_log: by[r["language"]][r["detected"]] += 1
print("   Simba-W language token the model produced (no language forced), per true language:"); [print("     ", k, dict(v)) for k, v in by.items()]
S = CLIPS["rw_rw"][:8]; B = m.transcribe(S, "rw_rw"); G = [m.transcribe([x], "rw_rw")[0] for x in S]
print("   Simba-W batch-vs-single mismatches on 8 Kinyarwanda clips:", sum(a != b for a, b in zip(B, G)), "(known: degenerate repetition loops differ in fp16)")
m.batch_size = 1; R1 = m.transcribe(S, "rw_rw"); R2 = m.transcribe(S, "rw_rw")
check("Simba-W: batch_size 1 is deterministic (two runs identical)", R1 == R2 == G)
del m; torch.cuda.empty_cache()

# ---------------- Simba-S
t = time.time(); m = SimbaSeamlessModel("UBC-NLP/Simba-S", {**CFG, "revision": "a87c8bf9f24edefe9df2d39fe3cb4163fa241b0f", "torch_dtype": "float16", "batch_size": 8,
        "generation_config": {"max_new_tokens": 256}})
print("loaded Simba-S in %.0fs" % (time.time() - t), m.get_model_info()); run(m, "Simba-S")
S = CLIPS["af_za"][:8]; check("Simba-S: batch of 8 == singles", m.transcribe(S, "af_za") == [m.transcribe([x], "af_za")[0] for x in S])
# language-conditioning probe: does a Seamless target-language code change the output, and what happens for codes it lacks?
for c, code in [("af_za", "afr"), ("sw_ke", "swh"), ("rw_rw", "kin"), ("rw_rw", "eng"), ("wo_sn", "wol")]:
    m.tgt_lang_map = {c: code}
    try:
        H = m.transcribe(CLIPS[c][:8], c); print("   Simba-S %s with tgt_lang=%s: rough WER/CER %.1f / %.1f | sample: %s" % (c, code, *rough(H, [x.transcription for x in CLIPS[c][:8]]), H[0][:80]))
    except Exception as e:
        print("   Simba-S %s with tgt_lang=%s raised %s: %s" % (c, code, type(e).__name__, str(e)[:100]))
m.tgt_lang_map = {}
print("\nSIMBA MODELS:", "ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
