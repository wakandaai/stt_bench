"""Acceptance check for the SimbaBench ASR adapter. Run on a compute node (decodes ~17 h of audio).
Exit code != 0 if any check fails."""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from stt_benchmark.datasets.registry import create_dataset

ROOT = "/work/nvme/bfbf/shared/datasets_16k/SimbaBench"
EXPECTED = {  # rows from the dataset card / HF file metadata (swa+swh = 500+500)
    "af_za": 1000, "am_et": 581, "bem": 1000, "fon_bj": 1000, "ha_ng": 681, "rw_rw": 1000, "lg_ug": 1000,
    "ny_mw": 428, "st_za": 1000, "sw_ke": 1000, "tn_za": 1000, "wo_sn": 1000, "xh_za": 1000, "yo_ng": 359, "zu_za": 1000,
}
FAILS = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok: FAILS.append(name)

ds = create_dataset("simbabench", local_path=ROOT, split="test", languages=sorted(EXPECTED))
check("list_languages == requested", ds.list_languages() == set(EXPECTED))
total_h = 0.0
for code, n in EXPECTED.items():
    t = time.time(); S = ds.get_language_samples(code)
    ids = [s.sample_id for s in S]
    sr_ok = all(s.sampling_rate == 16000 for s in S)
    dur = np.array([len(s.audio_array) / s.sampling_rate for s in S]); total_h += dur.sum() / 3600
    check(f"{code}: row count == {n}", len(S) == n, f"got {len(S)}")
    check(f"{code}: ids unique, text non-empty, audio 16 kHz and non-empty, language tag correct",
          len(set(ids)) == len(ids) and all(s.transcription for s in S) and sr_ok and dur.min() > 0 and all(s.language == code for s in S),
          f"{dur.sum()/3600:.2f} h, {dur.mean():.1f}s mean, {time.time()-t:.0f}s")
info = ds.get_dataset_info()["per_language"]
check("report: no row dropped for any language", all(v["rows_in_files"] == v["rows_used"] for v in info.values()),
      str({k: (v["rows_in_files"], v["rows_used"]) for k, v in info.items() if v["rows_in_files"] != v["rows_used"]}))
print("small-n (<300) languages:", [k for k, v in info.items() if v["small_n"]], "| total hours decoded: %.2f" % total_h)
for label, fn in [("unmapped language raises", lambda: ds.get_language_samples("ar_eg")),
                  ("language outside the allowed set raises", lambda: create_dataset("simbabench", local_path=ROOT, languages=["af_za"]).get_language_samples("am_et")),
                  ("non-test split raises", lambda: create_dataset("simbabench", local_path=ROOT, split="dev")),
                  ("missing path raises", lambda: create_dataset("simbabench", local_path="/nonexistent")),
                  ("unknown language in config raises", lambda: create_dataset("simbabench", local_path=ROOT, languages=["xx_yy"]))]:
    try: fn(); check(label, False, "no exception")
    except (ValueError, FileNotFoundError): check(label, True)
print("\nSIMBABENCH ADAPTER:", "ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
