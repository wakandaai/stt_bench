#!/usr/bin/env python3
"""Behaviour tests for scripts/run_leaderboard.py with stub models (light: 3 small groups, seconds)."""
import csv, json, os, shutil, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_leaderboard as R
from stt_benchmark.evaluation.config import load_eval_config
from stt_benchmark.models.base import BaseASRModel

passes, fails = 0, []


def check(name, cond, detail=""):
    global passes
    if cond:
        passes += 1
    else:
        fails.append(f"{name} {detail}")
        print("FAIL", name, detail)


class Stub(BaseASRModel):
    def __init__(self, mode="oracle", fail_every=None, langs=None, n_fail=0):
        self.mode, self.fail_every, self.langs, self.n_fail, self.calls, self.failed = mode, fail_every, langs, n_fail, 0, 0
    def transcribe(self, samples, language):
        self.calls += 1
        if self.mode == "crash" or (self.mode == "flaky" and self.failed < self.n_fail and self.calls % 50 == 0):
            self.failed += 1
            raise RuntimeError("CUDA out of memory (stub)")
        if self.mode == "wronglen": return ["x"]
        if self.mode == "silent": return ["" for _ in samples]
        if self.mode == "empty10" and self.calls <= 10: return ["" for _ in samples]
        return [s.transcription for s in samples]
    def get_model_info(self): return {"model_name": f"stub_{self.mode}"}
    def get_supported_languages(self): return self.langs or {"__all__"}
    def supports_language(self, language): return True if self.langs is None else language in self.langs


cfg = load_eval_config(os.path.join(os.path.dirname(__file__), "..", "configs", "leaderboard_25langs.yaml"))
G = ["cleaned__fleurs__som", "cleaned__simbabench__swh__Alffa_Public", "cleaned__mls__por"]
specs = [s for s in cfg.datasets if s.dataset_name in G]
check("three test groups found", len(specs) == 3)
out = tempfile.mkdtemp(prefix="runner_test_")

# 1 oracle
r = R.run_model(Stub(), "oracle", specs, out, n_boot=50)
check("oracle: all ok", all(x["status"] == "ok" for x in r))
check("oracle: WER/CER 0, no failures", all(x["wer"] == 0.0 and x["cer"] == 0.0 and x["n_failed"] == 0 for x in r))
x = r[0]
check("result records audio seconds and decode seconds", x["audio_seconds"] > 0 and x["seconds"] >= 0, x.get("audio_seconds"))
check("result carries provenance", "scorer" in x and x["scorer"]["policy"] == "v1" and "git" in x and "dataset" in x)
check("predictions.csv written with failed column", os.path.exists(f"{out}/oracle/{x['group']}/predictions.csv") and
      next(csv.reader(open(f"{out}/oracle/{x['group']}/predictions.csv")))[-1] == "failed")
check("run_info.json written", "git" in json.load(open(f"{out}/oracle/run_info.json")))
check("summary.jsonl has one line per group", len(open(f"{out}/oracle/summary.jsonl").read().strip().split("\n")) == 3)

# 2 crash everywhere -> aborted, predictions kept
r = R.run_model(Stub("crash"), "crash", specs[:1], out, n_boot=50)
check("crash-all: aborted, not a score", r[0]["status"] == "aborted" and "wer" not in r[0], r[0].get("status"))
rows = list(csv.DictReader(open(f"{out}/crash/{specs[0].dataset_name}/predictions.csv")))
check("crash-all: every utterance marked failed", rows and all(x["failed"] == "1" for x in rows))

# 3 occasional failure: som has 1005 utterances; 1 failed (0.1%) tolerated, 5 failed (0.5%) aborts
r1 = R.run_model(Stub("flaky", n_fail=1), "flaky1", specs[:1], out, n_boot=50)
check("1/1005 failed -> ok and counted", r1[0]["status"] == "ok" and r1[0]["n_failed"] == 1, r1[0])
r5 = R.run_model(Stub("flaky", n_fail=5), "flaky5", specs[:1], out, n_boot=50)
check("5/1005 failed -> aborted", r5[0]["status"] == "aborted", r5[0].get("status"))

# 4 unsupported language
r = R.run_model(Stub(langs={"xx_xx"}), "unsup", specs, out, n_boot=50)
check("unsupported: explicit status, no score", all(x["status"] == "unsupported" and "wer" not in x for x in r))
check("unsupported: no predictions file", not os.path.exists(f"{out}/unsup/{specs[0].dataset_name}/predictions.csv"))

# 5 adapter bug
try:
    R.run_model(Stub("wronglen"), "bug", specs[:1], out, batch_size=4, n_boot=50)
    check("wrong-length output raises", False, "did not raise")
except RuntimeError:
    check("wrong-length output raises", True)

# 6 silent model: ok with WER 100, empties counted
r = R.run_model(Stub("silent"), "silent", specs[:1], out, n_boot=50)
check("silent model: WER 100, status ok", r[0]["status"] == "ok" and abs(r[0]["wer"] - 100) < 1e-6)
check("silent model: n_empty_hyp counted", r[0]["n_empty_hyp"] == r[0]["n"])

# 7 limit marks the result
r = R.run_model(Stub(), "limited", specs[:1], out, n_boot=50, limit=20)
check("limit run marked limited, n=20", r[0]["limited"] is True and r[0]["n"] == 20)

# 8 resume / force
before = os.path.getmtime(f"{out}/oracle/{specs[0].dataset_name}/result.json")
R.run_model(Stub(), "oracle", specs, out, n_boot=50)
check("resume skips existing", os.path.getmtime(f"{out}/oracle/{specs[0].dataset_name}/result.json") == before)
R.run_model(Stub(), "oracle", specs[:1], out, n_boot=50, force=True)
check("--force reruns", os.path.getmtime(f"{out}/oracle/{specs[0].dataset_name}/result.json") > before)

# 9 group filter
r = R.run_model(Stub(), "filt", cfg.datasets, out, n_boot=20, limit=5, groups=["cleaned__mls__por"])
check("group glob filter selects exactly one", len(r) == 1 and r[0]["group"] == "cleaned__mls__por")

# 10 plan totals equal the approved stats
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    R.plan(cfg.datasets, None)
check("plan totals match approved 89,662 utterances", "89662 utterances" in buf.getvalue(), buf.getvalue().strip().split("\n")[-1])

shutil.rmtree(out)
print(f"\n{passes} PASS, {len(fails)} FAIL")
for f in fails:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
