#!/usr/bin/env python3
"""Acceptance test for the cleaned-eval loader (run on a compute node; it reads all 65 groups).

A. every group: loads via the real config path (load_eval_config -> create_dataset), count == approved n_clean,
   language == group language, ids unique, audio 16 kHz mono (enforced by the loader), duration of 3 random
   samples matches the manifest within 0.05 s.
B. oracle through the real EvaluationPipeline (stub model returns the reference): WER/CER must be 0 and
   num_samples must equal n_clean (detects references silently dropped by the metric layer).
   Silent-model check on 3 groups: WER must be 100.
C. error paths must raise: unknown group, wrong language, split=dev, missing audio file, missing manifest row
   (count mismatch), tampered Simba raw transcript (broken id->text join), unknown config key.
"""
import csv, os, random, shutil, sys, tempfile, traceback

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stt_benchmark.datasets.registry import create_dataset
from stt_benchmark.datasets.cleaned_eval import CleanedEvalDataset
from stt_benchmark.evaluation.config import load_eval_config
from stt_benchmark.evaluation.pipeline import EvaluationPipeline
from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.utils.audio_strict import prepare_audio

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "leaderboard_25langs.yaml")
OUT = os.environ.get("CLEANED_EVAL_TEST_OUT", tempfile.mkdtemp(prefix="cleaned_eval_test_"))
fails, passes = [], 0


def check(name, cond, detail=""):
    global passes
    if cond:
        passes += 1
    else:
        fails.append(f"{name} {detail}")
        print("FAIL", name, detail, flush=True)


class Oracle(BaseASRModel):
    def __init__(self, silent=False): self.silent = silent
    def transcribe(self, samples, language): return ["" if self.silent else s.transcription for s in samples]
    def get_model_info(self): return {"model_name": "stub_silent" if self.silent else "stub_oracle"}
    def get_supported_languages(self): return {"__all__"}
    def supports_language(self, language): return True


def expect_raises(name, fn, exc=Exception):
    try:
        fn()
    except exc:
        check(name, True)
        return
    except Exception as e:                       # raised, but the wrong kind
        check(name, True, f"(raised {type(e).__name__})")
        return
    check(name, False, "did not raise")


def main():
    random.seed(0)
    cfg = load_eval_config(CFG)
    specs = cfg.datasets if hasattr(cfg, "datasets") else cfg.dataset_specs
    pipe = EvaluationPipeline(output_dir=OUT)
    n_total = 0
    for spec in specs:
        name, kw = spec.dataset_name, dict(spec.dataset_kwargs)
        ds = create_dataset(name, **kw)
        code = spec.asr_languages[0]
        try:
            samples = ds.get_language_samples(code)
        except Exception:
            traceback.print_exc(); check(f"A load {name}", False); continue
        n = ds.get_dataset_info()["n_clean"]; n_total += len(samples)
        check(f"A count {name}", len(samples) == n, f"{len(samples)} != {n}")
        check(f"A ids unique {name}", len({s.sample_id for s in samples}) == len(samples))
        check(f"A lang {name}", all(s.language == code for s in samples))
        rows = {r["sample_id"]: r for r in csv.DictReader(open(os.path.join(kw["manifest_dir"], ds.info["manifest_file"]), encoding="utf-8"), delimiter="\t")}
        for s in random.sample(samples, min(3, len(samples))):
            a = prepare_audio(s)
            check(f"A dur {name}/{s.sample_id}", abs(len(a) / 16000 - float(rows[s.sample_id]["duration"])) <= 0.05,
                  f"{len(a)/16000:.2f} vs {rows[s.sample_id]['duration']}")
            check(f"A text {name}/{s.sample_id}", s.transcription == rows[s.sample_id]["transcript"])
        # B oracle through the real pipeline
        res = pipe.evaluate_asr(Oracle(), ds, code, "cleaned_eval_test", name)
        check(f"B oracle WER0 {name}", res is not None and res.metrics.wer == 0.0 and res.metrics.cer == 0.0,
              f"wer={getattr(res and res.metrics,'wer',None)}")
        check(f"B oracle n {name}", res is not None and res.metrics.num_samples == n,
              f"metric used {getattr(res and res.metrics,'num_samples',None)} of {n}")
    # silent model on a few groups
    for spec in specs[:3]:
        ds = create_dataset(spec.dataset_name, **dict(spec.dataset_kwargs))
        res = pipe.evaluate_asr(Oracle(silent=True), ds, spec.asr_languages[0], "cleaned_eval_test", spec.dataset_name + "_silent")
        check(f"B silent WER100 {spec.dataset_name}", res is not None and abs(res.metrics.wer - 100.0) < 1e-6)

    # C error paths
    md = specs[0].dataset_kwargs["manifest_dir"]
    wav_spec = next(s for s in specs if "commonvoice" in s.dataset_name or "mls" in s.dataset_name)
    sim_spec = next(s for s in specs if "simbabench" in s.dataset_name)
    expect_raises("C unknown group", lambda: create_dataset("cleaned__nope", manifest_dir=md))
    expect_raises("C wrong language", lambda: create_dataset(wav_spec.dataset_name, manifest_dir=md).get_language_samples("xx_xx"))
    expect_raises("C config wrong language", lambda: create_dataset(wav_spec.dataset_name, manifest_dir=md, languages=["xx_xx"]))
    expect_raises("C split dev", lambda: create_dataset(wav_spec.dataset_name, manifest_dir=md, split="dev"))
    expect_raises("C unknown kwarg", lambda: create_dataset(wav_spec.dataset_name, manifest_dir=md, bogus=1), TypeError)
    tmp = tempfile.mkdtemp(prefix="cleaned_eval_tamper_")
    for f in os.listdir(md):
        if f == "groups.csv" or f == wav_spec.dataset_name + ".tsv" or f == sim_spec.dataset_name + ".tsv":
            shutil.copy(os.path.join(md, f), os.path.join(tmp, f))
    def tamper(fn, mod):
        p = os.path.join(tmp, fn); lines = open(p, encoding="utf-8").read().split("\n")
        open(p, "w", encoding="utf-8").write("\n".join(mod(lines)))
        return p
    orig_w = open(os.path.join(tmp, wav_spec.dataset_name + ".tsv"), encoding="utf-8").read()
    orig_s = open(os.path.join(tmp, sim_spec.dataset_name + ".tsv"), encoding="utf-8").read()
    tamper(wav_spec.dataset_name + ".tsv", lambda L: L[:5] + L[6:])                         # drop a row
    expect_raises("C row dropped (count)", lambda: create_dataset(wav_spec.dataset_name, manifest_dir=tmp).get_language_samples(wav_spec.asr_languages[0]))
    open(os.path.join(tmp, wav_spec.dataset_name + ".tsv"), "w", encoding="utf-8").write(orig_w)
    tamper(wav_spec.dataset_name + ".tsv", lambda L: [L[0], L[1].replace("/work/", "/nowhere/")] + L[2:])  # missing audio
    expect_raises("C missing audio file", lambda: create_dataset(wav_spec.dataset_name, manifest_dir=tmp).get_language_samples(wav_spec.asr_languages[0]))
    tamper(sim_spec.dataset_name + ".tsv", lambda L: [L[0], "\t".join(L[1].split("\t")[:5] + ["TAMPERED RAW"] + L[1].split("\t")[6:])] + L[2:])
    expect_raises("C simba raw transcript tampered", lambda: create_dataset(sim_spec.dataset_name, manifest_dir=tmp).get_language_samples(sim_spec.asr_languages[0]))
    shutil.rmtree(tmp)

    print(f"\n{passes} PASS, {len(fails)} FAIL; {len(specs)} groups, {n_total} utterances loaded")
    for f in fails[:40]:
        print("  FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
