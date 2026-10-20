#!/usr/bin/env python3
"""Leaderboard runner: one model x the audited, cleaned groups, scored with strict_scoring (policy v1).

Design rules (nothing is hidden):
  * language not supported by the model  -> written as status "unsupported"; never scored, never silently skipped;
  * a batch that raises                  -> its utterances get hypothesis None (counted, reported); no retry loop;
  * a model returning the wrong number of outputs, or a non-string -> RuntimeError (adapter bug, stop);
  * failures > 0.2% or an empty reference after normalisation -> status "aborted" with the reason, predictions kept;
  * empty-string hypotheses are COUNTED (n_empty_hyp) because some adapters return "" on internal errors;
  * every result.json carries the scorer provenance, the dataset info, the model info, the code commit, the SLURM job.
  * --limit runs (debug) are marked "limited": true and must never be used for the leaderboard.
Output layout: <out_dir>/<model_id>/<group>/{predictions.csv, result.json}, <out_dir>/<model_id>/run_info.json,
               <out_dir>/<model_id>/summary.jsonl (one line per group).
Resume: a group with an existing result.json is skipped unless --force.
"""
import argparse, csv, fnmatch, json, os, platform, subprocess, sys, time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stt_benchmark.datasets.registry import create_dataset
from stt_benchmark.evaluation import strict_scoring as S
from stt_benchmark.evaluation.config import load_eval_config

DEFAULT_CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "leaderboard_25langs.yaml")


def _git_commit() -> str:
    try:
        root = os.path.join(os.path.dirname(__file__), "..")
        c = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        return c + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _public(res: Dict[str, object]) -> Dict[str, object]:
    return {k: (float(v) if hasattr(v, "item") else v) for k, v in res.items() if not k.startswith("_")}


def audio_seconds(samples) -> float:
    import soundfile as sf
    tot = 0.0
    for s in samples:
        tot += (len(s.audio_array) / s.sampling_rate) if s.audio_array is not None else sf.info(s.audio_path).frames / sf.info(s.audio_path).samplerate
    return tot


def infer(model, samples, language: str, batch_size: int):
    """Returns (hypotheses with None for failed utterances, list of error strings, seconds)."""
    hyps: List[Optional[str]] = []
    errors: List[str] = []
    t0 = time.time()
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        try:
            out = model.transcribe(batch, language)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {str(e)[:200]}")
            hyps.extend([None] * len(batch))
            continue
        if len(out) != len(batch) or any(not isinstance(h, str) for h in out):
            raise RuntimeError(f"adapter bug: {len(out)} outputs for {len(batch)} inputs (or non-string output)")
        hyps.extend(out)
    return hyps, errors, time.time() - t0


def run_model(model, model_id: str, specs, out_dir: str, batch_size: int = 1, n_boot: int = 1000,
              limit: int = 0, force: bool = False, groups: Optional[List[str]] = None,
              model_info: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    mdir = os.path.join(out_dir, model_id)
    os.makedirs(mdir, exist_ok=True)
    info = {"model_id": model_id, "model_info": model_info or model.get_model_info(), "git": _git_commit(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": platform.node(), "batch_size": batch_size,
            "n_boot": n_boot, "limit": limit, "scorer": S.provenance(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(info, open(os.path.join(mdir, "run_info.json"), "w"), indent=1, default=str)
    summary = []
    for spec in specs:
        name = spec.dataset_name
        if groups and not any(fnmatch.fnmatch(name, g) for g in groups):
            continue
        lang = spec.asr_languages[0]
        gdir = os.path.join(mdir, name)
        rpath = os.path.join(gdir, "result.json")
        if os.path.exists(rpath) and not force:
            print(f"[skip existing] {name}", flush=True)
            summary.append(json.load(open(rpath)))
            continue
        os.makedirs(gdir, exist_ok=True)
        base = {"group": name, "lang": lang, "model_id": model_id, "limited": bool(limit)}
        if not model.supports_language(lang):
            res = dict(base, status="unsupported")
        else:
            kw = dict(spec.dataset_kwargs)
            if limit and name.startswith("cleaned__"):
                kw["limit"] = limit                                    # verify/decode only what will be used
            ds = create_dataset(name, **kw)
            samples = ds.get_language_samples(lang)
            if limit:
                samples = samples[:limit]
            hyps, errors, secs = infer(model, samples, lang, batch_size)
            refs = [s.transcription for s in samples]
            with open(os.path.join(gdir, "predictions.csv"), "w", encoding="utf-8", newline="") as fo:
                w = csv.writer(fo)
                w.writerow(["sample_id", "reference", "hypothesis", "failed"])
                for s, h in zip(samples, hyps):
                    w.writerow([s.sample_id, s.transcription, "" if h is None else h, int(h is None)])
            res = dict(base, n_requested=len(samples), seconds=round(secs, 2), audio_seconds=round(audio_seconds(samples), 2), errors_sample=errors[:3],
                       n_empty_hyp=sum(1 for h in hyps if h == ""), dataset=ds.get_dataset_info(), model=info["model_info"],
                       scorer=info["scorer"], git=info["git"], slurm_job=info["slurm_job"])
            try:
                res.update(status="ok", **_public(S.score_group(hyps, refs, lang, n_boot=n_boot)))
            except S.ScoringAborted as e:
                res.update(status="aborted", reason=str(e), n_failed=sum(1 for h in hyps if h is None))
        json.dump(res, open(rpath, "w"), indent=1, default=str, ensure_ascii=False)
        with open(os.path.join(mdir, "summary.jsonl"), "a", encoding="utf-8") as fo:
            fo.write(json.dumps(res, default=str, ensure_ascii=False) + "\n")
        line = f"{name}: {res['status']}"
        if res["status"] == "ok":
            line += f"  WER {res['wer']:.2f} [{res['wer_ci_low']:.2f}-{res['wer_ci_high']:.2f}]  CER {res['cer']:.2f}  n={res['n']}  failed={res['n_failed']}"
        print(line, flush=True)
        summary.append(res)
    return summary


def plan(specs, groups):
    import csv as _csv
    gcsv = {r["group_key"]: r for r in _csv.DictReader(open(os.path.join(list(specs)[0].dataset_kwargs["manifest_dir"], "groups.csv")))}
    n = h = 0
    for s in specs:
        if groups and not any(fnmatch.fnmatch(s.dataset_name, g) for g in groups):
            continue
        r = gcsv[s.dataset_name]
        n += int(r["n_clean"]); h += float(r["hours_clean"])
        print(f"  {s.dataset_name:55s} {s.asr_languages[0]:6s} n={r['n_clean']:>6s}  {float(r['hours_clean']):6.2f} h")
    print(f"TOTAL selected: {n} utterances, {h:.1f} h of audio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--groups", nargs="*", help="glob patterns on group names, e.g. 'cleaned__fleurs__*'")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="DEBUG: first N utterances per group; marks results limited")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="print the plan (groups, utterances, hours); loads no model")
    a = ap.parse_args()
    cfg = load_eval_config(a.config)
    if a.dry_run:
        plan(cfg.datasets, a.groups); return 0
    from stt_benchmark.models.factory import ModelFactory
    model = ModelFactory.create_model(a.model_id)
    run_model(model, a.model_id, cfg.datasets, a.out_dir, a.batch_size, a.n_boot, a.limit, a.force, a.groups)
    return 0


if __name__ == "__main__":
    sys.exit(main())
