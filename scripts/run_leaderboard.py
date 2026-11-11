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
Throughput on ONE GPU: several processes can decode the same model at once (`--shard i/n`: process i takes manifest rows
i, i+n, i+2n, ... of every group and writes predictions.shard<i>of<n>.csv + decode.shard<i>of<n>.json, no score); then
`--merge n` (no model, CPU only) checks that every shard of every group is complete, puts the hypotheses back in
manifest order and scores exactly as an unsharded run does. A batch that raises is re-run utterance by utterance so one
bad utterance costs one failure, not a whole batch; the failure count and threshold are unchanged.
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
            if len(batch) == 1:
                hyps.append(None)
                continue
            out = []                                    # isolate the bad utterance(s): redo the batch one by one
            for s in batch:
                try:
                    o1 = model.transcribe([s], language)
                except Exception as e1:
                    errors.append(f"{type(e1).__name__}: {str(e1)[:200]}")
                    out.append(None)
                    continue
                if len(o1) != 1 or not isinstance(o1[0], str):
                    raise RuntimeError(f"adapter bug: {len(o1)} outputs for 1 input (or non-string output)")
                out.append(o1[0])
            hyps.extend(out)
            continue
        if len(out) != len(batch) or any(not isinstance(h, str) for h in out):
            raise RuntimeError(f"adapter bug: {len(out)} outputs for {len(batch)} inputs (or non-string output)")
        hyps.extend(out)
    return hyps, errors, time.time() - t0


def run_model(model, model_id: str, specs, out_dir: str, batch_size: int = 1, n_boot: int = 1000,
              limit: int = 0, force: bool = False, groups: Optional[List[str]] = None,
              model_info: Optional[Dict[str, object]] = None,
              shard: Optional[tuple] = None, ready_file: Optional[str] = None) -> List[Dict[str, object]]:
    mdir = os.path.join(out_dir, model_id)
    os.makedirs(mdir, exist_ok=True)
    info = {"model_id": model_id, "model_info": model_info or model.get_model_info(), "git": _git_commit(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": platform.node(), "batch_size": batch_size,
            "n_boot": n_boot, "limit": limit, "scorer": S.provenance(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if shard:
        info["shard"] = list(shard)
    json.dump(info, open(os.path.join(mdir, f"run_info{_stag(shard)}.json" if shard else "run_info.json"), "w"), indent=1, default=str)
    if ready_file:
        open(ready_file, "w").write("ready\n")
    summary = []
    for spec in specs:
        name = spec.dataset_name
        if groups and not any(fnmatch.fnmatch(name, g) for g in groups):
            continue
        lang = spec.asr_languages[0]
        gdir = os.path.join(mdir, name)
        rpath = os.path.join(gdir, "result.json")
        dpath = os.path.join(gdir, f"decode{_stag(shard)}.json") if shard else None
        if shard and os.path.exists(dpath) and not force:
            print(f"[skip existing shard] {name}", flush=True)
            continue
        if os.path.exists(rpath) and not force:
            print(f"[skip existing] {name}", flush=True)
            summary.append(json.load(open(rpath)))
            continue
        os.makedirs(gdir, exist_ok=True)
        base = {"group": name, "lang": lang, "model_id": model_id, "limited": bool(limit)}
        if not model.supports_language(lang):
            if shard and shard[0] != 0:
                continue                                         # shard 0 records the unsupported group once
            res = dict(base, status="unsupported")
        else:
            kw = dict(spec.dataset_kwargs)
            if limit and name.startswith("cleaned__"):
                kw["limit"] = limit                                    # verify/decode only what will be used
            if shard:
                kw["shard"] = shard
            ds = create_dataset(name, **kw)
            samples = ds.get_language_samples(lang)
            if limit:
                samples = samples[:limit]
            hyps, errors, secs = infer(model, samples, lang, batch_size)
            refs = [s.transcription for s in samples]
            if shard:
                _write_predictions(os.path.join(gdir, f"predictions{_stag(shard)}.csv"), samples, hyps)
                part = dict(group=name, lang=lang, model_id=model_id, shard=list(shard), n=len(samples), seconds=round(secs, 2),
                            audio_seconds=round(audio_seconds(samples), 2), errors=errors[:5], n_errors=len(errors),
                            n_empty_hyp=sum(1 for h in hyps if h == ""), n_failed=sum(1 for h in hyps if h is None))
                json.dump(part, open(dpath, "w"), indent=1, default=str)
                print(f"{name}: decoded shard {shard[0]}/{shard[1]} n={len(samples)} failed={part['n_failed']} {secs:.0f}s", flush=True)
                summary.append(part)
                continue
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


def _stag(shard) -> str:
    return f".shard{shard[0]}of{shard[1]}"


def _write_predictions(path, samples, hyps):
    with open(path, "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["sample_id", "reference", "hypothesis", "failed"])
        for s, h in zip(samples, hyps):
            w.writerow([s.sample_id, s.transcription, "" if h is None else h, int(h is None)])


def merge_model(model_id: str, specs, out_dir: str, n_shards: int, n_boot: int = 1000,
                groups: Optional[List[str]] = None, force: bool = False) -> List[Dict[str, object]]:
    """Combine the shard outputs of every group into the same result.json an unsharded run writes (CPU only, no model).
    Raises if any shard of a supported group is missing, if the ids differ from the manifest, or if shards disagree."""
    mdir = os.path.join(out_dir, model_id)
    infos = [json.load(open(os.path.join(mdir, f"run_info{_stag((i, n_shards))}.json"))) for i in range(n_shards)]
    info = dict(infos[0])
    info.pop("shard", None)
    for k in ("model_info", "git", "batch_size", "n_boot", "limit", "scorer"):
        if any(json.dumps(x[k], sort_keys=True, default=str) != json.dumps(info[k], sort_keys=True, default=str) for x in infos):
            raise RuntimeError(f"shards of {model_id} disagree on '{k}'")
    info["slurm_jobs"] = sorted({str(x["slurm_job"]) for x in infos})
    info["n_shards"] = n_shards
    json.dump(info, open(os.path.join(mdir, "run_info.json"), "w"), indent=1, default=str)
    summary = []
    for spec in specs:
        name = spec.dataset_name
        if groups and not any(fnmatch.fnmatch(name, g) for g in groups):
            continue
        gdir = os.path.join(mdir, name)
        rpath = os.path.join(gdir, "result.json")
        if os.path.exists(rpath) and not force:
            summary.append(json.load(open(rpath)))
            continue
        lang = spec.asr_languages[0]
        parts = []
        for i in range(n_shards):
            dp = os.path.join(gdir, f"decode{_stag((i, n_shards))}.json")
            if not os.path.exists(dp):
                raise RuntimeError(f"{model_id}/{name}: shard {i}/{n_shards} missing ({dp})")
            parts.append(json.load(open(dp)))
        ds = create_dataset(name, **dict(spec.dataset_kwargs))
        ids = ds.manifest_ids()
        if info["limit"]:
            ids = ids[: info["limit"]]
        rows = {}
        for i in range(n_shards):
            for r in csv.DictReader(open(os.path.join(gdir, f"predictions{_stag((i, n_shards))}.csv"), encoding="utf-8", newline="")):
                if r["sample_id"] in rows:
                    raise RuntimeError(f"{name}: sample {r['sample_id']} in two shards")
                rows[r["sample_id"]] = r
        if set(rows) != set(ids) or len(rows) != len(ids):
            raise RuntimeError(f"{name}: shard rows ({len(rows)}) do not match the manifest ({len(ids)})")
        hyps = [None if rows[i]["failed"] == "1" else rows[i]["hypothesis"] for i in ids]
        refs = [rows[i]["reference"] for i in ids]
        with open(os.path.join(gdir, "predictions.csv"), "w", encoding="utf-8", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["sample_id", "reference", "hypothesis", "failed"])
            for i, h in zip(ids, hyps):
                w.writerow([i, rows[i]["reference"], "" if h is None else h, int(h is None)])
        base = {"group": name, "lang": lang, "model_id": model_id, "limited": bool(info["limit"])}
        res = dict(base, n_requested=len(ids), seconds=round(sum(p["seconds"] for p in parts), 2),
                   wall_seconds_max_shard=round(max(p["seconds"] for p in parts), 2), n_shards=n_shards,
                   audio_seconds=round(sum(p["audio_seconds"] for p in parts), 2),
                   errors_sample=[e for p in parts for e in p["errors"]][:3],
                   n_empty_hyp=sum(p["n_empty_hyp"] for p in parts), dataset=ds.get_dataset_info(), model=info["model_info"],
                   scorer=info["scorer"], git=info["git"], slurm_job=",".join(info["slurm_jobs"]))
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
    ap.add_argument("--shard", help="i/n: decode only manifest rows i::n of every group (writes shard files, no score)")
    ap.add_argument("--merge", type=int, metavar="N", help="combine the N shards of --model_id into scored results (no model)")
    ap.add_argument("--ready_file", help="touched once the model is loaded (lets a launcher stagger the other shards)")
    ap.add_argument("--dry_run", action="store_true", help="print the plan (groups, utterances, hours); loads no model")
    a = ap.parse_args()
    cfg = load_eval_config(a.config)
    if a.dry_run:
        plan(cfg.datasets, a.groups); return 0
    if a.merge:
        merge_model(a.model_id, cfg.datasets, a.out_dir, a.merge, a.n_boot, a.groups, a.force)
        return 0
    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    from stt_benchmark.models.factory import ModelFactory
    model = ModelFactory.create_model(a.model_id)
    run_model(model, a.model_id, cfg.datasets, a.out_dir, a.batch_size, a.n_boot, a.limit, a.force, a.groups,
              shard=shard, ready_file=a.ready_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
