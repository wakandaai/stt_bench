#!/usr/bin/env python3
"""Audit evaluation sets before any model is scored on them (CPU only; run via srun/sbatch).

Metadata comes from the shared index (ASR_INDEX_V6_16k.csv); every claim in it is
re-measured on the real files.  One result block per (source, language, split).

HARD checks (any failure => group FAIL, must be fixed or excluded before scoring):
  H1 file exists                          H5 no empty / whitespace-only transcript
  H2 header: 16 kHz, 1 channel, frames>0  H6 no duplicate audio_id
  H3 header duration == index duration    H7 decoded audio finite and not all-zero
     (+-0.05 s)                           H8 speaker-disjoint from train/dev of same source+language
  H4 header sr == index sr                H9 sentence-disjoint from train/dev of same source+language
                                             (normalised: NFC, lowercase, punctuation stripped)
FLAG checks (reported with counts + row list; heuristic thresholds, need a human decision):
  F1 duration > 30 s (Whisper-family window)   F5 chars/sec outside [3, 35]  (likely misaligned text)
  F2 duration < 0.5 s                          F6 transcript not NFC / has control chars / markup tokens
  F3 clipping: >1% samples at |x|>=0.999       F7 duplicate audio within the set (decoded-sample hash)
  F4 near-silent: RMS < -60 dBFS               F8 n_utts < 300
INFO (never a verdict): sentence overlap with train rows of OTHER sources (possible leakage into
our own training data), speaker-id availability, unicode-category census of transcripts.

Nothing is normalised or altered; this script only measures.
"""
import argparse, collections, csv, hashlib, json, os, re, sys, unicodedata
from multiprocessing import Pool

import numpy as np
import soundfile as sf

OCEAN = "/ocean/projects/cis250145p/shared/datasets_16k/"
SHARED = "/work/nvme/bfbf/shared/datasets_16k/"
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_MARKUP = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def norm(t):
    t = unicodedata.normalize("NFC", t).lower()
    return " ".join(_PUNCT.sub(" ", t).split())


def measure(job):
    """Per-file measurements. Pure function of the file; returns a dict."""
    path, idx_sr, idx_dur = job
    out = {"path": path, "exists": os.path.isfile(path)}
    if not out["exists"]:
        return out
    try:
        info = sf.info(path)
        out.update(sr=info.samplerate, ch=info.channels, frames=info.frames,
                   dur=info.frames / info.samplerate if info.samplerate else 0.0,
                   fmt=f"{info.format}/{info.subtype}")
        x, _ = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        out["finite"] = bool(np.isfinite(x).all())
        if x.size and out["finite"]:
            a = np.abs(x)
            out["peak"] = float(a.max())
            out["clip_frac"] = float((a >= 0.999).mean())
            rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
            out["rms_db"] = 20 * np.log10(rms) if rms > 0 else -200.0
            out["dc"] = float(x.mean())
            out["hash"] = hashlib.md5((x * 32767).astype(np.int16).tobytes()).hexdigest()
        else:
            out["rms_db"] = -200.0
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=SHARED + "ASR_INDEX_V6_16k.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--manifest_dir", default=None, help="audit cleaned manifests (build_clean_manifests.py output) instead of raw index rows")
    ap.add_argument("--max_per_group", type=int, default=0, help="debug only: cap rows per group (0 = all)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    sel = collections.defaultdict(list)                       # (source, lang, split) -> rows to measure
    ref_spk = collections.defaultdict(set)                    # (source, lang) -> train/dev speakers
    ref_txt = collections.defaultdict(set)                    # (source, lang) -> train/dev sentences
    train_txt_by_src = collections.defaultdict(lambda: collections.defaultdict(set))  # lang -> src -> train sents
    with open(a.index, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            s, l, sp = r["source"], r["language"], r["split"]
            if s in a.sources and sp in a.splits:
                sel[(s, l, sp)].append(r)
            if sp == "train":
                train_txt_by_src[l][s].add(norm(r["transcript"]))
            if s in a.sources and sp != "test" and sp not in a.splits:
                ref_spk[(s, l)].add(r["speaker_id"]); ref_txt[(s, l)].add(norm(r["transcript"]))
    if a.manifest_dir:
        # Audit the CLEANED manifests instead of the raw index rows. Same checks, same thresholds.
        # Extra: every manifest row must exist in the index test rows, and kept + excluded == raw.
        exc = collections.Counter()
        with open(os.path.join(a.manifest_dir, "exclusions.tsv"), newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                exc[(r["source"], r["language"])] += 1
        raw_ids = {k: {r["audio_id"] for r in rows} for k, rows in sel.items()}
        new_sel, mismatches = {}, []
        for (s, l, sp), rows in sel.items():
            p = os.path.join(a.manifest_dir, f"{s}_{l}_test.tsv")
            if not os.path.isfile(p):
                print(f"[no manifest] {s}|{l}: excluded as a whole set")
                continue
            with open(p, newline="", encoding="utf-8") as fh:
                m = list(csv.DictReader(fh, delimiter="\t"))
            ids = {r["audio_id"] for r in m}
            if not ids <= raw_ids[(s, l, sp)]:
                mismatches.append(f"{s}|{l}: {len(ids - raw_ids[(s, l, sp)])} manifest ids not in index")
            if len(m) + exc[(s, l)] != len(rows):
                mismatches.append(f"{s}|{l}: kept {len(m)} + excluded {exc[(s, l)]} != raw {len(rows)}")
            new_sel[(s, l, sp)] = [dict(audio_id=r["audio_id"], path=r["path"], transcript=r["transcript"],
                                        speaker_id=r["speaker_id"], duration=r["duration"], sample_rate="16000",
                                        source=s, language=l, split=sp) for r in m]
        sel = new_sel
        print("MANIFEST CROSS-CHECKS:", "OK (ids subset of index; kept+excluded==raw)" if not mismatches else mismatches, flush=True)

    report = {}
    for key in sorted(sel):
        s, l, sp = key
        rows = sel[key][: a.max_per_group] if a.max_per_group else sel[key]
        name = f"{s}|{l}|{sp}"
        rep = {"n": len(rows), "hard": [], "flags": {}, "info": {}}
        report[name] = rep
        jobs = [(r["path"].replace(OCEAN, SHARED), r["sample_rate"], r["duration"]) for r in rows]
        with Pool(a.workers) as pool:
            m = pool.map(measure, jobs, chunksize=64)

        bad = collections.defaultdict(list)
        ids = collections.Counter(r["audio_id"] for r in rows)
        hashes = collections.defaultdict(list)
        durs, cps = [], []
        fmts = collections.Counter(); srs = collections.Counter(); chs = collections.Counter()
        for r, x in zip(rows, m):
            rid = r["audio_id"]
            txt = r["transcript"]
            if not x["exists"]: bad["H1_missing"].append(rid); continue
            if "error" in x: bad["H2_unreadable"].append(f"{rid}: {x['error']}"); continue
            srs[x["sr"]] += 1; chs[x["ch"]] += 1; fmts[x["fmt"]] += 1
            if x["sr"] != 16000 or x["ch"] != 1 or x["frames"] <= 0: bad["H2_not_16k_mono"].append(rid)
            if abs(x["dur"] - float(r["duration"])) > 0.05: bad["H3_dur_mismatch"].append(rid)
            if str(x["sr"]) != str(r["sample_rate"]): bad["H4_index_sr_wrong"].append(rid)
            if not txt.strip(): bad["H5_empty_text"].append(rid)
            if not x.get("finite", True) or x.get("peak", 1.0) == 0.0: bad["H7_nonfinite_or_zero"].append(rid)
            if x["dur"] > 30: bad["F1_gt_30s"].append(rid)
            if x["dur"] < 0.5: bad["F2_lt_0.5s"].append(rid)
            if x.get("clip_frac", 0) > 0.01: bad["F3_clipping"].append(rid)
            if x.get("rms_db", 0) < -60: bad["F4_near_silent"].append(rid)
            if txt.strip() and x["dur"] > 0:
                c = len(txt.strip()) / x["dur"]; cps.append(c)
                if not (3 <= c <= 35): bad["F5_chars_per_sec"].append(f"{rid}:{c:.1f}")
            if unicodedata.normalize("NFC", txt) != txt or _CTRL.search(txt) or _MARKUP.search(txt):
                bad["F6_text_form"].append(rid)
            if "hash" in x: hashes[x["hash"]].append(rid)
            durs.append(x["dur"])
        dup_ids = [i for i, c in ids.items() if c > 1]
        if dup_ids: bad["H6_duplicate_ids"] = dup_ids
        dup_audio = [v for v in hashes.values() if len(v) > 1]
        if dup_audio: bad["F7_duplicate_audio"] = [",".join(v) for v in dup_audio]
        if len(rows) < 300: bad["F8_small_n"] = [str(len(rows))]

        # split hygiene vs the other splits of the same source+language
        spk = {r["speaker_id"] for r in rows if r["speaker_id"]}
        txts = {norm(r["transcript"]) for r in rows}
        rep["info"]["speaker_ids_present"] = f"{len(spk)} distinct" if spk else "NONE (speaker leakage cannot be checked)"
        if sp == "test":
            ov_s, ov_t = spk & ref_spk[(s, l)], txts & ref_txt[(s, l)]
            if ov_s: bad["H8_speaker_overlap"] = [f"{len(ov_s)} speakers"]
            if ov_t: bad["H9_sentence_overlap"] = [f"{len(ov_t)} sentences"]
            rep["info"]["sentence_overlap_with_train_of_other_sources"] = {
                o: len(txts & v) for o, v in train_txt_by_src[l].items() if o != s and txts & v}

        rep["srs"], rep["channels"], rep["formats"] = dict(srs), dict(chs), dict(fmts)
        if durs:
            d = np.array(durs); rep["hours"] = round(d.sum() / 3600, 3)
            rep["dur_s"] = {k: round(float(v), 2) for k, v in zip(("min", "p5", "median", "p95", "max"),
                            np.percentile(d, [0, 5, 50, 95, 100]))}
        if cps:
            rep["chars_per_sec"] = {k: round(float(v), 2) for k, v in zip(("p1", "median", "p99"),
                                    np.percentile(cps, [1, 50, 99]))}
        rep["hard"] = sorted(k for k in bad if k.startswith("H"))
        rep["flags"] = {k: len(v) for k, v in bad.items() if k.startswith("F")}
        rep["hard_counts"] = {k: len(v) for k, v in bad.items() if k.startswith("H")}
        rep["verdict"] = "FAIL" if rep["hard"] else ("FLAG" if rep["flags"] else "PASS")
        with open(os.path.join(a.out_dir, f"rows_{s}_{l}_{sp}.json"), "w") as fo:
            json.dump({k: v[:2000] for k, v in bad.items()}, fo, ensure_ascii=False)
        print(name, rep["verdict"], json.dumps({k: rep[k] for k in ("n", "hours", "srs", "channels", "hard_counts", "flags")
              if k in rep}, ensure_ascii=False), flush=True)

    with open(os.path.join(a.out_dir, "audit_summary.json"), "w") as fo:
        json.dump(report, fo, indent=1, ensure_ascii=False)
    print("FAIL:", sum(v["verdict"] == "FAIL" for v in report.values()), "FLAG:", sum(v["verdict"] == "FLAG" for v in report.values()),
          "PASS:", sum(v["verdict"] == "PASS" for v in report.values()))


if __name__ == "__main__":
    sys.exit(main())
