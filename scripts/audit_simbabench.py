#!/usr/bin/env python3
"""Audit + clean the SimbaBench ASR test release (local parquet, audio embedded). CPU only.

One group per parquet file = one (language, source). Same hard checks / flags / cleaning rules as
audit_eval_sets.py + build_clean_manifests.py, except what the release makes impossible:
  * the release is TEST-ONLY, so speaker/sentence overlap with train/dev (H8/H9) is NOT CHECKABLE;
    this is stated in the output, not silently passed;
  * no speaker ids are released.
Additional INFO check: sentence overlap of each group's references with the train/dev/test rows of
OUR index for the same language (possible leakage into our own training data; not an exclusion).

Hard (group FAIL):  decodable audio, header 16 kHz / mono / frames>0, header duration == duration_s (+-0.05 s),
                    non-empty text, unique benchmark_id (global), finite and non-silent-zero audio.
Exclusions (per utterance, first reason):  gt30s | tag | empty_ref | chars_per_sec | duplicate_audio
  (tags: strip [um] [uhm] [cs] [pause]; any other [..]/<..>/{..} -> drop; cleaning = NFC + whitespace collapse)
Flags (reported): >30 s, <0.5 s, clipping >1%, near-silent (< -60 dBFS), chars/sec outside [3,35],
                  non-NFC / control chars / markup, duplicate audio, n<300.
"""
import argparse, collections, csv, glob, hashlib, io, json, os, re, sys, unicodedata
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

STRIP = {"[um]", "[uhm]", "[cs]", "[pause]"}
TAG = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_P = re.compile(r"[^\w\s]", re.UNICODE)
WORDCH = re.compile(r"\w", re.UNICODE)


def norm(t):
    return " ".join(_P.sub(" ", unicodedata.normalize("NFC", t).lower()).split())


def clean_ref(raw):
    t = unicodedata.normalize("NFC", raw)
    drop = False

    def sub(m):
        nonlocal drop
        if m.group(0).lower() in STRIP:
            return " "
        drop = True
        return m.group(0)
    t = TAG.sub(sub, t)
    if drop:
        return t, "tag"
    t = " ".join(t.split())
    return (t, None) if WORDCH.search(t) else (t, "empty_ref")


def do_file(args):
    path, out_dir = args
    stem = os.path.basename(path)[len("HF_test-"):-len(".parquet")]
    rep = {"file": os.path.basename(path), "n_raw": 0, "hard": collections.Counter(), "flags": collections.Counter(),
           "excluded": collections.Counter(), "srs": collections.Counter(), "channels": collections.Counter()}
    kept, ids, hashes, norms, raw_hours = [], [], {}, [], 0.0
    lang_iso = lang_name = None
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            rep["n_raw"] += 1
            bid, raw = str(row["benchmark_id"]), (row.get("text") or "")
            lang_iso, lang_name = row.get("lang_iso3"), row.get("lang_name")
            ids.append(bid)
            audio = row.get("audio") or {}
            try:
                b = audio["bytes"]
                info = sf.info(io.BytesIO(b))
                x, _ = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
            except Exception:
                rep["hard"]["H1_undecodable"] += 1
                rep["excluded"]["undecodable"] += 1
                continue
            rep["srs"][info.samplerate] += 1; rep["channels"][info.channels] += 1
            if info.samplerate != 16000 or info.channels != 1 or info.frames <= 0:
                rep["hard"]["H2_not_16k_mono"] += 1
            dur = info.frames / info.samplerate
            raw_hours += dur / 3600
            if abs(dur - float(row["duration_s"])) > 0.05:
                rep["hard"]["H3_dur_mismatch"] += 1
            if not raw.strip():
                rep["hard"]["H5_empty_text"] += 1
            if x.ndim > 1:
                x = x.mean(axis=1)
            finite = bool(np.isfinite(x).all())
            if not finite or (x.size and float(np.abs(x).max()) == 0.0):
                rep["hard"]["H7_nonfinite_or_zero"] += 1
            a = np.abs(x) if finite and x.size else np.zeros(1)
            rms = float(np.sqrt((x.astype(np.float64) ** 2).mean())) if finite and x.size else 0.0
            rms_db = 20 * np.log10(rms) if rms > 0 else -200.0
            h = hashlib.md5((x * 32767).astype(np.int16).tobytes()).hexdigest() if finite else None
            txt, st = clean_ref(raw) if raw.strip() else ("", "empty_ref")
            cps = len(raw.strip()) / dur if dur > 0 else 0
            # flags (on raw)
            if dur > 30: rep["flags"]["F1_gt_30s"] += 1
            if dur < 0.5: rep["flags"]["F2_lt_0.5s"] += 1
            if float((a >= 0.999).mean()) > 0.01: rep["flags"]["F3_clipping"] += 1
            if rms_db < -60: rep["flags"]["F4_near_silent"] += 1
            if raw.strip() and not (3 <= cps <= 35): rep["flags"]["F5_chars_per_sec"] += 1
            if unicodedata.normalize("NFC", raw) != raw or CTRL.search(raw) or TAG.search(raw): rep["flags"]["F6_text_form"] += 1
            # exclusions (first reason)
            why = None
            if h is not None and h in hashes:
                why = "duplicate_audio"; rep["flags"]["F7_duplicate_audio"] += 1
            elif dur > 30:
                why = "gt30s"
            elif st:
                why = st
            elif (cps < 1 and dur >= 5) or cps > 45:
                why = "chars_per_sec"
            if h is not None and h not in hashes:
                hashes[h] = bid
            if why:
                rep["excluded"][why] += 1
            else:
                kept.append((bid, txt, raw, dur)); norms.append((bid, norm(txt), norm(raw)))
    rep["n_clean"] = len(kept)
    rep["hours_raw"] = round(raw_hours, 3)
    rep["hours_clean"] = round(sum(k[3] for k in kept) / 3600, 3)
    rep["lang_iso3"], rep["lang_name"] = lang_iso, lang_name
    if len(kept) < 300: rep["flags"]["F8_small_n"] = 1
    dup_ids = [i for i, c in collections.Counter(ids).items() if c > 1]
    if dup_ids: rep["hard"]["H6_duplicate_ids"] = len(dup_ids)
    with open(os.path.join(out_dir, f"{stem}.clean.tsv".replace("*", "__")), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w.writerow(["benchmark_id", "transcript", "transcript_raw", "duration"])
        for bid, t, r, d in kept:
            w.writerow([bid, t, r.replace("\t", " ").replace("\n", "\\n"), round(d, 3)])
    rep["hard"], rep["flags"] = dict(rep["hard"]), dict(rep["flags"])
    rep["excluded"], rep["srs"], rep["channels"] = dict(rep["excluded"]), dict(rep["srs"]), dict(rep["channels"])
    return stem, rep, norms, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/nvme/bfbf/shared/datasets_16k/SimbaBench/asr_test")
    ap.add_argument("--index", default="/work/nvme/bfbf/shared/datasets_16k/ASR_INDEX_V6_16k.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.root, "HF_test-*.parquet")))
    with Pool(a.workers) as p:
        res = p.map(do_file, [(f, a.out_dir) for f in files], chunksize=1)

    # global unique benchmark_id
    allids = collections.Counter(i for _, _, _, ids in res for i in ids)
    gdup = {i for i, c in allids.items() if c > 1}

    # INFO: overlap with OUR index (any split, any source) for the same language
    LANG = {"afr": "afrikaans", "amh": "amharic", "bem": "bemba", "hau": "hausa", "ibo": "igbo", "kin": "kinyarwanda",
            "lug": "luganda", "swa": "swahili", "swh": "swahili", "yor": "yoruba", "xho": "xhosa", "zul": "zulu",
            "sot": "sotho", "tsn": "tswana", "tso": "tsonga", "tir": "tigrinya", "nso": "northern sotho"}
    need = {LANG[r["lang_iso3"]] for _, r, _, _ in res if r["lang_iso3"] in LANG}
    idx = collections.defaultdict(lambda: collections.defaultdict(set))
    with open(a.index, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["language"] in need:
                idx[row["language"]][(row["source"], row["split"])].add(norm(row["transcript"]))

    summary = {}
    for stem, rep, norms, ids in res:
        if gdup & set(ids): rep["hard"]["H6_duplicate_ids_across_files"] = len(gdup & set(ids))
        l = LANG.get(rep["lang_iso3"])
        ov = {}
        if l:
            mine = {n for _, n, _ in norms} | {r for _, _, r in norms}
            ov = {f"{s}/{sp}": len(mine & v) for (s, sp), v in idx[l].items() if mine & v}
        rep["info_overlap_with_our_index"] = ov
        rep["not_checkable"] = "speaker/sentence overlap with train/dev (test-only release; no speaker ids)"
        rep["verdict"] = "FAIL" if rep["hard"] else ("FLAG" if rep["flags"] else "PASS")
        summary[stem] = rep
        print(stem, rep["verdict"], rep["lang_iso3"], "raw", rep["n_raw"], "clean", rep["n_clean"], "h", rep["hours_clean"],
              "hard", rep["hard"], "srs", rep["srs"], "excl", rep["excluded"], "ovl", ov, flush=True)
    with open(os.path.join(a.out_dir, "simbabench_summary.json"), "w") as fo:
        json.dump(summary, fo, indent=1, ensure_ascii=False)
    print("FAIL:", sum(v["verdict"] == "FAIL" for v in summary.values()), "FLAG:", sum(v["verdict"] == "FLAG" for v in summary.values()),
          "PASS:", sum(v["verdict"] == "PASS" for v in summary.values()))


if __name__ == "__main__":
    sys.exit(main())
