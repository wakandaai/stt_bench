#!/usr/bin/env python3
"""Build + audit the Common Voice evaluation manifests (CPU only, run via srun).

Input : the shared ASR index (ASR_INDEX_V6_16k.csv), rows with source == commonvoice.
Output: <out_dir>/test_<code>.tsv  (audio_id, path, transcript, speaker_id, duration)
        <out_dir>/audit.json       (every check below, per language)

Checks per language (a failed HARD check aborts without writing that manifest):
  H1 every test clip exists on disk after path remap
  H2 every test clip header is 16 kHz, mono, frames > 0; header duration matches the index (+-0.05 s)
  H3 no empty transcript, no duplicate audio_id
  H4 test speakers are disjoint from CV train/dev speakers
  H5 test sentences (normalised) are disjoint from CV train/dev sentences
Reported (not hard): overlap of test sentences with train-split rows of ANY other source
in the index for the same language (possible leakage into our own training data).

Normalisation here is for overlap detection only. Nothing is normalised in the manifest.
"""
import argparse, collections, csv, json, os, re, sys, unicodedata

import soundfile as sf

OCEAN = "/ocean/projects/cis250145p/shared/datasets_16k/"
SHARED = "/work/nvme/bfbf/shared/datasets_16k/"
NAME_TO_CODE = {
    "afrikaans": "af_za", "amharic": "am_et", "hausa": "ha_ng", "igbo": "ig_ng",
    "kinyarwanda": "rw_rw", "luganda": "lg_ug", "swahili": "sw_ke", "yoruba": "yo_ng",
}
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t).lower()
    return " ".join(_PUNCT.sub(" ", t).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=SHARED + "ASR_INDEX_V6_16k.csv")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    test_rows = collections.defaultdict(list)
    cv_other_spk = collections.defaultdict(set)      # train/dev speakers (CV)
    cv_other_txt = collections.defaultdict(set)      # train/dev sentences (CV)
    other_src_txt = collections.defaultdict(lambda: collections.defaultdict(set))  # lang -> source -> train sentences
    n_cv = 0
    with open(a.index, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lang = r["language"]
            if lang not in NAME_TO_CODE:
                continue
            if r["source"] == "commonvoice":
                n_cv += 1
                if r["split"] == "test":
                    test_rows[lang].append(r)
                else:
                    cv_other_spk[lang].add(r["speaker_id"])
                    cv_other_txt[lang].add(norm(r["transcript"]))
            elif r["split"] == "train":
                other_src_txt[lang][r["source"]].add(norm(r["transcript"]))
    print(f"commonvoice rows read: {n_cv}", flush=True)

    audit = {}
    for lang in sorted(NAME_TO_CODE):
        code = NAME_TO_CODE[lang]
        rows = test_rows.get(lang, [])
        rep = {"code": code, "n_test_index": len(rows), "hard_failures": []}
        audit[code] = rep
        if not rows:
            rep["hard_failures"].append("no test rows in index")
            continue
        missing, bad_sr, bad_dur, empty, seen, dup = [], [], [], 0, set(), 0
        total_s = 0.0
        for r in rows:
            p = r["path"].replace(OCEAN, SHARED)
            r["_p"] = p
            if r["audio_id"] in seen:
                dup += 1
            seen.add(r["audio_id"])
            if not r["transcript"].strip():
                empty += 1
            if not os.path.isfile(p):
                missing.append(p)
                continue
            info = sf.info(p)
            if info.samplerate != 16000 or info.channels != 1 or info.frames <= 0:
                bad_sr.append((p, info.samplerate, info.channels, info.frames))
            if abs(info.duration - float(r["duration"])) > 0.05:
                bad_dur.append((p, info.duration, r["duration"]))
            total_s += info.duration
        rep.update(n_missing=len(missing), missing_examples=missing[:3],
                   n_not_16k_mono=len(bad_sr), not_16k_examples=bad_sr[:3],
                   n_duration_mismatch=len(bad_dur), n_empty_text=empty, n_duplicate_ids=dup,
                   hours=round(total_s / 3600, 3))
        if missing: rep["hard_failures"].append(f"H1 {len(missing)} missing files")
        if bad_sr: rep["hard_failures"].append(f"H2 {len(bad_sr)} not 16k mono")
        if bad_dur: rep["hard_failures"].append(f"H2 {len(bad_dur)} duration mismatches")
        if empty or dup: rep["hard_failures"].append(f"H3 empty={empty} dup={dup}")
        spk = {r["speaker_id"] for r in rows}
        rep["n_test_speakers"] = len(spk)
        rep["n_speaker_overlap_with_cv_train_dev"] = len(spk & cv_other_spk[lang])
        if rep["n_speaker_overlap_with_cv_train_dev"]:
            rep["hard_failures"].append("H4 speaker overlap with CV train/dev")
        txt = [norm(r["transcript"]) for r in rows]
        tset = set(txt)
        rep["n_unique_test_sentences"] = len(tset)
        rep["n_sentence_overlap_with_cv_train_dev"] = len(tset & cv_other_txt[lang])
        if rep["n_sentence_overlap_with_cv_train_dev"]:
            rep["hard_failures"].append("H5 sentence overlap with CV train/dev")
        rep["sentence_overlap_with_other_source_train"] = {
            s: len(tset & v) for s, v in other_src_txt[lang].items() if tset & v}
        if not rep["hard_failures"]:
            with open(os.path.join(a.out_dir, f"test_{code}.tsv"), "w", encoding="utf-8", newline="") as fo:
                w = csv.writer(fo, delimiter="\t", lineterminator="\n")
                w.writerow(["audio_id", "path", "transcript", "speaker_id", "duration"])
                for r in rows:
                    w.writerow([r["audio_id"], r["_p"], r["transcript"].strip(), r["speaker_id"], r["duration"]])
        print(code, json.dumps({k: v for k, v in rep.items() if k != "missing_examples"}, ensure_ascii=False), flush=True)

    with open(os.path.join(a.out_dir, "audit.json"), "w") as fo:
        json.dump(audit, fo, indent=1, ensure_ascii=False)
    n_fail = sum(1 for v in audit.values() if v["hard_failures"])
    print(f"languages with hard failures: {n_fail}/{len(audit)}")


if __name__ == "__main__":
    sys.exit(main())
