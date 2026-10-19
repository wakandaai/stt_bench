#!/usr/bin/env python3
"""Build cleaned evaluation manifests from the shared index (CPU only; run via srun/sbatch).

Applies the audit-approved rules in a fixed order. Every excluded utterance is written to
exclusions.tsv with its FIRST matching reason; nothing is dropped silently.

  R0 group-level: CV Afrikaans (test audio missing on disk), CV Swahili (speaker AND sentence
     overlap with train/dev; speaker-dependent split) are excluded as whole sets.
  R1 duration > 30 s                          -> 'gt30s'
  R2 reference annotation tags:
       strip  [um] [uhm] [cs] [pause] (case-insensitive)   (kept utterance, tag removed)
       drop   '[?]' (reference omits audible words) and any other [..]/<..>/{..} tag   -> 'tag'
  R3 reference clean-up: NFC, whitespace (incl. newlines) collapsed, edges stripped.
     Empty / no word character after cleaning   -> 'empty_ref'
  R4 misaligned text heuristics: chars/sec < 1 with duration >= 5 s, or > 45  -> 'chars_per_sec'
  R5 sentence also present (normalised) in train/dev of the same source+language   -> 'sentence_overlap'
  R6 duplicate audio inside the set (groups found by audit_eval_sets.py, F7): keep the first copy,
     drop the rest                                                              -> 'duplicate_audio'
  Near-silent clips (F4) are NOT excluded; they stay flagged in the audit.

Kept per row: transcript (cleaned), transcript_raw (as released), path (remapped to /work), speaker_id.
The speaker-disjointness of what remains is measured and written to the summary
(speaker_dependent=True means the release's split is not speaker-independent; it is NOT fixed here).
"""
import argparse, collections, csv, json, os, re, sys, unicodedata

OCEAN = "/ocean/projects/cis250145p/shared/datasets_16k/"
SHARED = "/work/nvme/bfbf/shared/datasets_16k/"
STRIP = {"[um]", "[uhm]", "[cs]", "[pause]"}
TAG = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WORDCH = re.compile(r"\w", re.UNICODE)
WHOLE_SET_EXCLUDED = {
    ("commonvoice", "afrikaans"): "test audio missing on disk (index lists 129 test rows, only train clips present)",
    ("commonvoice", "swahili"): "speaker overlap (266 speakers) and 64% sentence overlap with train/dev",
}


def norm(t):
    t = unicodedata.normalize("NFC", t).lower()
    return " ".join(_PUNCT.sub(" ", t).split())


def clean_ref(raw):
    """Return (cleaned_text, status) where status is None, 'tag' or 'empty_ref'."""
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
    if not WORDCH.search(t):
        return t, "empty_ref"
    return t, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=SHARED + "ASR_INDEX_V6_16k.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--audit_dir", required=True, help="output dir of audit_eval_sets.py on the RAW sets (for R6)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    test = collections.defaultdict(list)
    ref_txt = collections.defaultdict(set)
    ref_spk = collections.defaultdict(set)
    with open(a.index, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["source"] not in a.sources:
                continue
            k = (r["source"], r["language"])
            if r["split"] == "test":
                test[k].append(r)
            else:
                ref_txt[k].add(norm(r["transcript"]))
                if r["speaker_id"]:
                    ref_spk[k].add(r["speaker_id"])

    summary, excl_rows = {}, []
    for k in sorted(test):
        s, l = k
        rows = test[k]
        name = f"{s}|{l}"
        if k in WHOLE_SET_EXCLUDED:
            summary[name] = {"n_raw": len(rows), "n_clean": 0, "excluded_whole_set": WHOLE_SET_EXCLUDED[k]}
            continue
        reasons = collections.Counter()
        kept = []
        dup_drop = set()
        pj = os.path.join(a.audit_dir, f"rows_{s}_{l}_test.json")
        if os.path.isfile(pj):
            for grp in json.load(open(pj)).get("F7_duplicate_audio", []):
                dup_drop.update(grp.split(",")[1:])
        for r in rows:
            raw = r["transcript"]
            dur = float(r["duration"])
            why = None
            if r["audio_id"] in dup_drop:
                why = "duplicate_audio"
            elif dur > 30:
                why = "gt30s"
            else:
                txt, st = clean_ref(raw)
                if st:
                    why = st
                else:
                    cps = len(txt) / dur if dur > 0 else 0
                    if (cps < 1 and dur >= 5) or cps > 45:
                        why = "chars_per_sec"
                    elif norm(txt) in ref_txt[k] or norm(raw) in ref_txt[k]:
                        why = "sentence_overlap"
            if why:
                reasons[why] += 1
                excl_rows.append((s, l, r["audio_id"], why, raw[:120].replace("\t", " ").replace("\n", " ")))
            else:
                kept.append((r, txt))
        spk = {r["speaker_id"] for r, _ in kept if r["speaker_id"]}
        ov = spk & ref_spk[k]
        hours = sum(float(r["duration"]) for r, _ in kept) / 3600
        summary[name] = {
            "n_raw": len(rows), "n_clean": len(kept), "excluded": dict(reasons),
            "hours_clean": round(hours, 3), "n_speakers": len(spk),
            "speaker_ids_present": bool(spk),
            "speaker_dependent": bool(ov), "n_speakers_also_in_train_dev": len(ov),
            "small_n_lt_300": len(kept) < 300,
        }
        with open(os.path.join(a.out_dir, f"{s}_{l}_test.tsv"), "w", encoding="utf-8", newline="") as fo:
            w = csv.writer(fo, delimiter="\t", lineterminator="\n")
            w.writerow(["audio_id", "path", "transcript", "transcript_raw", "speaker_id", "duration", "source", "language"])
            for r, txt in kept:
                w.writerow([r["audio_id"], r["path"].replace(OCEAN, SHARED), txt, r["transcript"].replace("\t", " ").replace("\n", "\\n"),
                            r["speaker_id"], r["duration"], s, l])
        print(name, json.dumps(summary[name], ensure_ascii=False), flush=True)

    with open(os.path.join(a.out_dir, "exclusions.tsv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w.writerow(["source", "language", "audio_id", "reason", "reference_head"])
        w.writerows(excl_rows)
    with open(os.path.join(a.out_dir, "clean_summary.json"), "w") as fo:
        json.dump(summary, fo, indent=1, ensure_ascii=False)
    for n, v in summary.items():
        if v.get("excluded_whole_set"):
            print(n, "EXCLUDED WHOLE SET:", v["excluded_whole_set"])


if __name__ == "__main__":
    sys.exit(main())
