#!/usr/bin/env python3
"""Turn the downloaded FLEURS test split (+ train/dev transcripts for overlap checks) into an
index-format CSV that audit_eval_sets.py / build_clean_manifests.py can read (source = 'fleurs_new').

FLEURS tsv columns (no header): id, file_name, raw_transcription, transcription, phonemes, num_samples, gender.
We keep the RAW (cased, punctuated) transcription: normalisation is the scorer's job, never the data's.
Only test rows carry a real path; train/dev rows exist solely so the sentence-overlap check can see them.
FLEURS has no speaker ids, so speaker overlap is NOT checkable here (reported as such by the audit).
"""
import argparse, csv, os, sys
from huggingface_hub import hf_hub_download

CODES = ["af_za", "am_et", "ar_eg", "en_us", "fr_fr", "ha_ng", "ig_ng", "lg_ug", "ln_cd", "ny_mw",
         "pt_br", "sn_zw", "so_so", "sw_ke", "wo_sn", "xh_za", "yo_ng", "zu_za"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)      # .../FLEURS/google_fleurs_test
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = ["audio_id", "path", "transcript", "language", "split", "source", "speaker_id", "sample_rate", "duration"]
    n = {"test": 0, "dev": 0, "train": 0}
    with open(a.out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(cols)
        for c in CODES:
            for split in ("test", "dev", "train"):
                if split == "test":
                    tsv = os.path.join(a.root, "data", c, "test.tsv")
                else:
                    tsv = hf_hub_download("google/fleurs", f"data/{c}/{split}.tsv", repo_type="dataset",
                                          local_dir=os.path.join(a.root, "_split_tsv"), token=False)
                with open(tsv, newline="", encoding="utf-8") as fh:
                    for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                        fid, fname, raw, _norm, _ph, nsamp = row[0], row[1], row[2], row[3], row[4], row[5]
                        path = ""
                        if split == "test":
                            path = os.path.join(a.root, "data", c, "audio", "test_extracted", "test", fname)
                            if not os.path.isfile(path):          # extraction layout check: fail loudly
                                alt = os.path.join(a.root, "data", c, "audio", "test_extracted", fname)
                                if not os.path.isfile(alt):
                                    raise FileNotFoundError(f"{c}: {fname} not found under test_extracted")
                                path = alt
                        w.writerow([f"fleurs_{c}_{split}_{fid}_{fname}", path, raw, c, split, "fleurs_new", "",
                                    "16000", round(int(nsamp) / 16000, 3)])
                        n[split] += 1
    print("rows written:", n)


if __name__ == "__main__":
    sys.exit(main())
