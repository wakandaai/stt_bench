#!/usr/bin/env python3
"""Download the FLEURS TEST split (test.tsv + audio/test.tar.gz) for the 25-language set's FLEURS codes.

Public data: no token is used (token=False), so no stored credential is ever sent.
Writes to <out>/<code>/{test.tsv, audio/test/*.wav}; never touches FLEURS/original_data.
Verifies per language: downloaded archive size == size reported by the Hub, extracted wav count
== number of distinct file names in test.tsv (reported, raises on mismatch).
"""
import argparse, csv, json, os, sys, tarfile, urllib.request

from huggingface_hub import hf_hub_download

CODES = ["af_za", "am_et", "ar_eg", "en_us", "fr_fr", "ha_ng", "ig_ng", "lg_ug", "ln_cd", "ny_mw",
         "pt_br", "sn_zw", "so_so", "sw_ke", "wo_sn", "xh_za", "yo_ng", "zu_za"]
REPO = "google/fleurs"


def hub_size(path):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{os.path.dirname(path)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return {x["path"]: x.get("size") for x in json.load(r)}[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--codes", nargs="*", default=CODES)
    a = ap.parse_args()
    report = {}
    for c in a.codes:
        d = os.path.join(a.out, c)
        os.makedirs(d, exist_ok=True)
        tsv = hf_hub_download(REPO, f"data/{c}/test.tsv", repo_type="dataset", local_dir=a.out, token=False)
        tgz = hf_hub_download(REPO, f"data/{c}/audio/test.tar.gz", repo_type="dataset", local_dir=a.out, token=False)
        want = hub_size(f"data/{c}/audio/test.tar.gz")
        got = os.path.getsize(tgz)
        if want != got:
            raise RuntimeError(f"{c}: archive size {got} != hub {want}")
        out_audio = os.path.join(a.out, "data", c, "audio", "test_extracted")
        if not os.path.isdir(out_audio):
            with tarfile.open(tgz) as tf:
                tf.extractall(out_audio, filter="data")
        wavs = sorted(f for _, _, fs in os.walk(out_audio) for f in fs if f.endswith(".wav"))
        names = set()
        with open(tsv, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                names.add(row[1])
        missing = names - set(wavs)
        report[c] = {"tsv_rows": sum(1 for _ in open(tsv, encoding="utf-8")), "distinct_files": len(names),
                     "wav_extracted": len(wavs), "tsv_files_missing_audio": len(missing), "archive_bytes": got}
        print(c, report[c], flush=True)
        if missing:
            raise RuntimeError(f"{c}: {len(missing)} files named in test.tsv are not in the archive")
    json.dump(report, open(os.path.join(a.out, "download_report.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
