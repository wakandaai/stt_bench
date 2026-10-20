#!/usr/bin/env python3
"""Assemble the audited + cleaned evaluation groups into ONE uniform manifest set.

Input : EVAL_DATASET_STATS_25langs.csv (the approved list of groups, with clean counts) and the cleaned
        manifests written by build_clean_manifests.py / audit_simbabench.py.
Output: <out_dir>/groups.csv                      one row per group (key, language code, counts, manifest file)
        <out_dir>/<group_key>.tsv                 uniform schema (below), one per group
        <config_out>                              ready-to-run stt_bench eval config, one dataset block per group

Uniform manifest columns:
  sample_id, audio_kind (wav|parquet), audio_path, parquet_id, transcript (cleaned), transcript_raw, duration

Every count is cross-checked against the approved stats CSV; any mismatch or missing file raises.
"""
import argparse, csv, os, sys

STATS = "/work/nvme/bfbf/shared/datasets_16k/EVAL_DATASET_STATS_25langs.csv"
D = "/work/nvme/bfbf/etanghang/stt_bench_data"
SIMBA_ROOT = "/work/nvme/bfbf/shared/datasets_16k/SimbaBench/asr_test"

ISO_TO_CODE = {"afr": "af_za", "amh": "am_et", "ara": "ar_eg", "bem": "bem", "eng": "en_us", "fon": "fon_bj",
               "fra": "fr_fr", "hau": "ha_ng", "ibo": "ig_ng", "kin": "rw_rw", "lin": "ln_cd", "lug": "lg_ug",
               "mlg": "mg_mg", "nya": "ny_mw", "por": "pt_br", "sna": "sn_zw", "som": "so_so", "sot": "st_za",
               "swa": "sw_ke", "tir": "ti_et", "tsn": "tn_za", "wol": "wo_sn", "xho": "xh_za", "yor": "yo_ng",
               "zul": "zu_za"}
ISO_TO_NAME = {"afr": "afrikaans", "amh": "amharic", "ara": "arabic", "bem": "bemba", "eng": "english",
               "fra": "french", "hau": "hausa", "ibo": "igbo", "kin": "kinyarwanda", "lin": "lingala",
               "lug": "luganda", "mlg": "malagasy", "por": "portuguese", "sna": "shona", "sot": "sotho",
               "swa": "swahili", "tir": "tigrinya", "tsn": "tswana", "xho": "xhosa", "yor": "yoruba", "zul": "zulu"}
SCHEMA = ["sample_id", "audio_kind", "audio_path", "parquet_id", "transcript", "transcript_raw", "duration"]


def group_key(config_name):
    return "cleaned__" + config_name.replace("/", "__")


def read_tsv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=STATS)
    ap.add_argument("--out_dir", default=f"{D}/eval_manifests_25langs")
    ap.add_argument("--config_out", default="/projects/bfbf/etanghang/stt_bench/configs/leaderboard_25langs.yaml")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    stats = list(csv.DictReader(open(a.stats, encoding="utf-8")))
    groups, seen_keys = [], set()
    for r in stats:
        cfg, iso, n_clean = r["Config Name"], r["ISO"], int(r["# Samples (clean)"])
        parts = cfg.split("/")
        src = parts[0]
        key = group_key(cfg)
        if key in seen_keys:
            raise ValueError(f"duplicate group key {key}")
        seen_keys.add(key)
        if src == "simbabench":
            siso, ssrc = parts[1], parts[2]
            disk_src = "NCHTL" if ssrc == "NCHLT" else ssrc           # the release spells it NCHTL on disk
            stem = f"{siso}*{disk_src}"
            parquet = os.path.join(SIMBA_ROOT, f"HF_test-{stem}.parquet")
            if not os.path.isfile(parquet):
                raise FileNotFoundError(parquet)
            rows = read_tsv(f"{D}/simbabench_audit/{stem.replace('*', '__')}.clean.tsv")
            out = [dict(sample_id=f"simbabench_{x['benchmark_id']}", audio_kind="parquet", audio_path=parquet,
                        parquet_id=x["benchmark_id"], transcript=x["transcript"], transcript_raw=x["transcript_raw"],
                        duration=x["duration"]) for x in rows]
        else:
            if src == "fleurs":
                code = ISO_TO_CODE[iso]
                rows = read_tsv(f"{D}/fleurs_new/clean_eval/fleurs_new_{code}_test.tsv")
            else:
                rows = read_tsv(f"{D}/clean_eval/{src}_{ISO_TO_NAME[iso]}_test.tsv")
            out = [dict(sample_id=x["audio_id"], audio_kind="wav", audio_path=x["path"], parquet_id="",
                        transcript=x["transcript"], transcript_raw=x["transcript_raw"], duration=x["duration"])
                   for x in rows]
        if len(out) != n_clean:
            raise ValueError(f"{cfg}: manifest has {len(out)} rows, approved stats say {n_clean}")
        if len({o["sample_id"] for o in out}) != len(out):
            raise ValueError(f"{cfg}: duplicate sample ids")
        if any(not o["transcript"].strip() for o in out):
            raise ValueError(f"{cfg}: empty transcript")
        fn = f"{key}.tsv"
        with open(os.path.join(a.out_dir, fn), "w", encoding="utf-8", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=SCHEMA, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(out)
        groups.append(dict(group_key=key, config_name=cfg, language_code=ISO_TO_CODE[iso], iso=iso,
                           n_clean=n_clean, hours_clean=r["# Hours (clean)"], audio_kind=out[0]["audio_kind"],
                           manifest_file=fn))
    with open(os.path.join(a.out_dir, "groups.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(groups[0].keys()))
        w.writeheader()
        w.writerows(groups)

    with open(a.config_out, "w", encoding="utf-8") as fo:
        fo.write("# GENERATED by scripts/assemble_eval_manifests.py - one dataset block per audited, cleaned group.\n")
        fo.write("# Do not edit by hand; regenerate. References are the cleaned transcripts; audio is verified 16 kHz mono.\n")
        fo.write("experiment_name: leaderboard_25langs\n\ndatasets:\n")
        for g in groups:
            fo.write(f"  {g['group_key']}:\n    manifest_dir: {a.out_dir}\n    split: test\n"
                     f"    asr:\n      languages: [{g['language_code']}]\n")
    print(f"{len(groups)} groups, {sum(g['n_clean'] for g in groups)} utterances -> {a.out_dir}")
    print(f"config -> {a.config_out}")


if __name__ == "__main__":
    sys.exit(main())
