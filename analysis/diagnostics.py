#!/usr/bin/env python3
"""
§4 automatic diagnostics pass for the Aura-ASR error analysis.

Reads every included ASR prediction CSV (see analysis/common.py for the filter),
recomputes all metrics under the ONE shared normalizer, and writes one row per
(model, language, benchmark) to error_diagnostics.csv. This is the automatic
foundation the LLM judge sits on top of -- it populates the provenance layer
(Tier A rates) and the segmentation-tax / boundary / diacritic proxies so the
judge only handles genuine judgment calls.

No inference, no external services. Deterministic.

Usage:
  python analysis/diagnostics.py results/african_eval -o analysis/outputs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

from jiwer import cer as jiwer_cer, process_words, wer as jiwer_wer

from analysis import common as c


# =========================================================================
# Loading
# =========================================================================

def load_cell(path: Path) -> List[Dict[str, Any]]:
    """Load and normalize one prediction CSV. Drops rows with empty reference.

    Each returned row carries the shared-normalized ref/hyp plus the derived
    per-utterance features every downstream metric needs.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            ref = (raw.get("reference") or "").strip()
            hyp = (raw.get("hypothesis") or "").strip()  # may be empty (truncation)

            nref = c.NORMALIZER.normalize(ref)
            nhyp = c.NORMALIZER.normalize(hyp)
            if not nref.strip():
                continue  # no defined WER; matches ASRMetricsCalculator

            ref_words = nref.split()
            hyp_words = nhyp.split()
            n_ref = len(ref_words)
            len_ratio = (len(hyp_words) / n_ref) if n_ref else 0.0

            rows.append({
                "sample_id": raw.get("sample_id", ""),
                "nref": nref,
                "nhyp": nhyp,
                "n_ref_words": n_ref,
                "len_ratio": len_ratio,
                "tier_a": c.tier_a_label(nref, nhyp),
                # per-utterance CER/WER for the seg-dominated flag
                "u_cer": jiwer_cer(nref, nhyp) * 100 if nref else 0.0,
                "u_wer": jiwer_wer(nref, nhyp) * 100 if nref else 0.0,
                "nref_ns": c.remove_spaces(nref),
                "nhyp_ns": c.remove_spaces(nhyp),
            })
    return rows


# =========================================================================
# Corpus metrics
# =========================================================================

def _corpus_cer(refs: List[str], hyps: List[str]) -> float:
    return jiwer_cer(refs, hyps) * 100 if refs else 0.0


def summarize_cell(benchmark: str, model: str, lang: str,
                   rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    refs = [r["nref"] for r in rows]
    hyps = [r["nhyp"] for r in rows]
    info = c.lang_info(lang)

    # Corpus WER + I/D/S in one alignment pass.
    w = process_words(refs, hyps)
    wer = w.wer * 100
    total_ref_words = w.hits + w.substitutions + w.deletions

    cer = _corpus_cer(refs, hyps)
    scer = _corpus_cer([r["nref_ns"] for r in rows], [r["nhyp_ns"] for r in rows])

    # B2 proxy: WER recovered by stripping tone marks / diacritics from both sides.
    d_refs = [c.strip_diacritics(r["nref"]) for r in rows]
    d_hyps = [c.strip_diacritics(r["nhyp"]) for r in rows]
    wer_nodiac = process_words(d_refs, d_hyps).wer * 100

    # Per-utterance flag aggregates.
    len_ratios = [r["len_ratio"] for r in rows]
    n_a1 = sum(r["tier_a"] == "A1_script" for r in rows)
    n_a2 = sum(r["tier_a"] == "A2_loop" for r in rows)
    n_a3 = sum(r["tier_a"] == "A3_truncation" for r in rows)
    n_clean = sum(r["tier_a"] == "CLEAN" for r in rows)
    n_seg = sum(r["u_cer"] < c.SEG_CER_MAX and r["u_wer"] > c.SEG_WER_MIN for r in rows)
    # B1 boundary proxy: spaceless strings match but spaced strings differ
    # (words wrong ONLY because of segmentation).
    n_bound = sum(r["nref_ns"] == r["nhyp_ns"] and r["nref"] != r["nhyp"] for r in rows)

    pct = lambda k: round(100.0 * k / n, 2) if n else 0.0
    return {
        "benchmark": benchmark,
        "model": model,
        "lang": lang,
        "lang_name": info.name,
        "typology": info.typology,
        "ortho": info.ortho,
        "has_lm": c.MODEL_HAS_LM.get(model, ""),
        "n": n,
        "wer": round(wer, 2),
        "cer": round(cer, 2),
        "scer": round(scer, 2),
        "wer_cer_ratio": round(wer / cer, 2) if cer > 0 else None,
        "len_ratio_mean": round(mean(len_ratios), 3) if n else 0.0,
        "len_ratio_median": round(median(len_ratios), 3) if n else 0.0,
        # I/D/S as % of reference words
        "sub_pct": round(100.0 * w.substitutions / total_ref_words, 2) if total_ref_words else 0.0,
        "del_pct": round(100.0 * w.deletions / total_ref_words, 2) if total_ref_words else 0.0,
        "ins_pct": round(100.0 * w.insertions / total_ref_words, 2) if total_ref_words else 0.0,
        # Tier A provenance rates (pathologies, reported separately)
        "pct_A1_script": pct(n_a1),
        "pct_A2_loop": pct(n_a2),
        "pct_A3_truncation": pct(n_a3),
        "pct_clean": pct(n_clean),
        # Segmentation-tax detectors
        "pct_seg_dominated": pct(n_seg),
        "pct_boundary_exact": pct(n_bound),
        # B2 diacritic/tone proxy
        "wer_nodiacritic": round(wer_nodiac, 2),
        "wer_diacritic_recovered": round(wer - wer_nodiac, 2),
    }


# =========================================================================
# Output
# =========================================================================

FIELDS = [
    "benchmark", "model", "lang", "lang_name", "typology", "ortho", "has_lm", "n",
    "wer", "cer", "scer", "wer_cer_ratio",
    "len_ratio_mean", "len_ratio_median",
    "sub_pct", "del_pct", "ins_pct",
    "pct_A1_script", "pct_A2_loop", "pct_A3_truncation", "pct_clean",
    "pct_seg_dominated", "pct_boundary_exact",
    "wer_nodiacritic", "wer_diacritic_recovered",
]


def write_diagnostics(out_dir: Path, records: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "error_diagnostics.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"[ok] wrote {len(records)} cells -> {path}")


def write_file_log(out_dir: Path, included: List[Path], excluded: List[Path],
                   root: Path) -> None:
    for name, files in (("included_files.txt", included), ("excluded_files.txt", excluded)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for p in files:
                f.write(str(p.relative_to(root)) + "\n")
    print(f"[ok] logged {len(included)} included / {len(excluded)} excluded files")


# =========================================================================
# Main
# =========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="§4 automatic ASR error diagnostics.")
    ap.add_argument("results_dir", type=Path, help="e.g. results/african_eval")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("analysis/outputs"))
    args = ap.parse_args()

    if not args.results_dir.exists():
        print(f"[error] {args.results_dir} does not exist", file=sys.stderr)
        return 1

    included, excluded = c.collect_prediction_files(args.results_dir)
    if not included:
        print(f"[error] no included ASR CSVs under {args.results_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_file_log(args.output_dir, included, excluded, args.results_dir)

    records: List[Dict[str, Any]] = []
    for path in included:
        benchmark, model, lang = c.parse_path(path)
        rows = load_cell(path)
        if not rows:
            print(f"[warn] no scorable rows in {path}", file=sys.stderr)
            continue
        records.append(summarize_cell(benchmark, model, lang, rows))

    records.sort(key=lambda r: (r["benchmark"], r["lang"], r["model"]))
    write_diagnostics(args.output_dir, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
