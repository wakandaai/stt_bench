#!/usr/bin/env python3
"""
Per-instance prediction analysis for STT Benchmark ASR runs.

Corpus WER hides the *shape* of the error distribution. This script walks the
per-instance prediction CSVs (which carry `cer,wer` per row) and, for every
language-benchmark cell, reports the distribution plus three diagnostic signals
that separate real model error from formatting artifact and from outright bugs:

  * collapse tail        - rows with WER >= 100 (truncations / loops / wrong
                           script). A handful of these can dominate corpus WER.
  * formatting signature - rows where WER is high but CER is low. Characters are
                           nearly right, words are not: case/punct/spelling/
                           morphology, i.e. a normalization win, not a model
                           failure.
  * length ratio         - len(hyp_words)/len(ref_words). <<1 flags collapse,
                           >>1 flags insertion/hallucination.

Two views are produced:
  1. A per-cell SUMMARY table (one row per language-benchmark CSV): n, corpus
     WER/CER, corpus WER with the top-1% worst rows trimmed, distribution
     percentiles, %exact, %collapse, %high-WER-low-CER, mean length ratio.
  2. An OUTLIERS table: the individual rows flagged as collapses or formatting
     artifacts, so they can be eyeballed and triaged (work items 1 & 2).

Inputs are read as-is; the `cer`/`wer` columns produced at inference time are
used when present, otherwise recomputed per row with jiwer. Everything here is
descriptive and dev-only.

Expected on-disk layout (produced by EvaluationPipeline):
  <results_dir>/<experiment>/<dataset>/predictions/<model>_<task>_<lang(s)>.csv
  columns: sample_id,reference,hypothesis,audio_path[,cer,wer]

Usage:
  # Analyze every cell under the dev results tree
  python utils/analyze_predictions.py results_dev/african_eval

  # Write the summary + outliers CSVs to a report dir
  python utils/analyze_predictions.py results_dev/african_eval -o reports/dev_analysis

  # Tune the "formatting artifact" rule (WER >= ratio * CER and CER below floor)
  python utils/analyze_predictions.py results_dev/african_eval \
      --fmt-wer-cer-ratio 3.0 --fmt-cer-max 15.0
"""

import argparse
import csv
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

from jiwer import wer as jiwer_wer, cer as jiwer_cer

from stt_benchmark.utils.text_normalize import DEFAULT_NORMALIZER


# =========================================================================
# CSV loading
# =========================================================================

def collect_prediction_files(root: Path) -> List[Path]:
    """Find every prediction CSV under <root>/.../predictions/*.csv."""
    return sorted(p for p in root.rglob("predictions/*.csv"))


def _derive_dataset_lang(path: Path) -> tuple:
    """Return (dataset, lang) from a predictions CSV path.

    Layout: .../<dataset>/predictions/<model>_<task>_<lang(s)>.csv
    The language tag is whatever follows the '..._asr_' / '..._transcribe_'
    marker in the stem; we fall back to the full stem if no marker is found.
    """
    try:
        dataset = path.parent.parent.name
    except Exception:
        dataset = ""

    stem = path.stem
    lang = stem
    for marker in ("_asr_", "_transcribe_"):
        idx = stem.rfind(marker)
        if idx != -1:
            lang = stem[idx + len(marker):]
            break
    return dataset, lang


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Load prediction rows, ensuring each has per-instance cer/wer/len_ratio.

    Uses the `cer`/`wer` columns when present and parseable; otherwise recomputes
    them per row from the normalized reference/hypothesis with jiwer. Rows with
    an empty reference are dropped (no defined WER), matching ASRMetricsCalculator.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            ref = (raw.get("reference") or "").strip()
            hyp = (raw.get("hypothesis") or "").strip()

            norm_ref = DEFAULT_NORMALIZER.normalize(ref)
            norm_hyp = DEFAULT_NORMALIZER.normalize(hyp)
            if not norm_ref.strip():
                continue

            cer = _to_float(raw.get("cer"))
            wer = _to_float(raw.get("wer"))
            if cer is None:
                cer = jiwer_cer(norm_ref, norm_hyp) * 100
            if wer is None:
                wer = jiwer_wer(norm_ref, norm_hyp) * 100

            ref_words = len(norm_ref.split())
            hyp_words = len(norm_hyp.split())
            len_ratio = (hyp_words / ref_words) if ref_words else 0.0

            rows.append({
                "sample_id": raw.get("sample_id", ""),
                "reference": ref,
                "hypothesis": hyp,
                "norm_ref": norm_ref,
                "norm_hyp": norm_hyp,
                "cer": cer,
                "wer": wer,
                "ref_words": ref_words,
                "hyp_words": hyp_words,
                "len_ratio": len_ratio,
            })
    return rows


# =========================================================================
# Statistics
# =========================================================================

def _percentile(sorted_vals: List[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]) on an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 100:
        return sorted_vals[-1]
    rank = int(round((q / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[rank]


def _corpus_wer(rows: List[Dict[str, Any]]) -> float:
    refs = [r["norm_ref"] for r in rows]
    hyps = [r["norm_hyp"] for r in rows]
    if not refs:
        return 0.0
    return jiwer_wer(refs, hyps) * 100


def _corpus_cer(rows: List[Dict[str, Any]]) -> float:
    refs = [r["norm_ref"] for r in rows]
    hyps = [r["norm_hyp"] for r in rows]
    if not refs:
        return 0.0
    return jiwer_cer(refs, hyps) * 100


def summarize_cell(
    dataset: str,
    lang: str,
    rows: List[Dict[str, Any]],
    collapse_wer: float,
    fmt_wer_cer_ratio: float,
    fmt_cer_max: float,
    trim_frac: float,
) -> Dict[str, Any]:
    """Compute the per-cell summary record."""
    n = len(rows)
    wers = sorted(r["wer"] for r in rows)
    len_ratios = [r["len_ratio"] for r in rows]

    n_exact = sum(1 for w in wers if w == 0.0)
    n_collapse = sum(1 for w in wers if w >= collapse_wer)
    n_fmt = sum(
        1 for r in rows
        if r["wer"] >= fmt_wer_cer_ratio * r["cer"] and r["cer"] <= fmt_cer_max
        and r["wer"] > 0
    )

    # Corpus WER after trimming the worst `trim_frac` rows by per-instance WER.
    keep = sorted(rows, key=lambda r: r["wer"])
    n_trim = int(n * trim_frac)
    trimmed_rows = keep[: n - n_trim] if n_trim < n else keep
    corpus_wer_trim = _corpus_wer(trimmed_rows)

    pct = lambda c: (100.0 * c / n) if n else 0.0
    return {
        "dataset": dataset,
        "lang": lang,
        "n": n,
        "corpus_wer": round(_corpus_wer(rows), 2),
        "corpus_cer": round(_corpus_cer(rows), 2),
        f"corpus_wer_trim{int(trim_frac*100)}pct": round(corpus_wer_trim, 2),
        "mean_wer": round(sum(wers) / n, 2) if n else 0.0,
        "median_wer": round(median(wers), 2) if n else 0.0,
        "p90_wer": round(_percentile(wers, 90), 2),
        "p99_wer": round(_percentile(wers, 99), 2),
        "pct_exact": round(pct(n_exact), 2),
        "pct_collapse": round(pct(n_collapse), 2),
        "pct_fmt_artifact": round(pct(n_fmt), 2),
        "n_collapse": n_collapse,
        "n_fmt_artifact": n_fmt,
        "mean_len_ratio": round(sum(len_ratios) / n, 3) if n else 0.0,
    }


def collect_outliers(
    dataset: str,
    lang: str,
    rows: List[Dict[str, Any]],
    collapse_wer: float,
    fmt_wer_cer_ratio: float,
    fmt_cer_max: float,
) -> List[Dict[str, Any]]:
    """Flag individual rows as 'collapse' and/or 'fmt_artifact' for triage."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        flags = []
        if r["wer"] >= collapse_wer:
            flags.append("collapse")
        if (r["wer"] >= fmt_wer_cer_ratio * r["cer"]
                and r["cer"] <= fmt_cer_max and r["wer"] > 0):
            flags.append("fmt_artifact")
        if not flags:
            continue
        out.append({
            "dataset": dataset,
            "lang": lang,
            "flags": "|".join(flags),
            "sample_id": r["sample_id"],
            "wer": round(r["wer"], 2),
            "cer": round(r["cer"], 2),
            "len_ratio": round(r["len_ratio"], 3),
            "ref_words": r["ref_words"],
            "hyp_words": r["hyp_words"],
            "reference": r["reference"],
            "hypothesis": r["hypothesis"],
        })
    # Worst (highest WER) first.
    out.sort(key=lambda d: d["wer"], reverse=True)
    return out


# =========================================================================
# Output
# =========================================================================

SUMMARY_FIELDS_PREFIX = ["dataset", "lang", "n", "corpus_wer", "corpus_cer"]


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        print(f"[warn] nothing to write to {path}", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[ok] wrote {len(records)} rows -> {path}")


def print_summary(records: List[Dict[str, Any]]) -> None:
    cols = [
        ("dataset", 16), ("lang", 7), ("n", 6),
        ("corpus_wer", 12), ("median_wer", 12), ("p99_wer", 9),
        ("pct_exact", 11), ("pct_collapse", 14), ("pct_fmt_artifact", 18),
        ("mean_len_ratio", 16),
    ]
    header = "".join(name.rjust(width) for name, width in cols)
    print(header)
    print("-" * len(header))
    for rec in records:
        line = "".join(str(rec.get(name, "")).rjust(width) for name, width in cols)
        print(line)


# =========================================================================
# Main
# =========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-instance WER/CER distribution analysis for ASR predictions.",
    )
    parser.add_argument("results_dir", type=Path,
                        help="Root to scan, e.g. results_dev/african_eval")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Directory for summary.csv and outliers.csv "
                             "(default: print only)")
    parser.add_argument("--collapse-wer", type=float, default=100.0,
                        help="WER threshold (>=) for a collapse/truncation outlier")
    parser.add_argument("--fmt-wer-cer-ratio", type=float, default=3.0,
                        help="Flag formatting artifact when WER >= ratio * CER")
    parser.add_argument("--fmt-cer-max", type=float, default=15.0,
                        help="...and CER is at or below this floor (chars ~right)")
    parser.add_argument("--trim-frac", type=float, default=0.01,
                        help="Fraction of worst rows to trim for corpus_wer_trim")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"[error] {args.results_dir} does not exist", file=sys.stderr)
        return 1

    files = collect_prediction_files(args.results_dir)
    if not files:
        print(f"[error] no predictions/*.csv under {args.results_dir}", file=sys.stderr)
        return 1

    summaries: List[Dict[str, Any]] = []
    outliers: List[Dict[str, Any]] = []
    for path in files:
        dataset, lang = _derive_dataset_lang(path)
        rows = load_rows(path)
        if not rows:
            print(f"[warn] no scorable rows in {path}", file=sys.stderr)
            continue
        summaries.append(summarize_cell(
            dataset, lang, rows,
            args.collapse_wer, args.fmt_wer_cer_ratio,
            args.fmt_cer_max, args.trim_frac,
        ))
        outliers.extend(collect_outliers(
            dataset, lang, rows,
            args.collapse_wer, args.fmt_wer_cer_ratio, args.fmt_cer_max,
        ))

    summaries.sort(key=lambda r: (r["dataset"], r["lang"]))
    print_summary(summaries)

    n_collapse = sum(r["n_collapse"] for r in summaries)
    n_fmt = sum(r["n_fmt_artifact"] for r in summaries)
    print(f"\nflagged across all cells: {n_collapse} collapses, "
          f"{n_fmt} formatting-artifact rows")

    if args.output_dir:
        write_csv(args.output_dir / "summary.csv", summaries)
        write_csv(args.output_dir / "outliers.csv", outliers)

    return 0


if __name__ == "__main__":
    sys.exit(main())
