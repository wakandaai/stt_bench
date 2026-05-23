#!/usr/bin/env python3
"""
Summarize STT Benchmark Results

Scans experiment result directories for per-language/pair metric JSON files
and produces consolidated summary CSVs — one for ASR, one for AST.

Expects the new layout:
  results/<experiment>/<dataset>/metrics/<file>.json

Usage:
  python utils/summarize_results.py results/african_eval
  python utils/summarize_results.py results/
  python utils/summarize_results.py results/african_eval -o reports/
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any


def collect_metric_files(root: Path) -> List[Path]:
    """Recursively find all *_metrics.json files under root."""
    return sorted(root.rglob("*_metrics.json"))


def load_metric(path: Path) -> Dict[str, Any]:
    """Load a single metrics JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _derive_experiment_dataset(path: Path) -> tuple:
    """Derive (experiment, dataset) from a metrics file path.

    Expected layout: results/<experiment>/<dataset>/metrics/<file>.json
    Falls back gracefully if the dataset segment is missing.
    """
    # path.parent is .../metrics, parent.parent is the dataset dir,
    # parent.parent.parent is the experiment dir.
    try:
        dataset = path.parent.parent.name
        experiment = path.parent.parent.parent.name
    except Exception:
        dataset = ""
        experiment = path.parent.parent.name
    return experiment, dataset


def build_summaries(metric_files: List[Path]):
    """Parse metric files into separate ASR and AST record lists."""
    asr_rows: List[Dict[str, Any]] = []
    ast_rows: List[Dict[str, Any]] = []

    for path in metric_files:
        try:
            data = load_metric(path)
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  Skipping {path}: {e}", file=sys.stderr)
            continue

        task = data.get("task", "")
        experiment, dataset_from_path = _derive_experiment_dataset(path)
        # Prefer the dataset name written into the JSON (authoritative),
        # fall back to the path-derived one.
        dataset = data.get("dataset", dataset_from_path)

        if task == "asr":
            asr_rows.append({
                "experiment": experiment,
                "dataset": dataset,
                "model_name": data.get("model_name", ""),
                "language": data.get("language", ""),
                "wer": data.get("wer"),
                "cer": data.get("cer"),
                "num_samples": data.get("num_samples"),
                "total_time": data.get("total_time"),
                "avg_time_per_sample": data.get("avg_time_per_sample"),
            })
        elif task == "ast":
            ast_rows.append({
                "experiment": experiment,
                "dataset": dataset,
                "model_name": data.get("model_name", ""),
                "source_lang": data.get("source_lang", ""),
                "target_lang": data.get("target_lang", ""),
                "bleu": data.get("bleu"),
                "chrf": data.get("chrf"),
                "num_samples": data.get("num_samples"),
                "total_time": data.get("total_time"),
                "avg_time_per_sample": data.get("avg_time_per_sample"),
            })
        else:
            print(f"⚠️  Unknown task '{task}' in {path}", file=sys.stderr)

    asr_rows.sort(key=lambda r: (
        r["experiment"], r["dataset"], r["model_name"], r["language"]
    ))
    ast_rows.sort(key=lambda r: (
        r["experiment"], r["dataset"], r["model_name"],
        r["source_lang"], r["target_lang"],
    ))

    return asr_rows, ast_rows


def write_csv(rows: List[Dict[str, Any]], path: Path):
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Wrote {len(rows)} rows → {path}")


def print_asr_table(rows: List[Dict[str, Any]]):
    """Pretty-print ASR summary to stdout, grouped by (model, dataset)."""
    if not rows:
        return
    groups = sorted(set((r["model_name"], r["dataset"]) for r in rows))
    for model, dataset in groups:
        group_rows = [
            r for r in rows
            if r["model_name"] == model and r["dataset"] == dataset
        ]
        print(f"\n{'─'*60}")
        print(f"  Model: {model}   Dataset: {dataset}")
        print(f"{'─'*60}")
        print(f"  {'Language':<12} {'WER':>8} {'CER':>8} {'Samples':>8} {'Time(s)':>9}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*9}")
        for r in sorted(group_rows, key=lambda x: x["language"]):
            wer = f"{r['wer']:.2f}%" if r["wer"] is not None else "N/A"
            cer = f"{r['cer']:.2f}%" if r["cer"] is not None else "N/A"
            n = r["num_samples"] or ""
            t = f"{r['total_time']:.1f}" if r["total_time"] is not None else "N/A"
            print(f"  {r['language']:<12} {wer:>8} {cer:>8} {str(n):>8} {t:>9}")
        wers = [r["wer"] for r in group_rows if r["wer"] is not None]
        cers = [r["cer"] for r in group_rows if r["cer"] is not None]
        if wers:
            print(f"  {'─'*12} {'─'*8} {'─'*8}")
            print(f"  {'Average':<12} {sum(wers)/len(wers):>7.2f}% {sum(cers)/len(cers):>7.2f}%")


def print_ast_table(rows: List[Dict[str, Any]]):
    """Pretty-print AST summary to stdout, grouped by (model, dataset)."""
    if not rows:
        return
    groups = sorted(set((r["model_name"], r["dataset"]) for r in rows))
    for model, dataset in groups:
        group_rows = [
            r for r in rows
            if r["model_name"] == model and r["dataset"] == dataset
        ]
        print(f"\n{'─'*60}")
        print(f"  Model: {model}   Dataset: {dataset}")
        print(f"{'─'*60}")
        print(f"  {'Pair':<20} {'BLEU':>8} {'chrF++':>8} {'Samples':>8} {'Time(s)':>9}")
        print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*9}")
        for r in sorted(group_rows, key=lambda x: (x["source_lang"], x["target_lang"])):
            pair = f"{r['source_lang']}→{r['target_lang']}"
            bleu = f"{r['bleu']:.2f}" if r["bleu"] is not None else "N/A"
            chrf = f"{r['chrf']:.2f}" if r["chrf"] is not None else "N/A"
            n = r["num_samples"] or ""
            t = f"{r['total_time']:.1f}" if r["total_time"] is not None else "N/A"
            print(f"  {pair:<20} {bleu:>8} {chrf:>8} {str(n):>8} {t:>9}")
        bleus = [r["bleu"] for r in group_rows if r["bleu"] is not None]
        chrfs = [r["chrf"] for r in group_rows if r["chrf"] is not None]
        if bleus:
            print(f"  {'─'*20} {'─'*8} {'─'*8}")
            print(f"  {'Average':<20} {sum(bleus)/len(bleus):>8.2f} {sum(chrfs)/len(chrfs):>8.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize STT benchmark metric files into CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        help="Path to results directory (experiment or parent of experiments)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for CSVs (default: <results_dir>)",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Skip printing tables to stdout",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    metric_files = collect_metric_files(results_dir)
    if not metric_files:
        print(f"No *_metrics.json files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(metric_files)} metric file(s) under {results_dir}\n")

    asr_rows, ast_rows = build_summaries(metric_files)

    if asr_rows:
        write_csv(asr_rows, output_dir / "asr_summary.csv")
    if ast_rows:
        write_csv(ast_rows, output_dir / "ast_summary.csv")

    if not asr_rows and not ast_rows:
        print("No valid ASR or AST results found.")
        sys.exit(0)

    if not args.no_print:
        if asr_rows:
            print(f"\n{'='*60}")
            print("  ASR Results Summary")
            print(f"{'='*60}")
            print_asr_table(asr_rows)
        if ast_rows:
            print(f"\n{'='*60}")
            print("  AST Results Summary")
            print(f"{'='*60}")
            print_ast_table(ast_rows)

    print()


if __name__ == "__main__":
    main()