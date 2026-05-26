#!/usr/bin/env python3
"""
Recompute STT Benchmark metrics from existing prediction CSVs.

Walks a results directory, finds prediction CSVs, and updates the matching
metrics JSON files in place — no model inference required. Useful for:

  - Adding SSA-COMET to existing AST runs without re-translating
  - Recomputing BLEU/chrF after a sacrebleu upgrade or config change
  - Switching the ASR text normalizer and refreshing WER/CER
  - Adding spBLEU-1K to runs that predate the spBLEU support

Expected on-disk layout (produced by EvaluationPipeline):
  <results_dir>/<experiment>/<dataset>/predictions/<model>_<task>_<lang(s)>.csv
  <results_dir>/<experiment>/<dataset>/metrics/<model>_<task>_<lang(s)>_metrics.json

Usage:
  # Recompute all AST metrics for everything under results/
  python utils/recompute_metrics.py results/

  # Add SSA-COMET to one experiment, leave BLEU/chrF/spBLEU alone
  python utils/recompute_metrics.py results/african_eval \
      --task ast --metrics ssa_comet

  # Recompute just one cell of the matrix
  python utils/recompute_metrics.py results/african_eval \
      --model seamless_m4t_v2_large \
      --dataset fleurs \
      --source-lang yo_ng --target-lang en_us \
      --metrics bleu chrf spbleu ssa_comet

  # Preview without writing
  python utils/recompute_metrics.py results/ --dry-run
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from stt_benchmark.evaluation.metrics import (
    ASRMetricsCalculator, ASTMetricsCalculator, SsaCometScorer,
)
from stt_benchmark.utils.text_normalize import TextNormalizer


ALL_ASR_METRICS = {"wer", "cer"}
ALL_AST_METRICS = {"bleu", "chrf", "spbleu", "ssa_comet"}


# =========================================================================
# CSV → field lists
# =========================================================================

def _read_asr_csv(path: Path) -> Tuple[List[str], List[str]]:
    """Return (hypotheses, references) from an ASR predictions CSV."""
    hyps, refs = [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hyps.append(row.get("hypothesis", ""))
            refs.append(row.get("reference", ""))
    return hyps, refs


def _read_ast_csv(path: Path) -> Tuple[List[str], List[str], List[str]]:
    """Return (sources, hypotheses, references) from an AST predictions CSV.

    `sources` is the source_transcription column — needed by SSA-COMET as `src`.
    """
    srcs, hyps, refs = [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            srcs.append(row.get("source_transcription", ""))
            hyps.append(row.get("hypothesis", ""))
            refs.append(row.get("reference", ""))
    return srcs, hyps, refs


# =========================================================================
# File pairing
# =========================================================================

def _metrics_path_for_predictions(pred_csv: Path) -> Path:
    """Map .../predictions/<stem>.csv → .../metrics/<stem>_metrics.json."""
    ds_dir = pred_csv.parent.parent  # .../<dataset>
    return ds_dir / "metrics" / f"{pred_csv.stem}_metrics.json"


def _parse_pred_filename(pred_csv: Path) -> Optional[Dict[str, str]]:
    """Parse <model>_<task>_<lang(s)>.csv into its components.

    Returns None if the filename doesn't fit the expected layout. The model
    name itself may contain underscores (e.g. 'whisper_large_v3'), so we
    anchor on the task token which is either '_asr_' or '_ast_'.
    """
    stem = pred_csv.stem
    for task in ("asr", "ast"):
        marker = f"_{task}_"
        if marker not in stem:
            continue
        idx = stem.rfind(marker)
        model = stem[:idx]
        tail = stem[idx + len(marker):]
        if task == "asr":
            return {"model": model, "task": "asr", "language": tail}
        # AST tail = <src>_<tgt> where each is a FLEURS-style code that itself
        # may contain underscores (e.g. cmn_hans_cn). Split on the LAST '_'
        # by default — works for the vast majority of cases (af_za, en_us, …).
        # The metrics JSON is authoritative on src/tgt so we just need a
        # plausible split here for filtering.
        if "_" not in tail:
            return None
        src, tgt = tail.rsplit("_", 1)
        # If tail looks like 'cmn_hans_cn_en_us', the naive rsplit gives
        # ('cmn_hans_cn_en', 'us'), which is wrong. Cross-check against the
        # metrics JSON below to resolve ambiguity.
        return {"model": model, "task": "ast", "source_lang": src, "target_lang": tgt}
    return None


def _file_matches_filter(
    info: Dict[str, str],
    metrics_json: Dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    """Check a (parsed filename, loaded metrics) pair against CLI filters.

    Uses the metrics JSON as the authoritative source of model/lang ids
    (filename parsing is best-effort).
    """
    if args.task and info["task"] != args.task:
        return False

    model = metrics_json.get("model_name") or info.get("model", "")
    if args.model and args.model != model:
        return False

    dataset = metrics_json.get("dataset", "")
    if args.dataset and args.dataset != dataset:
        return False

    if info["task"] == "asr":
        lang = metrics_json.get("language") or info.get("language", "")
        if args.language and args.language != lang:
            return False
    else:  # ast
        src = metrics_json.get("source_lang") or info.get("source_lang", "")
        tgt = metrics_json.get("target_lang") or info.get("target_lang", "")
        if args.source_lang and args.source_lang != src:
            return False
        if args.target_lang and args.target_lang != tgt:
            return False

    return True


# =========================================================================
# Recompute
# =========================================================================

class MetricRecomputer:
    """Lazy holder for metric calculators — only constructs what's needed."""

    def __init__(
        self,
        bleu_config: Dict[str, Any],
        chrf_config: Dict[str, Any],
        normalizer_config: Optional[Dict[str, Any]],
        ssa_comet_batch_size: int,
        ssa_comet_model: str,
    ):
        self.bleu_config = bleu_config
        self.chrf_config = chrf_config
        self.normalizer_config = normalizer_config
        self.ssa_comet_batch_size = ssa_comet_batch_size
        self.ssa_comet_model = ssa_comet_model

        self._asr_calc: Optional[ASRMetricsCalculator] = None
        # Two AST calculators — one without spBLEU (cheap), one with. We
        # build whichever is needed and remember it. This keeps us from
        # downloading the spBLEU-1K SPM model if the user only asked for
        # bleu+chrf.
        self._ast_calc_no_spbleu: Optional[ASTMetricsCalculator] = None
        self._ast_calc_with_spbleu: Optional[ASTMetricsCalculator] = None
        self._comet: Optional[SsaCometScorer] = None

    def asr_calc(self) -> ASRMetricsCalculator:
        if self._asr_calc is None:
            normalizer = None
            if self.normalizer_config:
                normalizer = TextNormalizer(**self.normalizer_config)
            self._asr_calc = ASRMetricsCalculator(normalizer)
        return self._asr_calc

    def ast_calc(self, with_spbleu: bool) -> ASTMetricsCalculator:
        if with_spbleu:
            if self._ast_calc_with_spbleu is None:
                self._ast_calc_with_spbleu = ASTMetricsCalculator(
                    self.bleu_config, self.chrf_config, compute_spbleu=True,
                )
            return self._ast_calc_with_spbleu
        if self._ast_calc_no_spbleu is None:
            self._ast_calc_no_spbleu = ASTMetricsCalculator(
                self.bleu_config, self.chrf_config, compute_spbleu=False,
            )
        return self._ast_calc_no_spbleu

    def comet(self) -> SsaCometScorer:
        if self._comet is None:
            self._comet = SsaCometScorer(
                model_name=self.ssa_comet_model,
                batch_size=self.ssa_comet_batch_size,
            )
        return self._comet


def _recompute_asr(
    pred_csv: Path,
    metrics_json: Dict[str, Any],
    metrics_to_compute: set,
    rc: MetricRecomputer,
) -> Dict[str, Any]:
    """Return a dict of updates to apply to metrics_json."""
    hyps, refs = _read_asr_csv(pred_csv)
    if not hyps:
        print(f"  ⚠️  {pred_csv.name}: empty CSV, skipping")
        return {}

    updates: Dict[str, Any] = {}
    # WER and CER are computed together by ASRMetricsCalculator; if either
    # is requested we recompute both, since splitting them out would
    # duplicate normalization work for no gain.
    if metrics_to_compute & {"wer", "cer"}:
        m = rc.asr_calc().calculate(hyps, refs)
        if "wer" in metrics_to_compute:
            updates["wer"] = m.wer
        if "cer" in metrics_to_compute:
            updates["cer"] = m.cer
        # Refresh num_samples and the metric_config snapshot, but only when
        # we actually recomputed.
        updates["num_samples"] = m.num_samples
        existing_cfg = metrics_json.get("metric_config", {}) or {}
        existing_cfg.update(m.metric_config)
        updates["metric_config"] = existing_cfg

    return updates


def _recompute_ast(
    pred_csv: Path,
    metrics_json: Dict[str, Any],
    metrics_to_compute: set,
    rc: MetricRecomputer,
) -> Dict[str, Any]:
    """Return a dict of updates to apply to metrics_json."""
    srcs, hyps, refs = _read_ast_csv(pred_csv)
    if not hyps:
        print(f"  ⚠️  {pred_csv.name}: empty CSV, skipping")
        return {}

    updates: Dict[str, Any] = {}
    existing_cfg = metrics_json.get("metric_config", {}) or {}

    # Sacrebleu-based metrics: bleu, chrf, spbleu. These three share the
    # ASTMetricsCalculator. If any of them is requested, build the calc;
    # toggle compute_spbleu based on whether spBLEU is asked for.
    sacrebleu_metrics = metrics_to_compute & {"bleu", "chrf", "spbleu"}
    if sacrebleu_metrics:
        want_spbleu = "spbleu" in sacrebleu_metrics
        calc = rc.ast_calc(with_spbleu=want_spbleu)
        m = calc.calculate(hyps, refs)
        if "bleu" in sacrebleu_metrics:
            updates["bleu"] = m.bleu
            existing_cfg["bleu_config"] = m.metric_config.get("bleu_config")
        if "chrf" in sacrebleu_metrics:
            updates["chrf"] = m.chrf
            existing_cfg["chrf_config"] = m.metric_config.get("chrf_config")
        if "spbleu" in sacrebleu_metrics:
            updates["spbleu"] = m.spbleu
            existing_cfg["spbleu_config"] = m.metric_config.get("spbleu_config")
        # num_samples is set by every BLEU/chrF call regardless of which
        # subset was asked for; refresh it.
        updates["num_samples"] = m.num_samples

    # SSA-COMET: independent of sacrebleu metrics, separate model load.
    if "ssa_comet" in metrics_to_compute:
        score = rc.comet().score(srcs, hyps, refs)
        updates["ssa_comet"] = score
        if score is not None:
            existing_cfg["ssa_comet_model"] = rc.comet().model_name

    if existing_cfg:
        updates["metric_config"] = existing_cfg
    return updates


# =========================================================================
# Main
# =========================================================================

def _resolve_metrics_arg(metrics: List[str], all_tasks_metrics: set) -> set:
    """Validate and expand a --metrics list against allowed names."""
    requested = set()
    for m in metrics:
        for piece in m.split(","):
            piece = piece.strip().lower()
            if piece == "all":
                return set(all_tasks_metrics)
            if piece:
                requested.add(piece)
    unknown = requested - all_tasks_metrics
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {sorted(unknown)}. "
            f"Valid: {sorted(all_tasks_metrics)} (or 'all')."
        )
    return requested


def main():
    parser = argparse.ArgumentParser(
        description="Recompute metrics from existing prediction CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument("results_dir", help="Path to results dir (experiment or parent)")

    # What to recompute
    parser.add_argument(
        "--task", choices=["asr", "ast"], default=None,
        help="Restrict to one task. Default: both.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["all"],
        help=(
            "Metrics to recompute. Default: 'all'. "
            "ASR metrics: wer, cer. "
            "AST metrics: bleu, chrf, spbleu, ssa_comet. "
            "Pass space- or comma-separated names."
        ),
    )

    # Filters (combine to target one cell of the matrix)
    parser.add_argument("--model", help="Only process files for this model_name.")
    parser.add_argument("--dataset", help="Only process files for this dataset.")
    parser.add_argument("--language", help="ASR-only: only this language.")
    parser.add_argument("--source-lang", help="AST-only: only this source language.")
    parser.add_argument("--target-lang", help="AST-only: only this target language.")

    # Metric config (mirrors evaluate.py defaults).
    # AST sacrebleu BLEU/spBLEU score case-insensitively by default: our
    # MMS-based ASR / CTC encoder outputs are lowercase, while NLLB / Whisper
    # / SeamlessM4T produce cased text, and FLEURS-style references are
    # lowercase. Lowercasing both sides avoids penalizing cased outputs for
    # capitalization differences that aren't translation errors.
    parser.add_argument(
        "--bleu-case-sensitive", action="store_true",
        help="Pass lowercase=False to sacrebleu BLEU/spBLEU (case-sensitive). "
             "Default: case-insensitive, matching scripts/evaluate.py.",
    )
    parser.add_argument(
        "--chrf-word-order", type=int, default=2,
        help="sacrebleu CHRF word_order (default: 2 for chrF++).",
    )
    parser.add_argument(
        "--ssa-comet-model", default="McGill-NLP/ssa-comet-mtl",
        help="HF checkpoint id for SSA-COMET.",
    )
    parser.add_argument(
        "--ssa-comet-batch-size", type=int, default=8,
        help="Batch size for SSA-COMET scoring.",
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned changes without writing.",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Resolve --metrics against the chosen task(s).
    if args.task == "asr":
        all_metrics = ALL_ASR_METRICS
    elif args.task == "ast":
        all_metrics = ALL_AST_METRICS
    else:
        all_metrics = ALL_ASR_METRICS | ALL_AST_METRICS

    try:
        metrics_to_compute = _resolve_metrics_arg(args.metrics, all_metrics)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not metrics_to_compute:
        print("Error: no metrics requested", file=sys.stderr)
        sys.exit(1)

    rc = MetricRecomputer(
        bleu_config={"lowercase": not args.bleu_case_sensitive},
        chrf_config={"word_order": args.chrf_word_order},
        normalizer_config=None,   # defer to ASRMetricsCalculator's default
        ssa_comet_batch_size=args.ssa_comet_batch_size,
        ssa_comet_model=args.ssa_comet_model,
    )

    # Walk results dir for prediction CSVs.
    pred_csvs = sorted(results_dir.rglob("predictions/*.csv"))
    if not pred_csvs:
        print(f"No prediction CSVs found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pred_csvs)} prediction CSV(s) under {results_dir}")
    print(f"Metrics to recompute: {sorted(metrics_to_compute)}")
    if args.dry_run:
        print("(dry-run: no files will be written)\n")
    else:
        print()

    updated = 0
    skipped_no_metrics = 0
    skipped_filtered = 0
    skipped_no_match = 0
    errors = 0

    for pred_csv in pred_csvs:
        info = _parse_pred_filename(pred_csv)
        if info is None:
            print(f"⏭️  {pred_csv.name}: unparseable filename, skipping")
            skipped_no_match += 1
            continue

        metrics_path = _metrics_path_for_predictions(pred_csv)
        if not metrics_path.exists():
            print(f"⏭️  {pred_csv.name}: no metrics JSON at {metrics_path.name}")
            skipped_no_metrics += 1
            continue

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_json = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  {metrics_path.name}: failed to read: {e}")
            errors += 1
            continue

        if not _file_matches_filter(info, metrics_json, args):
            skipped_filtered += 1
            continue

        # Restrict to metrics relevant for this task
        task = info["task"]
        if task == "asr":
            task_metrics = metrics_to_compute & ALL_ASR_METRICS
        else:
            task_metrics = metrics_to_compute & ALL_AST_METRICS

        if not task_metrics:
            # User asked for AST-only metrics but this is an ASR file (or v.v.)
            continue

        # Describe the file we're working on.
        if task == "asr":
            label = (
                f"{metrics_json.get('model_name', info['model'])} | "
                f"{metrics_json.get('dataset', '?')} | "
                f"asr/{metrics_json.get('language', info.get('language', '?'))}"
            )
        else:
            label = (
                f"{metrics_json.get('model_name', info['model'])} | "
                f"{metrics_json.get('dataset', '?')} | "
                f"ast/{metrics_json.get('source_lang', '?')}→"
                f"{metrics_json.get('target_lang', '?')}"
            )

        try:
            if task == "asr":
                updates = _recompute_asr(pred_csv, metrics_json, task_metrics, rc)
            else:
                updates = _recompute_ast(pred_csv, metrics_json, task_metrics, rc)
        except Exception as e:
            print(f"⚠️  {label}: failed to recompute: {e}")
            errors += 1
            continue

        if not updates:
            continue

        # Build a human-readable diff line.
        diff_parts = []
        for k, new in updates.items():
            if k in ("num_samples", "metric_config"):
                continue
            old = metrics_json.get(k)
            if isinstance(new, float) and isinstance(old, (int, float)):
                diff_parts.append(f"{k}: {old:.4f} → {new:.4f}")
            elif new is None and old is None:
                continue
            else:
                diff_parts.append(f"{k}: {old} → {new}")

        diff_str = ", ".join(diff_parts) if diff_parts else "(no scalar changes)"
        print(f"✏️  {label}\n     {diff_str}")

        if not args.dry_run:
            metrics_json.update(updates)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_json, f, indent=2)
            updated += 1

    print()
    print(f"{'Would update' if args.dry_run else 'Updated'}: {updated} file(s)")
    if skipped_filtered:
        print(f"Skipped (filtered out): {skipped_filtered}")
    if skipped_no_metrics:
        print(f"Skipped (no metrics JSON): {skipped_no_metrics}")
    if skipped_no_match:
        print(f"Skipped (unparseable filename): {skipped_no_match}")
    if errors:
        print(f"Errors: {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()