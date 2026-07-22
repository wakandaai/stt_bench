#!/usr/bin/env python3
"""
STT Benchmark Evaluation Script

Evaluate speech models against one or more datasets declared in an eval config.

Usage:
  python scripts/evaluate.py whisper_large_v3 --eval-config configs/african_evaluation.yaml
  python scripts/evaluate.py seamless_m4t_v2_large --eval-config configs/african_evaluation.yaml --batch-size 4
  python scripts/evaluate.py seamless_m4t_v2_large --eval-config configs/african_evaluation.yaml --ssa-comet

By default, cells (model × dataset × language or pair) that already have a
metrics JSON on disk under --output-dir are skipped. Pass --force to re-run
everything regardless.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
from typing import List, Tuple

from stt_benchmark.models.factory import ModelFactory
from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.datasets.registry import create_dataset
from stt_benchmark.evaluation.pipeline import EvaluationPipeline
from stt_benchmark.evaluation.config import (
    load_eval_config, DatasetEvalSpec,
)


# =========================================================================
# Skip-already-done helpers
# =========================================================================

def _sanitize_model_name(model_name: str) -> str:
    """Mirror EvaluationPipeline's filename transform of model_name.

    Pipeline does `model_name.replace("/", "_")` when writing predictions and
    metrics; we replicate that exactly so existence checks point at the right
    file. If the pipeline ever changes this transform, this helper must move
    in lockstep.
    """
    return model_name.replace("/", "_")


def asr_metrics_path(
    output_dir: str,
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    language: str,
) -> Path:
    """Path EvaluationPipeline writes ASR metrics to for this cell."""
    safe = _sanitize_model_name(model_name)
    return (
        Path(output_dir)
        / experiment_name
        / dataset_name
        / "metrics"
        / f"{safe}_asr_{language}_metrics.json"
    )


def ast_metrics_path(
    output_dir: str,
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    source_lang: str,
    target_lang: str,
) -> Path:
    """Path EvaluationPipeline writes AST metrics to for this cell."""
    safe = _sanitize_model_name(model_name)
    return (
        Path(output_dir)
        / experiment_name
        / dataset_name
        / "metrics"
        / f"{safe}_ast_{source_lang}_{target_lang}_metrics.json"
    )


def filter_already_done(
    asr_valid: List[str],
    ast_valid: List[Tuple[str, str]],
    *,
    output_dir: str,
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    force: bool,
) -> Tuple[List[str], List[Tuple[str, str]], List[str], List[Tuple[str, str]]]:
    """Split valid cells into (to-run, already-done) lists.

    When force=True, everything stays in to-run. Otherwise we check for the
    metrics JSON the pipeline would have written and move existing ones to
    the already-done bucket.

    Returns:
        asr_to_run, ast_to_run, asr_done, ast_done
    """
    if force:
        return list(asr_valid), list(ast_valid), [], []

    asr_to_run: List[str] = []
    asr_done: List[str] = []
    for lang in asr_valid:
        path = asr_metrics_path(
            output_dir, experiment_name, dataset_name, model_name, lang,
        )
        (asr_done if path.exists() else asr_to_run).append(lang)

    ast_to_run: List[Tuple[str, str]] = []
    ast_done: List[Tuple[str, str]] = []
    for src, tgt in ast_valid:
        path = ast_metrics_path(
            output_dir, experiment_name, dataset_name, model_name, src, tgt,
        )
        (ast_done if path.exists() else ast_to_run).append((src, tgt))

    return asr_to_run, ast_to_run, asr_done, ast_done


# =========================================================================
# Pre-flight validation (per-dataset)
# =========================================================================

def validate_asr_languages(
    languages: List[str],
    model: BaseASRModel,
    dataset,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Filter ASR languages against model support and dataset availability."""
    model_supported = model.get_supported_languages()
    dataset_available = dataset.list_languages()

    valid = []
    skipped = []

    for lang in languages:
        if lang not in dataset_available:
            skipped.append((lang, "not in dataset"))
        elif lang not in model_supported:
            skipped.append((lang, "not supported by model"))
        else:
            valid.append(lang)

    return valid, skipped


def validate_ast_pairs(
    pairs: List[Tuple[str, str]],
    model,
    dataset,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """Filter AST pairs against model support and dataset availability."""
    dataset_pairs = dataset.list_language_pairs()
    has_ast = hasattr(model, "supports_language_pair")

    valid = []
    skipped = []

    for src, tgt in pairs:
        if (src, tgt) not in dataset_pairs:
            skipped.append((src, tgt, "not supported by dataset"))
        elif not has_ast:
            skipped.append((src, tgt, "model does not support AST"))
        elif not model.supports_language_pair(src, tgt):
            skipped.append((src, tgt, "not supported by model"))
        else:
            valid.append((src, tgt))

    return valid, skipped


def print_validation_report(
    dataset_name: str,
    asr_to_run: List[str],
    asr_done: List[str],
    asr_skipped: List[Tuple[str, str]],
    ast_to_run: List[Tuple[str, str]],
    ast_done: List[Tuple[str, str]],
    ast_skipped: List[Tuple[str, str, str]],
    run_asr: bool,
    run_ast: bool,
):
    print(f"\n📋 Dataset: {dataset_name}")

    if run_asr:
        print(f"   ASR Validation:")
        print(f"     ✅ Will evaluate: {len(asr_to_run)} language(s)")
        if asr_to_run:
            print(f"        {', '.join(asr_to_run)}")
        if asr_done:
            print(f"     ✓  Already done (skipping; use --force to re-run): "
                  f"{len(asr_done)} language(s)")
            print(f"        {', '.join(asr_done)}")
        if asr_skipped:
            print(f"     ⏭️  Skipping: {len(asr_skipped)} language(s)")
            for lang, reason in asr_skipped:
                print(f"        {lang}: {reason}")

    if run_ast:
        print(f"   AST Validation:")
        print(f"     ✅ Will evaluate: {len(ast_to_run)} pair(s)")
        if ast_to_run and len(ast_to_run) <= 20:
            for src, tgt in ast_to_run:
                print(f"        {src} → {tgt}")
        elif ast_to_run:
            for src, tgt in ast_to_run[:10]:
                print(f"        {src} → {tgt}")
            print(f"        ... and {len(ast_to_run) - 10} more")
        if ast_done:
            print(f"     ✓  Already done (skipping; use --force to re-run): "
                  f"{len(ast_done)} pair(s)")
            if len(ast_done) <= 20:
                for src, tgt in ast_done:
                    print(f"        {src} → {tgt}")
            else:
                for src, tgt in ast_done[:10]:
                    print(f"        {src} → {tgt}")
                print(f"        ... and {len(ast_done) - 10} more")
        if ast_skipped:
            print(f"     ⏭️  Skipping: {len(ast_skipped)} pair(s)")
            for src, tgt, reason in ast_skipped:
                print(f"        {src} → {tgt}: {reason}")


# =========================================================================
# Per-dataset evaluation
# =========================================================================

def evaluate_one_dataset(
    spec: DatasetEvalSpec,
    model,
    evaluator: EvaluationPipeline,
    experiment_name: str,
    output_dir: str,
    force: bool,
):
    """Instantiate one dataset and run its ASR + AST plan."""
    run_asr = spec.has_asr()
    run_ast = spec.has_ast()

    if not (run_asr or run_ast):
        print(f"\n⚠️  Dataset '{spec.dataset_name}' has no ASR or AST plan — skipping.")
        return

    # ── Skip-existing check (BEFORE loading the dataset) ──────────────────
    # Loading some datasets is expensive (parquet streaming, HF downloads, XML
    # parsing across 5 languages). If everything in this dataset's plan is
    # already on disk, bail out before paying that cost.
    model_name = model.get_model_info()["model_name"]

    asr_to_run, ast_to_run, asr_done_pre, ast_done_pre = filter_already_done(
        asr_valid=spec.asr_languages,
        ast_valid=spec.ast_pairs,
        output_dir=output_dir,
        experiment_name=experiment_name,
        dataset_name=spec.dataset_name,
        model_name=model_name,
        force=force,
    )

    if not force and not asr_to_run and not ast_to_run \
            and (asr_done_pre or ast_done_pre):
        print(f"\n📋 Dataset: {spec.dataset_name}")
        if asr_done_pre:
            print(f"   ✓ ASR already done for {len(asr_done_pre)} language(s); "
                  f"skipping dataset load.")
        if ast_done_pre:
            print(f"   ✓ AST already done for {len(ast_done_pre)} pair(s); "
                  f"skipping dataset load.")
        print(f"   (use --force to re-run)")
        return

    # Build the set of languages the dataset needs to know about.
    # IMPORTANT: scope by the ORIGINAL config-requested languages, not by what
    # survives skip-filtering. Otherwise the dataset can't validate "language
    # X was already done" against its own catalog later.
    scoped_languages = set(spec.asr_languages)
    for src, tgt in spec.ast_pairs:
        scoped_languages.add(src)
        scoped_languages.add(tgt)

    # Inject scope into dataset kwargs. The base kwargs (split, root, ...)
    # come from the YAML; languages/pairs come from the eval spec.
    ds_kwargs = dict(spec.dataset_kwargs)
    if scoped_languages:
        ds_kwargs.setdefault("languages", sorted(scoped_languages))
    if spec.ast_pairs:
        ds_kwargs.setdefault("ast_pairs", spec.ast_pairs)

    print(f"\n{'='*60}")
    print(f"Loading dataset: {spec.dataset_name}")
    print(f"{'='*60}")
    dataset = create_dataset(spec.dataset_name, **ds_kwargs)

    # Pre-flight validation against the model + dataset (still on the full
    # requested set so the "not in dataset" / "not supported by model" report
    # stays accurate).
    asr_valid, asr_skipped = [], []
    ast_valid, ast_skipped = [], []

    if run_asr:
        asr_valid, asr_skipped = validate_asr_languages(
            spec.asr_languages, model, dataset
        )
    if run_ast:
        ast_valid, ast_skipped = validate_ast_pairs(
            spec.ast_pairs, model, dataset
        )

    # Re-apply skip-existing on the post-validation set. This may differ from
    # the pre-load filter if validation drops some cells.
    asr_to_run, ast_to_run, asr_done, ast_done = filter_already_done(
        asr_valid=asr_valid,
        ast_valid=ast_valid,
        output_dir=output_dir,
        experiment_name=experiment_name,
        dataset_name=spec.dataset_name,
        model_name=model_name,
        force=force,
    )

    print_validation_report(
        spec.dataset_name,
        asr_to_run, asr_done, asr_skipped,
        ast_to_run, ast_done, ast_skipped,
        run_asr, run_ast,
    )

    nothing_to_do = (
        (not run_asr or len(asr_to_run) == 0) and
        (not run_ast or len(ast_to_run) == 0)
    )
    if nothing_to_do:
        print(f"   ⚠️  Nothing to evaluate for {spec.dataset_name}.")
        return

    # ── ASR ──────────────────────────────────────────────────────────────
    if asr_to_run:
        print(f"\n--- ASR: {spec.dataset_name} ({len(asr_to_run)} language(s)) ---")
        for lang in asr_to_run:
            try:
                result = evaluator.evaluate_asr(
                    model=model,
                    dataset=dataset,
                    language=lang,
                    experiment_name=experiment_name,
                    dataset_name=spec.dataset_name,
                )
                if result:
                    print(
                        f"  {lang}: WER={result.metrics.wer:.2f}%, "
                        f"CER={result.metrics.cer:.2f}%"
                    )
            except Exception as e:
                print(f"  Error on {lang}: {e}")
                continue

    # ── AST ──────────────────────────────────────────────────────────────
    if ast_to_run:
        print(f"\n--- AST: {spec.dataset_name} ({len(ast_to_run)} pair(s)) ---")
        for src, tgt in ast_to_run:
            try:
                result = evaluator.evaluate_ast(
                    model=model,
                    dataset=dataset,
                    source_lang=src,
                    target_lang=tgt,
                    experiment_name=experiment_name,
                    dataset_name=spec.dataset_name,
                )
                if result:
                    msg = (
                        f"  {src}→{tgt}: BLEU={result.metrics.bleu:.2f}, "
                        f"chrF++={result.metrics.chrf:.2f}"
                    )
                    if result.metrics.spbleu is not None:
                        msg += f", spBLEU-1K={result.metrics.spbleu:.2f}"
                    if result.metrics.ssa_comet is not None:
                        msg += f", SSA-COMET={result.metrics.ssa_comet:.4f}"
                    print(msg)
            except Exception as e:
                print(f"  Error on {src}→{tgt}: {e}")
                continue


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="STT Benchmark: Evaluate speech models against datasets in an eval config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/evaluate.py whisper_large_v3 \\
      --eval-config configs/african_evaluation.yaml

  # Re-run everything, even cells that already have metrics on disk
  python scripts/evaluate.py whisper_large_v3 \\
      --eval-config configs/african_evaluation.yaml --force
        """,
    )

    parser.add_argument("model_id", help="Model ID from config (e.g., whisper_large_v3)")
    parser.add_argument(
        "--eval-config", required=True,
        help="Path to evaluation config YAML.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--experiment-name", help="Override experiment name")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run every cell (model × dataset × language or pair), even if "
             "a metrics JSON already exists under --output-dir. Default: skip "
             "cells whose metrics file is already on disk.",
    )
    parser.add_argument(
        "--no-spbleu", action="store_true",
        help="Skip the spBLEU-1K AST metric (otherwise computed by default).",
    )
    parser.add_argument(
        "--ssa-comet", action="store_true",
        help="Also compute SSA-COMET-MTL for AST. Loads "
             "McGill-NLP/ssa-comet-mtl on first AST scoring call (GPU recommended).",
    )
    parser.add_argument(
        "--ssa-comet-batch-size", type=int, default=8,
        help="Batch size for SSA-COMET scoring (default: 8).",
    )

    args = parser.parse_args()

    # CUDA setup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        eval_cfg = load_eval_config(args.eval_config)
        experiment_name = args.experiment_name or eval_cfg.experiment_name

        print(f"\n{'='*60}")
        print(f"Loaded eval config: {args.eval_config}")
        print(eval_cfg.summary())
        if args.force:
            print(f"\n--force: re-running all cells regardless of existing metrics.")
        print(f"{'='*60}")

        print(f"\nLoading model: {args.model_id}")
        model = ModelFactory.create_model(args.model_id)

        evaluator = EvaluationPipeline(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            # AST scoring is case-insensitive: our MMS-based ASR and CTC
            # encoders only emit lowercase, while NLLB/Whisper/SeamlessM4T
            # produce cased output. Lowercasing both sides at scoring time
            # avoids penalizing the cased models for capitalization mismatches
            # against the lowercase FLEURS-style references.
            bleu_config={"lowercase": True},
            chrf_config={"word_order": 2},
            skip_unsupported=False,
            compute_spbleu=not args.no_spbleu,
            compute_ssa_comet=args.ssa_comet,
            ssa_comet_batch_size=args.ssa_comet_batch_size,
        )

        for spec in eval_cfg.datasets:
            evaluate_one_dataset(
                spec, model, evaluator, experiment_name,
                output_dir=args.output_dir,
                force=args.force,
            )

        print(f"\nResults saved to {args.output_dir}/{experiment_name}/")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()