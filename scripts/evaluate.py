#!/usr/bin/env python3
"""
STT Benchmark Evaluation Script

Evaluate speech models against one or more datasets declared in an eval config.

Usage:
  python scripts/evaluate.py whisper_large_v3 --eval-config configs/african_evaluation.yaml
  python scripts/evaluate.py seamless_m4t_v2_large --eval-config configs/african_evaluation.yaml --batch-size 4
"""

import argparse
import sys
import os
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
    asr_valid: List[str],
    asr_skipped: List[Tuple[str, str]],
    ast_valid: List[Tuple[str, str]],
    ast_skipped: List[Tuple[str, str, str]],
    run_asr: bool,
    run_ast: bool,
):
    print(f"\n📋 Dataset: {dataset_name}")

    if run_asr:
        print(f"   ASR Validation:")
        print(f"     ✅ Will evaluate: {len(asr_valid)} language(s)")
        if asr_valid:
            print(f"        {', '.join(asr_valid)}")
        if asr_skipped:
            print(f"     ⏭️  Skipping: {len(asr_skipped)} language(s)")
            for lang, reason in asr_skipped:
                print(f"        {lang}: {reason}")

    if run_ast:
        print(f"   AST Validation:")
        print(f"     ✅ Will evaluate: {len(ast_valid)} pair(s)")
        if ast_valid and len(ast_valid) <= 20:
            for src, tgt in ast_valid:
                print(f"        {src} → {tgt}")
        elif ast_valid:
            for src, tgt in ast_valid[:10]:
                print(f"        {src} → {tgt}")
            print(f"        ... and {len(ast_valid) - 10} more")
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
):
    """Instantiate one dataset and run its ASR + AST plan."""
    run_asr = spec.has_asr()
    run_ast = spec.has_ast()

    if not (run_asr or run_ast):
        print(f"\n⚠️  Dataset '{spec.dataset_name}' has no ASR or AST plan — skipping.")
        return

    # Build the set of languages the dataset needs to know about.
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

    # Pre-flight validation
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

    print_validation_report(
        spec.dataset_name,
        asr_valid, asr_skipped,
        ast_valid, ast_skipped,
        run_asr, run_ast,
    )

    nothing_to_do = (
        (not run_asr or len(asr_valid) == 0) and
        (not run_ast or len(ast_valid) == 0)
    )
    if nothing_to_do:
        print(f"   ⚠️  Nothing to evaluate for {spec.dataset_name}.")
        return

    # ── ASR ──────────────────────────────────────────────────────────────
    if asr_valid:
        print(f"\n--- ASR: {spec.dataset_name} ({len(asr_valid)} language(s)) ---")
        for lang in asr_valid:
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
    if ast_valid:
        print(f"\n--- AST: {spec.dataset_name} ({len(ast_valid)} pair(s)) ---")
        for src, tgt in ast_valid:
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
                    print(
                        f"  {src}→{tgt}: BLEU={result.metrics.bleu:.2f}, "
                        f"chrF++={result.metrics.chrf:.2f}"
                    )
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
        print(f"{'='*60}")

        print(f"\nLoading model: {args.model_id}")
        model = ModelFactory.create_model(args.model_id)

        evaluator = EvaluationPipeline(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            bleu_config={"lowercase": False},
            chrf_config={"word_order": 2},
            skip_unsupported=False,
        )

        for spec in eval_cfg.datasets:
            evaluate_one_dataset(spec, model, evaluator, experiment_name)

        print(f"\nResults saved to {args.output_dir}/{experiment_name}/")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()