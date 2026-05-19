#!/usr/bin/env python3
"""
STT Benchmark Evaluation Script

Evaluate speech models on the HuggingFace `google/fleurs` dataset for ASR and/or AST tasks.

Usage with eval config (recommended):
  python scripts/evaluate.py whisper_large_v3 --eval-config configs/african_evaluation.yaml
  python scripts/evaluate.py seamless_m4t_v2_large --eval-config configs/african_evaluation.yaml --split validation

Usage with CLI flags (ad-hoc):
  python scripts/evaluate.py whisper_large_v3 --task asr --language sw_ke
  python scripts/evaluate.py whisper_large_v3 --task ast --source-lang sw_ke --target-lang en_us
"""

import argparse
import sys
import os
import torch
from typing import List, Tuple

from stt_benchmark.models.factory import ModelFactory
from stt_benchmark.models.base import BaseASRModel
from stt_benchmark.datasets.fleurs import FleursDataset
from stt_benchmark.evaluation.pipeline import EvaluationPipeline
from stt_benchmark.evaluation.config import load_eval_config


# =========================================================================
# Pre-flight validation
# =========================================================================

def validate_asr_languages(
    languages: List[str],
    model: BaseASRModel,
    dataset: FleursDataset,
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
    dataset: FleursDataset,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """Filter AST pairs against model support and dataset availability."""
    dataset_pairs = dataset.list_language_pairs()
    has_ast = hasattr(model, "supports_language_pair")

    valid = []
    skipped = []

    for src, tgt in pairs:
        if (src, tgt) not in dataset_pairs:
            skipped.append((src, tgt, "not in dataset scope"))
        elif not has_ast:
            skipped.append((src, tgt, "model does not support AST"))
        elif not model.supports_language_pair(src, tgt):
            skipped.append((src, tgt, "not supported by model"))
        else:
            valid.append((src, tgt))

    return valid, skipped


def print_validation_report(
    asr_valid: List[str],
    asr_skipped: List[Tuple[str, str]],
    ast_valid: List[Tuple[str, str]],
    ast_skipped: List[Tuple[str, str, str]],
    run_asr: bool,
    run_ast: bool,
):
    if run_asr:
        print(f"\n📋 ASR Validation:")
        print(f"   ✅ Will evaluate: {len(asr_valid)} language(s)")
        if asr_valid:
            print(f"      {', '.join(asr_valid)}")
        if asr_skipped:
            print(f"   ⏭️  Skipping: {len(asr_skipped)} language(s)")
            for lang, reason in asr_skipped:
                print(f"      {lang}: {reason}")

    if run_ast:
        print(f"\n📋 AST Validation:")
        print(f"   ✅ Will evaluate: {len(ast_valid)} pair(s)")
        if ast_valid and len(ast_valid) <= 20:
            for src, tgt in ast_valid:
                print(f"      {src} → {tgt}")
        elif ast_valid:
            for src, tgt in ast_valid[:10]:
                print(f"      {src} → {tgt}")
            print(f"      ... and {len(ast_valid) - 10} more")
        if ast_skipped:
            print(f"   ⏭️  Skipping: {len(ast_skipped)} pair(s)")
            for src, tgt, reason in ast_skipped:
                print(f"      {src} → {tgt}: {reason}")

    print()


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="STT Benchmark: Evaluate speech models on google/fleurs (HF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use an eval config (recommended)
  python scripts/evaluate.py whisper_large_v3 \\
      --eval-config configs/african_evaluation.yaml

  # Ad-hoc ASR on one language
  python scripts/evaluate.py whisper_large_v3 --task asr --language sw_ke

  # Ad-hoc AST on one pair
  python scripts/evaluate.py whisper_large_v3 --task ast --source-lang sw_ke --target-lang en_us
        """,
    )

    parser.add_argument("model_id", help="Model ID from config (e.g., whisper_large_v3)")

    # ── Eval config mode ─────────────────────────────────────────────────
    parser.add_argument(
        "--eval-config",
        help="Path to evaluation config YAML. Overrides --task, --language, etc.",
    )

    # ── Ad-hoc mode ──────────────────────────────────────────────────────
    parser.add_argument(
        "--task",
        choices=["asr", "ast", "both"],
        default="asr",
        help="Evaluation task (default: asr). Ignored when --eval-config is set.",
    )
    parser.add_argument("--language", help="Single FLEURS language for ASR")
    parser.add_argument("--languages", nargs="+", help="Multiple FLEURS languages for ASR")
    parser.add_argument("--source-lang", help="Source FLEURS language for AST")
    parser.add_argument("--target-lang", help="Target FLEURS language for AST")
    parser.add_argument(
        "--ast-pairs", nargs="+",
        help="Multiple AST pairs as src:tgt (e.g., sw_ke:en_us yo_ng:en_us)",
    )

    # ── General options ──────────────────────────────────────────────────
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="HuggingFace FLEURS split to evaluate on (default: test)",
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
        # ── Resolve what to evaluate ─────────────────────────────────────
        if args.eval_config:
            eval_cfg = load_eval_config(args.eval_config)
            experiment_name = args.experiment_name or eval_cfg.experiment_name
            asr_languages = eval_cfg.asr_languages
            ast_pairs = eval_cfg.ast_pairs
            run_asr = len(asr_languages) > 0
            run_ast = len(ast_pairs) > 0

            print(f"\n{'='*60}")
            print(f"Loaded eval config: {args.eval_config}")
            print(eval_cfg.summary())
            print(f"{'='*60}")
        else:
            experiment_name = args.experiment_name or args.model_id
            asr_languages = _resolve_asr_languages(args)
            ast_pairs = _resolve_ast_pairs(args)
            run_asr = args.task in ("asr", "both")
            run_ast = args.task in ("ast", "both")

        # ── Load model & dataset ─────────────────────────────────────────
        print(f"\nLoading model: {args.model_id}")
        model = ModelFactory.create_model(args.model_id)

        # Build the set of languages the dataset needs to know about:
        # ASR languages + every language mentioned in any AST pair.
        scoped_languages = set(asr_languages or [])
        for src, tgt in ast_pairs or []:
            scoped_languages.add(src)
            scoped_languages.add(tgt)

        print(f"Loading dataset: google/fleurs (split={args.split})")
        dataset = FleursDataset(
            split=args.split,
            languages=sorted(scoped_languages) if scoped_languages else None,
            ast_pairs=ast_pairs if ast_pairs else None,
        )

        # ── Pre-flight validation ────────────────────────────────────────
        asr_valid, asr_skipped = [], []
        ast_valid, ast_skipped = [], []

        if run_asr and asr_languages:
            asr_valid, asr_skipped = validate_asr_languages(
                asr_languages, model, dataset
            )
        elif run_asr:
            asr_valid = None  # sentinel: means evaluate all

        if run_ast and ast_pairs:
            ast_valid, ast_skipped = validate_ast_pairs(
                ast_pairs, model, dataset
            )

        print_validation_report(
            asr_valid or [], asr_skipped,
            ast_valid, ast_skipped,
            run_asr, run_ast,
        )

        nothing_to_do = True
        if run_asr and (asr_valid is None or len(asr_valid) > 0):
            nothing_to_do = False
        if run_ast and len(ast_valid) > 0:
            nothing_to_do = False

        if nothing_to_do:
            print("⚠️  Nothing to evaluate — all languages/pairs were skipped.")
            print("   Check that your eval config matches your model and dataset.")
            sys.exit(0)

        # ── Create pipeline ──────────────────────────────────────────────
        evaluator = EvaluationPipeline(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            bleu_config={"lowercase": False},
            chrf_config={"word_order": 2},
            skip_unsupported=False,
        )

        # ── ASR ──────────────────────────────────────────────────────────
        if run_asr:
            if asr_valid is None:
                print(f"\n{'='*60}")
                print("ASR Evaluation: all languages in dataset")
                print(f"{'='*60}")
                evaluator.skip_unsupported = True
                results = evaluator.evaluate_asr_all_languages(
                    model=model,
                    dataset=dataset,
                    experiment_name=experiment_name,
                )
                evaluator.skip_unsupported = False
                print(f"\nCompleted ASR for {len(results)} languages")
            elif asr_valid:
                print(f"\n{'='*60}")
                print(f"ASR Evaluation: {len(asr_valid)} language(s)")
                print(f"{'='*60}")
                for lang in asr_valid:
                    result = evaluator.evaluate_asr(
                        model=model,
                        dataset=dataset,
                        language=lang,
                        experiment_name=experiment_name,
                    )
                    if result:
                        print(
                            f"  {lang}: WER={result.metrics.wer:.2f}%, "
                            f"CER={result.metrics.cer:.2f}%"
                        )

        # ── AST ──────────────────────────────────────────────────────────
        if run_ast and ast_valid:
            print(f"\n{'='*60}")
            print(f"AST Evaluation: {len(ast_valid)} pair(s)")
            print(f"{'='*60}")
            for src, tgt in ast_valid:
                result = evaluator.evaluate_ast(
                    model=model,
                    dataset=dataset,
                    source_lang=src,
                    target_lang=tgt,
                    experiment_name=experiment_name,
                )
                if result:
                    print(
                        f"  {src}→{tgt}: BLEU={result.metrics.bleu:.2f}, "
                        f"chrF++={result.metrics.chrf:.2f}"
                    )

        print(f"\nResults saved to {args.output_dir}/{experiment_name}/")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _resolve_asr_languages(args):
    if args.language:
        return [args.language]
    elif args.languages:
        return args.languages
    return []


def _resolve_ast_pairs(args):
    pairs = []
    if args.source_lang and args.target_lang:
        pairs.append((args.source_lang, args.target_lang))
    if args.ast_pairs:
        for pair_str in args.ast_pairs:
            src, tgt = pair_str.split(":")
            pairs.append((src, tgt))
    return pairs


if __name__ == "__main__":
    main()