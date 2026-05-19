# stt_benchmark/evaluation/pipeline.py

import time
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from stt_benchmark.models.base import BaseASRModel, BaseASTModel
from stt_benchmark.datasets.base import (
    BaseASRDataset, BaseASTDataset, AudioSample, ParallelAudioSample,
)
from stt_benchmark.evaluation.base import (
    ASRPrediction, ASTPrediction,
    ASREvaluationResult, ASTEvaluationResult,
)
from stt_benchmark.evaluation.metrics import ASRMetricsCalculator, ASTMetricsCalculator
from stt_benchmark.utils.text_normalize import TextNormalizer


def _is_cascaded_model(model) -> bool:
    """Check if a model is a cascaded ASR→MT model with intermediate transcripts."""
    return hasattr(model, "get_last_intermediate_transcripts")


class EvaluationPipeline:
    """Main evaluation pipeline for ASR and AST models."""

    def __init__(
        self,
        output_dir: str = "results",
        batch_size: int = 1,
        normalizer: TextNormalizer = None,
        bleu_config: Dict[str, Any] = None,
        chrf_config: Dict[str, Any] = None,
        skip_unsupported: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.skip_unsupported = skip_unsupported

        self.asr_metrics = ASRMetricsCalculator(normalizer)
        self.ast_metrics = ASTMetricsCalculator(bleu_config, chrf_config)

    # ====================================================================
    # ASR Evaluation
    # ====================================================================

    def evaluate_asr(
        self,
        model: BaseASRModel,
        dataset: BaseASRDataset,
        language: str,
        experiment_name: str,
        batch_size: Optional[int] = None,
    ) -> Optional[ASREvaluationResult]:
        if self.skip_unsupported and not model.supports_language(language):
            print(f"⏭️  Skipping ASR {language}: not supported by model")
            return None

        model_name = model.get_model_info()["model_name"]
        print(f"🎙️  ASR: {model_name} on {language}")

        samples = dataset.get_language_samples(language)
        if not samples:
            raise ValueError(f"No samples for {language}")

        bs = batch_size or self.batch_size
        start_time = time.time()
        predictions = self._run_asr_inference(model, samples, bs, language)
        total_time = time.time() - start_time

        hypotheses = [p.hypothesis for p in predictions]
        references = [p.reference for p in predictions]
        metrics = self.asr_metrics.calculate(hypotheses, references)

        result = ASREvaluationResult(
            language=language,
            model_name=model_name.replace("/", "_"),
            predictions=predictions,
            metrics=metrics,
            experiment_name=experiment_name,
            total_time=total_time,
        )

        self._save_asr_result(result)
        print(f"  WER: {metrics.wer:.2f}%, CER: {metrics.cer:.2f}%")
        return result

    def evaluate_asr_all_languages(
        self,
        model: BaseASRModel,
        dataset: BaseASRDataset,
        experiment_name: str,
        languages: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ) -> List[ASREvaluationResult]:
        langs = languages or sorted(dataset.list_languages())
        model_name = model.get_model_info()["model_name"]
        print(f"Evaluating ASR: {model_name} on {len(langs)} languages")

        results = []
        for lang in tqdm(langs, desc="ASR languages"):
            try:
                result = self.evaluate_asr(
                    model, dataset, lang, experiment_name, batch_size
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"  Error on {lang}: {e}")
                continue

        if results:
            self._save_asr_summary(results, experiment_name)
        return results

    def _run_asr_inference(
        self,
        model: BaseASRModel,
        samples: List[AudioSample],
        batch_size: int,
        language: str,
    ) -> List[ASRPrediction]:
        """Run ASR inference — passes samples directly to the model."""
        predictions = []

        for i in tqdm(
            range(0, len(samples), batch_size), desc="ASR inference", leave=False
        ):
            batch = samples[i : i + batch_size]

            start = time.time()
            try:
                hypotheses = model.transcribe(batch, language)
                pred_time = (time.time() - start) / len(batch)
            except Exception as e:
                print(f"    Transcription error: {e}")
                hypotheses = [""] * len(batch)
                pred_time = None

            for sample, hyp in zip(batch, hypotheses):
                predictions.append(
                    ASRPrediction(
                        sample_id=sample.sample_id,
                        reference=sample.transcription,
                        hypothesis=hyp,
                        audio_path=sample.audio_path or "",
                        language=language,
                        prediction_time=pred_time,
                    )
                )

        return predictions

    # ====================================================================
    # AST Evaluation
    # ====================================================================

    def evaluate_ast(
        self,
        model: BaseASTModel,
        dataset: BaseASTDataset,
        source_lang: str,
        target_lang: str,
        experiment_name: str,
        batch_size: Optional[int] = None,
    ) -> Optional[ASTEvaluationResult]:
        if self.skip_unsupported and not model.supports_language_pair(
            source_lang, target_lang
        ):
            print(f"⏭️  Skipping AST {source_lang}→{target_lang}: not supported")
            return None

        model_name = model.get_model_info()["model_name"]
        print(f"🌐 AST: {model_name} on {source_lang}→{target_lang}")

        samples = dataset.get_parallel_samples(source_lang, target_lang)
        if not samples:
            raise ValueError(f"No parallel samples for {source_lang}→{target_lang}")

        bs = batch_size or self.batch_size
        start_time = time.time()
        predictions = self._run_ast_inference(
            model, samples, bs, source_lang, target_lang
        )
        total_time = time.time() - start_time

        hypotheses = [p.hypothesis for p in predictions]
        references = [p.reference for p in predictions]
        metrics = self.ast_metrics.calculate(hypotheses, references)

        result = ASTEvaluationResult(
            source_lang=source_lang,
            target_lang=target_lang,
            model_name=model_name.replace("/", "_"),
            predictions=predictions,
            metrics=metrics,
            experiment_name=experiment_name,
            total_time=total_time,
        )

        self._save_ast_result(result)
        print(f"  BLEU: {metrics.bleu:.2f}, chrF++: {metrics.chrf:.2f}")
        return result

    def evaluate_ast_all_pairs(
        self,
        model: BaseASTModel,
        dataset: BaseASTDataset,
        experiment_name: str,
        pairs: Optional[List[tuple]] = None,
        batch_size: Optional[int] = None,
    ) -> List[ASTEvaluationResult]:
        if pairs is None:
            pairs = sorted(dataset.list_language_pairs())

        model_name = model.get_model_info()["model_name"]
        print(f"Evaluating AST: {model_name} on {len(pairs)} pairs")

        results = []
        for src, tgt in tqdm(pairs, desc="AST pairs"):
            try:
                result = self.evaluate_ast(
                    model, dataset, src, tgt, experiment_name, batch_size
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"  Error on {src}→{tgt}: {e}")
                continue

        if results:
            self._save_ast_summary(results, experiment_name)
        return results

    def _run_ast_inference(
        self,
        model: BaseASTModel,
        samples: List[ParallelAudioSample],
        batch_size: int,
        source_lang: str,
        target_lang: str,
    ) -> List[ASTPrediction]:
        """Run AST inference — passes samples directly to the model."""
        predictions = []
        is_cascaded = _is_cascaded_model(model)

        for i in tqdm(
            range(0, len(samples), batch_size), desc="AST inference", leave=False
        ):
            batch = samples[i : i + batch_size]

            start = time.time()
            try:
                hypotheses = model.translate(batch, source_lang, target_lang)
                pred_time = (time.time() - start) / len(batch)

                if is_cascaded:
                    intermediate = model.get_last_intermediate_transcripts()
                else:
                    intermediate = [None] * len(batch)

            except Exception as e:
                print(f"    Translation error: {e}")
                hypotheses = [""] * len(batch)
                intermediate = [None] * len(batch)
                pred_time = None

            for sample, hyp, inter in zip(batch, hypotheses, intermediate):
                predictions.append(
                    ASTPrediction(
                        sample_id=sample.sample_id,
                        source_transcription=sample.source_transcription,
                        reference=sample.target_transcription,
                        hypothesis=hyp,
                        audio_path=sample.source_audio_path or "",
                        source_lang=source_lang,
                        target_lang=target_lang,
                        prediction_time=pred_time,
                        intermediate_transcript=inter,
                    )
                )

        return predictions

    # ====================================================================
    # Save Results
    # ====================================================================

    def _save_asr_result(self, result: ASREvaluationResult):
        exp_dir = self.output_dir / result.experiment_name

        pred_dir = exp_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        csv_path = pred_dir / f"{result.model_name}_asr_{result.language}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "reference", "hypothesis", "audio_path"])
            for pred in result.predictions:
                writer.writerow(
                    [pred.sample_id, pred.reference, pred.hypothesis, pred.audio_path]
                )

        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = (
            metrics_dir / f"{result.model_name}_asr_{result.language}_metrics.json"
        )
        metrics_data = {
            "task": "asr",
            "model_name": result.model_name,
            "language": result.language,
            "wer": result.metrics.wer,
            "cer": result.metrics.cer,
            "num_samples": result.metrics.num_samples,
            "total_time": result.total_time,
            "avg_time_per_sample": (
                result.total_time / len(result.predictions)
                if result.total_time
                else None
            ),
            "metric_config": result.metrics.metric_config,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

    def _save_ast_result(self, result: ASTEvaluationResult):
        exp_dir = self.output_dir / result.experiment_name

        has_intermediate = any(
            p.intermediate_transcript is not None for p in result.predictions
        )

        pred_dir = exp_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        csv_path = (
            pred_dir
            / f"{result.model_name}_ast_{result.source_lang}_{result.target_lang}.csv"
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "sample_id",
                "source_transcription",
                "reference",
                "hypothesis",
                "audio_path",
            ]
            if has_intermediate:
                header.append("intermediate_transcript")
            writer.writerow(header)

            for pred in result.predictions:
                row = [
                    pred.sample_id,
                    pred.source_transcription,
                    pred.reference,
                    pred.hypothesis,
                    pred.audio_path,
                ]
                if has_intermediate:
                    row.append(pred.intermediate_transcript or "")
                writer.writerow(row)

        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = (
            metrics_dir
            / f"{result.model_name}_ast_{result.source_lang}_{result.target_lang}_metrics.json"
        )
        metrics_data = {
            "task": "ast",
            "model_name": result.model_name,
            "source_lang": result.source_lang,
            "target_lang": result.target_lang,
            "bleu": result.metrics.bleu,
            "chrf": result.metrics.chrf,
            "num_samples": result.metrics.num_samples,
            "total_time": result.total_time,
            "avg_time_per_sample": (
                result.total_time / len(result.predictions)
                if result.total_time
                else None
            ),
            "metric_config": result.metrics.metric_config,
            "model_type": "cascaded" if has_intermediate else "end_to_end",
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

    def _save_asr_summary(
        self, results: List[ASREvaluationResult], experiment_name: str
    ):
        exp_dir = self.output_dir / experiment_name
        summary_path = exp_dir / f"{experiment_name}_asr_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["model_name", "language", "wer", "cer", "num_samples", "total_time"]
            )
            for r in results:
                writer.writerow(
                    [
                        r.model_name,
                        r.language,
                        r.metrics.wer,
                        r.metrics.cer,
                        r.metrics.num_samples,
                        r.total_time,
                    ]
                )
        print(f"ASR summary saved to {summary_path}")

    def _save_ast_summary(
        self, results: List[ASTEvaluationResult], experiment_name: str
    ):
        exp_dir = self.output_dir / experiment_name
        summary_path = exp_dir / f"{experiment_name}_ast_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model_name",
                    "language_pair",
                    "source_lang",
                    "target_lang",
                    "bleu",
                    "chrf",
                    "num_samples",
                    "total_time",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.model_name,
                        f"{r.source_lang}→{r.target_lang}",
                        r.source_lang,
                        r.target_lang,
                        r.metrics.bleu,
                        r.metrics.chrf,
                        r.metrics.num_samples,
                        r.total_time,
                    ]
                )
        print(f"AST summary saved to {summary_path}")