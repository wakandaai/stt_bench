# stt_benchmark/evaluation/metrics.py

import sacrebleu
from jiwer import wer, cer
from typing import List, Dict, Any, Optional
from stt_benchmark.evaluation.base import ASRMetrics, ASTMetrics
from stt_benchmark.utils.text_normalize import TextNormalizer, DEFAULT_NORMALIZER


class ASRMetricsCalculator:
    """Calculator for ASR metrics (WER, CER)."""

    def __init__(self, normalizer: TextNormalizer = None):
        self.normalizer = normalizer or DEFAULT_NORMALIZER

    def calculate(self, hypotheses: List[str], references: List[str]) -> ASRMetrics:
        if len(hypotheses) != len(references):
            raise ValueError(
                f"Mismatch: {len(hypotheses)} hypotheses vs {len(references)} references"
            )

        norm_hyps = [self.normalizer.normalize(h) for h in hypotheses]
        norm_refs = [self.normalizer.normalize(r) for r in references]

        # Filter out empty references
        valid_pairs = [(h, r) for h, r in zip(norm_hyps, norm_refs) if r.strip()]
        if not valid_pairs:
            return ASRMetrics(
                wer=100.0, cer=100.0, num_samples=0,
                metric_config=self.normalizer.get_config(),
            )

        valid_hyps, valid_refs = zip(*valid_pairs)

        corpus_wer = wer(list(valid_refs), list(valid_hyps))
        corpus_cer = cer(list(valid_refs), list(valid_hyps))

        return ASRMetrics(
            wer=corpus_wer * 100,
            cer=corpus_cer * 100,
            num_samples=len(valid_pairs),
            metric_config=self.normalizer.get_config(),
        )


class ASTMetricsCalculator:
    """Calculator for AST metrics (BLEU, chrF++, spBLEU-1K)."""

    def __init__(
        self,
        bleu_config: Dict[str, Any] = None,
        chrf_config: Dict[str, Any] = None,
        compute_spbleu: bool = True,
    ):
        self.bleu_config = bleu_config or {}
        self.chrf_config = chrf_config or {"word_order": 2}
        self.compute_spbleu = compute_spbleu

        self.bleu_metric = sacrebleu.BLEU(**self.bleu_config)
        self.chrf_metric = sacrebleu.CHRF(**self.chrf_config)

        # spBLEU-1K is sacrebleu BLEU with the spBLEU-1K SPM tokenizer (Toucan,
        # ACL 2024). Requires sacrebleu >= 2.6.0. The SPM model is auto-fetched
        # on first call into ~/.sacrebleu/models/.
        self.spbleu_metric = None
        if self.compute_spbleu:
            spbleu_config = dict(self.bleu_config)
            spbleu_config["tokenize"] = "spBLEU-1K"
            try:
                self.spbleu_metric = sacrebleu.BLEU(**spbleu_config)
            except Exception as e:
                print(
                    f"  ⚠️  spBLEU-1K unavailable: {e}\n"
                    f"      (requires sacrebleu>=2.6.0 and network access to "
                    f"fetch the SPM model on first use). Skipping spBLEU."
                )
                self.spbleu_metric = None

    def calculate(self, hypotheses: List[str], references: List[str]) -> ASTMetrics:
        if len(hypotheses) != len(references):
            raise ValueError(
                f"Mismatch: {len(hypotheses)} hypotheses vs {len(references)} references"
            )

        bleu_score = self.bleu_metric.corpus_score(hypotheses, [references])
        chrf_score = self.chrf_metric.corpus_score(hypotheses, [references])

        spbleu_value: Optional[float] = None
        spbleu_config_out: Optional[Dict[str, Any]] = None
        if self.spbleu_metric is not None:
            try:
                spbleu_score = self.spbleu_metric.corpus_score(hypotheses, [references])
                spbleu_value = spbleu_score.score
                spbleu_config_out = {"tokenize": "spBLEU-1K", **self.bleu_config}
            except Exception as e:
                print(f"  ⚠️  spBLEU-1K scoring failed: {e}")

        return ASTMetrics(
            bleu=bleu_score.score,
            chrf=chrf_score.score,
            num_samples=len(hypotheses),
            metric_config={
                "bleu_config": self.bleu_config,
                "chrf_config": self.chrf_config,
                "spbleu_config": spbleu_config_out,
            },
            spbleu=spbleu_value,
        )


class SsaCometScorer:
    """Wrapper around McGill-NLP/ssa-comet-mtl for AST reference-based scoring.

    Lazy-loads the COMET model on first call. The model is heavy (~XL-R large)
    so we keep this opt-in via the EvaluationPipeline `compute_ssa_comet` flag.

    Inputs: lists of (src, mt, ref) strings — that is, source-language text,
    the model's hypothesis, and the reference target-language text.
    """

    DEFAULT_MODEL = "McGill-NLP/ssa-comet-mtl"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 8,
        gpus: Optional[int] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        # If gpus is None, auto-detect: 1 if CUDA available else 0.
        if gpus is None:
            try:
                import torch
                gpus = 1 if torch.cuda.is_available() else 0
            except ImportError:
                gpus = 0
        self.gpus = gpus
        self._model = None
        # Cache the most recent prediction so that callers asking for both the
        # system score and the per-segment scores on the same inputs (e.g. the
        # recompute tool computing a corpus number and CSV columns) only pay
        # for one COMET forward pass.
        self._cache_key = None
        self._cache_output = None

    def _load(self):
        if self._model is not None:
            return
        from comet import download_model, load_from_checkpoint
        print(f"  Loading SSA-COMET model: {self.model_name}")
        model_path = download_model(self.model_name)
        self._model = load_from_checkpoint(model_path)

    def _predict(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ):
        """Run COMET and return its PredictionOutput, or None on failure.

        All three lists must be the same length. Empty hypotheses are scored
        as-is (COMET handles them); we do not filter, to keep counts aligned
        with the rest of the AST metrics on this run. Results are cached on the
        exact input triple so a follow-up call with identical inputs is free.
        """
        if not (len(sources) == len(hypotheses) == len(references)):
            raise ValueError(
                f"SSA-COMET length mismatch: src={len(sources)}, "
                f"mt={len(hypotheses)}, ref={len(references)}"
            )
        if not hypotheses:
            return None

        key = (tuple(sources), tuple(hypotheses), tuple(references))
        if key == self._cache_key:
            return self._cache_output

        try:
            self._load()
        except Exception as e:
            print(f"  ⚠️  Failed to load SSA-COMET model: {e}")
            return None

        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        try:
            output = self._model.predict(
                data, batch_size=self.batch_size, gpus=self.gpus
            )
        except Exception as e:
            print(f"  ⚠️  SSA-COMET prediction failed: {e}")
            return None

        self._cache_key = key
        self._cache_output = output
        return output

    def score(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> Optional[float]:
        """Return system-level SSA-COMET score, or None on failure."""
        output = self._predict(sources, hypotheses, references)
        if output is None:
            return None

        # comet's PredictionOutput exposes .system_score (a float in [0, 1]).
        # Fall back to averaging .scores if for some reason that attribute is
        # missing in this comet version.
        sys_score = getattr(output, "system_score", None)
        if sys_score is None:
            scores = getattr(output, "scores", None)
            if not scores:
                return None
            sys_score = sum(scores) / len(scores)
        return float(sys_score)

    def score_segments(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> Optional[List[float]]:
        """Return per-segment SSA-COMET scores (one float per input), or None.

        Order matches the inputs. Useful for writing a per-instance column
        alongside predictions.
        """
        output = self._predict(sources, hypotheses, references)
        if output is None:
            return None
        scores = getattr(output, "scores", None)
        if scores is None:
            return None
        return [float(s) for s in scores]