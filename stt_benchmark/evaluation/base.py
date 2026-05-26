# stt_benchmark/evaluation/base.py

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class ASRPrediction:
    """A single ASR prediction."""
    sample_id: str
    reference: str
    hypothesis: str
    audio_path: str
    language: str
    prediction_time: Optional[float] = None


@dataclass
class ASTPrediction:
    """A single AST prediction."""
    sample_id: str
    source_transcription: str
    reference: str
    hypothesis: str
    audio_path: str
    source_lang: str
    target_lang: str
    prediction_time: Optional[float] = None
    intermediate_transcript: Optional[str] = None  # ASR output for cascaded models


@dataclass
class ASRMetrics:
    """Corpus-level ASR metrics."""
    wer: float
    cer: float
    num_samples: int
    metric_config: Dict[str, Any]


@dataclass
class ASTMetrics:
    """Corpus-level AST metrics."""
    bleu: float
    chrf: float
    num_samples: int
    metric_config: Dict[str, Any]
    spbleu: Optional[float] = None       # spBLEU-1K (sacrebleu --tokenize spBLEU-1K)
    ssa_comet: Optional[float] = None    # SSA-COMET-MTL system-level score


@dataclass
class ASREvaluationResult:
    """Complete ASR evaluation result for a language."""
    language: str
    model_name: str
    predictions: List[ASRPrediction]
    metrics: ASRMetrics
    experiment_name: str
    total_time: Optional[float] = None


@dataclass
class ASTEvaluationResult:
    """Complete AST evaluation result for a language pair."""
    source_lang: str
    target_lang: str
    model_name: str
    predictions: List[ASTPrediction]
    metrics: ASTMetrics
    experiment_name: str
    total_time: Optional[float] = None