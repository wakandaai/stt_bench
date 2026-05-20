# stt_benchmark/models/factory.py

import yaml
from typing import Dict, Any
from pathlib import Path

from stt_benchmark.models.base import BaseASRModel, BaseASTModel, BaseSTTModel
from stt_benchmark.models.hf.whisper import WhisperModel
from stt_benchmark.models.hf.mms import MMSModel
from stt_benchmark.models.hf.seamless import SeamlessModel
from stt_benchmark.models.hf.cascaded import CascadedMmsNllbModel
from stt_benchmark.models.hf.speech_aura import SpeechAuraModel
from stt_benchmark.models.hf.ctc_encoder import CTCEncoderModel


class ModelFactory:
    """Factory for creating STT models from YAML configuration."""

    _model_registry = {
        "whisper": WhisperModel,
        "mms": MMSModel,
        "seamless": SeamlessModel,
        "cascaded": CascadedMmsNllbModel,
        "speech_aura": SpeechAuraModel,
        "ctc_encoder": CTCEncoderModel,
    }

    @classmethod
    def create_model(cls, model_id: str, config_path: str = None):
        """Create a model instance from configuration.

        Args:
            model_id: Model identifier from config (e.g., 'whisper_large_v3')
            config_path: Path to models.yaml (default: stt_benchmark/config/models.yaml)

        Returns:
            Model instance.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "models.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if model_id not in config:
            available = list(config.keys())
            raise ValueError(
                f"Model '{model_id}' not found in config. Available: {available}"
            )

        model_config = config[model_id]
        model_name = model_config.get("model_name", model_id)
        model_type = model_config.get("model_type")

        if model_type is None:
            model_type = cls._infer_model_type(model_id, model_name)

        if model_type not in cls._model_registry:
            raise ValueError(
                f"Unknown model_type '{model_type}' for {model_id}. "
                f"Supported: {list(cls._model_registry.keys())}"
            )

        model_class = cls._model_registry[model_type]
        print(f"Creating {model_type} model: {model_name}")

        # For cascaded models, pass model_id as model_name since there's no
        # single HuggingFace model_name — it composes multiple models.
        if model_type == "cascaded":
            return model_class(model_id, model_config)

        return model_class(model_name, model_config)

    @classmethod
    def _infer_model_type(cls, model_id: str, model_name: str) -> str:
        """Infer model type from model_id or model_name."""
        combined = f"{model_id} {model_name}".lower()
        if "cascaded" in combined:
            return "cascaded"
        elif "whisper" in combined:
            return "whisper"
        elif "mms" in combined:
            return "mms"
        elif "seamless" in combined or "m4t" in combined:
            return "seamless"
        raise ValueError(
            f"Cannot infer model_type for {model_id} ({model_name}). "
            f"Please add 'model_type' to the YAML config."
        )

    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """Register a new model type."""
        cls._model_registry[model_type] = model_class

    @classmethod
    def list_available_models(cls, config_path: str = None) -> Dict[str, Dict[str, Any]]:
        """List all available models from config."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def get_model_tasks(cls, model_id: str, config_path: str = None) -> list:
        """Get the tasks a model supports ('asr', 'ast')."""
        models = cls.list_available_models(config_path)
        if model_id not in models:
            raise ValueError(f"Model '{model_id}' not found")
        return models[model_id].get("tasks", [])