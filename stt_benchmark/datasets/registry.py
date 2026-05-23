# stt_benchmark/datasets/registry.py

"""
Dataset registry.

Maps eval-config dataset names (e.g. 'fleurs', 'bembaspeech') to their loader
classes, and instantiates them with the kwargs from the eval config.

To add a new dataset, import its class and add it to DATASET_REGISTRY.
"""

from typing import Any, Dict, Type

from stt_benchmark.datasets.fleurs import FleursDataset
from stt_benchmark.datasets.mbaza_fleurs_rw import MbazaFleursRwDataset
from stt_benchmark.datasets.bembaspeech import BembaSpeechDataset
from stt_benchmark.datasets.bigc import BigCDataset
from stt_benchmark.datasets.nchlt import NchltDataset


# Map config name -> dataset class.
DATASET_REGISTRY: Dict[str, Type] = {
    "fleurs": FleursDataset,
    "mbaza_fleurs_rw": MbazaFleursRwDataset,
    "bembaspeech": BembaSpeechDataset,
    "bigc": BigCDataset,
    "nchlt": NchltDataset,
}


def create_dataset(name: str, **kwargs) -> Any:
    """Instantiate a dataset by name.

    Args:
        name: Registry key from the eval config (e.g. 'fleurs').
        **kwargs: Passed through to the dataset constructor.

    Returns:
        Dataset instance.
    """
    if name not in DATASET_REGISTRY:
        available = sorted(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Registered datasets: {available}"
        )
    cls = DATASET_REGISTRY[name]
    return cls(**kwargs)


def list_datasets() -> list:
    """List all registered dataset names."""
    return sorted(DATASET_REGISTRY.keys())