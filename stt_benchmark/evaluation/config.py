# stt_benchmark/evaluation/config.py

"""
Evaluation config loader.

Reads a YAML eval config and produces a list of per-dataset evaluation specs.
Each dataset block declares which languages / pairs to evaluate against
*that dataset specifically*.

Schema:
    experiment_name: my_eval

    datasets:
      <dataset_name>:
        # Arbitrary dataset-specific kwargs (split, root, etc.) pass through
        # to the dataset constructor via the registry.
        split: test
        root: /path/to/local/data            # only for local datasets

        asr:
          languages: [sw_ke, yo_ng, ...]

        ast:
          # Either anchor-style:
          sources: [sw_ke, yo_ng]             # defaults to asr.languages
          anchors: [en_us]
          direction: forward                   # "forward" | "reverse" | "both"

          # Or explicit pairs (additive with anchor expansion):
          pairs:
            - [bem, en_us]

Dataset-specific config keys (anything not 'asr' or 'ast') is passed to the
dataset constructor as **kwargs.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Any


@dataclass
class DatasetEvalSpec:
    """Evaluation plan for a single dataset."""
    dataset_name: str
    dataset_kwargs: Dict[str, Any]
    asr_languages: List[str]
    ast_pairs: List[Tuple[str, str]]

    def has_asr(self) -> bool:
        return len(self.asr_languages) > 0

    def has_ast(self) -> bool:
        return len(self.ast_pairs) > 0

    def summary(self) -> str:
        lines = [f"  Dataset: {self.dataset_name}"]
        if self.dataset_kwargs:
            kw = ", ".join(f"{k}={v}" for k, v in self.dataset_kwargs.items())
            lines.append(f"    kwargs: {kw}")
        lines.append(f"    ASR languages: {len(self.asr_languages)}")
        if self.asr_languages:
            lines.append(f"      {', '.join(self.asr_languages)}")
        lines.append(f"    AST pairs: {len(self.ast_pairs)}")
        if self.ast_pairs and len(self.ast_pairs) <= 10:
            for src, tgt in self.ast_pairs:
                lines.append(f"      {src} → {tgt}")
        elif self.ast_pairs:
            for src, tgt in self.ast_pairs[:5]:
                lines.append(f"      {src} → {tgt}")
            lines.append(f"      ... and {len(self.ast_pairs) - 5} more")
        return "\n".join(lines)


@dataclass
class EvalConfig:
    """Parsed evaluation configuration — one experiment, many datasets."""
    experiment_name: str
    datasets: List[DatasetEvalSpec] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Datasets: {len(self.datasets)}",
        ]
        for spec in self.datasets:
            lines.append(spec.summary())
        return "\n".join(lines)


def _expand_ast(ast_section: Dict[str, Any],
                default_sources: List[str]) -> List[Tuple[str, str]]:
    """Expand the ast: block into a list of (source, target) pairs.

    Supports anchor-style expansion + explicit `pairs:` list, additive.
    """
    sources: List[str] = ast_section.get("sources", default_sources)
    anchors: List[str] = ast_section.get("anchors", [])
    direction: str = ast_section.get("direction", "both")

    if direction not in ("forward", "reverse", "both"):
        raise ValueError(
            f"Invalid ast.direction: '{direction}'. "
            f"Must be 'forward', 'reverse', or 'both'."
        )

    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    def _add(src: str, tgt: str):
        pair = (src, tgt)
        if src != tgt and pair not in seen:
            pairs.append(pair)
            seen.add(pair)

    for src in sources:
        for anchor in anchors:
            if direction in ("forward", "both"):
                _add(src, anchor)
            if direction in ("reverse", "both"):
                _add(anchor, src)

    # Explicit pairs are always added regardless of direction.
    explicit_pairs = ast_section.get("pairs", [])
    for entry in explicit_pairs:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            _add(entry[0], entry[1])
        elif isinstance(entry, dict) and "source" in entry and "target" in entry:
            _add(entry["source"], entry["target"])
        else:
            raise ValueError(
                f"Invalid ast.pairs entry: {entry}. "
                f"Expected [src, tgt] or {{source: ..., target: ...}}."
            )

    return pairs


def _parse_dataset_block(name: str, block: Dict[str, Any]) -> DatasetEvalSpec:
    """Parse a single dataset block from the YAML."""
    block = block or {}

    asr_section = block.get("asr", {}) or {}
    ast_section = block.get("ast", {}) or {}

    asr_languages: List[str] = asr_section.get("languages", [])
    ast_pairs = _expand_ast(ast_section, default_sources=asr_languages)

    # Everything that isn't asr/ast is passed to the dataset constructor.
    dataset_kwargs = {k: v for k, v in block.items() if k not in ("asr", "ast")}

    return DatasetEvalSpec(
        dataset_name=name,
        dataset_kwargs=dataset_kwargs,
        asr_languages=asr_languages,
        ast_pairs=ast_pairs,
    )


def load_eval_config(config_path: str) -> EvalConfig:
    """Load and parse an evaluation config YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Eval config not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    experiment_name = raw.get("experiment_name", config_path.stem)

    datasets_section = raw.get("datasets", {})
    if not datasets_section:
        raise ValueError(
            f"No 'datasets:' block found in {config_path}. "
            f"Eval configs must specify at least one dataset."
        )

    specs = [
        _parse_dataset_block(name, block)
        for name, block in datasets_section.items()
    ]

    return EvalConfig(experiment_name=experiment_name, datasets=specs)