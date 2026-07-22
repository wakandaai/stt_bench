#!/usr/bin/env python3
"""
Deterministic sampling + judge-ready input builder for the §5 LLM-judge pass.

Produces the input the Gemini judge will consume -- but makes NO API call. For
each sampled utterance we emit reference + hypothesis + the pre-computed aligned
edit operations (from jiwer) + the automatic features (sCER, len_ratio, diacritic
delta, Tier A label). Per §5 rule 2 the judge is fed the ALIGNMENT, not raw
strings alone, so it adjudicates real errors instead of hallucinating them.

Sampling is fixed-seed and records the sampled sample_ids. By default we sample
the story-critical (benchmark, language) cells for the CTC / Aura / Omni decoders
(the B4->B5 contrast), ~100 utterances each.

Usage:
  python -m analysis.sample_for_judge results/african_eval -o analysis/outputs
  python -m analysis.sample_for_judge results/african_eval --per-cell 100 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from jiwer import cer as jiwer_cer, process_words, wer as jiwer_wer

from analysis import common as c


# Story-critical (benchmark, lang) cells to prioritise (§5 scope).
PRIORITY_CELLS = [
    ("waxal", "am_et"), ("waxal", "lg_ug"), ("waxal", "ln_cd"),
    ("waxal", "mg_mg"), ("waxal", "sn_zw"), ("waxal", "ti_et"),  # 6 Waxal langs
    ("bembaspeech", "bem"), ("bigc", "bem"),                     # both Bemba
    ("fleurs", "yo_ng"), ("fleurs", "ig_ng"),                    # tonal FLEURS
    ("fleurs", "ar_eg"),                                         # Arabic failure case
    ("nchlt", "st_za"),                                          # NCHLT collapse case
]

# Decoders to label (CTC + Aura at minimum for the B4->B5 shift; Omni for the
# English-LM contrast).
PRIORITY_MODELS = ["Conformer-CTC", "Aura-ASR", "OmniASR-LLM-1B"]


def build_ops(nref: str, nhyp: str) -> List[Dict[str, Any]]:
    """Aligned non-equal edit operations (word level) as ref/hyp spans."""
    ref_words = nref.split()
    hyp_words = nhyp.split()
    out = process_words([nref], [nhyp])
    ops: List[Dict[str, Any]] = []
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            continue
        ops.append({
            "type": chunk.type,  # substitute | insert | delete
            "ref_span": " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx]),
            "hyp_span": " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]),
        })
    return ops


def utterance_features(nref: str, nhyp: str) -> Dict[str, Any]:
    """Automatic features the judge conditions on (all recomputed, normalized)."""
    ref_words = nref.split()
    hyp_words = nhyp.split()
    n_ref = len(ref_words)
    scer = jiwer_cer(c.remove_spaces(nref), c.remove_spaces(nhyp)) * 100 if nref else 0.0
    u_wer = jiwer_wer(nref, nhyp) * 100 if nref else 0.0
    u_wer_nd = jiwer_wer(c.strip_diacritics(nref), c.strip_diacritics(nhyp)) * 100 if nref else 0.0
    return {
        "cer": round(jiwer_cer(nref, nhyp) * 100, 2) if nref else 0.0,
        "wer": round(u_wer, 2),
        "scer": round(scer, 2),
        "len_ratio": round(len(hyp_words) / n_ref, 3) if n_ref else 0.0,
        "diacritic_wer_delta": round(u_wer - u_wer_nd, 2),
    }


def load_and_build(path: Path) -> List[Dict[str, Any]]:
    """Load a CSV and build a judge-input object per scorable utterance."""
    objs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            ref = (raw.get("reference") or "").strip()
            hyp = (raw.get("hypothesis") or "").strip()
            nref = c.NORMALIZER.normalize(ref)
            nhyp = c.NORMALIZER.normalize(hyp)
            if not nref.strip():
                continue
            objs.append({
                "sample_id": raw.get("sample_id", ""),
                "reference": nref,
                "hypothesis": nhyp,
                "auto_provenance": c.tier_a_label(nref, nhyp),
                "features": utterance_features(nref, nhyp),
                "aligned_ops": build_ops(nref, nhyp),
            })
    return objs


def sample_cell(objs: List[Dict[str, Any]], per_cell: int, seed: int) -> List[Dict[str, Any]]:
    if len(objs) <= per_cell:
        return objs
    rng = random.Random(seed)
    return rng.sample(objs, per_cell)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build judge-ready sample (no API call).")
    ap.add_argument("results_dir", type=Path, help="e.g. results/african_eval")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("analysis/outputs"))
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-cells", action="store_true",
                    help="Sample every included cell instead of only the priority set.")
    args = ap.parse_args()

    included, _ = c.collect_prediction_files(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    n_written = 0
    jsonl_path = args.output_dir / "judge_input.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for path in included:
            benchmark, model, lang = c.parse_path(path)
            if not args.all_cells:
                if (benchmark, lang) not in PRIORITY_CELLS or model not in PRIORITY_MODELS:
                    continue
            objs = load_and_build(path)
            if not objs:
                continue
            # Seed varies per cell (deterministically) so cells don't all pick row 0..k.
            cell_seed = args.seed + hash((benchmark, model, lang)) % 100000
            sampled = sample_cell(objs, args.per_cell, cell_seed)
            for o in sampled:
                record = {"benchmark": benchmark, "model": model, "lang": lang, **o}
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.append({"benchmark": benchmark, "model": model,
                                 "lang": lang, "sample_id": o["sample_id"]})
                n_written += 1

    if n_written == 0:
        print("[warn] no utterances sampled (check priority set / --all-cells)", file=sys.stderr)
        return 1

    man_path = args.output_dir / "judge_sample_manifest.csv"
    with open(man_path, "w", encoding="utf-8", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=["benchmark", "model", "lang", "sample_id"])
        writer.writeheader()
        writer.writerows(manifest)

    n_cells = len({(m["benchmark"], m["model"], m["lang"]) for m in manifest})
    print(f"[ok] wrote {n_written} judge-input rows across {n_cells} cells -> {jsonl_path}")
    print(f"[ok] wrote manifest -> {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
