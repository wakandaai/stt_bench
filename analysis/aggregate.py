#!/usr/bin/env python3
"""
§8.3 aggregation: turn per-utterance judge labels into category proportions.

Reads taxonomy_labels.jsonl (the Gemini judge output) and writes:
  * taxonomy_summary.csv   -- one row per (benchmark, model, lang): Tier A rates,
                              n_clean, Tier B category counts + proportions (on
                              CLEAN utts only), and the B4/B5 split.
  * taxonomy_rollup.csv    -- Tier B proportions rolled up by (typology, model),
                              on the set of cells where the model was sampled.

Tier A pathologies (A1/A2/A3) are reported as rates but EXCLUDED from the Tier B
proportions (guardrail: pathologies out of linguistic stats). The B4-vs-B5 split
(sub_phonetic vs sub_lm_plausible) is surfaced explicitly -- it is the sharpest
H2 evidence (no-LM CTC -> phonetic; African-LM Aura -> fluent substitution).

Usage:
  python -m analysis.aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from analysis import common as c

TIER_A = ["CLEAN", "A1_script", "A2_loop", "A3_truncation"]
CATS = ["boundary", "tone_diacritic", "morphology",
        "sub_phonetic", "sub_lm_plausible", "function_word"]


def load(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _cat_row(errors_counter: Counter, n_err: int) -> Dict[str, Any]:
    """Category counts, proportions (%) and the B4/B5 split."""
    row: Dict[str, Any] = {"n_errors": n_err}
    for cat in CATS:
        row[f"pct_{cat}"] = round(100.0 * errors_counter.get(cat, 0) / n_err, 2) if n_err else 0.0
    b4 = errors_counter.get("sub_phonetic", 0)
    b5 = errors_counter.get("sub_lm_plausible", 0)
    row["b5_share_of_subs"] = round(100.0 * b5 / (b4 + b5), 2) if (b4 + b5) else 0.0
    return row


def summarize_cells(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cells: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cells[(r["benchmark"], r["model"], r["lang"])].append(r)

    out: List[Dict[str, Any]] = []
    for (benchmark, model, lang), rs in sorted(cells.items()):
        n = len(rs)
        prov = Counter(r["provenance"] for r in rs)
        clean = [r for r in rs if r["provenance"] == "CLEAN"]
        errc = Counter(e["category"] for r in clean for e in r["errors"])
        info = c.lang_info(lang)
        rec = {
            "benchmark": benchmark, "model": model, "lang": lang,
            "lang_name": info.name, "typology": info.typology, "ortho": info.ortho,
            "has_lm": c.MODEL_HAS_LM.get(model, ""),
            "n_sampled": n, "n_clean": len(clean),
        }
        for k in TIER_A:
            rec[f"pct_{k}"] = round(100.0 * prov.get(k, 0) / n, 2) if n else 0.0
        rec.update(_cat_row(errc, sum(errc.values())))
        out.append(rec)
    return out


def rollup_typology(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tier B proportions by (typology, model), pooling CLEAN errors."""
    groups: Dict[tuple, Counter] = defaultdict(Counter)
    n_clean: Dict[tuple, int] = defaultdict(int)
    for r in rows:
        if r["provenance"] != "CLEAN":
            continue
        info = c.lang_info(r["lang"])
        key = (info.typology, r["model"])
        n_clean[key] += 1
        for e in r["errors"]:
            groups[key][e["category"]] += 1

    out: List[Dict[str, Any]] = []
    for (typology, model), errc in sorted(groups.items()):
        rec = {"typology": typology, "model": model,
               "has_lm": c.MODEL_HAS_LM.get(model, ""), "n_clean": n_clean[(typology, model)]}
        rec.update(_cat_row(errc, sum(errc.values())))
        out.append(rec)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {len(rows)} rows -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="§8.3 taxonomy aggregation.")
    ap.add_argument("-i", "--input", type=Path,
                    default=Path("analysis/outputs/taxonomy_labels.jsonl"))
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("analysis/outputs"))
    args = ap.parse_args()

    rows = load(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "taxonomy_summary.csv", summarize_cells(rows))
    write_csv(args.output_dir / "taxonomy_rollup.csv", rollup_typology(rows))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
