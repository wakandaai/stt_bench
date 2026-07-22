#!/usr/bin/env python3
"""
Stacked-normalization re-scoring sweep for ASR prediction CSVs (dev-only).

Recomputes corpus WER and CER over existing predictions under a STACK of
progressively heavier, language-agnostic normalization schemes, so that real
model error can be separated from formatting artifact before any scheme is
frozen for test.

The stack (each adds to the previous):

  raw       no normalization (score the literal strings)
  +case     lowercase
  +punct    strip punctuation (keep in-word apostrophes/hyphens)
  +nfc      Unicode NFC (reconcile composed/decomposed diacritics; tone marks
            are PRESERVED -- diacritics are never stripped). This is the final,
            language-agnostic scheme intended for freezing.

CER is reported beside WER at every step because CER is segmentation-invariant:
for agglutinative languages, a large WER with a small CER is word-boundary
disagreement (a scoring artifact), not a model error.

NUMERIC TAX (separate, honest measurement):
  Digit<->spelled-out mismatch (ref "1755" vs hyp "igihumbi kimwe...") cannot be
  fixed by masking, because the spelled-out side carries no digits to mask;
  collapsing it back to a number needs a per-language number grammar (true ITN).
  So instead of faking it, we partition utterances by whether the REFERENCE
  contains a digit and report WER on full / digit-free / digit-only subsets. The
  (full - digit_free) gap is the real upper bound on what per-language ITN could
  recover -- and it needs no lookup table.

Output:
  * printed per-language table of WER@scheme (and the WER removed from raw->final)
  * printed numeric-tax table (full / digit-free / digit-only at the +nfc scheme)
  * --output-dir writes sweep_wer.csv, sweep_cer.csv, sweep_long.csv, numeric_tax.csv

Everything here is descriptive re-scoring on results_dev. No inference. The
chosen scheme should be frozen (and recorded as a normalizer_config) before
test is touched.

Usage:
  python utils/normalization_sweep.py results_dev/african_eval
  python utils/normalization_sweep.py results_dev/african_eval -o reports/norm_sweep
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jiwer import wer as jiwer_wer, cer as jiwer_cer

from stt_benchmark.utils.text_normalize import TextNormalizer


# Ordered, cumulative stack. Each entry is (scheme_name, normalizer_kwargs).
# kwargs are the FULL config at that step (cumulative), so the frozen choice can
# be dropped straight into recompute_metrics.py via --normalizer / normalizer_config.
SCHEME_STACK: List[Tuple[str, Dict[str, Any]]] = [
    ("raw",    dict(lowercase=False, remove_punctuation=False)),
    ("+case",  dict(lowercase=True,  remove_punctuation=False)),
    ("+punct", dict(lowercase=True,  remove_punctuation=True)),
    ("+nfc",   dict(lowercase=True,  remove_punctuation=True, unicode_form="NFC")),
]

# Config used for the numeric-tax partition (the final frozen scheme).
FROZEN_KWARGS: Dict[str, Any] = dict(
    lowercase=True, remove_punctuation=True, unicode_form="NFC",
)

import re
_HAS_DIGIT = re.compile(r"\d")


def collect_prediction_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("predictions/*.csv"))


def _derive_dataset_lang(path: Path) -> Tuple[str, str]:
    dataset = path.parent.parent.name if path.parent.parent else ""
    stem = path.stem
    lang = stem
    for marker in ("_asr_", "_transcribe_"):
        idx = stem.rfind(marker)
        if idx != -1:
            lang = stem[idx + len(marker):]
            break
    return dataset, lang


def read_pairs(path: Path) -> Tuple[List[str], List[str]]:
    """Return (references, hypotheses) as raw strings (no normalization here)."""
    refs, hyps = [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            refs.append((row.get("reference") or ""))
            hyps.append((row.get("hypothesis") or ""))
    return refs, hyps


def numeric_tax(refs: List[str], hyps: List[str]) -> Dict[str, Any]:
    """WER on full / digit-free / digit-only reference subsets at the frozen scheme.

    The (full - digit_free) gap is the upper bound on what true per-language ITN
    could recover; digit_only shows how badly the numeric utterances score.
    """
    full = [(r, h) for r, h in zip(refs, hyps)]
    free = [(r, h) for r, h in full if not _HAS_DIGIT.search(r)]
    only = [(r, h) for r, h in full if _HAS_DIGIT.search(r)]

    def _w(pairs):
        if not pairs:
            return None, 0
        rs, hs = zip(*pairs)
        w, _ = score_scheme(list(rs), list(hs), FROZEN_KWARGS)
        return round(w, 2), len(pairs)

    w_full, n_full = _w(full)
    w_free, _ = _w(free)
    w_only, n_only = _w(only)
    tax = round(w_full - w_free, 2) if (w_full is not None and w_free is not None) else 0.0
    return {
        "wer_full": w_full, "wer_digit_free": w_free, "wer_digit_only": w_only,
        "n_digit": n_only, "n_total": n_full, "numeric_tax": tax,
    }


def score_scheme(
    refs: List[str], hyps: List[str], kwargs: Dict[str, Any]
) -> Tuple[float, int]:
    """Corpus WER and CER under one normalization scheme. Returns (wer, cer).

    Rows whose reference normalizes to empty are dropped (no defined WER),
    matching ASRMetricsCalculator. The kept count can vary by scheme (e.g. a
    reference that is purely a number becomes empty under +digit), so it is not
    returned here; counts are reported separately at the base scheme.
    """
    norm = TextNormalizer(**kwargs)
    nrefs, nhyps = [], []
    for r, h in zip(refs, hyps):
        nr = norm.normalize(r)
        if not nr.strip():
            continue
        nrefs.append(nr)
        nhyps.append(norm.normalize(h))
    if not nrefs:
        return 0.0, 0
    w = jiwer_wer(nrefs, nhyps) * 100
    c = jiwer_cer(nrefs, nhyps) * 100
    return w, c


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stacked-normalization WER/CER sweep over ASR prediction CSVs.",
    )
    parser.add_argument("results_dir", type=Path,
                        help="Root to scan, e.g. results_dev/african_eval")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Directory for sweep_wer.csv / sweep_cer.csv / sweep_long.csv")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"[error] {args.results_dir} does not exist", file=sys.stderr)
        return 1
    files = collect_prediction_files(args.results_dir)
    if not files:
        print(f"[error] no predictions/*.csv under {args.results_dir}", file=sys.stderr)
        return 1

    scheme_names = [name for name, _ in SCHEME_STACK]
    wer_rows: List[Dict[str, Any]] = []
    cer_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    tax_rows: List[Dict[str, Any]] = []

    for path in files:
        dataset, lang = _derive_dataset_lang(path)
        refs, hyps = read_pairs(path)
        n = sum(1 for r in refs if r.strip())

        tax = numeric_tax(refs, hyps)
        tax_rows.append({"dataset": dataset, "lang": lang, **tax})

        wer_rec: Dict[str, Any] = {"dataset": dataset, "lang": lang, "n": n}
        cer_rec: Dict[str, Any] = {"dataset": dataset, "lang": lang, "n": n}
        prev_w = None
        for name, kwargs in SCHEME_STACK:
            w, c = score_scheme(refs, hyps, kwargs)
            wer_rec[name] = round(w, 2)
            cer_rec[name] = round(c, 2)
            long_rows.append({
                "dataset": dataset, "lang": lang, "scheme": name,
                "wer": round(w, 2), "cer": round(c, 2),
                "step_wer_delta": round((prev_w - w), 2) if prev_w is not None else 0.0,
            })
            prev_w = w
        wer_rec["wer_removed"] = round(wer_rec[scheme_names[0]] - wer_rec[scheme_names[-1]], 2)
        wer_rows.append(wer_rec)
        cer_rows.append(cer_rec)

    wer_rows.sort(key=lambda r: (r["dataset"], r["lang"]))
    cer_rows.sort(key=lambda r: (r["dataset"], r["lang"]))

    # Printed WER table.
    cols = [("dataset", 16), ("lang", 8), ("n", 6)] + \
           [(name, 9) for name in scheme_names] + [("wer_removed", 13)]
    header = "".join(name.rjust(w) for name, w in cols)
    print("WER by normalization scheme (corpus %, cumulative stack)")
    print(header)
    print("-" * len(header))
    for rec in wer_rows:
        print("".join(str(rec.get(name, "")).rjust(w) for name, w in cols))

    # Numeric-tax table (only languages that actually have digit refs).
    tax_rows.sort(key=lambda r: (r["dataset"], r["lang"]))
    tcols = [("dataset", 16), ("lang", 8), ("n_digit", 9), ("wer_full", 10),
             ("wer_digit_free", 16), ("wer_digit_only", 16), ("numeric_tax", 13)]
    theader = "".join(name.rjust(w) for name, w in tcols)
    print("\nNumeric tax at +nfc (full vs digit-free vs digit-only reference subsets)")
    print(theader)
    print("-" * len(theader))
    for rec in tax_rows:
        if not rec["n_digit"]:
            continue
        print("".join(str(rec.get(name, "")).rjust(w) for name, w in tcols))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for fname, rows in (("sweep_wer.csv", wer_rows),
                            ("sweep_cer.csv", cer_rows),
                            ("sweep_long.csv", long_rows),
                            ("numeric_tax.csv", tax_rows)):
            p = args.output_dir / fname
            with open(p, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[ok] wrote {len(rows)} rows -> {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
