#!/usr/bin/env python3
"""
§5 LLM-judge pass (Gemini) for the Aura-ASR error analysis.

The judge CATEGORIZES errors; it never ranks systems (§5 rule 1 -- asking which
transcription is "better" would reintroduce the very English-LM prior the paper
studies). It is an adjudicator ON TOP OF the automatic pass: it is fed the
reference, hypothesis, the pre-computed aligned edit operations and the automatic
features (from analysis/sample_for_judge.py), and returns:
  * a Tier A provenance label (A1_script / A2_loop / A3_truncation / CLEAN), and
  * for CLEAN hypotheses, one Tier B label per aligned error, first-matching in
    the B1->B6 precedence order.

Deterministic: temperature 0, fixed prompt, pinned model, structured JSON output.
The exact model string and full prompt are written to judge_appendix.md for the
paper appendix.

Reads the API key from a file (default /jet/home/gichamba/gemini_key); the key is
never logged. Output is resumable: already-judged (benchmark, model, lang,
sample_id) rows in the output JSONL are skipped on re-run.

Usage:
  # small dry-run to validate prompt/schema first
  python -m analysis.judge --limit 20
  # full pass
  python -m analysis.judge
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from typing import Literal

from google import genai
from google.genai import types


# =========================================================================
# Structured output schema (mirrors the §5 JSONL schema)
# =========================================================================

PROVENANCE = Literal["A1_script", "A2_loop", "A3_truncation", "CLEAN"]
CATEGORY = Literal[
    "boundary", "tone_diacritic", "morphology",
    "sub_phonetic", "sub_lm_plausible", "function_word",
]


class ErrorItem(BaseModel):
    ref_span: str
    hyp_span: str
    category: CATEGORY
    confidence: float


class Judgment(BaseModel):
    item_id: int  # echoes the batch position so we can map back reliably
    provenance: PROVENANCE
    errors: List[ErrorItem]
    notes: str


# =========================================================================
# Prompt (fixed; recorded verbatim in the appendix)
# =========================================================================

SYSTEM_INSTRUCTION = """\
You are a linguistic error-categorization tool for African-language ASR output.
You are given, for one utterance: a reference transcript, a hypothesis transcript,
the pre-computed word-level aligned edit operations between them, and automatic
diagnostic features. Your job is to LABEL the errors, using a fixed two-tier
taxonomy. You must follow these rules exactly.

CRITICAL RULES
1. CATEGORIZE, NEVER RANK. Never judge which transcript is "better" or score
   transcription quality. Only assign the taxonomy labels below.
2. Use the PROVIDED aligned edit operations. Do not invent errors that are not in
   the alignment. Each Tier B label you emit must correspond to one aligned
   substitute/insert/delete operation (or a directly adjacent group of them).
3. Assign labels by PRECEDENCE, first-matching. Tier A is decided before Tier B.
4. Output MUST match the requested JSON schema. `confidence` is your 0.0-1.0
   confidence in that single label.

TIER A -- provenance / whole-hypothesis pathology. Choose exactly one:
  A1_script      : the hypothesis is largely in the WRONG script or language
                   relative to the reference (e.g. Latin text for an Arabic ref).
  A2_loop        : degeneration loop -- the hypothesis repeats token bursts and is
                   much longer than the reference.
  A3_truncation  : the hypothesis is near-empty or collapsed (far shorter than ref).
  CLEAN          : none of the above.
The `auto_provenance` field is an automatic first guess; correct it if the strings
show otherwise. If provenance is A1/A2/A3, return an EMPTY errors list (Tier B is
only assigned to CLEAN hypotheses).

TIER B -- linguistic error type, one label PER aligned error, first-matching in
this precedence order (B1 highest):
  boundary        (B1): word-boundary disagreement only -- the SAME characters,
                        different spacing. e.g. ref "alimumuputule" vs hyp
                        "ali mu muputule". Conjunctive/disjunctive orthography.
  tone_diacritic  (B2): the segmental letters are correct but tone marks / accents
                        / sub-dot diacritics differ. (See diacritic_wer_delta.)
  morphology      (B3): an affix / agreement-prefix / clitic error INSIDE an
                        otherwise correct word boundary.
  sub_phonetic    (B4): the wrong word is an acoustically similar garble -- a
                        phonetic substitution the sound could force.
  sub_lm_plausible(B5): the wrong word is fluent and in-language but semantically
                        wrong -- a substitution a language model would make, not one
                        the acoustics force (semantic drift, plausible completion).
  function_word   (B6): a residual grammatical / function-word error.

The B4-vs-B5 distinction is important: B4 = sounds like the target (phonetic),
B5 = sounds different but is a fluent in-language word (LM-driven). Judge by
whether the acoustics plausibly force the confusion.
"""

USER_TEMPLATE = """\
Categorize the errors in each of the following {n} utterances. Return one
judgment object per utterance, echoing its `item_id`. Utterances are independent.

{items}
"""


def _item_view(i: int, rec: Dict[str, Any]) -> Dict[str, Any]:
    """The per-utterance payload the judge sees (item_id + ref/hyp/ops/features)."""
    return {
        "item_id": i,
        "language": rec["lang"],
        "reference": rec["reference"],
        "hypothesis": rec["hypothesis"],
        "auto_provenance": rec["auto_provenance"],
        "features": rec["features"],
        "aligned_ops": rec["aligned_ops"],
    }


def build_user_content(batch: List[Dict[str, Any]]) -> str:
    items = [_item_view(i, rec) for i, rec in enumerate(batch)]
    return USER_TEMPLATE.format(n=len(batch),
                                items=json.dumps(items, ensure_ascii=False, indent=1))


# =========================================================================
# Judging
# =========================================================================

_ID_FIELDS = ("benchmark", "model", "lang", "sample_id")


def _key(rec: Dict[str, Any]) -> tuple:
    return tuple(rec[k] for k in _ID_FIELDS)


def _retry_delay_seconds(err: Exception, attempt: int) -> float:
    """Honor the server's suggested retryDelay on 429s; else exponential backoff."""
    msg = str(err)
    m = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", msg)
    if m:
        return float(m.group(1)) + 1.0
    return min(2 ** attempt, 60)


def _error_record(rec: Dict[str, Any], err: Exception) -> Dict[str, Any]:
    return {
        **{k: rec[k] for k in _ID_FIELDS},
        "provenance": None,
        "errors": [],
        "notes": f"ERROR: {type(err).__name__}: {err}",
    }


def judge_batch(client: genai.Client, model: str, batch: List[Dict[str, Any]],
                max_retries: int = 6) -> List[Dict[str, Any]]:
    """Judge a batch of utterances in one request; returns one record per input.

    The model echoes `item_id`; we map judgments back by it, so mis-ordering or a
    missing item is detected rather than silently misattributed.
    """
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=list[Judgment],
    )
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=build_user_content(batch),
                config=config,
            )
            judgments: List[Judgment] = resp.parsed or []
            by_id = {j.item_id: j for j in judgments}
            out: List[Dict[str, Any]] = []
            for i, rec in enumerate(batch):
                j = by_id.get(i)
                if j is None:
                    out.append(_error_record(rec, ValueError(f"no judgment for item_id {i}")))
                    continue
                out.append({
                    **{k: rec[k] for k in _ID_FIELDS},
                    "provenance": j.provenance,
                    "errors": [e.model_dump() for e in j.errors],
                    "notes": j.notes,
                })
            return out
        except Exception as e:  # rate limit / transient / parse
            last_err = e
            time.sleep(_retry_delay_seconds(e, attempt))
    return [_error_record(rec, last_err) for rec in batch]


def chunk(seq: List[Any], size: int) -> List[List[Any]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def load_inputs(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("provenance") is not None:  # only completed rows count
                    done.add(tuple(r[k] for k in _ID_FIELDS))
            except json.JSONDecodeError:
                continue
    return done


def write_appendix(out_dir: Path, model: str) -> None:
    path = out_dir / "judge_appendix.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# LLM-judge configuration (paper appendix)\n\n")
        f.write(f"- **Model (pinned):** `{model}`\n")
        f.write("- **Provider/SDK:** Google Gemini via `google-genai` client\n")
        f.write("- **Decoding:** temperature 0.0, structured JSON output "
                "(`response_schema=Judgment`), fixed prompt\n")
        f.write("- **Input:** reference + hypothesis + pre-computed word-level "
                "aligned edit ops + automatic features (categorize, not rank)\n\n")
        f.write("## System instruction (verbatim)\n\n```\n")
        f.write(SYSTEM_INSTRUCTION)
        f.write("\n```\n\n## Per-utterance user template (verbatim)\n\n```\n")
        f.write(USER_TEMPLATE)
        f.write("\n```\n")
    print(f"[ok] wrote appendix -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="§5 Gemini LLM-judge categorization pass.")
    ap.add_argument("-i", "--input", type=Path,
                    default=Path("analysis/outputs/judge_input.jsonl"))
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("analysis/outputs/taxonomy_labels.jsonl"))
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--key-file", type=Path, default=Path("/jet/home/gichamba/gemini_key"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Judge only the first N un-done utterances (dry-run).")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="Utterances per API request (cuts request count & cost).")
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent requests (keep low for free-tier rate limits).")
    args = ap.parse_args()

    key = args.key_file.read_text().strip()
    if not key:
        print(f"[error] key file {args.key_file} is empty", file=sys.stderr)
        return 1
    client = genai.Client(api_key=key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_appendix(args.output.parent, args.model)

    inputs = load_inputs(args.input)
    done = load_done(args.output)
    todo = [r for r in inputs if _key(r) not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"[info] {len(inputs)} inputs, {len(done)} already done, "
          f"judging {len(todo)} with {args.model}")
    if not todo:
        print("[ok] nothing to do")
        return 0

    batches = chunk(todo, args.batch_size)
    lock = threading.Lock()
    n_ok = n_err = n_done = 0
    with open(args.output, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge_batch, client, args.model, b): b for b in batches}
            for bi, fut in enumerate(as_completed(futs), 1):
                results = fut.result()
                with lock:
                    for res in results:
                        out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        n_done += 1
                        if res["provenance"] is None:
                            n_err += 1
                        else:
                            n_ok += 1
                    out_f.flush()
                if bi % 5 == 0 or bi == len(batches):
                    print(f"  batch [{bi}/{len(batches)}]  utts done={n_done} ok={n_ok} err={n_err}")

    print(f"[ok] judged {n_ok} utterances ({n_err} errors) -> {args.output}")
    return 1 if n_ok == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
