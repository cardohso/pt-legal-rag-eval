"""Run one model over a list of items, writing every prediction as it is made.

Resumable: an interrupted run keeps what it finished, and a rerun does only what is missing
(items that errored are retried). One JSONL file holds any number of (model, condition,
prompt version) runs; for a key that appears twice, the latest line wins.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ptlegal_eval.chunks import Chunk
from ptlegal_eval.models import GenerationResult, clean_output
from ptlegal_eval.prompts import PROMPT_VERSION, build_messages

Key = tuple[str, str, str, str]  # model, condition, prompt version, item id


@dataclass
class RunSummary:
    generated: int = 0
    skipped: int = 0
    errors: int = 0


def read_predictions(path: Path) -> dict[Key, dict[str, Any]]:
    out: dict[Key, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["model"], r["condition"], r["prompt_version"], r["item_id"])] = r
    return out


def _sha256(messages: list[dict[str, str]]) -> str:
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_items(
    items: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Sequence[Chunk] | None],
    model_key: str,
    condition: str,
    generate: Callable[[list[dict[str, str]]], GenerationResult],
    out_path: Path,
    prompt_version: str = PROMPT_VERSION,
    log: Callable[[str], None] = print,
) -> RunSummary:
    done = read_predictions(out_path)
    summary = RunSummary()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for n, item in enumerate(items, start=1):
            key = (model_key, condition, prompt_version, item["id"])
            if key in done and done[key]["error"] is None:
                summary.skipped += 1
                continue
            chunks = contexts[item["id"]]
            messages = build_messages(item["question"], chunks)
            record: dict[str, Any] = {
                "model": model_key,
                "condition": condition,
                "prompt_version": prompt_version,
                "item_id": item["id"],
                "context_ids": None if chunks is None else [c.id for c in chunks],
                "prompt_sha256": _sha256(messages),
            }
            start = time.perf_counter()
            try:
                result = generate(messages)
            except Exception as e:
                record |= {
                    "answer_raw": None,
                    "answer": None,
                    "n_input_tokens": None,
                    "n_new_tokens": None,
                    "finish": None,
                    "error": f"{type(e).__name__}: {e}",
                    "valid": False,
                }
                summary.errors += 1
            else:
                answer = clean_output(result.text)
                record |= {
                    "answer_raw": result.text,
                    "answer": answer,
                    "n_input_tokens": result.n_input_tokens,
                    "n_new_tokens": result.n_new_tokens,
                    "finish": result.finish,
                    "error": None,
                    "valid": bool(answer) and result.finish == "stop",
                }
                summary.generated += 1
            record["latency_s"] = round(time.perf_counter() - start, 3)
            record["created"] = datetime.now(UTC).isoformat(timespec="seconds")
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            log(
                f"[{model_key} {condition}] {n}/{len(items)} {item['id']} "
                f"{record['latency_s']:.1f}s {record['n_new_tokens']} tok "
                f"{'ERROR ' + record['error'] if record['error'] else record['finish']}"
            )
    return summary
