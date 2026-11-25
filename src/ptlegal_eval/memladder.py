"""Which context budget can a model actually run? Rising sizes, stop at the first failure.

Motivated by the BF16 smoke test, where Gemma ran out of memory at the 12,000-character budget. Each
step fills the budget with the longest articles that fit (the largest prompt the study could send
at that size), forces the full 400 new tokens so the KV cache is as large as it can get, and records
peak GPU memory. The study lowers the budget for every model to one that works (pre-registration,
section 6).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ptlegal_eval.chunks import Chunk, read_chunks
from ptlegal_eval.contexts import verify_chunks
from ptlegal_eval.models import MODELS, ModelSpec
from ptlegal_eval.prompts import build_messages
from ptlegal_eval.smoke import stress_context

DEFAULT_BUDGETS = (7500, 8000, 9000, 10000, 11000, 12000)
Step = Callable[[list[dict[str, str]]], dict[str, Any]]


def run_ladder(
    budgets: Sequence[int],
    chunks: Sequence[Chunk],
    question: str,
    step_fn: Step,
    out_dir: Path,
    info: dict[str, Any],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"info": info, "steps": [], "largest_passing_budget": None}
    for budget in sorted(set(budgets)):
        context = stress_context(chunks, budget)
        entry: dict[str, Any] = {
            "budget": budget,
            "context_chars": sum(len(c.text) for c in context),
        }
        log(f"=== budget {budget} ({entry['context_chars']} characters of articles)")
        try:
            res = step_fn(build_messages(question, context))
        except Exception as e:
            entry |= {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        else:
            total = info.get("total_gib", {})
            spare = [total[g] - peak for g, peak in res["peak_gib"].items() if g in total]
            entry |= {"ok": True, **res, "headroom_gib": round(min(spare), 2) if spare else None}
            result["largest_passing_budget"] = budget
        result["steps"].append(entry)
        (out_dir / "ladder.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not entry["ok"]:
            break
    return result


def format_ladder(result: dict[str, Any]) -> str:
    info = result["info"]
    lines = [
        f"Memory ladder: loader {info.get('loader_used')}, GPU totals {info.get('total_gib')}, "
        f"allocated after load {info.get('after_load_gib')}"
    ]
    for s in result["steps"]:
        head = f"  budget {s['budget']:>6} (context {s['context_chars']} chars)"
        if s["ok"]:
            lines.append(
                f"{head}: ok, {s['input_tokens']} input + {s['new_tokens']} new tokens, "
                f"{s['seconds']:.0f} s, peak GiB {s['peak_gib']}, headroom {s['headroom_gib']} GiB"
            )
        else:
            lines.append(f"{head}: FAILED: {s['error']}")
    best = result["largest_passing_budget"]
    if best is None:
        lines.append("Result: no budget tried works.")
    else:
        spare = next(s["headroom_gib"] for s in result["steps"] if s["budget"] == best)
        lines.append(f"Result: largest budget that works: {best} (headroom {spare} GiB)")
    return "\n".join(lines)


def model_session(spec: ModelSpec) -> tuple[Step, dict[str, Any], Callable[[], Any]]:
    """Load once, exactly as the study does, and return (one ladder step, info, close)."""
    from ptlegal_eval import models as m

    loaded = m.load_model(spec)
    cfg = m.GenerationConfig(max_new_tokens=400, min_new_tokens=400)
    generate = m.make_generate(loaded, cfg)

    def step(messages: list[dict[str, str]]) -> dict[str, Any]:
        m.free_gpu_cache()
        m.reset_gpu_peaks()
        start = time.perf_counter()
        r = generate(messages)
        return {
            "input_tokens": r.n_input_tokens,
            "new_tokens": r.n_new_tokens,
            "seconds": round(time.perf_counter() - start, 1),
            "peak_gib": m.gpu_peaks_gib(),
        }

    info = {
        "loader_used": loaded.loader_used,
        "load_seconds": round(loaded.load_seconds, 1),
        "dtype": m.DTYPE,
        "total_gib": m.gpu_totals_gib(),
        "after_load_gib": m.gpu_allocated_gib(),
    }
    return step, info, lambda: m.unload(loaded)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gemma", choices=sorted(MODELS))
    p.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    p.add_argument("--chunks", type=Path, required=True, help="data/chunks.jsonl (the law text)")
    p.add_argument("--dev", type=Path, default=Path("data/golden/dev.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.json"))
    p.add_argument("--out", type=Path, default=Path("runs/memladder"))
    args = p.parse_args(argv)

    chunks = read_chunks(args.chunks)
    verify_chunks(chunks, json.loads(args.manifest.read_text(encoding="utf-8")))
    print(f"chunks verified against {args.manifest} ({len(chunks)} chunks)")
    dev = [json.loads(x) for x in args.dev.read_text(encoding="utf-8").splitlines() if x.strip()]
    question = min(dev, key=lambda i: i["id"])["question"]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    step, info, close = model_session(MODELS[args.model])
    try:
        result = run_ladder(budgets, chunks, question, step, args.out, info)
    finally:
        close()
    print(format_ladder(result))


if __name__ == "__main__":
    main()
