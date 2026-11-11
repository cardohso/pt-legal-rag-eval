"""Step 6 smoke test: 5 dev questions per model, to confirm loading, templates and memory.

Not a result. Dev items only, condition A, prompt v0. The one place it departs from the
pre-registered design: a question with no gold article gets the BM25 top-3 as context (the real
runs use the hybrid retriever, which needs bge-m3). Everything is written to an output folder as it
happens, so a crash loses at most one item.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ptlegal_eval.bm25 import BM25
from ptlegal_eval.chunks import Chunk, read_chunks
from ptlegal_eval.contexts import distractor_context, gold_context, verify_chunks
from ptlegal_eval.golden_check import CATEGORIES
from ptlegal_eval.models import MODELS, GenerationResult, ModelSpec
from ptlegal_eval.prompts import PROMPT_VERSION, build_messages
from ptlegal_eval.retrieve import indexed_text
from ptlegal_eval.runner import read_predictions, run_items


@dataclass
class Backend:
    generate: Callable[[list[dict[str, str]]], GenerationResult]
    render: Callable[[list[dict[str, str]]], str]
    info: dict[str, Any]
    peaks: Callable[[], dict[str, float]]
    close: Callable[[], None]


def real_backend(spec: ModelSpec) -> Backend:
    import time

    from ptlegal_eval import models as m

    m.reset_gpu_peaks()
    start = time.perf_counter()
    loaded = m.load_model(spec)
    return Backend(
        generate=m.make_generate(loaded),
        render=lambda messages: m.render_prompt(loaded, messages),
        info={
            "loader_used": loaded.loader_used,
            "load_seconds": round(time.perf_counter() - start, 1),
            "load_errors": loaded.load_errors,
        },
        peaks=m.gpu_peaks_gib,
        close=lambda: m.unload(loaded),
    )


def select_smoke_items(dev_items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The first dev item (by id) of each category: 5 items."""
    chosen = []
    for cat in CATEGORIES:
        pool = sorted(
            (i for i in dev_items if i["metadata"]["category"] == cat), key=lambda i: i["id"]
        )
        chosen.append(pool[0])
    return chosen


def _env() -> dict[str, Any]:
    from importlib import metadata

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    for pkg in ("torch", "transformers", "accelerate"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg] = None
    try:
        env["git_sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        env["git_sha"] = None
    try:
        import torch

        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        env["gpus"] = []
    return env


def _stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in recs if r["error"] is None]
    time_s = sum(r["latency_s"] for r in ok) or 1e-9
    return {
        "valid": sum(r["valid"] for r in recs),
        "errors": sum(r["error"] is not None for r in recs),
        "truncated": sum(r["finish"] == "length" for r in ok),
        "tokens_per_s": round(sum(r["n_new_tokens"] for r in ok) / time_s, 2),
        "max_input_tokens": max((r["n_input_tokens"] for r in ok), default=None),
        "total_seconds": round(time_s, 1),
    }


def run_smoke(
    model_keys: Sequence[str],
    chunks: Sequence[Chunk],
    dev_items: Sequence[dict[str, Any]],
    out_dir: Path,
    backend_factory: Callable[[ModelSpec], Backend] = real_backend,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = select_smoke_items(dev_items)
    bm25 = BM25({c.id: indexed_text(c) for c in chunks})
    contexts: dict[str, Sequence[Chunk] | None] = {}
    kinds: dict[str, str] = {}
    for it in items:
        if it.get("gold_passage_ids"):
            contexts[it["id"]], kinds[it["id"]] = gold_context(it, chunks), "gold"
        else:
            contexts[it["id"]] = distractor_context(it, bm25, chunks, k=3)
            kinds[it["id"]] = "bm25_smoke_only"

    meta_path = out_dir / "meta.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "condition": "A",
        "max_new_tokens": 400,
        "env": _env(),
        "items": [
            {
                "id": i["id"],
                "category": i["metadata"]["category"],
                "context": kinds[i["id"]],
                "context_ids": [c.id for c in contexts[i["id"]] or []],
            }
            for i in items
        ],
        "models": dict(old.get("models", {})),
    }
    for key in model_keys:
        log(f"=== {key}")
        try:
            backend = backend_factory(MODELS[key])
        except Exception as e:
            meta["models"][key] = {"load_error": f"{type(e).__name__}: {e}"}
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"    could not load {key}: {e}")
            continue
        try:
            path = out_dir / f"{key}.jsonl"
            run_items(items, contexts, key, "A", backend.generate, path, log=log)
            rendered = {
                i["id"]: backend.render(build_messages(i["question"], contexts[i["id"]]))
                for i in items
            }
            (out_dir / f"{key}_rendered_prompts.json").write_text(
                json.dumps(rendered, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            recs = [
                r
                for k, r in read_predictions(path).items()
                if k[0] == key and k[2] == PROMPT_VERSION
            ]
            meta["models"][key] = {**backend.info, "peak_gib": backend.peaks(), **_stats(recs)}
        finally:
            backend.close()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def format_report(out_dir: Path) -> str:
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    env = meta["env"]
    lines = [
        f"Smoke test, prompt {meta['prompt_version']}, condition {meta['condition']}, "
        f"commit {env.get('git_sha')}",
        f"torch {env.get('torch')}, transformers {env.get('transformers')}, GPUs {env.get('gpus')}",
        "",
    ]
    for key, m in meta["models"].items():
        if "load_error" in m:
            lines += [f"## {key}: LOAD FAILED", f"   {m['load_error']}", ""]
            continue
        lines += [
            f"## {key}: loader {m['loader_used']}, load {m['load_seconds']} s, "
            f"peak GiB {m['peak_gib']}",
            f"   valid {m['valid']}/5, errors {m['errors']}, truncated {m['truncated']}, "
            f"{m['tokens_per_s']} tok/s, longest prompt {m['max_input_tokens']} tokens",
        ]
        recs = [
            r
            for k, r in read_predictions(out_dir / f"{key}.jsonl").items()
            if k[2] == meta["prompt_version"]
        ]
        cats = {i["id"]: i["category"] for i in meta["items"]}
        for r in sorted(recs, key=lambda r: r["item_id"]):
            answer = (r["answer"] or r["error"] or "").replace("\n", " ")[:170]
            tokens = f"{r['n_input_tokens']}->{r['n_new_tokens']}"
            category = cats.get(r["item_id"], "?")
            lines.append(f"   {r['item_id']} {category:<15} {tokens} {r['finish']} | {answer}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--chunks", type=Path, required=True, help="data/chunks.jsonl (the law text)")
    r.add_argument("--dev", type=Path, default=Path("data/golden/dev.jsonl"))
    r.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.json"))
    r.add_argument("--models", default=",".join(MODELS))
    r.add_argument("--out", type=Path, default=Path("runs/smoke"))
    rep = sub.add_parser("report")
    rep.add_argument("out", type=Path)
    args = p.parse_args(argv)

    if args.cmd == "report":
        print(format_report(args.out))
        return
    chunks = read_chunks(args.chunks)
    verify_chunks(chunks, json.loads(args.manifest.read_text(encoding="utf-8")))
    print(f"chunks verified against {args.manifest} ({len(chunks)} chunks)")
    dev = [json.loads(x) for x in args.dev.read_text(encoding="utf-8").splitlines() if x.strip()]
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        sys.exit(f"unknown models {unknown}; choose from {list(MODELS)}")
    run_smoke(keys, chunks, dev, args.out)
    print(format_report(args.out))


if __name__ == "__main__":
    main()
