"""Numerical probe: is a model sane in a given precision, and how fast is it?

Motivated by the step 6 smoke test, where AMALIA produced non-finite logits in FP16 while the other
three models did not (pre-registration, section 3 says FP16 for every model). Run on the GPU, one
model and a few precisions at a time; the result decides what precision policy the study can use.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ptlegal_eval.models import MODELS, ModelSpec, probe_dtype


def run_probe(
    model_keys: Sequence[str],
    dtypes: Sequence[str],
    out_dir: Path,
    probe_fn: Callable[[ModelSpec, str], dict[str, Any]] = probe_dtype,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for key in model_keys:
        results[key] = {}
        for dtype in dtypes:
            log(f"=== {key} {dtype}")
            try:
                results[key][dtype] = probe_fn(MODELS[key], dtype)
            except Exception as e:
                results[key][dtype] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            (out_dir / "probe.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return results


def format_probe(results: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for key, by_dtype in results.items():
        for dtype, r in by_dtype.items():
            if "error" in r:
                lines += [f"## {key} {dtype}: FAILED", f"   {r['error']}", ""]
                continue
            lines.append(f"## {key} {dtype}: loader {r['loader_used']}, load {r['load_seconds']} s")
            lines.append(f"   sanity: {r['sanity']}")
            rep = r.get("layer_report", {})
            if "error" in rep:
                lines.append(f"   layer report failed: {rep['error']}")
            else:
                lines.append(f"   first non-finite module: {rep.get('first_non_finite')}")
                top = [(s["module"], s["max_abs"]) for s in rep.get("largest_activations", [])[:3]]
                lines.append(f"   largest activations (limit in FP16 is 65504): {top}")
            if "generation_error" in r:
                lines.append(f"   generation failed: {r['generation_error']}")
            else:
                speed, ids = r.get("tokens_per_s"), r.get("first_token_ids")
                lines.append(f"   generation: {speed} tok/s, first ids {ids}")
                lines.append(f"   sample: {(r.get('sample') or '')[:160]!r}")
            lines.append(
                f"   peak GiB {r.get('peak_gib')}, still allocated after unload "
                f"{r.get('allocated_after_unload_gib')}"
            )
            lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="amalia")
    p.add_argument("--dtypes", default="float16,bfloat16")
    p.add_argument("--out", type=Path, default=Path("runs/probe"))
    args = p.parse_args(argv)
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    results = run_probe(keys, [d.strip() for d in args.dtypes.split(",") if d.strip()], args.out)
    print(format_probe(results))


if __name__ == "__main__":
    main()
