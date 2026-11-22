"""Fail-fast hardware check and a ten-second precision benchmark, before any model is downloaded.

The study's compute plan is Kaggle's free "GPU T4 x2". A notebook that lost that setting would
waste a long model download (or fail with an obscure CUDA error), so this runs first and stops with
a plain explanation. The benchmark multiplies two large matrices in each precision, which shows
whether BF16 works on a T4 at all and how much slower it is than FP16, before committing to a
3-minute model load.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

EXPECTED_COUNT = 2
EXPECTED_NAME = "T4"


def evaluate_hardware(
    names: Sequence[str], expected_count: int = EXPECTED_COUNT, name_contains: str = EXPECTED_NAME
) -> tuple[bool, str]:
    """(ok, message) for the list of GPU names this session has."""
    fix = (
        "Fix: in the right panel under Session options set Accelerator = GPU T4 x2, "
        "then Run All again."
    )
    if not names:
        return False, f"This session has no GPU. {fix}"
    found = ", ".join(names)
    if len(names) != expected_count or not all(name_contains in n for n in names):
        return False, (
            f"Found {len(names)} GPU(s): {found}. The study plan needs {expected_count} x "
            f"{name_contains} (T4 x2). {fix}"
        )
    return True, f"Hardware ok: {len(names)} GPUs ({found})"


def gpu_names() -> list[str]:
    import torch

    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def benchmark(n: int = 4096, reps: int = 20) -> dict[str, dict[str, Any]]:
    """Matrix-multiply speed and finiteness on GPU 0 in each precision."""
    import torch

    results: dict[str, dict[str, Any]] = {}
    for name in ("float16", "bfloat16", "float32"):
        dtype = getattr(torch, name)
        try:
            a = torch.randn(n, n, device="cuda:0", dtype=dtype)
            b = torch.randn(n, n, device="cuda:0", dtype=dtype)
            c = a @ b  # warm-up
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(reps):
                c = a @ b
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            results[name] = {
                "tflops": round(2 * n**3 * reps / elapsed / 1e12, 1),
                "finite": bool(torch.isfinite(c).all()),
            }
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return results


def format_benchmark(results: dict[str, dict[str, Any]]) -> str:
    lines = ["Matrix-multiply benchmark on GPU 0 (higher is faster):"]
    for name, r in results.items():
        if "error" in r:
            lines.append(f"  {name:<9} unsupported or failed: {r['error']}")
        else:
            flag = "" if r["finite"] else "  (result NOT finite)"
            lines.append(f"  {name:<9} {r['tflops']} TFLOPS{flag}")
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    names_fn: Callable[[], list[str]] = gpu_names,
    bench_fn: Callable[[], dict[str, dict[str, Any]]] = benchmark,
    log: Callable[[str], None] = print,
) -> int:
    try:
        names = names_fn()
    except ImportError as e:
        log(f"torch is not installed in this environment ({e}); run the setup cell first.")
        return 1
    ok, message = evaluate_hardware(names)
    log(message)
    if not ok:
        return 1
    log(format_benchmark(bench_fn()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
