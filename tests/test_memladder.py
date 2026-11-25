import json

from ptlegal_eval.chunks import Chunk
from ptlegal_eval.memladder import format_ladder, run_ladder

# chunk texts of many sizes so any budget can be filled closely
CHUNKS = [
    Chunk(f"CT-{n}", f"{n}.º", "E", "S", "x" * size)
    for n, size in enumerate([3900, 3800, 3000, 900, 700, 500, 300, 200, 100, 50], start=1)
]
INFO = {"loader_used": "multimodal_lm", "total_gib": {"cuda:0": 14.56, "cuda:1": 14.56}}


def fits_up_to(limit: int, seen: list):
    def step(messages):
        chars = len(messages[-1]["content"])
        seen.append(chars)
        if chars > limit:
            raise RuntimeError("OutOfMemoryError: CUDA out of memory")
        return {
            "input_tokens": chars // 4,
            "new_tokens": 400,
            "seconds": 90.0,
            "peak_gib": {"cuda:0": 13.9, "cuda:1": 11.0},
        }

    return step


def run(budgets, limit, tmp_path, seen=None):
    return run_ladder(
        budgets,
        CHUNKS,
        "Pergunta?",
        fits_up_to(limit, [] if seen is None else seen),
        tmp_path,
        INFO,
        log=lambda s: None,
    )


def test_budgets_run_in_ascending_order_whatever_the_input_order(tmp_path) -> None:
    result = run([10000, 7500, 9000], 10**9, tmp_path)
    assert [s["budget"] for s in result["steps"]] == [7500, 9000, 10000]


def test_each_context_fills_its_budget_closely_and_never_exceeds_it(tmp_path) -> None:
    result = run([7500, 9000, 12000], 10**9, tmp_path)
    for s in result["steps"]:
        assert 0.95 * s["budget"] <= s["context_chars"] <= s["budget"]


def test_stops_at_the_first_failure_and_reports_the_largest_passing_budget(tmp_path) -> None:
    seen: list = []
    result = run([7500, 8000, 9000, 10000, 12000], 8600, tmp_path, seen)
    assert [(s["budget"], s["ok"]) for s in result["steps"]] == [
        (7500, True),
        (8000, True),
        (9000, False),
    ]
    assert (
        result["largest_passing_budget"] == 8000 and "out of memory" in result["steps"][-1]["error"]
    )
    assert len(seen) == 3  # nothing after the failure


def test_when_even_the_smallest_fails_there_is_no_passing_budget(tmp_path) -> None:
    result = run([7500, 8000], 0, tmp_path)
    assert result["largest_passing_budget"] is None and len(result["steps"]) == 1


def test_headroom_is_the_tightest_gpu_and_the_result_is_saved(tmp_path) -> None:
    result = run([7500], 10**9, tmp_path)
    assert result["steps"][0]["headroom_gib"] == 0.66
    saved = json.loads((tmp_path / "ladder.json").read_text(encoding="utf-8"))
    assert (
        saved["largest_passing_budget"] == 7500 and saved["info"]["loader_used"] == "multimodal_lm"
    )


def test_the_report_states_each_step_and_the_answer(tmp_path) -> None:
    text = format_ladder(run([7500, 9000], 8600, tmp_path))
    assert "7500" in text and "headroom 0.66" in text and "FAILED" in text
    assert "largest budget that works: 7500" in text
    assert "no budget" in format_ladder(run([7500], 0, tmp_path))


def test_the_longer_forced_output_is_configurable() -> None:
    from ptlegal_eval.models import GenerationConfig

    assert GenerationConfig().min_new_tokens is None
    assert GenerationConfig(min_new_tokens=400).min_new_tokens == 400


def test_the_real_session_wrapper_works_against_a_strict_fake_gpu(monkeypatch) -> None:
    """Exercises the glue between the ladder and the GPU helpers, where both earlier Kaggle failures
    were. Like real torch, the fake rejects memory calls before CUDA is initialised."""
    import sys
    import types

    from ptlegal_eval import memladder, models
    from tests.test_cuda_init import FakeCuda

    class Cuda(FakeCuda):
        def get_device_properties(self, i):
            self._need_init()
            return types.SimpleNamespace(total_memory=int(14.56 * 2**30))

    cuda = Cuda()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    seen = {}
    monkeypatch.setattr(
        models,
        "load_model",
        lambda spec: types.SimpleNamespace(loader_used="multimodal_lm", load_seconds=200.0),
    )

    def fake_make_generate(loaded, cfg=None):
        seen["cfg"] = cfg
        return lambda messages: models.GenerationResult("x", 3000, 400, "length")

    monkeypatch.setattr(models, "make_generate", fake_make_generate)
    monkeypatch.setattr(models, "unload", lambda loaded: seen.setdefault("closed", True))

    step, info, close = memladder.model_session(models.MODELS["gemma"])
    assert seen["cfg"].min_new_tokens == 400 and seen["cfg"].max_new_tokens == 400
    assert info["total_gib"] == {"cuda:0": 14.56, "cuda:1": 14.56} and info["dtype"] == "bfloat16"
    out = step([{"role": "user", "content": "x"}])
    assert (
        out["input_tokens"] == 3000
        and out["new_tokens"] == 400
        and out["peak_gib"] == {"cuda:0": 9.0, "cuda:1": 9.0}
    )
    assert ("reset", 0) in cuda.calls
    close()
    assert seen["closed"]
