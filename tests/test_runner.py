import json

import pytest

from ptlegal_eval.chunks import Chunk
from ptlegal_eval.models import GenerationResult
from ptlegal_eval.runner import read_predictions, run_items

CH = Chunk("CT-1", "1.º", "Epígrafe", "S", "1 - Texto.")


def item(i: int):
    return {"id": f"g{i:03d}", "question": f"Pergunta {i}?"}


ITEMS = [item(1), item(2), item(3)]
CONTEXTS = {"g001": [CH], "g002": [CH], "g003": None}


def echo(messages):
    return GenerationResult(
        text=f"  Resposta a: {messages[-1]['content'][-12:]}",
        n_input_tokens=100,
        n_new_tokens=7,
        finish="stop",
    )


def run(path, generate=echo, **kw):
    return run_items(ITEMS, CONTEXTS, "amalia", "A", generate, path, **kw)


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_writes_one_record_per_item_with_the_fields_analysis_needs(tmp_path) -> None:
    out = tmp_path / "p.jsonl"
    run(out)
    recs = lines(out)
    assert [r["item_id"] for r in recs] == ["g001", "g002", "g003"]
    r = recs[0]
    assert r["model"] == "amalia" and r["condition"] == "A" and r["prompt_version"] == "v0"
    assert r["answer"] == r["answer_raw"].strip() and r["valid"] is True
    assert r["context_ids"] == ["CT-1"] and recs[2]["context_ids"] is None
    assert r["n_input_tokens"] == 100 and r["n_new_tokens"] == 7 and r["finish"] == "stop"
    assert r["latency_s"] >= 0 and len(r["prompt_sha256"]) == 64 and r["error"] is None


def test_a_rerun_skips_finished_items(tmp_path) -> None:
    out, calls = tmp_path / "p.jsonl", []

    def counting(messages):
        calls.append(1)
        return echo(messages)

    run(out, counting)
    run(out, counting)
    assert len(calls) == 3 and len(lines(out)) == 3


def test_an_interrupted_run_keeps_what_it_finished_and_resumes(tmp_path) -> None:
    out = tmp_path / "p.jsonl"
    seen = []

    def dies_on_third(messages):
        seen.append(1)
        if len(seen) == 3:
            raise KeyboardInterrupt
        return echo(messages)

    with pytest.raises(KeyboardInterrupt):
        run(out, dies_on_third)
    assert [r["item_id"] for r in lines(out)] == ["g001", "g002"]
    summary = run(out)
    assert summary.generated == 1 and summary.skipped == 2


def test_a_failing_item_is_recorded_and_does_not_stop_the_run(tmp_path) -> None:
    out = tmp_path / "p.jsonl"

    def flaky(messages):
        if "Pergunta 2?" in messages[-1]["content"]:
            raise RuntimeError("CUDA out of memory")
        return echo(messages)

    summary = run(out, flaky)
    recs = read_predictions(out)
    assert summary.errors == 1 and recs[("amalia", "A", "v0", "g002")]["valid"] is False
    assert "out of memory" in recs[("amalia", "A", "v0", "g002")]["error"]
    assert recs[("amalia", "A", "v0", "g003")]["valid"] is True


def test_errored_items_are_retried_on_rerun_and_the_latest_record_wins(tmp_path) -> None:
    out = tmp_path / "p.jsonl"

    def fail_two(messages):
        if "Pergunta 2?" in messages[-1]["content"]:
            raise RuntimeError("boom")
        return echo(messages)

    run(out, fail_two)
    summary = run(out)
    assert summary.generated == 1 and summary.errors == 0
    assert read_predictions(out)[("amalia", "A", "v0", "g002")]["valid"] is True


def test_truncated_or_empty_answers_are_not_valid(tmp_path) -> None:
    out = tmp_path / "p.jsonl"
    results = iter(
        [
            GenerationResult("texto cortado", 10, 400, "length"),
            GenerationResult("   ", 10, 3, "stop"),
            GenerationResult("<think>\n\n</think>\n\nOk.", 10, 9, "stop"),
        ]
    )
    run(out, lambda m: next(results))
    recs = lines(out)
    assert [r["valid"] for r in recs] == [False, False, True]
    assert recs[2]["answer"] == "Ok." and recs[2]["answer_raw"].startswith("<think>")


def test_different_models_and_conditions_do_not_collide(tmp_path) -> None:
    out = tmp_path / "p.jsonl"
    run_items(ITEMS, CONTEXTS, "amalia", "A", echo, out)
    run_items(ITEMS, CONTEXTS, "qwen", "A", echo, out)
    run_items(ITEMS, {k: None for k in CONTEXTS}, "amalia", "C", echo, out)
    assert len(lines(out)) == 9


def test_the_prompt_hash_changes_with_the_context(tmp_path) -> None:
    out = tmp_path / "p.jsonl"
    run(out)
    recs = lines(out)
    assert recs[0]["prompt_sha256"] != recs[2]["prompt_sha256"]


def test_progress_is_logged(tmp_path) -> None:
    logged = []
    run(tmp_path / "p.jsonl", log=logged.append)
    assert len(logged) >= 3
