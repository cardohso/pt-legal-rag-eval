import json

from ptlegal_eval.chunks import Chunk
from ptlegal_eval.golden_check import CATEGORIES, CONTEXT_BUDGET
from ptlegal_eval.models import DTYPE, GenerationResult
from ptlegal_eval.smoke import (
    STRESS_ID,
    Backend,
    format_report,
    run_smoke,
    select_smoke_items,
    stress_context,
)

CHUNKS = [
    Chunk("CT-1", "1.º", "Um", "S", "texto sobre ferias e descanso"),
    Chunk("CT-2", "2.º", "Dois", "S", "texto sobre salario"),
    Chunk("CT-3", "3.º", "Três", "S", "texto sobre horario de trabalho"),
]


def dev_items():
    items = []
    for n, cat in enumerate(CATEGORIES * 2, start=1):
        refuse = cat == "unanswerable"
        items.append(
            {
                "id": f"d{n:02d}",
                "question": f"Pergunta {n} sobre ferias?",
                "gold_passage_ids": [] if refuse else ["CT-1"],
                "metadata": {"category": cat, "behaviour": "refuse" if refuse else "answer"},
            }
        )
    return items


def fake_backend(spec):
    def generate(messages):
        return GenerationResult("Resposta (art. 1.º).", 500, 20, "stop")

    return Backend(
        generate=generate,
        render=lambda messages: "RENDERED " + messages[-1]["content"][-20:],
        info={"loader_used": "causal_lm", "load_seconds": 1.5, "load_errors": []},
        peaks=lambda: {"cuda:0": 12.0, "cuda:1": 9.0},
        close=lambda: None,
    )


def test_one_item_per_category_the_first_by_id() -> None:
    chosen = select_smoke_items(dev_items())
    assert [i["metadata"]["category"] for i in chosen] == list(CATEGORIES)
    assert [i["id"] for i in chosen] == ["d01", "d02", "d03", "d04", "d05"]


def test_run_smoke_writes_predictions_prompts_and_meta(tmp_path) -> None:
    meta = run_smoke(
        ["amalia", "qwen"], CHUNKS, dev_items(), tmp_path, fake_backend, log=lambda s: None
    )
    for m in ("amalia", "qwen"):
        recs = [
            json.loads(x)
            for x in (tmp_path / f"{m}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(recs) == 6 and all(r["valid"] for r in recs)
        prompts = json.loads((tmp_path / f"{m}_rendered_prompts.json").read_text(encoding="utf-8"))
        assert len(prompts) == 6
    saved = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert saved["models"]["amalia"]["loader_used"] == "causal_lm"
    assert saved["models"]["amalia"]["peak_gib"] == {"cuda:0": 12.0, "cuda:1": 9.0}
    assert saved["models"]["amalia"]["valid"] == 6 and saved["models"]["amalia"]["tokens_per_s"] > 0
    assert saved["prompt_version"] == "v0" and len(saved["items"]) == 6
    assert meta["models"].keys() == saved["models"].keys()


def test_unanswerable_smoke_item_gets_a_search_context_not_an_empty_one(tmp_path) -> None:
    run_smoke(["amalia"], CHUNKS, dev_items(), tmp_path, fake_backend, log=lambda s: None)
    recs = [
        json.loads(x) for x in (tmp_path / "amalia.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    unanswerable = next(r for r in recs if r["item_id"] == "d04")
    assert unanswerable["context_ids"]  # smoke-only BM25 stand-in for the retriever
    assert recs[0]["context_ids"] == ["CT-1"]


def test_a_model_that_fails_to_load_is_recorded_and_the_others_still_run(tmp_path) -> None:
    def factory(spec):
        if spec.key == "gemma":
            raise RuntimeError("CUDA out of memory while loading")
        return fake_backend(spec)

    meta = run_smoke(
        ["gemma", "amalia"], CHUNKS, dev_items(), tmp_path, factory, log=lambda s: None
    )
    assert "out of memory" in meta["models"]["gemma"]["load_error"]
    assert meta["models"]["amalia"]["valid"] == 6


def test_rerunning_does_not_repeat_finished_work(tmp_path) -> None:
    calls = []

    def counting(spec):
        b = fake_backend(spec)

        def generate(messages):
            calls.append(1)
            return b.generate(messages)

        return Backend(generate, b.render, b.info, b.peaks, b.close)

    run_smoke(["amalia"], CHUNKS, dev_items(), tmp_path, counting, log=lambda s: None)
    run_smoke(["amalia"], CHUNKS, dev_items(), tmp_path, counting, log=lambda s: None)
    assert len(calls) == 6


def test_report_summarises_every_model_in_plain_text(tmp_path) -> None:
    run_smoke(["amalia", "gemma"], CHUNKS, dev_items(), tmp_path, fake_backend, log=lambda s: None)
    text = format_report(tmp_path)
    assert "amalia" in text and "gemma" in text and "causal_lm" in text and "12.0" in text
    assert "Resposta (art. 1.º)." in text


def test_memory_left_after_unloading_is_recorded_so_a_leak_is_visible(tmp_path) -> None:
    def factory(spec):
        b = fake_backend(spec)
        return Backend(
            b.generate, b.render, b.info, b.peaks, lambda: {"cuda:0": 8.5, "cuda:1": 8.5}
        )

    meta = run_smoke(["amalia"], CHUNKS, dev_items(), tmp_path, factory, log=lambda s: None)
    assert meta["models"]["amalia"]["allocated_after_unload_gib"] == {"cuda:0": 8.5, "cuda:1": 8.5}
    assert "LEAK" in format_report(tmp_path)


def test_report_shows_why_an_answer_is_empty_and_any_sanity_check(tmp_path) -> None:
    def factory(spec):
        b = fake_backend(spec)
        info = {**b.info, "sanity": {"finite_logits": False, "top5": ["<pad>"]}}

        def generate(messages):
            return GenerationResult("", 50, 400, "length", first_token_ids=(0, 0, 0))

        return Backend(generate, b.render, info, b.peaks, b.close)

    run_smoke(["amalia"], CHUNKS, dev_items(), tmp_path, factory, log=lambda s: None)
    text = format_report(tmp_path)
    assert "finite_logits" in text and "empty" in text and "[0, 0, 0]" in text


def test_the_study_precision_is_bf16_and_is_recorded(tmp_path) -> None:
    meta = run_smoke(["amalia"], CHUNKS, dev_items(), tmp_path, fake_backend, log=lambda s: None)
    assert DTYPE == "bfloat16" and meta["dtype"] == "bfloat16"
    recs = (tmp_path / "amalia.jsonl").read_text(encoding="utf-8").splitlines()
    assert {json.loads(x)["dtype"] for x in recs} == {"bfloat16"}
    assert "bfloat16" in format_report(tmp_path)


def test_the_stress_item_fills_the_context_budget_without_exceeding_it() -> None:
    big = [
        Chunk(f"CT-{n}", f"{n}.º", "E", "S", "palavra " * size)
        for n, size in ((1, 480), (2, 470), (3, 460), (4, 450), (5, 3))
    ]
    context = stress_context(big)
    total = sum(len(c.text) for c in context)
    assert total <= CONTEXT_BUDGET and total > 0.9 * CONTEXT_BUDGET
    assert [c.id for c in context] == sorted(
        c.id for c in context
    )  # article order, as in condition A


def test_smoke_runs_the_five_dev_items_plus_the_stress_prompt(tmp_path) -> None:
    meta = run_smoke(["amalia"], CHUNKS, dev_items(), tmp_path, fake_backend, log=lambda s: None)
    kinds = {i["id"]: i["context"] for i in meta["items"]}
    assert len(kinds) == 6 and kinds[STRESS_ID] == "stress_budget"
    assert meta["models"]["amalia"]["valid"] == 6
    assert "valid 6/6" in format_report(tmp_path)
