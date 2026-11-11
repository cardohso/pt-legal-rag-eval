import json

from ptlegal_eval.chunks import Chunk
from ptlegal_eval.golden_check import CATEGORIES
from ptlegal_eval.models import GenerationResult
from ptlegal_eval.smoke import Backend, format_report, run_smoke, select_smoke_items

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
        assert len(recs) == 5 and all(r["valid"] for r in recs)
        prompts = json.loads((tmp_path / f"{m}_rendered_prompts.json").read_text(encoding="utf-8"))
        assert len(prompts) == 5
    saved = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert saved["models"]["amalia"]["loader_used"] == "causal_lm"
    assert saved["models"]["amalia"]["peak_gib"] == {"cuda:0": 12.0, "cuda:1": 9.0}
    assert saved["models"]["amalia"]["valid"] == 5 and saved["models"]["amalia"]["tokens_per_s"] > 0
    assert saved["prompt_version"] == "v0" and len(saved["items"]) == 5
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
    assert meta["models"]["amalia"]["valid"] == 5


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
    assert len(calls) == 5


def test_report_summarises_every_model_in_plain_text(tmp_path) -> None:
    run_smoke(["amalia", "gemma"], CHUNKS, dev_items(), tmp_path, fake_backend, log=lambda s: None)
    text = format_report(tmp_path)
    assert "amalia" in text and "gemma" in text and "causal_lm" in text and "12.0" in text
    assert "Resposta (art. 1.º)." in text
