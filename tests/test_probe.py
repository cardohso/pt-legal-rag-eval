import json

from ptlegal_eval.probe import format_probe, run_probe


def fake_probe(spec, dtype):
    if dtype == "bfloat16" and spec.key == "gemma":
        raise RuntimeError("bf16 not supported here")
    finite = dtype == "bfloat16"
    return {
        "dtype": dtype,
        "loader_used": "causal_lm",
        "load_seconds": 100.0,
        "sanity": {"finite_logits": finite, "top5": ["O"] if finite else []},
        "layer_report": {
            "first_non_finite": None if finite else "model.layers.7.mlp",
            "largest_activations": [
                {"module": "model.layers.3", "max_abs": 1234.5, "finite": True}
            ],
        },
        "tokens_per_s": 4.5 if finite else 0.0,
        "peak_gib": {"cuda:0": 9.7},
        "allocated_after_unload_gib": {"cuda:0": 0.01},
    }


def test_probe_runs_every_model_and_dtype_and_writes_json(tmp_path) -> None:
    out = run_probe(["amalia"], ["float16", "bfloat16"], tmp_path, fake_probe, log=lambda s: None)
    assert set(out["amalia"]) == {"float16", "bfloat16"}
    saved = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
    assert saved["amalia"]["float16"]["sanity"]["finite_logits"] is False


def test_a_failing_combination_is_recorded_and_the_rest_still_run(tmp_path) -> None:
    out = run_probe(["gemma", "amalia"], ["bfloat16"], tmp_path, fake_probe, log=lambda s: None)
    assert "bf16 not supported" in out["gemma"]["bfloat16"]["error"]
    assert out["amalia"]["bfloat16"]["tokens_per_s"] == 4.5


def test_format_names_the_first_bad_module_and_compares_dtypes(tmp_path) -> None:
    out = run_probe(["amalia"], ["float16", "bfloat16"], tmp_path, fake_probe, log=lambda s: None)
    text = format_probe(out)
    assert "model.layers.7.mlp" in text and "finite_logits" in text
    assert "float16" in text and "bfloat16" in text and "4.5 tok/s" in text


def test_format_shows_errors_plainly() -> None:
    text = format_probe({"gemma": {"bfloat16": {"error": "RuntimeError: boom"}}})
    assert "boom" in text
