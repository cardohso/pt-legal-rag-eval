from pathlib import Path

from ptlegal_eval.models import MODELS, clean_output

PREREG = (Path(__file__).resolve().parents[1] / "docs" / "PREREGISTRATION.md").read_text(
    encoding="utf-8"
)


def test_the_four_systems_match_the_preregistration() -> None:
    assert set(MODELS) == {"amalia", "eurollm", "gemma", "qwen"}
    for spec in MODELS.values():
        assert spec.hf_id in PREREG and spec.revision in PREREG, spec.key


def test_thinking_is_off_for_every_model() -> None:
    assert all(spec.template_kwargs == {"enable_thinking": False} for spec in MODELS.values())


def test_loader_order_tries_the_right_class_first() -> None:
    assert MODELS["amalia"].loaders[0] == "causal_lm"
    assert MODELS["eurollm"].loaders[0] == "causal_lm"
    assert MODELS["gemma"].loaders[0] == "multimodal_lm"
    assert MODELS["qwen"].loaders[0] in {"multimodal_lm", "image_text_to_text"}


def test_clean_output_strips_whitespace_and_think_blocks() -> None:
    assert clean_output("  Olá.\n") == "Olá."
    assert clean_output("<think>\nraciocínio\n</think>\n\nResposta.") == "Resposta."
    assert clean_output("<think>\n\n</think>\n\nResposta.") == "Resposta."


def test_clean_output_handles_an_unclosed_think_block_by_dropping_it() -> None:
    assert clean_output("<think>\nnunca fechou") == ""


def test_clean_output_leaves_normal_text_alone() -> None:
    text = "O prazo é de 30 dias (art. 396.º, n.º 1)."
    assert clean_output(text) == text
