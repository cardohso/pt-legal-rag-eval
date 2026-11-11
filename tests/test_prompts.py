from ptlegal_eval.chunks import Chunk
from ptlegal_eval.prompts import PROMPT_VERSION, build_messages, render_context

CHUNKS = [
    Chunk("CT-229", "229.º", "Descanso compensatório", "S", "1 - Texto A.\n2 - Texto B."),
    Chunk("CT-40#2", "40.º", "Licença parental inicial", "S", "Segunda parte."),
]


def test_context_lists_each_article_with_number_epigraph_and_text() -> None:
    text = render_context(CHUNKS)
    assert "[Artigo 229.º — Descanso compensatório]\n1 - Texto A.\n2 - Texto B." in text
    assert text.index("229.º") < text.index("40.º")  # order is preserved


def test_open_book_messages_have_system_then_user_with_context_and_question() -> None:
    msgs = build_messages("Qual é a pergunta?", CHUNKS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Texto A." in msgs[1]["content"] and msgs[1]["content"].endswith(
        "Pergunta: Qual é a pergunta?"
    )
    assert "artigos fornecidos" in msgs[0]["content"]


def test_closed_book_messages_have_no_context_and_a_different_system_prompt() -> None:
    open_book = build_messages("P?", CHUNKS)
    closed = build_messages("P?", None)
    assert closed[1]["content"] == "Pergunta: P?"
    assert closed[0]["content"] != open_book[0]["content"]
    assert "artigos fornecidos" not in closed[0]["content"]


def test_an_empty_context_is_not_the_same_as_closed_book() -> None:
    assert build_messages("P?", [])[0]["content"] == build_messages("P?", CHUNKS)[0]["content"]


def test_prompts_are_in_european_portuguese_and_versioned() -> None:
    assert PROMPT_VERSION == "v0"
    system = build_messages("P?", CHUNKS)[0]["content"]
    assert "português europeu" in system and "premissa" in system
    # pt-BR forms the prompt itself must not use
    assert not any(w in system.lower() for w in ("você", "usuário", "celular"))


def test_building_messages_does_not_depend_on_the_model() -> None:
    assert build_messages("P?", CHUNKS) == build_messages("P?", list(CHUNKS))


def test_every_prompt_version_in_code_is_logged_with_its_hashes() -> None:
    import hashlib
    from pathlib import Path

    from ptlegal_eval.prompts import SYSTEM_CLOSED_BOOK, SYSTEM_OPEN_BOOK

    log = (Path(__file__).resolve().parents[1] / "docs" / "PROMPT_LOG.md").read_text(
        encoding="utf-8"
    )
    assert f"## {PROMPT_VERSION} " in log
    for text in (SYSTEM_OPEN_BOOK, SYSTEM_CLOSED_BOOK):
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() in log
