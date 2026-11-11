from ptlegal_eval.chunks import Chunk
from ptlegal_eval.golden_check import check_references, number_values

CHUNKS = [
    Chunk(
        "CT-229",
        "229.º",
        "Descanso",
        "S",
        "1 - (Revogado).\n"
        "4 - O trabalhador tem direito a um dia de descanso, num dos três dias úteis seguintes.\n"
        "5 - O descanso é marcado por acordo ou pelo empregador.",
    ),
    Chunk(
        "CT-228",
        "228.º",
        "Limites",
        "S",
        "1 - Limite de cento e setenta e cinco horas por ano, ver o n.º 2 do artigo 207.º\n"
        "2 - Pode ir até duzentas horas.",
    ),
    Chunk("CT-273", "273.º", "RMMG", "S", "1 - O valor é fixado por legislação específica."),
    Chunk("CT-207", "207.º", "Período de referência", "S", "1 - Texto."),
]


def item(ref, gold=("CT-229",), question="Pergunta?", behaviour="answer"):
    return {
        "id": "g1",
        "question": question,
        "reference_answer": ref,
        "gold_passage_ids": list(gold),
        "metadata": {"category": "single_article", "behaviour": behaviour},
    }


def problems(it):
    return check_references([it], CHUNKS)


def test_number_values_reads_digits_and_compound_words() -> None:
    assert number_values("cento e setenta e cinco horas") >= {175}
    assert number_values("quarenta e oito horas e 22 dias") >= {48, 22}
    assert 1 not in number_values("um dia")  # "um" is mostly an article


def test_number_values_ignores_ordinals_and_paragraph_references() -> None:
    assert number_values("art. 229.º, n.º 5, alínea a) e n.os 2 e 3 do artigo 207.º") == set()


def test_grounded_reference_has_no_problems() -> None:
    ref = "Um dia de descanso, num dos três dias úteis seguintes (art. 229.º, n.º 4)."
    assert problems(item(ref)) == []


def test_number_missing_from_gold_text_is_reported() -> None:
    assert any("7" in p for p in problems(item("Tem sete dias (art. 229.º, n.º 4).")))


def test_number_from_the_question_is_allowed() -> None:
    ref = "A premissa está errada: não são 35 horas (art. 229.º, n.º 4)."
    assert problems(item(ref, question="Porque são 35 horas?")) == []


def test_partial_number_word_is_grounded_by_its_compound() -> None:
    assert problems(item("São setenta horas (art. 228.º, n.º 1).", gold=("CT-228",))) == []


def test_digits_match_number_words_in_the_law() -> None:
    assert problems(item("175 horas por ano (art. 228.º, n.º 1).", gold=("CT-228",))) == []


def test_cited_article_must_be_gold_or_mentioned_in_the_gold_text() -> None:
    assert any("229" in p for p in problems(item("Ver art. 229.º, n.º 4.", gold=("CT-228",))))
    assert problems(item("Ver art. 207.º.", gold=("CT-228",))) == []  # named in 228.º n.º 1


def test_cited_paragraph_must_exist() -> None:
    assert any("n.º 9" in p for p in problems(item("Ver art. 229.º, n.º 9.")))
    assert problems(item("Ver art. 229.º, n.º 5.")) == []


def test_unanswerable_items_may_cite_any_corpus_article_but_not_invent_one() -> None:
    ok = item("O art. 273.º, n.º 1, remete para legislação.", gold=(), behaviour="refuse")
    assert problems(ok) == []
    bad = item("Ver o art. 999.º.", gold=(), behaviour="refuse")
    assert any("999" in p for p in problems(bad))
