from ptlegal_eval.bm25 import BM25, tokenize

DOCS = {
    "a1": "O período normal de trabalho não pode exceder oito horas por dia.",
    "a2": "As férias têm a duração mínima de 22 dias úteis.",
    "a3": "O trabalhador tem direito a descanso semanal obrigatório.",
}


def test_tokenize_folds_case_and_accents() -> None:
    assert tokenize("Férias, DESCANSO n.º 22") == ["ferias", "descanso", "n", "22"]


def test_tokenize_keeps_numbers() -> None:
    assert "30" in tokenize("prazo de 30 dias")


def test_rank_puts_matching_document_first() -> None:
    bm25 = BM25(DOCS)
    assert bm25.rank("duração das férias", k=3)[0] == "a2"


def test_rank_ignores_accents_in_query() -> None:
    bm25 = BM25(DOCS)
    assert bm25.rank("ferias", k=3) == ["a2"]


def test_rank_omits_documents_without_any_query_term() -> None:
    bm25 = BM25(DOCS)
    assert bm25.rank("descanso", k=3) == ["a3"]


def test_rank_is_deterministic_and_breaks_ties_by_id() -> None:
    bm25 = BM25({"b": "alfa beta", "a": "alfa beta"})
    assert bm25.rank("alfa", k=2) == ["a", "b"]


def test_rank_respects_k() -> None:
    bm25 = BM25({"a": "x", "b": "x", "c": "x"})
    assert len(bm25.rank("x", k=2)) == 2


def test_rare_term_outweighs_common_term() -> None:
    docs = {"d1": "trabalho trabalho trabalho", "d2": "trabalho despedimento", "d3": "trabalho"}
    assert BM25(docs).rank("trabalho despedimento", k=3)[0] == "d2"


def test_empty_query_returns_nothing() -> None:
    assert BM25(DOCS).rank("", k=3) == []
