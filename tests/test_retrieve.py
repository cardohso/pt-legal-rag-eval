from collections.abc import Sequence

import numpy as np

from ptlegal_eval.bm25 import tokenize
from ptlegal_eval.chunks import Chunk
from ptlegal_eval.retrieve import HybridRetriever

VOCAB = ["ferias", "dias", "descanso", "salario", "despedimento", "aviso", "horas", "trabalho"]


class FakeEmbedder:
    """Bag-of-words vectors over a tiny vocabulary: deterministic, no model, no network."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(VOCAB)))
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                if tok in VOCAB:
                    out[row, VOCAB.index(tok)] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def _chunks() -> list[Chunk]:
    rows = {
        "CT-1": ("Férias", "As ferias duram 22 dias."),
        "CT-2": ("Descanso", "O descanso semanal e obrigatorio."),
        "CT-3": ("Despedimento", "O despedimento exige aviso previo."),
        "CT-4": ("Salário", "O salario e pago por horas de trabalho."),
    }
    return [Chunk(i, f"{i}.º", ep, "S", text) for i, (ep, text) in rows.items()]


def test_retrieve_returns_top_k_ids_in_rank_order() -> None:
    r = HybridRetriever(_chunks(), FakeEmbedder())
    result = r.retrieve("quantos dias de ferias", top_k=2)
    assert [h.chunk_id for h in result][0] == "CT-1"
    assert len(result) == 2


def test_query_matching_one_chunk_ranks_it_first() -> None:
    r = HybridRetriever(_chunks(), FakeEmbedder())
    assert r.retrieve("aviso", top_k=1)[0].chunk_id == "CT-3"


def test_chunk_found_only_by_dense_ranking_still_appears() -> None:
    class DenseOnly(FakeEmbedder):
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            # map the query word "remuneracao" onto the "salario" dimension
            return super().encode([t.replace("remuneracao", "salario") for t in texts])

    hits = HybridRetriever(_chunks(), DenseOnly()).retrieve("remuneracao", top_k=1)
    assert hits[0].chunk_id == "CT-4"
    assert hits[0].bm25_rank is None and hits[0].dense_rank == 1


def test_hits_expose_both_component_ranks_for_diagnosis() -> None:
    r = HybridRetriever(_chunks(), FakeEmbedder())
    hit = r.retrieve("ferias", top_k=1)[0]
    assert hit.bm25_rank == 1 and hit.dense_rank == 1


def test_retrieval_is_deterministic() -> None:
    r = HybridRetriever(_chunks(), FakeEmbedder())
    first = [h.chunk_id for h in r.retrieve("trabalho horas", top_k=4)]
    again = [h.chunk_id for h in r.retrieve("trabalho horas", top_k=4)]
    assert first == again


def test_top_k_larger_than_corpus_returns_everything_retrievable() -> None:
    r = HybridRetriever(_chunks(), FakeEmbedder())
    assert len(r.retrieve("ferias", top_k=50)) <= 4
