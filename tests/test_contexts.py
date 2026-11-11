import pytest

from ptlegal_eval.bm25 import BM25
from ptlegal_eval.chunks import Chunk
from ptlegal_eval.contexts import chunk_order_key, distractor_context, gold_context, verify_chunks
from ptlegal_eval.ingest import article_hashes

CHUNKS = [
    Chunk("CT-9", "9.º", "Nove", "S", "texto sobre ferias"),
    Chunk("CT-10-A", "10.º-A", "Dez A", "S", "texto sobre salario"),
    Chunk("CT-10", "10.º", "Dez", "S", "texto sobre horario"),
    Chunk("CT-40#2", "40.º", "Licença", "S", "parte dois licenca"),
    Chunk("CT-40#1", "40.º", "Licença", "S", "parte um licenca"),
    Chunk("CT-229", "229.º", "Descanso", "S", "texto sobre descanso compensatorio"),
]


def test_chunks_are_ordered_by_article_number_then_suffix_then_part() -> None:
    ordered = sorted(CHUNKS, key=chunk_order_key)
    assert [c.id for c in ordered] == ["CT-9", "CT-10", "CT-10-A", "CT-40#1", "CT-40#2", "CT-229"]


def test_gold_context_is_in_article_order_whatever_the_order_in_the_item() -> None:
    item = {"gold_passage_ids": ["CT-229", "CT-40#2", "CT-9"]}
    assert [c.id for c in gold_context(item, CHUNKS)] == ["CT-9", "CT-40#2", "CT-229"]


def test_gold_context_of_an_unanswerable_item_is_empty() -> None:
    assert gold_context({"gold_passage_ids": []}, CHUNKS) == []


def test_unknown_gold_id_is_an_error() -> None:
    with pytest.raises(KeyError):
        gold_context({"gold_passage_ids": ["CT-777"]}, CHUNKS)


def test_distractor_context_is_the_top_ranked_chunks_in_rank_order() -> None:
    bm25 = BM25({c.id: f"{c.epigraph} {c.text}" for c in CHUNKS})
    got = distractor_context({"question": "descanso compensatorio"}, bm25, CHUNKS, k=2)
    assert got[0].id == "CT-229" and len(got) <= 2


def test_verify_chunks_accepts_the_same_text_and_rejects_changes() -> None:
    manifest = {"chunk_sha256": article_hashes(CHUNKS)}
    verify_chunks(CHUNKS, manifest)
    tampered = [*CHUNKS[:-1], Chunk("CT-229", "229.º", "Descanso", "S", "texto alterado")]
    with pytest.raises(ValueError, match="CT-229"):
        verify_chunks(tampered, manifest)
    with pytest.raises(ValueError, match="missing"):
        verify_chunks(CHUNKS[:-1], manifest)
