import json

import pytest

from ptlegal_eval.chunks import Chunk
from ptlegal_eval.golden_check import CATEGORIES
from ptlegal_eval.split import expand_gold, split_items


def make(i: int, category: str, gold=("CT-1",)):
    return {
        "id": f"g{i:03d}",
        "question": f"Pergunta {i}?",
        "gold_passage_ids": list(gold),
        "metadata": {"category": category, "split": None, "reviewed": True},
    }


def pool(per: int = 10):
    return [make(i, c) for i, c in enumerate(CATEGORIES * per, start=1)]


def test_split_is_stratified_complete_and_disjoint() -> None:
    dev, test = split_items(pool(10), dev_per_category=3, seed=1)
    assert len(dev) == 3 * len(CATEGORIES) and len(test) == 7 * len(CATEGORIES)
    for cat in CATEGORIES:
        assert sum(i["metadata"]["category"] == cat for i in dev) == 3
        assert sum(i["metadata"]["category"] == cat for i in test) == 7
    ids = [i["id"] for i in dev + test]
    assert len(ids) == len(set(ids)) == 50


def test_split_is_deterministic_and_seed_dependent() -> None:
    a = [i["id"] for i in split_items(pool(), 3, seed=5)[0]]
    assert a == [i["id"] for i in split_items(pool(), 3, seed=5)[0]]
    assert a != [i["id"] for i in split_items(pool(), 3, seed=6)[0]]


def test_input_order_does_not_change_the_split() -> None:
    items = pool()
    forward = [i["id"] for i in split_items(items, 3, seed=9)[0]]
    backward = [i["id"] for i in split_items(list(reversed(items)), 3, seed=9)[0]]
    assert forward == backward


def test_items_are_labelled_with_their_split_and_sorted_by_id() -> None:
    dev, test = split_items(pool(), 3, seed=2)
    assert {i["metadata"]["split"] for i in dev} == {"dev"}
    assert {i["metadata"]["split"] for i in test} == {"test"}
    assert [i["id"] for i in test] == sorted(i["id"] for i in test)


def test_unreviewed_or_too_few_items_are_errors() -> None:
    items = pool()
    items[0]["metadata"]["reviewed"] = False
    with pytest.raises(ValueError, match="reviewed"):
        split_items(items, 3, seed=1)
    with pytest.raises(ValueError, match="single_article"):
        split_items(pool(per=2), 3, seed=1)


def test_input_is_not_mutated() -> None:
    items = pool()
    before = json.dumps(items)
    split_items(items, 3, seed=1)
    assert json.dumps(items) == before


def test_expand_gold_turns_articles_into_chunk_ids_and_keeps_labels() -> None:
    chunks = [
        Chunk("CT-1", "1.º", "E", "S", "x"),
        Chunk("CT-40#1", "40.º", "E", "S", "x"),
        Chunk("CT-40#2", "40.º", "E", "S", "x"),
    ]
    item = make(1, "numeric", gold=("CT-1", "CT-40"))
    out = expand_gold(item, chunks)
    assert out["gold_passage_ids"] == ["CT-1", "CT-40#1", "CT-40#2"]
    assert out["metadata"]["gold_articles"] == ["1.º", "40.º"]
    assert item["gold_passage_ids"] == ["CT-1", "CT-40"]  # original untouched
