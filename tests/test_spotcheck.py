import json

import pytest

from ptlegal_eval.golden_check import CATEGORIES
from ptlegal_eval.spotcheck import merge_spotcheck, sample_items

BULK = "Bulk-approved by J on 2026-11-11 without item-by-item review."


def make(i: int, category: str, bulk: bool = True):
    return {
        "id": f"g{i:03d}",
        "question": f"Pergunta {i}?",
        "reference_answer": f"Resposta {i}.",
        "metadata": {
            "category": category,
            "behaviour": "answer",
            "reviewed": True,
            "reviewed_by": "J",
            **({"review_comment": BULK} if bulk else {}),
        },
    }


def pool(per=6):
    return [make(i, c) for i, c in enumerate(CATEGORIES * per, start=1)]


def test_sample_is_stratified() -> None:
    chosen = sample_items(pool(), per_category=3, seed=1)
    assert len(chosen) == 3 * len(CATEGORIES)
    for cat in CATEGORIES:
        assert sum(i["metadata"]["category"] == cat for i in chosen) == 3


def test_sample_is_reproducible_and_depends_on_the_seed() -> None:
    a = [i["id"] for i in sample_items(pool(), 3, seed=7)]
    assert a == [i["id"] for i in sample_items(pool(), 3, seed=7)]
    assert a != [i["id"] for i in sample_items(pool(), 3, seed=8)]


def test_only_bulk_approved_items_are_drawn() -> None:
    items = pool() + [make(100 + n, "numeric", bulk=False) for n in range(20)]
    assert all(i["id"] < "g100" for i in sample_items(items, 3, seed=1))


def test_too_small_a_pool_is_an_error() -> None:
    with pytest.raises(ValueError, match="single_article"):
        sample_items(pool(per=2), per_category=3, seed=1)


def test_merge_marks_accepted_items_as_spot_checked_and_keeps_the_history() -> None:
    items = pool(per=1)
    out = merge_spotcheck(items, {"g001": {"status": "accept"}}, reviewer="J", date="2026-11-12")
    first = out.items[0]["metadata"]
    assert first["spot_checked"] is True and "Bulk-approved" in first["review_comment"]
    assert "Spot-checked 2026-11-12: accept" in first["review_comment"]
    assert "spot_checked" not in out.items[1]["metadata"]


def test_merge_applies_edits_and_keeps_the_comment() -> None:
    items = pool(per=1)
    d = {"g002": {"status": "edit", "question": "Nova?", "comment": "ambígua"}}
    out = merge_spotcheck(items, d, reviewer="J", date="2026-11-12")
    meta = out.items[1]["metadata"]
    assert out.items[1]["question"] == "Nova?" and meta["edited"] is True
    assert "ambígua" in meta["review_comment"] and out.edited == ["g002"]


def test_merge_removes_rejected_items_and_reports_them() -> None:
    items = pool(per=1)
    out = merge_spotcheck(items, {"g003": {"status": "reject"}}, reviewer="J", date="2026-11-12")
    assert [i["id"] for i in out.items] == ["g001", "g002", "g004", "g005"]
    assert out.rejected == ["g003"]


def test_merge_does_not_mutate_its_input() -> None:
    items = pool(per=1)
    before = json.dumps(items)
    merge_spotcheck(items, {"g001": {"status": "edit", "question": "X?"}}, reviewer="J", date="d")
    assert json.dumps(items) == before
