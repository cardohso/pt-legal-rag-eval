import pytest

from ptlegal_eval.fusion import rrf


def test_item_ranked_high_in_both_lists_wins() -> None:
    assert rrf([["a", "b", "c"], ["a", "c", "b"]])[0][0] == "a"


def test_score_formula_uses_rank_starting_at_one() -> None:
    fused = dict(rrf([["a"], ["a"]], k=60))
    assert fused["a"] == pytest.approx(2 / 61)


def test_item_in_one_list_still_appears() -> None:
    assert {"a", "z"} <= {i for i, _ in rrf([["a"], ["z"]])}


def test_ties_break_by_id() -> None:
    assert [i for i, _ in rrf([["b"], ["a"]])] == ["a", "b"]


def test_empty_input() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_duplicate_in_one_list_counts_once_at_best_rank() -> None:
    fused = dict(rrf([["a", "b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
