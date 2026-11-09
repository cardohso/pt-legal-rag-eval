import pytest

from ptlegal_eval.retrieval_metrics import evaluate_retrieval


def test_perfect_retrieval() -> None:
    result = evaluate_retrieval({"q1": ["a", "b"]}, {"q1": {"a"}}, k=5)
    assert result.recall == 1.0 and result.mrr == 1.0 and result.hit_rate == 1.0 and result.n == 1


def test_gold_at_rank_two() -> None:
    result = evaluate_retrieval({"q1": ["x", "a"]}, {"q1": {"a"}}, k=5)
    assert result.mrr == 0.5 and result.hit_rate == 1.0


def test_gold_outside_top_k_is_a_miss() -> None:
    result = evaluate_retrieval({"q1": ["x", "y", "a"]}, {"q1": {"a"}}, k=2)
    assert result.recall == 0.0 and result.hit_rate == 0.0


def test_multi_article_recall_is_a_share() -> None:
    result = evaluate_retrieval({"q1": ["a", "x"]}, {"q1": {"a", "b"}}, k=5)
    assert result.recall == 0.5


def test_items_without_gold_are_skipped_not_scored_zero() -> None:
    result = evaluate_retrieval(
        {"q1": ["a"], "refuse1": ["z"]}, {"q1": {"a"}, "refuse1": set()}, k=5
    )
    assert result.n == 1 and result.recall == 1.0


def test_means_over_items() -> None:
    result = evaluate_retrieval({"q1": ["a"], "q2": ["x"]}, {"q1": {"a"}, "q2": {"b"}}, k=5)
    assert result.hit_rate == 0.5 and result.recall == pytest.approx(0.5)


def test_no_scorable_items_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_retrieval({"r": ["a"]}, {"r": set()}, k=5)
