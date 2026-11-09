"""Retrieval quality, measured separately from generation. Uses goldpass's definitions."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from goldpass.metrics import hit_at_k, recall_at_k, reciprocal_rank


@dataclass(frozen=True)
class RetrievalResult:
    n: int  # items scored (those with at least one gold article)
    recall: float
    mrr: float
    hit_rate: float


def evaluate_retrieval(
    rankings: Mapping[str, Sequence[str]],
    gold: Mapping[str, Collection[str]],
    k: int,
) -> RetrievalResult:
    """Mean recall@k, MRR and hit rate over items that have gold articles.

    Items with no gold articles (unanswerable questions) have nothing to retrieve, so they are
    skipped rather than scored 0. An item missing from `rankings` counts as retrieving nothing.
    """
    ids = [i for i, g in gold.items() if g]
    if not ids:
        raise ValueError("no item has gold articles: retrieval cannot be scored")
    ranked = {i: list(rankings.get(i, ())) for i in ids}
    return RetrievalResult(
        n=len(ids),
        recall=fmean(recall_at_k(ranked[i], set(gold[i]), k) for i in ids),
        mrr=fmean(reciprocal_rank(ranked[i], set(gold[i]), k) for i in ids),
        hit_rate=fmean(float(hit_at_k(ranked[i], set(gold[i]), k)) for i in ids),
    )
