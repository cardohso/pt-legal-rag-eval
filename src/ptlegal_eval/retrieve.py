"""Hybrid retrieval: BM25 plus dense embeddings, fused with reciprocal rank fusion.

Identical for every model under test. Retrieval runs once per question and its output is reused.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ptlegal_eval.bm25 import BM25
from ptlegal_eval.chunks import Chunk
from ptlegal_eval.fusion import rrf


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """One row per text. Rows need not be normalised."""
        ...


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float  # fused RRF score
    bm25_rank: int | None  # 1-based rank in the BM25 list; None if it was not in it
    dense_rank: int | None


def indexed_text(chunk: Chunk) -> str:
    """What is indexed and embedded: the article heading plus its text."""
    return f"Artigo {chunk.article} {chunk.epigraph}. {chunk.text}"


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        k_rrf: int = 60,
        candidates: int = 50,
    ):
        self._ids = [c.id for c in chunks]
        texts = [indexed_text(c) for c in chunks]
        self._bm25 = BM25(dict(zip(self._ids, texts, strict=True)))
        self._embedder = embedder
        self._matrix = _unit_rows(np.asarray(embedder.encode(texts), dtype=float))
        self.k_rrf, self.candidates = k_rrf, candidates

    def _dense_ranking(self, query: str) -> list[str]:
        q = _unit_rows(np.asarray(self._embedder.encode([query]), dtype=float))[0]
        sims = self._matrix @ q
        order = sorted(range(len(self._ids)), key=lambda i: (-float(sims[i]), self._ids[i]))
        return [self._ids[i] for i in order[: self.candidates]]

    def retrieve(self, query: str, top_k: int = 5) -> list[Hit]:
        bm25 = self._bm25.rank(query, self.candidates)
        dense = self._dense_ranking(query)
        bm25_rank = {cid: r for r, cid in enumerate(bm25, start=1)}
        dense_rank = {cid: r for r, cid in enumerate(dense, start=1)}
        fused = rrf([bm25, dense], self.k_rrf)[:top_k]
        return [Hit(cid, s, bm25_rank.get(cid), dense_rank.get(cid)) for cid, s in fused]
