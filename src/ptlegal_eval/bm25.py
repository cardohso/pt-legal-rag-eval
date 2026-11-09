"""Okapi BM25 over article chunks. Pure Python, deterministic, no stemming.

Accents and case are folded so that "férias" and "Ferias" match. Ordinal indicators are dropped,
so "253.º" tokenises as "253" (not "253", "o").
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    text = text.replace("º", "").replace("ª", "")
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return _TOKEN.findall(folded.lower())


class BM25:
    def __init__(self, docs: Mapping[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tf = {doc_id: Counter(tokenize(text)) for doc_id, text in docs.items()}
        self._len = {doc_id: sum(tf.values()) for doc_id, tf in self._tf.items()}
        self._avgdl = sum(self._len.values()) / len(self._len) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        n = len(self._tf)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> dict[str, float]:
        terms = set(tokenize(query)) & self._idf.keys()
        out: dict[str, float] = {}
        for doc_id, tf in self._tf.items():
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    norm = 1 - self.b + self.b * self._len[doc_id] / self._avgdl
                    score += self._idf[term] * f * (self.k1 + 1) / (f + self.k1 * norm)
            if score > 0:
                out[doc_id] = score
        return out

    def rank(self, query: str, k: int) -> list[str]:
        """Ids of the top k documents with a positive score; ties are broken by id."""
        scored = self.scores(query)
        return sorted(scored, key=lambda d: (-scored[d], d))[:k]
