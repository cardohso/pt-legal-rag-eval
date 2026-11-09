"""The real embedding model. Heavy, so imported lazily and never used in tests or CI."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BGE_M3 = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"  # pinned in the pre-registration


class BgeM3Embedder:
    def __init__(self, model_id: str = BGE_M3, revision: str = BGE_M3_REVISION):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "BgeM3Embedder needs sentence-transformers: uv sync --extra retrieval"
            ) from e
        self._model = SentenceTransformer(model_id, revision=revision)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True, batch_size=16),
            dtype=float,
        )
