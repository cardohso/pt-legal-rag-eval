"""What goes into the prompt as context, and a check that the chunks are the frozen ones."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ptlegal_eval.bm25 import BM25
from ptlegal_eval.chunks import Chunk
from ptlegal_eval.golden_check import resolve_gold
from ptlegal_eval.ingest import article_hashes

_ID = re.compile(r"^CT-(\d+)(?:-([A-Z]+))?(?:#(\d+))?$")


def chunk_order_key(chunk: Chunk) -> tuple[int, str, int]:
    """Article number, then letter suffix ("10.º" before "10.º-A"), then part of a split article."""
    m = _ID.match(chunk.id)
    if m is None:
        raise ValueError(f"unexpected chunk id {chunk.id!r}")
    return int(m.group(1)), m.group(2) or "", int(m.group(3) or 0)


def gold_context(item: Mapping[str, Any], chunks: Sequence[Chunk]) -> list[Chunk]:
    """Condition A: the gold articles, in article-number order (pre-registration, section 6)."""
    by_id = {c.id: c for c in chunks}
    ids = resolve_gold(item.get("gold_passage_ids", []), by_id.keys())
    return sorted((by_id[i] for i in ids), key=chunk_order_key)


def distractor_context(
    item: Mapping[str, Any], bm25: BM25, chunks: Sequence[Chunk], k: int
) -> list[Chunk]:
    """The top-k retrieved chunks, in rank order, for questions that have no gold article."""
    by_id = {c.id: c for c in chunks}
    return [by_id[i] for i in bm25.rank(item["question"], k)]


def verify_chunks(chunks: Sequence[Chunk], manifest: Mapping[str, Any]) -> None:
    """Raise unless the chunk texts are exactly those recorded in data/corpus_manifest.json."""
    expected: Mapping[str, str] = manifest["chunk_sha256"]
    actual = article_hashes(chunks)
    missing = sorted(set(expected) - set(actual))
    changed = sorted(i for i in expected if i in actual and actual[i] != expected[i])
    extra = sorted(set(actual) - set(expected))
    if missing or changed or extra:
        raise ValueError(
            "chunks do not match the manifest: "
            f"missing {missing[:5]}, changed {changed[:5]}, unexpected {extra[:5]}"
        )
