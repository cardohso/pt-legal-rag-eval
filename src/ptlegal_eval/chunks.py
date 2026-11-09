"""Article chunks: data type, JSONL storage, splitting of long articles, context budget."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

_ITEM_START = re.compile(r"^\d+ - ")


@dataclass(frozen=True)
class Chunk:
    id: str  # stable, e.g. "CT-253" or "CT-253#2" for the second part of a long article
    article: str  # "253.º"
    epigraph: str
    section: str
    text: str


def write_chunks(chunks: Sequence[Chunk], path: Path) -> None:
    lines = (json.dumps(asdict(c), ensure_ascii=False) for c in chunks)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_chunks(path: Path) -> list[Chunk]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return [Chunk(**row) for row in rows]


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split at numbered items ("1 - ", "2 - ", ...), never inside one.

    Items are packed greedily up to `max_chars`. An item longer than the limit is kept whole, and
    text without numbered items is never split. Joining the parts with newlines gives the original.
    """
    if len(text) <= max_chars:
        return [text]
    groups: list[list[str]] = [[]]
    for line in text.split("\n"):
        if _ITEM_START.match(line) and groups[-1] and _ITEM_START.match(groups[-1][0]):
            groups.append([])
        groups[-1].append(line)
    if len(groups) < 2:
        return [text]
    parts: list[str] = []
    current = ""
    for group in ("\n".join(g) for g in groups):
        if current and len(current) + 1 + len(group) > max_chars:
            parts.append(current)
            current = group
        else:
            current = f"{current}\n{group}" if current else group
    parts.append(current)
    return parts


def fit_budget(chunks: Sequence[Chunk], max_chars: int) -> list[Chunk]:
    """The longest prefix of the ranked chunks whose total text fits the character budget.

    Whole chunks only, lowest-ranked dropped first. Stops at the first chunk that does not fit, so
    a lower-ranked short chunk never jumps ahead of a higher-ranked long one.
    """
    kept: list[Chunk] = []
    used = 0
    for chunk in chunks:
        used += len(chunk.text)
        if used > max_chars:
            break
        kept.append(chunk)
    return kept
