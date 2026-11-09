from pathlib import Path

from ptlegal_eval.chunks import Chunk, fit_budget, read_chunks, split_long_text, write_chunks

ITEMS = "\n".join(f"{n} - " + ("palavra " * 30).strip() + "." for n in range(1, 9))


def test_short_text_is_not_split() -> None:
    assert split_long_text("1 - a\n2 - b", max_chars=100) == ["1 - a\n2 - b"]


def test_long_text_splits_at_numbered_items_only() -> None:
    parts = split_long_text(ITEMS, max_chars=600)
    assert len(parts) > 1
    assert all(p.startswith(tuple(f"{n} - " for n in range(1, 9))) for p in parts)
    assert "\n".join(parts) == ITEMS


def test_parts_respect_the_limit_when_items_are_small_enough() -> None:
    assert all(len(p) <= 600 for p in split_long_text(ITEMS, max_chars=600))


def test_single_oversized_item_is_kept_whole() -> None:
    text = "1 - " + "x" * 500 + "\n2 - curto"
    parts = split_long_text(text, max_chars=100)
    assert parts[0].startswith("1 - xxx") and len(parts[0]) > 100
    assert parts[-1] == "2 - curto"


def test_unnumbered_text_is_never_split() -> None:
    text = "frase " * 200
    assert split_long_text(text, max_chars=100) == [text]


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, article="1.º", epigraph="E", section="S", text=text)


def test_fit_budget_keeps_longest_prefix_that_fits() -> None:
    chunks = [_chunk("a", "x" * 40), _chunk("b", "x" * 40), _chunk("c", "x" * 40)]
    assert [c.id for c in fit_budget(chunks, max_chars=100)] == ["a", "b"]


def test_fit_budget_stops_at_first_chunk_that_does_not_fit() -> None:
    chunks = [_chunk("a", "x" * 60), _chunk("b", "x" * 60), _chunk("c", "x" * 5)]
    assert [c.id for c in fit_budget(chunks, max_chars=100)] == ["a"]


def test_fit_budget_empty_when_first_chunk_alone_is_too_long() -> None:
    assert fit_budget([_chunk("a", "x" * 200)], max_chars=100) == []


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    chunks = [_chunk("CT-1", "Texto com acentuação: ção.")]
    write_chunks(chunks, path)
    assert read_chunks(path) == chunks
    assert "ç" in path.read_text(encoding="utf-8")
