"""Dev/test split of the golden set, done once, then frozen by hash.

Dev items are for prompt tuning; test items are never looked at until the pre-registered runs.
The split is stratified by category and reproducible from the seed. The CLI refuses to overwrite an
existing test file, so the split cannot be quietly re-rolled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ptlegal_eval.chunks import Chunk, read_chunks
from ptlegal_eval.golden_check import CATEGORIES, resolve_gold


def split_items(
    items: Sequence[Mapping[str, Any]], dev_per_category: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(dev, test), each sorted by id. Input order does not matter."""
    unreviewed = [i["id"] for i in items if not i["metadata"].get("reviewed")]
    if unreviewed:
        raise ValueError(f"items not reviewed: {unreviewed[:5]}")
    rng = random.Random(seed)
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        pool = sorted(
            (copy.deepcopy(dict(i)) for i in items if i["metadata"]["category"] == cat),
            key=lambda i: i["id"],
        )
        if len(pool) <= dev_per_category:
            raise ValueError(f"{cat}: {len(pool)} items, need more than {dev_per_category}")
        chosen = {i["id"] for i in rng.sample(pool, dev_per_category)}
        for it in pool:
            it["metadata"]["split"] = "dev" if it["id"] in chosen else "test"
            (dev if it["id"] in chosen else test).append(it)
    return sorted(dev, key=lambda i: i["id"]), sorted(test, key=lambda i: i["id"])


def expand_gold(item: Mapping[str, Any], chunks: Sequence[Chunk]) -> dict[str, Any]:
    """Article-level gold ids become chunk ids (what retrieval metrics compare); labels are kept."""
    out = copy.deepcopy(dict(item))
    by_id = {c.id: c for c in chunks}
    resolved = resolve_gold(item.get("gold_passage_ids", []), by_id.keys())
    out["gold_passage_ids"] = resolved
    articles = list(dict.fromkeys(by_id[c].article for c in resolved))
    out.setdefault("metadata", {})["gold_articles"] = articles
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, items: Sequence[Mapping[str, Any]]) -> str:
    path.write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8"
    )
    digest = _sha256(path)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path, help="reviewed_batch*.jsonl")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dev-per-category", type=int, default=6)
    p.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/golden"))
    args = p.parse_args(argv)

    test_path = args.out_dir / "test.jsonl"
    if test_path.exists():
        raise SystemExit(f"{test_path} exists: the split is frozen and is not redone")
    items = [
        json.loads(x) for f in args.files for x in f.read_text(encoding="utf-8").splitlines() if x
    ]
    chunks = read_chunks(args.chunks)
    dev, test = split_items(items, args.dev_per_category, args.seed)
    dev = [expand_gold(i, chunks) for i in dev]
    test = [expand_gold(i, chunks) for i in test]
    print(f"dev  {len(dev)} items  sha256 {_write(args.out_dir / 'dev.jsonl', dev)}")
    print(f"test {len(test)} items  sha256 {_write(test_path, test)}")
    dev_art = {a for i in dev for a in i["metadata"]["gold_articles"]}
    test_art = {a for i in test for a in i["metadata"]["gold_articles"]}
    print(f"gold articles shared by dev and test: {len(dev_art & test_art)} of {len(dev_art)} dev")


if __name__ == "__main__":
    main()
