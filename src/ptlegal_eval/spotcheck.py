"""Spot check of the bulk-approved golden-set items.

Most items were approved without being read (docs/DEVIATIONS.md). A random, stratified sample of
them goes through the normal review page; this module draws it and merges the answers back, keeping
the bulk-approval history and recording what the spot check found.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ptlegal_eval.chunks import read_chunks
from ptlegal_eval.golden_check import CATEGORIES
from ptlegal_eval.review import ApplyResult, apply_decisions, render_review_page

BULK_MARK = "Bulk-approved"


def sample_items(
    items: Sequence[Mapping[str, Any]], per_category: int, seed: int
) -> list[dict[str, Any]]:
    """`per_category` random bulk-approved items from every category, reproducible from `seed`."""
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        pool = sorted(
            (
                dict(i)
                for i in items
                if i["metadata"]["category"] == cat
                and BULK_MARK in i["metadata"].get("review_comment", "")
            ),
            key=lambda i: i["id"],
        )
        if len(pool) < per_category:
            raise ValueError(f"only {len(pool)} bulk-approved {cat} items, need {per_category}")
        chosen += rng.sample(pool, per_category)
    return sorted(chosen, key=lambda i: i["id"])


def merge_spotcheck(
    items: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    reviewer: str,
    date: str,
) -> ApplyResult:
    """Apply spot-check decisions to items; undecided items are left exactly as they were."""
    own = {k: {**d, "comment": ""} for k, d in decisions.items()}  # keep the old comment ourselves
    result = apply_decisions(items, own, reviewer)
    result.pending = []  # everything not sampled is simply not part of the spot check
    for it in result.items:
        d = decisions.get(it["id"])
        if d is None:
            continue
        meta = it["metadata"]
        meta["spot_checked"] = True
        note = f"Spot-checked {date}: {d['status']}"
        extra = str(d.get("comment", "")).strip()
        meta["review_comment"] = f"{meta.get('review_comment', '')} {note}" + (
            f" ({extra})." if extra else "."
        )
        meta["review_comment"] = meta["review_comment"].strip()
    return result


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw the sample and build its review page")
    s.add_argument("files", nargs="+", type=Path, help="reviewed_batch*.jsonl")
    s.add_argument("--per-category", type=int, default=3)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    s.add_argument("--out-items", type=Path, default=Path("data/golden/spotcheck_sample.jsonl"))
    s.add_argument("--out-page", type=Path, default=Path("review/review_spotcheck.html"))
    m = sub.add_parser("merge", help="merge downloaded decisions into the batch files")
    m.add_argument("files", nargs="+", type=Path, help="reviewed_batch*.jsonl, rewritten in place")
    m.add_argument("--decisions", type=Path, required=True)
    m.add_argument("--reviewer", required=True)
    m.add_argument("--date", required=True)
    args = p.parse_args(argv)

    if args.cmd == "sample":
        items = [i for f in args.files for i in _read(f)]
        chosen = sample_items(items, args.per_category, args.seed)
        args.out_items.parent.mkdir(parents=True, exist_ok=True)
        args.out_items.write_text(
            "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in chosen), encoding="utf-8"
        )
        args.out_page.parent.mkdir(parents=True, exist_ok=True)
        args.out_page.write_text(
            render_review_page(chosen, read_chunks(args.chunks)), encoding="utf-8"
        )
        print(f"{len(chosen)} items (seed {args.seed}): {[i['id'] for i in chosen]}")
        return

    raw = json.loads(args.decisions.read_text(encoding="utf-8"))
    decisions = raw.get("decisions", raw)
    seen: set[str] = set()
    totals = {"rejected": [], "edited": []}
    for f in args.files:
        items = _read(f)
        mine = {k: v for k, v in decisions.items() if k in {i["id"] for i in items}}
        result = merge_spotcheck(items, mine, args.reviewer, args.date)
        f.write_text(
            "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in result.items),
            encoding="utf-8",
        )
        seen |= set(mine)
        totals["rejected"] += result.rejected
        totals["edited"] += result.edited
    missing = set(decisions) - seen
    if missing:
        raise SystemExit(f"decisions for items in no batch file: {sorted(missing)}")
    print(f"merged {len(seen)} decisions: edited {totals['edited']}, rejected {totals['rejected']}")


if __name__ == "__main__":
    main()
