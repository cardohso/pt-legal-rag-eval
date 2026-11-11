"""Mechanical checks on golden-set items, run before the human review and again before the freeze.

Items follow goldpass's golden-item schema. `metadata` carries what goldpass ignores:

- `category`: single_article, multi_article, numeric, unanswerable, false_premise
- `behaviour`: answer, refuse (unanswerable) or correct (false premise)

Gold ids may be article-level ("CT-40"); `resolve_gold` expands them to chunk ids when the
article was split ("CT-40#1", "CT-40#2").
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import jsonschema
from goldpass.golden import schema as goldpass_schema

from ptlegal_eval.chunks import Chunk

CATEGORIES = ("single_article", "multi_article", "numeric", "unanswerable", "false_premise")
BEHAVIOUR_OF = {
    "single_article": "answer",
    "multi_article": "answer",
    "numeric": "answer",
    "unanswerable": "refuse",
    "false_premise": "correct",
}
CONTEXT_BUDGET = 12_000  # characters, from the pre-registration
SUSPENDED = {"263.º", "264.º"}  # suspended by Lei n.º 11/2013; see docs/CORPUS.md


def resolve_gold(gold_ids: Sequence[str], chunk_ids: Collection[str]) -> list[str]:
    """Chunk ids for article-level gold ids. Raises KeyError for an unknown id."""
    out: list[str] = []
    for gid in gold_ids:
        if gid in chunk_ids:
            out.append(gid)
            continue
        parts = sorted(c for c in chunk_ids if c.startswith(f"{gid}#"))
        if not parts:
            raise KeyError(gid)
        out.extend(parts)
    return out


def category_counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(CATEGORIES, 0)
    for it in items:
        cat = it.get("metadata", {}).get("category")
        if cat in counts:
            counts[cat] += 1
    return counts


def check_items(
    items: Sequence[Mapping[str, Any]], chunks: Sequence[Chunk], budget: int = CONTEXT_BUDGET
) -> list[str]:
    """Every problem found, as readable strings; an empty list means the items are consistent."""
    by_id = {c.id: c for c in chunks}
    validator = jsonschema.Draft202012Validator(goldpass_schema())
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}

    for it in items:
        iid = str(it.get("id", "?"))

        def bad(msg: str, iid: str = iid) -> None:
            problems.append(f"{iid}: {msg}")

        for err in validator.iter_errors(it):
            where = "/".join(map(str, err.path))
            bad(f"schema: {where + ': ' if where else ''}{err.message}")
        if iid in seen_ids:
            bad("duplicate id")
        seen_ids.add(iid)
        question = " ".join(str(it.get("question", "")).split()).casefold()
        if question in seen_questions:
            bad(f"duplicate question (same as {seen_questions[question]})")
        seen_questions.setdefault(question, iid)

        meta = it.get("metadata", {})
        category, behaviour = meta.get("category"), meta.get("behaviour")
        if category not in CATEGORIES:
            bad(f"unknown category {category!r}")
            continue
        if behaviour != BEHAVIOUR_OF[category]:
            bad(
                f"behaviour {behaviour!r} does not match category {category} "
                f"(expected {BEHAVIOUR_OF[category]!r})"
            )
        if (it.get("expected") == "refuse") != (behaviour == "refuse"):
            bad("`expected` and metadata.behaviour disagree about refusing")
        if not str(it.get("reference_answer", "")).strip():
            bad("reference answer is empty (for unanswerable items it explains why)")

        gold = list(it.get("gold_passage_ids", []))
        if behaviour == "refuse":
            if gold:
                bad("an unanswerable item must have no gold articles")
            continue
        if not gold:
            bad("no gold articles")
            continue
        try:
            resolved = resolve_gold(gold, by_id.keys())
        except KeyError as e:
            bad(f"unknown gold id {e.args[0]}")
            continue
        articles = {by_id[c].article for c in resolved}
        for art in sorted(articles & SUSPENDED):
            bad(f"gold article {art} is suspended")
        if category == "multi_article" and len(articles) < 2:
            bad("a multi_article item needs at least two gold articles")
        if category == "single_article" and len(articles) != 1:
            bad("a single_article item needs exactly one article")
        total = sum(len(by_id[c].text) for c in resolved)
        if total > budget:
            bad(f"gold context is {total} characters, over the {budget} budget")
    return problems
