"""Mechanical checks on golden-set items, run before the human review and again before the freeze.

Items follow goldpass's golden-item schema. `metadata` carries what goldpass ignores:

- `category`: single_article, multi_article, numeric, unanswerable, false_premise
- `behaviour`: answer, refuse (unanswerable) or correct (false premise)

Gold ids may be article-level ("CT-40"); `resolve_gold` expands them to chunk ids when the
article was split ("CT-40#1", "CT-40#2").
"""

from __future__ import annotations

import re
import unicodedata
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


# --- reference grounding -------------------------------------------------------------------------
# A reference answer may only cite articles, paragraphs and numbers that the gold text supports.
# Cheap, mechanical, and no substitute for reading, but it catches the typos and mix-ups a drafter
# makes: a number from the wrong paragraph, an article that is not in the gold set.

_UNITS = {
    "dois": 2,
    "duas": 2,
    "tres": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
}  # fmt: skip  ("um/uma" are left out: they are mostly articles)
_TEENS = {"dez": 10, "onze": 11, "doze": 12, "treze": 13, "catorze": 14, "quatorze": 14,
          "quinze": 15, "dezasseis": 16, "dezesseis": 16, "dezassete": 17, "dezoito": 18,
          "dezanove": 19}  # fmt: skip
_TENS = {"vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60,
         "setenta": 70, "oitenta": 80, "noventa": 90}  # fmt: skip
_HUNDREDS = {"cem": 100, "cento": 100, "duzentos": 200, "duzentas": 200, "trezentos": 300,
             "trezentas": 300, "quatrocentos": 400, "quinhentos": 500, "seiscentos": 600,
             "setecentos": 700, "oitocentos": 800, "novecentos": 900}  # fmt: skip
_WORDS = {**_UNITS, **_TEENS, **_TENS, **_HUNDREDS}

_STRIP = [
    re.compile(r"art(?:igos?|s?\.)?\s*\d+\.º(?:-[A-Z]+)?", re.IGNORECASE),
    re.compile(r"n\.(?:º|os)\s*\d+(?:\s*(?:,|e|a)\s*\d+)*", re.IGNORECASE),
    re.compile(r"alíneas?\s+[a-z]\)(?:\s*(?:,|e|a)\s*[a-z]\))*", re.IGNORECASE),
    re.compile(r"\d+\.º(?:-[A-Z]+)?"),  # ordinals such as "1.º grau"
    re.compile(r"(?m)^\d+ - "),  # paragraph markers of the law's own text
]
_TOKEN = re.compile(r"\d+(?:,\d+)?|[a-z]+")
_CITE = re.compile(r"art(?:igo)?\.?\s*(\d+)\.º(?:-([A-Z]+))?(?:,\s*n\.º\s*(\d+))?", re.IGNORECASE)
_MENTION = re.compile(r"artigos?\s+(\d+)\.º(?:-([A-Z]+))?", re.IGNORECASE)


def _label(number: str, suffix: str | None) -> str:
    return f"{number}.º" + (f"-{suffix.upper()}" if suffix else "")


def number_values(text: str, partial: bool = False) -> set[int | str]:
    """Numbers asserted in `text`, as digits or Portuguese words ("cento e setenta e cinco" is 175).

    Article and paragraph references, ordinals and list markers are not numbers. With `partial`,
    the parts of a compound ("setenta", "cinco") count too, which is what a source text needs.
    """
    for pattern in _STRIP:
        text = pattern.sub(" ", text)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    toks = _TOKEN.findall(text.lower())
    out: set[int | str] = set()
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.isdigit():
            out.add(int(tok))
        elif "," in tok:
            out.add(tok)
        elif tok in _WORDS:
            parts = [_WORDS[tok]]
            while i + 2 < len(toks) and toks[i + 1] == "e" and toks[i + 2] in _WORDS:
                last, nxt = parts[-1], _WORDS[toks[i + 2]]
                if (last >= 100 and nxt < 100) or (
                    20 <= last < 100 and last % 10 == 0 and nxt < 10
                ):
                    parts.append(nxt)
                    i += 2
                else:
                    break
            out.add(sum(parts))
            if partial:
                out.update(parts)
        i += 1
    return out


def check_references(items: Sequence[Mapping[str, Any]], chunks: Sequence[Chunk]) -> list[str]:
    """Problems in reference answers: ungrounded citations, missing paragraphs, odd numbers."""
    by_id = {c.id: c for c in chunks}
    texts_of: dict[str, list[str]] = {}
    for c in chunks:
        texts_of.setdefault(c.article, []).append(c.text)
    problems: list[str] = []
    for it in items:
        iid = str(it.get("id", "?"))
        ref = str(it.get("reference_answer", ""))
        refuse = it.get("metadata", {}).get("behaviour") == "refuse"
        gold_text, allowed_articles = "", set()
        if not refuse:
            try:
                resolved = resolve_gold(it.get("gold_passage_ids", []), by_id.keys())
            except KeyError:
                continue  # check_items reports unknown gold ids
            gold_chunks = [by_id[c] for c in resolved]
            gold_text = "\n".join(c.text for c in gold_chunks)
            allowed_articles = {c.article for c in gold_chunks}
            allowed_articles |= {_label(n, s) for n, s in _MENTION.findall(gold_text)}
        for number, suffix, para in _CITE.findall(ref):
            label = _label(number, suffix)
            if label not in texts_of:
                problems.append(f"{iid}: reference cites art. {label}, which is not in the corpus")
                continue
            if not refuse and label not in allowed_articles:
                problems.append(
                    f"{iid}: reference cites art. {label}, not a gold article and not named in one"
                )
            if para and not any(
                line.startswith(f"{para} - ") for t in texts_of[label] for line in t.split("\n")
            ):
                problems.append(
                    f"{iid}: art. {label} has no paragraph n.º {para}, cited in the reference"
                )
        if not refuse:
            allowed = number_values(gold_text + "\n" + str(it.get("question", "")), partial=True)
            for value in sorted(number_values(ref) - allowed, key=str):
                problems.append(
                    f"{iid}: number {value} in the reference is not in the gold text/question"
                )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m ptlegal_eval.golden_check FILE.jsonl ...`: run all checks on item files."""
    import argparse
    import json
    from pathlib import Path

    from ptlegal_eval.chunks import read_chunks

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    args = p.parse_args(argv)
    items: list[dict[str, Any]] = []
    for f in args.files:
        items += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    chunks = read_chunks(args.chunks)
    found = check_items(items, chunks) + check_references(items, chunks)
    for line in found:
        print(line)
    counts = category_counts(items)
    print(f"{len(items)} items {counts}: {len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
