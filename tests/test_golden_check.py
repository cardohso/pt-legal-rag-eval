from ptlegal_eval.chunks import Chunk
from ptlegal_eval.golden_check import category_counts, check_items, resolve_gold


def _chunk(cid: str, article: str, size: int = 100) -> Chunk:
    return Chunk(cid, article, "Epígrafe", "Secção", "x" * size)


CHUNKS = [
    _chunk("CT-10", "10.º"),
    _chunk("CT-11", "11.º"),
    _chunk("CT-40#1", "40.º"),
    _chunk("CT-40#2", "40.º"),
    _chunk("CT-263", "263.º"),
    _chunk("CT-99", "99.º", size=9000),
    _chunk("CT-98", "98.º", size=9000),
]


def item(**over):
    base = {
        "id": "g001",
        "question": "Qual é a pergunta?",
        "expected": "answer",
        "reference_answer": "Esta é a resposta.",
        "gold_passage_ids": ["CT-10"],
        "source_doc_ids": ["CT"],
        "tags": ["single-article"],
        "metadata": {"category": "single_article", "behaviour": "answer"},
    }
    base.update(over)
    return base


def problems(*items):
    return check_items(list(items), CHUNKS)


def test_valid_item_has_no_problems() -> None:
    assert problems(item()) == []


def test_resolve_gold_expands_split_articles() -> None:
    assert resolve_gold(["CT-40"], {c.id for c in CHUNKS}) == ["CT-40#1", "CT-40#2"]
    assert resolve_gold(["CT-10"], {c.id for c in CHUNKS}) == ["CT-10"]


def test_unknown_gold_id_is_reported() -> None:
    assert any("CT-777" in p for p in problems(item(gold_passage_ids=["CT-777"])))


def test_answer_item_needs_gold_articles() -> None:
    assert any("gold" in p for p in problems(item(gold_passage_ids=[])))


def test_unanswerable_item_must_have_no_gold_but_an_explanation() -> None:
    ok = item(
        expected="refuse",
        gold_passage_ids=[],
        metadata={"category": "unanswerable", "behaviour": "refuse"},
        tags=["unanswerable"],
    )
    assert problems(ok) == []
    assert any("gold" in p for p in problems({**ok, "gold_passage_ids": ["CT-10"]}))
    assert any("reference" in p for p in problems({**ok, "reference_answer": ""}))


def test_suspended_articles_are_rejected() -> None:
    assert any("263" in p and "suspended" in p for p in problems(item(gold_passage_ids=["CT-263"])))


def test_gold_context_must_fit_the_budget() -> None:
    too_big = item(
        gold_passage_ids=["CT-99", "CT-98"],
        metadata={"category": "multi_article", "behaviour": "answer"},
    )
    assert any("budget" in p for p in problems(too_big))


def test_category_and_behaviour_must_match() -> None:
    bad = item(metadata={"category": "numeric", "behaviour": "refuse"})
    assert any("behaviour" in p for p in problems(bad))


def test_multi_article_needs_two_articles_and_single_needs_one() -> None:
    multi = item(metadata={"category": "multi_article", "behaviour": "answer"})
    assert any("two" in p for p in problems(multi))
    single = item(gold_passage_ids=["CT-10", "CT-11"])
    assert any("one article" in p for p in problems(single))
    assert problems(item(gold_passage_ids=["CT-40"])) == []  # two parts, one article


def test_duplicate_questions_and_ids_are_reported() -> None:
    a, b = item(), item(id="g002", question="  qual é a PERGUNTA? ")
    assert any("duplicate question" in p for p in problems(a, b))
    assert any("duplicate id" in p for p in problems(a, item()))


def test_schema_violation_is_reported() -> None:
    bad = item()
    del bad["question"]
    assert any("question" in p for p in problems(bad))


def test_category_counts() -> None:
    assert category_counts([item(), item(id="g2")]) == {
        "single_article": 2,
        "multi_article": 0,
        "numeric": 0,
        "unanswerable": 0,
        "false_premise": 0,
    }
