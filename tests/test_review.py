import json
import re
import shutil
import subprocess

import pytest

from ptlegal_eval.chunks import Chunk
from ptlegal_eval.review import CORE_JS, apply_decisions, render_review_page

CHUNKS = [
    Chunk("CT-10", "10.º", "Epígrafe dez", "Livro I", "Texto do artigo dez."),
    Chunk("CT-40#1", "40.º", "Licença", "Livro I", "Primeira parte."),
    Chunk("CT-40#2", "40.º", "Licença", "Livro I", "Segunda parte."),
]


def make(i: int, **over):
    base = {
        "id": f"g{i:03d}",
        "question": f"Pergunta {i}?",
        "expected": "answer",
        "reference_answer": f"Resposta {i}.",
        "gold_passage_ids": ["CT-10"],
        "source_doc_ids": ["CT"],
        "tags": ["single-article"],
        "metadata": {"category": "single_article", "behaviour": "answer", "reviewed": False},
    }
    base.update(over)
    return base


ITEMS = [make(1), make(2), make(3), make(4)]


def test_accept_marks_reviewed_and_records_the_reviewer() -> None:
    result = apply_decisions(ITEMS, {"g001": {"status": "accept"}}, reviewer="João")
    first = result.items[0]
    assert first["metadata"]["reviewed"] is True and first["metadata"]["reviewed_by"] == "João"
    assert first["question"] == "Pergunta 1?"


def test_edit_replaces_question_and_reference_and_notes_the_edit() -> None:
    decision = {
        "g002": {
            "status": "edit",
            "question": "  Nova pergunta?  ",
            "reference_answer": "Nova resposta.",
            "comment": "mais claro",
        }
    }
    result = apply_decisions(ITEMS, decision, reviewer="João")
    second = result.items[1]
    assert second["question"] == "Nova pergunta?" and second["reference_answer"] == "Nova resposta."
    assert (
        second["metadata"]["edited"] is True
        and second["metadata"]["review_comment"] == "mais claro"
    )
    assert result.edited == ["g002"]


def test_accept_with_unchanged_text_is_not_an_edit() -> None:
    same = {"status": "accept", "question": "Pergunta 1?", "reference_answer": "Resposta 1."}
    result = apply_decisions(ITEMS, {"g001": same}, reviewer="João")
    assert result.edited == [] and "edited" not in result.items[0]["metadata"]


def test_reject_removes_the_item_and_pending_items_are_kept_unreviewed() -> None:
    result = apply_decisions(
        ITEMS, {"g001": {"status": "reject"}, "g002": {"status": "accept"}}, "J"
    )
    assert [i["id"] for i in result.items] == ["g002", "g003", "g004"]
    assert result.rejected == ["g001"] and result.pending == ["g003", "g004"]
    assert result.items[1]["metadata"]["reviewed"] is False


def test_unknown_item_or_status_is_an_error() -> None:
    with pytest.raises(ValueError, match="g999"):
        apply_decisions(ITEMS, {"g999": {"status": "accept"}}, "J")
    with pytest.raises(ValueError, match="status"):
        apply_decisions(ITEMS, {"g001": {"status": "maybe"}}, "J")


def test_inputs_are_not_mutated() -> None:
    before = json.dumps(ITEMS)
    apply_decisions(ITEMS, {"g001": {"status": "edit", "question": "X?"}}, "J")
    assert json.dumps(ITEMS) == before


def test_page_contains_items_and_gold_text_and_expands_split_articles() -> None:
    page = render_review_page([make(1, gold_passage_ids=["CT-40"])], CHUNKS)
    assert "Pergunta 1?" in page and "Primeira parte." in page and "Segunda parte." in page


def test_page_escapes_script_end_tags_in_item_text() -> None:
    page = render_review_page([make(1, question="x </script><b>boom</b>")], CHUNKS)
    assert "</script><b>boom" not in page
    assert page.count("</script>") == 3  # data, core, app


def test_page_makes_no_network_requests() -> None:
    page = render_review_page([make(1)], CHUNKS)
    assert "http://" not in page and "https://" not in page and "fetch(" not in page


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_core_js_counts_and_navigation() -> None:
    script = (
        CORE_JS
        + """
const items = [{id:'a',question:'q',reference_answer:'r'},{id:'b',question:'q',reference_answer:'r'},
               {id:'c',question:'q',reference_answer:'r'}];
const d = {a:{status:'accept'}, c:{status:'reject'}};
console.log(JSON.stringify({
  s: summarize(items, d),
  next: nextPending(items, d, -1),
  wrap: nextPending(items, {b:{status:'edit'}}, 0),
  none: nextPending(items, {a:{status:'accept'},b:{status:'accept'},c:{status:'accept'}}, 0),
  eff: effective(items[0], {status:'edit', question:'novo', reference_answer:'', comment:'c'}),
  effNone: effective(items[1], undefined),
}));
"""
    )
    out = json.loads(
        subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    )
    assert out["s"] == {"total": 3, "accepted": 1, "edited": 0, "rejected": 1, "pending": 1}
    assert out["next"] == 1
    assert out["wrap"] == 2  # next pending after index 0, skipping the decided 'b'
    assert out["none"] == -1
    assert out["eff"]["question"] == "novo" and out["eff"]["reference_answer"] == "r"
    assert out["effNone"] == {"question": "q", "reference_answer": "r", "comment": ""}


def test_cli_build_then_apply_round_trip(tmp_path) -> None:
    from ptlegal_eval.chunks import write_chunks
    from ptlegal_eval.review import main

    drafts, chunks = tmp_path / "drafts.jsonl", tmp_path / "chunks.jsonl"
    drafts.write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in ITEMS), encoding="utf-8"
    )
    write_chunks(CHUNKS, chunks)
    page = tmp_path / "out" / "review.html"
    main(["build", str(drafts), "--chunks", str(chunks), "--out", str(page)])
    assert "Pergunta 3?" in page.read_text(encoding="utf-8")

    decisions = tmp_path / "review_decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "exported_at": "x",
                "decisions": {"g001": {"status": "accept"}, "g002": {"status": "reject"}},
            }
        ),
        encoding="utf-8",
    )
    reviewed = tmp_path / "reviewed.jsonl"
    main(["apply", str(drafts), str(decisions), "--reviewer", "João", "--out", str(reviewed)])
    kept = [json.loads(x) for x in reviewed.read_text(encoding="utf-8").splitlines()]
    assert [k["id"] for k in kept] == ["g001", "g003", "g004"]
    assert kept[0]["metadata"]["reviewed_by"] == "João" and kept[1]["metadata"]["reviewed"] is False


def test_storage_key_differs_between_item_sets() -> None:
    def key(items):
        page = render_review_page(items, CHUNKS)
        return re.search(r'"key": "([^"]+)"', page).group(1)

    assert key([make(1), make(2)]) != key([make(3), make(4)])
    assert key([make(1), make(2)]) == key([make(1), make(2)])


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_export_only_contains_decisions_for_items_on_the_page() -> None:
    script = (
        CORE_JS
        + """
const items = [{id:'a'},{id:'b'}];
const d = {a:{status:'accept'}, b:{status:'maybe'}, zzz:{status:'accept'}};
console.log(JSON.stringify(exportable(items, d)));
"""
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == {"a": {"status": "accept"}}
