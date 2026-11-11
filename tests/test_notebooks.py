"""Notebooks are thin wrappers around repo code: no outputs, no logic."""

import json
from pathlib import Path

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def test_there_is_at_least_the_smoke_notebook() -> None:
    assert any(p.name == "step6_smoke.ipynb" for p in NOTEBOOKS)


def test_notebooks_are_valid_clean_and_thin() -> None:
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert all(c["outputs"] == [] and c["execution_count"] is None for c in code), path.name
        lines = [
            x for c in code for x in c["source"] if x.strip() and not x.lstrip().startswith("#")
        ]
        assert len(lines) <= 30, (
            f"{path.name} has {len(lines)} code lines: move logic into the package"
        )
        assert any("ptlegal_eval" in x for x in lines), path.name


def test_the_smoke_notebook_clones_the_step_6_branch_and_never_embeds_secrets() -> None:
    text = (NOTEBOOKS[0].parent / "step6_smoke.ipynb").read_text(encoding="utf-8")
    assert "step-6-runner" in text and "ptlegal_eval.smoke run" in text
    assert "hf_" not in text and "token" not in text.lower()
