import sys

import pytest

from ptlegal_eval.embedders import BGE_M3_REVISION, BgeM3Embedder


def test_missing_optional_dependency_gives_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes the import fail
    with pytest.raises(ImportError, match="uv sync --extra retrieval"):
        BgeM3Embedder()


def test_revision_matches_the_preregistration() -> None:
    from pathlib import Path

    prereg = (Path(__file__).resolve().parents[1] / "docs" / "PREREGISTRATION.md").read_text(
        encoding="utf-8"
    )
    assert BGE_M3_REVISION in prereg
