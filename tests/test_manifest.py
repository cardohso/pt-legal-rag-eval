import json
from pathlib import Path

from ptlegal_eval.ingest import ParseReport, build_manifest, main, parse_code
from tests.test_ingest import LONG


def _manifest():
    report = ParseReport()
    chunks = parse_code(LONG, report=report)
    raw = LONG.encode("utf-8")
    return build_manifest(
        raw,
        chunks,
        report,
        source_url="https://example.test/ct",
        version_date="2026-11-09",
        retrieved="2026-11-09",
    )


def test_manifest_counts_and_flags() -> None:
    m = _manifest()
    assert m["n_chunks"] == 5 and m["n_articles"] == 5
    assert m["revoked_articles"] == ["2.º-B"] and m["suspended_articles"] == ["2.º-C"]
    assert m["version_date"] == "2026-11-09" and len(m["raw_sha256"]) == 64


def test_manifest_holds_hashes_not_text() -> None:
    dumped = json.dumps(_manifest(), ensure_ascii=False)
    assert "trinta dias" not in dumped and "Texto aditado" not in dumped
    assert set(_manifest()["chunk_sha256"]) == {"CT-1", "CT-2", "CT-2-A", "CT-2-C", "CT-3"}


def test_manifest_is_deterministic() -> None:
    assert _manifest() == _manifest()


def test_cli_writes_chunks_and_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text(LONG, encoding="utf-8")
    main(
        [
            str(raw),
            "--source-url", "https://example.test/ct",
            "--version-date", "2026-11-09",
            "--retrieved", "2026-11-09",
            "--chunks-out", str(tmp_path / "chunks.jsonl"),
            "--manifest-out", str(tmp_path / "manifest.json"),
        ]
    )  # fmt: skip
    assert (tmp_path / "chunks.jsonl").read_text(encoding="utf-8").count("\n") == 5
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["n_chunks"] == 5
