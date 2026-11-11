"""The frozen golden set must not change. See docs/FREEZE.md."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden"
FREEZE = (ROOT / "docs" / "FREEZE.md").read_text(encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_test_set_matches_its_recorded_hash() -> None:
    recorded = (GOLDEN / "test.sha256").read_text(encoding="utf-8").split()[0]
    assert sha(GOLDEN / "test.jsonl") == recorded, "the frozen test set was changed"


def test_dev_set_matches_its_recorded_hash() -> None:
    recorded = (GOLDEN / "dev.sha256").read_text(encoding="utf-8").split()[0]
    assert sha(GOLDEN / "dev.jsonl") == recorded, "the dev set was changed after the split"


def test_freeze_log_states_the_same_hashes() -> None:
    assert sha(GOLDEN / "test.jsonl") in FREEZE
    assert sha(GOLDEN / "dev.jsonl") in FREEZE
