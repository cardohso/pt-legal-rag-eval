"""The pre-registration must not change after its first commit.

`docs/PREREGISTRATION.sha256` was written in the same commit as the file. Editing the file without
rewriting the hash fails this test; rewriting the hash shows up in git history. Deviations belong in
REPORT.md.
"""

import hashlib
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_preregistration_matches_recorded_hash() -> None:
    recorded = (DOCS / "PREREGISTRATION.sha256").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256((DOCS / "PREREGISTRATION.md").read_bytes()).hexdigest()
    assert actual == recorded, "PREREGISTRATION.md changed after it was registered"
