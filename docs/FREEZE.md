# Freeze log

Append-only. Each entry records something fixed **before** any model was run on a test-set question.
Per `docs/PREREGISTRATION.md`, nothing is tuned on the test set. A change to anything recorded here
after its entry is a deviation and goes in `docs/DEVIATIONS.md`.

## 1. Golden set split and test set frozen (2026-11-11)

| | |
|---|---|
| Source | the 150 items in `data/golden/reviewed_batch1.jsonl` to `reviewed_batch5.jsonl`, all passing `python -m ptlegal_eval.golden_check` |
| Rule | stratified by category: 6 dev and 24 test per category (30 dev, 120 test) |
| Seed | 31415, arbitrary, chosen before the split was run, run once |
| Command | `python -m ptlegal_eval.split data/golden/reviewed_batch*.jsonl --seed 31415` (refuses to run again if `test.jsonl` exists) |
| Test set | `data/golden/test.jsonl`, 120 items, SHA-256 `bce45e636e50f06ba28137040c8dc0ecf37531063de4f6de6485a64ccbaa3987` |
| Dev set | `data/golden/dev.jsonl`, 30 items, SHA-256 `f33bd270e0f3bea573849eeb66f58e11cc4b411045bdb1ba31bdfd4b6da12ae5` |
| Guard | `tests/test_freeze.py` fails if either file no longer matches its hash |

Per split: 5 categories x (6 dev, 24 test). Behaviours: dev 18 answer, 6 refuse, 6 correct; test
72 answer, 24 refuse, 24 correct. Gold ids in these files are chunk-level (an article split in two
chunks lists both parts); the article labels are in `metadata.gold_articles`.

**Review status of the test set.** Of the 120 test items, 20 were individually read by João (items
from batch 1 and from the 15-item spot check); the other 100 were bulk-approved without being read.
See `docs/DEVIATIONS.md`, section 1, for the consequences and the spot-check result.

**Known weaknesses of the split.** 5 of the 30 dev items share a gold article with a test item, so
prompts tuned on dev have seen some of the same articles; the effect is expected to be small. The
split is random, so the test set contains whichever of the weaker items fell into it, including any
`g033`-style arguable item.

## Still to record before the first test-set run

Appended here as separate dated entries, not edited into the entry above:

- the frozen pt-BR lexicon and its hash (pre-registration, section 9),
- the final judge `reasoning_effort` and prompt version (section 8),
- the refusal patterns, frozen after dev (section 8),
- the context budget, if the step 6 smoke test forces a change from 12,000 characters (section 6).
