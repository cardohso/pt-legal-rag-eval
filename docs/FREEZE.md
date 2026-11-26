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

## 2. Context budget lowered to 7,500 characters (2026-11-25)

Recorded before any test-set run, as pre-registration section 6 provides: "If the step 6 smoke test
shows it doesn't fit, it is lowered for all models and recorded in `docs/FREEZE.md` before the first
test-set run." The pre-registered budget was 12,000.

| | |
|---|---|
| New budget | 7,500 characters of article text, **all four models, every condition** |
| Why | in BF16, Gemma 4 12B ran out of memory on a prompt at 12,000 characters, and again at 10,000 in the memory ladder (`runs/memladder_2026-11-25/`) |
| Margin | at 7,500 the ladder left 0.78 GiB free on the tighter GPU; 8,000 and 9,000 also ran, with 0.27 and 0.33 GiB, margins within the measurement noise |
| Effect on condition A (primary) | none: the largest gold context is 7,151 characters (a dev item; 4,600 in test), so no item is removed or trimmed |
| Effect on condition B | lower-ranked passages are dropped more often; a BM25-only stand-in trims 20 of 150 questions (mean 4.81 of 5 passages kept). The real hybrid retriever has not run |
| Chosen by | the author, choosing the larger memory margin over keeping more condition B passages |
| Enforced | `CONTEXT_BUDGET` in `src/ptlegal_eval/golden_check.py`; `tests/test_freeze.py` |

A budget confirmation smoke run for all four models is still to be done.

## 3. Precision and context budget confirmed (2026-11-26)

The confirmation smoke run for all four models passed (`runs/smoke_bf16_7500_2026-11-25/`, run
2026-11-25, commit `c7ab250`): finite output in BF16, 6/6 valid answers per model, no errors, and the
worst-case prompt at the 7,500-character budget fits. This ends the "provisional" status of the BF16
decision (`docs/DEVIATIONS.md`, section 3) and of entry 2. Recorded before any test-set run.

**Residual risk.** Gemma peaked at 14.07 of 14.56 GiB on GPU 0, 0.49 GiB spare. That is less than the
0.78 GiB of the clean memory test, because other prompts had been processed first. The smoke answers
were short, so the longest prompt, a full 400-token answer and hundreds of earlier prompts have not
been tested together. If Gemma runs out of memory in a test run, those failures are counted and
reported separately, not hidden in its scores.
