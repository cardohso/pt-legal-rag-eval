# Deviations from the pre-registration

`docs/PREREGISTRATION.md` is frozen. Anything done differently is logged here, with the date, and
must be reported in the "Deviations" section of `REPORT.md`.

## 1. 140 of 150 golden-set items were bulk-approved (2026-11-11)

- **Pre-registration (section 4):** Claude drafts items and João "reviews every question and
  answer".
- **What happened:** João reviewed batch 1 (10 items, `g001` to `g010`, all accepted unchanged). For
  batch 2 (35 items, `g011` to `g045`) João instructed "approve all of them", and for batches 3 to 5
  (105 items, `g046` to `g150`) "draft them all and approve them all". None of these 140 items was
  read by João. Each carries `metadata.review_comment` saying so; `metadata.reviewed_by` records
  who approved, not that they read the item.
- **Effect on the study:** only 25 of 150 items (17 %) have had a human check (10 read in full, 15
  from a random spot check, see below). The other 125 may have a
  wrong gold article, a wrong reference answer, an ambiguous question, or an "unanswerable" item that
  a reasonable reader would consider answerable. An error scores against every model equally, so it
  adds noise to all comparisons rather than favouring one model, but it could go unnoticed and it
  weakens any claim that the reference answers are correct.
- **What was done instead (all mechanical, none a substitute for reading):**
  - Every reference answer was drafted from the article text read in full, not from memory (several
    memory errors were caught this way, for example collective-dismissal compensation is 14 days, not
    12).
  - `python -m ptlegal_eval.golden_check` checks gold ids, suspended articles, the 12,000-character
    context budget, category rules and duplicates, plus that every article, paragraph and number cited
    in a reference answer appears in its gold text (Portuguese number words handled). All 150 items pass.
  - Each "unanswerable" item names a topic whose search pattern has no hit in the corpus; this is
    asserted when the item is generated.
- **What those checks cannot catch:** a swap between two numbers inside the same article (tested: "12
  dias" for art. 366.º passes because "12 vezes" appears elsewhere in that article), a reference that
  is incomplete or phrased misleadingly, a question with two defensible answers, and a trap that is
  arguably answerable.
- **Items flagged by the drafter as arguable:** `g033` (public-sector holidays: other legislation may
  apply parts of the Código to public workers).
- **Spot check (2026-11-11):** before the freeze, 15 of the 140 bulk-approved items were drawn at
  random, stratified (3 per category), with seed 20261111, and committed before review
  (`data/golden/spotcheck_sample.jsonl`, commit `09431eb`). João reviewed each against its gold
  article text and accepted all 15 unchanged, with no comments: 0 errors found in 15. This does not
  show the set is error-free: the exact (Clopper-Pearson) 95 % upper bound on the error rate is
  21.8 %, so up to about 1 item in 5 of the unread items could still be wrong. Each spot-checked item
  carries `metadata.spot_checked`. The other 125 items remain unread.
- **Not covered by the sample:** `g033` (flagged by the drafter as arguable) was not drawn and has not
  been read.
- **Report wording:** say "no errors in a random sample of 15 of the 140 unread items", not "no
  errors".

## 2. Dates in earlier records (2026-11-11)

The batch 2 approval comments and this file first said 2026-11-12. The correct date is 2026-11-11;
all records were corrected before any further commit.

## 3. FP16 breaks AMALIA: the "FP16 for every model" policy is under review (opened 2026-11-22)

- **Pre-registration (section 3):** all four models on Kaggle T4 x2 in FP16, no quantisation.
- **What happened:** in the step 6 smoke test (`runs/smoke_fp16_2026-11-22/`, commit `f5ef6d3`) AMALIA
  produced non-finite logits in FP16 (`finite_logits: False`); every generated token id was 0 and all
  5 answers were empty. EuroLLM (same architecture and size, AMALIA's parent), Gemma 4 12B and
  Qwen3.5-9B were finite and gave valid answers in the same run. Chat templates were checked and are
  not the cause.
- **Status: open, not yet a deviation.** Nothing in the pre-registered policy has been changed. The
  options, to be chosen with evidence from `python -m ptlegal_eval.probe` (where the values overflow,
  and BF16 on a T4):
  1. a precision that works for all four models, if one does at acceptable speed (all four were
     trained in BF16, so BF16 for everyone would be defensible), recorded here as a deviation;
  2. a different precision for AMALIA only, recorded as a deviation, which weakens "same treatment";
  3. report that AMALIA could not be run in the pre-registered setup, if neither works.
- **Rule:** whatever is chosen is decided before any test-set run and logged here with the evidence.
- **Evidence so far (2026-11-23, Kaggle T4 x2, `python -m ptlegal_eval.gpucheck`):** a 4096 x 4096 matrix
  multiply runs at 24.4 TFLOPS in FP16, 4.2 in FP32 and 2.6 in BF16. BF16 is emulated on a T4, about
  9x slower than FP16 and slower than FP32. Estimated cost if BF16 were used for all four models:
  prompt processing is compute-bound, so a 3,000-token prompt would take roughly 20 s instead of
  about 2 s, adding on the order of 8 GPU hours over the full test run. This is an estimate from the
  benchmark, not a measured run. The probe that measures AMALIA itself in FP16 and BF16 failed on a
  bug of mine (CUDA was not initialised before a memory-stat call; fixed, with a test that
  reproduces it) and has not yet produced results.
