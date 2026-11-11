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
