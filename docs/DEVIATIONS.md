# Deviations from the pre-registration

`docs/PREREGISTRATION.md` is frozen. Anything done differently is logged here, with the date, and
must be reported in the "Deviations" section of `REPORT.md`.

## 1. Golden-set batch 2 was bulk-approved (2026-11-12)

- **Pre-registration (section 4):** Claude drafts items and João "reviews every question and
  answer".
- **What happened:** João reviewed batch 1 (10 items, all accepted unchanged). For batch 2 (35 items,
  `g011` to `g045`) João instructed "approve all of them" without reviewing the items one by one.
  Each of the 35 items carries `metadata.review_comment` saying so; `metadata.reviewed_by` records
  who approved, not that they read the item.
- **Effect on the study:** these items have had no human check for wrong gold articles, wrong
  reference answers or unfair questions, beyond the mechanical checks (gold ids exist, no suspended
  articles, context budget, category rules, duplicates) and the drafter's own verification against
  the article text. Errors in them would show up as noise in every model's score, and could go
  unnoticed.
- **Items flagged by the drafter as arguable:** `g033` (public-sector holidays: other legislation
  may apply parts of the Código to public workers).
- **Mitigation, if any:** none yet. A later item-by-item review of batch 2, or a spot check, would
  reduce the risk and should be recorded here.
