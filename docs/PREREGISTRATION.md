# Pre-registration: small open models for RAG over Portuguese labour law

**Author:** João Pedro Cardoso. **Registered:** the date of the first commit that contains this file (`git log --follow docs/PREREGISTRATION.md`).

Independent work, not affiliated with the AMALIA project. Not legal advice. This file is not edited
after its commit. Any deviation goes in the "Deviations" section of `REPORT.md`.

## 1. Question

When given the same retrieved Portuguese legal text, which small open model answers most faithfully,
cites correctly and refuses when it should?

Any result is acceptable. No hypothesis below predicts a direction.

## 2. Hypotheses

All are two-sided, on the primary metric, in condition A.

- **H1:** AMALIA and EuroLLM differ.
- **H2:** AMALIA and Gemma differ.
- **H3:** AMALIA and Qwen differ.

These three are the primary comparisons. The other three model pairs are reported as secondary,
with confidence intervals and no significance claims.

## 3. Systems

| Name | Hugging Face ID | Revision | Parameters |
|---|---|---|---|
| AMALIA | `amalia-llm/AMALIA-9B-0626-DPO` | `c4614b9b5c7b4fe303fa20902e692fc7dface640` | 9.15B |
| EuroLLM | `utter-project/EuroLLM-9B-Instruct-2512` | `def82454026b0353f12e48a0de081a9ed380d91e` | 9.15B |
| Gemma | `google/gemma-4-12B-it` | `43c9d014b231474fa7212dbb48a762566c2b255d` | 11.96B |
| Qwen | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 9.65B |

Known differences, to be stated in the report:

- Gemma is 31% larger than AMALIA. Qwen's count includes a vision encoder.
- EuroLLM has no DPO step, and the AMALIA card does not name its exact parent checkpoint.
  AMALIA vs EuroLLM therefore measures Portuguese data and DPO together.
- AMALIA's training data includes outputs from Gemma 3 and Qwen 3 models (technical report).

Identical for all systems:

- Kaggle T4 x2, FP16, no quantisation, `device_map="auto"`, `max_memory` 13 GiB per GPU, batch size 1.
- `transformers` 5.22.2 and the torch version of the pinned Kaggle image, both recorded per run.
- Each model's own chat template, with the same system and user messages.
- Thinking modes off (`enable_thinking=False` where the template supports it).
- `do_sample=False`, no repetition penalty, `max_new_tokens=400`. Greedy decoding, so one run per setting.

If any model runs out of memory, all four move to the same 8-bit setup and the change is a deviation.

## 4. Data

**Corpus.** Código do Trabalho (Lei n.º 7/2009), consolidated text from Diário da República. Source
URL, version date and SHA-256 are recorded in `docs/CORPUS.md` when the snapshot is taken. That
snapshot is the corpus even if the law changes later. One chunk per article. Articles over 4,000
characters are split at "n.º" boundaries, keeping article number, epigraph and section.

**Golden set.** 150 items in European Portuguese: 30 per category, split at random within category
into 30 dev items (6 per category) and 120 test items (24 per category).

| Category | Expected behaviour |
|---|---|
| Single-article factual | answer |
| Multi-article | answer |
| Numeric (deadlines, days, percentages) | answer |
| Unanswerable from the corpus | refuse |
| False premise | correct or refuse |

An "article amended after cutoff" category is dropped. The only change after AMALIA's June 2024
cutoff is one article added by Lei n.º 32/2025, which is too few.

Each item has: question, gold article(s), reference answer, expected behaviour, category. Claude
(Anthropic, not a compared family) may draft items. João reviews every question and answer. No
compared model sees any item before the freeze. The SHA-256 of the test file is committed in
`docs/FREEZE.md` before the first test-set run. Nothing is tuned on test.

## 5. Primary metric

**Task Success (TS)**, binary per question, in condition A.

- **Answerable items:** TS = 1 if the answer is not a refusal, the judge marks it correct against the
  reference, the judge finds no unsupported claim, and it cites at least one article with every cited
  article present in the supplied context.
- **Unanswerable items:** TS = 1 if the model declines.
- **False-premise items:** TS = 1 if the model explicitly corrects the premise or declines.

Invalid or failed outputs score TS = 0.

**Smallest difference that matters:** 15 percentage points.

Exact McNemar power for one comparison at n = 120 and alpha = 0.0167 (Holm's first step):

| Discordant pairs | 10 points | 12.5 points | 15 points |
|---|---|---|---|
| 15% | 0.63 | 0.91 | 1.00 |
| 25% | 0.37 | 0.60 | 0.81 |
| 35% | 0.25 | 0.43 | 0.62 |

Smaller real differences may go undetected. Per-category results (n = 24) are descriptive only.

## 6. Conditions

- **A (primary): gold context.** The gold articles, in article-number order. Unanswerable items have
  no gold article, so they get the top passages from the condition B retriever, identical for all models.
- **B (secondary): real retrieval.** Top 5 passages from a hybrid of BM25 and `BAAI/bge-m3`
  (revision `5617a9f61b028005a4858fdac845db406aefb181`), combined by reciprocal rank fusion with
  k = 60. Retrieval runs once and its output is reused for every model. Retrieval quality is reported
  separately (recall@k, MRR, hit rate).
- **C (secondary): closed book.** No context. Shows reliance on memory. Correctness and refusal only.

Context budget: 12,000 characters in every condition. In B, the lowest-ranked whole passages are
dropped until the budget fits. Items whose gold articles exceed the budget are removed before the
freeze. The budget was chosen from one Gemma memory test at about 3,000 prompt tokens. If the step 6
smoke test shows it doesn't fit, it is lowered for all models and recorded in `docs/FREEZE.md` before
the first test-set run.

## 7. Prompts

One shared prompt, in European Portuguese, for all four models. At most 5 versions, tuned on dev only,
in condition A. Every version runs on all four models. The selected version has the highest mean TS
across the four models, and any tie goes to the shorter prompt. Every version is logged.

## 8. Judging

**Deterministic checks run first:** citation existence, numbers absent from the context, and refusal
patterns. The refusal patterns are frozen after dev.

**Judge:** `openai/gpt-oss-120b`, on Groq's free tier. Temperature 0, `reasoning_effort` low, one host
for all test-set judging, exact ID and host recorded per call. `reasoning_effort` moves to medium only
if the dev audit below misses more than 20% of planted errors, and the final value is recorded in
`docs/FREEZE.md` before the first test-set run. OpenAI is not a compared family, and the AMALIA report does not name an OpenAI model as a data
generator, translator or judge. One caveat: its SFT data includes a WildChat subset, which comes from
ChatGPT user conversations, and the report doesn't say whether any ChatGPT responses were kept. A small
style overlap between AMALIA and this judge therefore can't be ruled out. Google, Alibaba and DeepSeek
models are excluded because they share a family with a compared model or appear in AMALIA's training
data.

The judge sees no model identity. Answers are shuffled across models before judging. It returns:

1. Atomic claims, each marked supported or unsupported by the context, with the articles that support it.
2. Correct or incorrect against the reference answer.
3. Declined or not, and premise corrected or not.
4. Portuguese variety: pt-PT, pt-BR, mixed or undetermined.

Judge prompts are versioned and tuned on dev answers only. A goldpass-style audit with planted errors,
written in Portuguese, runs on dev answers. Its miss rate is reported.

Judge failures are retried, then reported per model. They are excluded from judged metrics. A
sensitivity analysis counts them as failures.

**Human check.** João scores a random sample of 80 condition A items: 4 per model and category, drawn
with a fixed seed after predictions are frozen. Each item is scored for correctness, unsupported
claims, declined or corrected, and Portuguese variety, without seeing the model or the judge's verdict.

Agreement is Cohen's kappa per metric, with a bootstrap interval and raw agreement.

| Kappa | Treatment |
|---|---|
| 0.60 or above | Can be headlined |
| 0.40 to 0.59 | Reported with a warning, not headlined |
| Below 0.40 | Reported as unreliable, excluded from conclusions |

If correctness or faithfulness falls below 0.60, TS is not headlined, the report says which part of the
question cannot be answered, and the deterministic results lead. The primary metric is not replaced.

## 9. Secondary metrics

- Answer correctness; claim-level faithfulness
- Citation correctness: cited articles exist, and they support the claim
- Number fidelity: values not found in the context
- Refusal: correct-refusal rate and false-refusal rate, always reported as a pair
- European Portuguese quality
- Valid-output rate, latency, throughput, peak GPU memory

**European Portuguese quality** has two parts.

- A lexicon of pt-BR forms and syntactic markers. Every entry cites a published source, such as
  Barreiro & Wittmann's contrastive pairs or the FRMT dataset (CC BY-SA 3.0). It is built without
  looking at test-set outputs, reviewed by João, and hashed in `docs/FREEZE.md` before the first
  test-set run. Reported as markers per 1,000 words and share of answers with at least one marker.
- The judged and human-rated variety labels from section 8.

Text copied from the supplied context is excluded. Pre-1990-agreement spellings (for example
"colectiva") never count as pt-BR.

## 10. Analysis

- 95% confidence intervals from a paired bootstrap across questions: 10,000 resamples, seed 20261109.
- Exact McNemar test for each binary paired outcome.
- Holm correction across H1 to H3 only.

Reading a primary comparison:

| Result | Wording |
|---|---|
| Holm-adjusted p < 0.05 | The models differ; direction and size reported |
| Not different, and the interval lies within ±15 points | Practically equivalent at this margin |
| Otherwise | Cannot be told apart at this sample size |

No overall ranking is claimed unless these comparisons support it. Everything outside H1 to H3 is
secondary. Every reported number comes from a repo script over saved predictions.

## 11. Limitations stated in advance

One legal domain, one reviewer and scorer, one hardware and precision setup, thinking modes off, a
small sample, model sizes that differ, a judge that may be weak on Portuguese, and possible style cues
that could weaken blinding.
