# Runs

Raw model outputs and run metadata, kept so every number in the report can be recomputed without a
GPU. Rendered prompts are not stored: they contain the law text (see `docs/CORPUS.md`).

## `smoke_fp16_2026-11-22/` — step 6 smoke test, FP16, prompt v0 (not a result)

5 dev items per model, condition A, code at commit `f5ef6d3`, Kaggle T4 x2, torch 2.11.0,
transformers 5.24.0. One unanswerable item used a BM25 stand-in context (smoke only).

| Model | Loader | Peak GiB (GPU 0/1) | Valid | Notes |
|---|---|---|---|---|
| AMALIA | causal_lm | 9.74 / 9.73 | 0/5 | **non-finite logits in FP16**: every generated token id is 0, answers empty |
| EuroLLM | causal_lm | 9.73 / 9.73 | 5/5 | |
| Gemma 4 12B | multimodal_lm | 13.03 / 10.38 | 5/5 | |
| Qwen3.5-9B | multimodal_lm | 10.20 / 9.36 | 5/5 | |

GPU memory after unloading each model: 0.01 GiB (an earlier version leaked 8.5 GiB and made the
next three models fail to load; fixed in `f5ef6d3`). See `docs/DEVIATIONS.md` for what this means for
the pre-registered FP16 policy.

## `probe_2026-11-25/` — AMALIA in FP16 and BF16 (diagnostic, not a result)

Transcribed summary of the precision probe: where AMALIA overflows in FP16 and that it is finite in
BF16. See `docs/DEVIATIONS.md`, section 3.

## `smoke_bf16_2026-11-25/` — step 6 smoke test, BF16, prompt v0 (not a result)

5 dev items per model plus one worst-case prompt at the 12,000-character budget (about 3,100 to 3,200
tokens), condition A, code at commit `72d4271`, Kaggle T4 x2, torch 2.11.0, transformers 5.24.1.

| Model | Finite | Valid | Peak GiB (GPU 0/1) | Seconds on the worst-case prompt |
|---|---|---|---|---|
| AMALIA | yes | 6/6 | 12.13 / 12.14 | 33 (19 new tokens) |
| EuroLLM | yes | 6/6 | 12.13 / 12.13 | 46 (100 new tokens) |
| Gemma 4 12B | yes | 5/6 | 14.35 / 11.01 | **out of memory** |
| Qwen3.5-9B | yes | 6/6 | 11.71 / 10.86 | 26 (26 new tokens) |

Gemma's peak includes the failed attempt. It needed 154 MiB with 83 MiB free on GPU 0. GPU memory
after unloading each model: 0.01 GiB.

## `memladder_2026-11-25/` — which context budget can Gemma run in BF16? (diagnostic, not a result)

Gemma 4 12B loaded as in the study (code at commit `6e9f497`), longest articles at each budget, 400
new tokens forced, rising budgets, stop at first failure. `ladder.json` holds the full record.
