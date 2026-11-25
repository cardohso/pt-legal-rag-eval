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
