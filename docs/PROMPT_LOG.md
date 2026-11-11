# Prompt log

Every version of the prompt, in order (pre-registration, section 7). The prompt text lives in
`src/ptlegal_eval/prompts.py`; this log records each version's hashes, why it exists, and what it was
run on. `tests/test_prompts.py` fails if the prompt in code has no entry here.

Rules: tuned on **dev only**, at most 5 versions, every version run on **all four** models, the
selected version has the highest mean Task Success across the four models, ties go to the shorter
prompt.

## v0 (2026-11-11): first draft, not tuned

- Written once by Claude, in European Portuguese, before any model was run.
- Used by the step 6 smoke test (5 dev items per model). Not a result.
- System prompt, open book (conditions A and B), SHA-256 `e3c5bb6f4a16a3f8d7efd9ab95ea29760860fa4a9fe68b58b4e90da4264e64f2`.
- System prompt, closed book (condition C), SHA-256 `a6a3abe82e25de0ea524bb2a3b513b835763b3a42a0b4f52625d5c1eca8b57f7`.
- User message: the articles as `[Artigo N.º — epígrafe]` blocks in article-number order (condition
  A) or retrieval rank order (condition B), then `Pergunta: ...`. Closed book: `Pergunta: ...` only.
- Settings shared by every model: the model's own chat template, `enable_thinking=False`, greedy
  decoding, `max_new_tokens=400`, FP16.
