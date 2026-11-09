# pt-legal-rag-eval

[![CI](https://github.com/cardohso/pt-legal-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/cardohso/pt-legal-rag-eval/actions/workflows/ci.yml)

Independent evaluation of small open models for RAG over Portuguese labour law: which one answers
most faithfully, cites correctly and refuses when it should, given the same retrieved text?

**Status: in progress. No results yet.** The design was pre-registered before any model was run:
[docs/PREREGISTRATION.md](docs/PREREGISTRATION.md).

**Not legal advice.** The author is not affiliated with the AMALIA project, and this is neither an
attack on it nor promotion of it. Corpus: Código do Trabalho (Lei n.º 7/2009), consolidated text from
Diário da República, which states that the consolidated version has no legal value.

## Compared systems

`amalia-llm/AMALIA-9B-0626-DPO`, `utter-project/EuroLLM-9B-Instruct-2512`,
`google/gemma-4-12B-it`, `Qwen/Qwen3.5-9B`. Exact revisions are in the pre-registration.

## Develop

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

Tests run offline with no GPU and no API keys. Secrets go in `.env` (see `.env.example`), never in
git. `scripts/secret_scan.sh` scans history and files with gitleaks.

## Licences

Code: MIT. Each model keeps its own licence (all four are Apache 2.0). The corpus keeps its source's
terms; see `docs/CORPUS.md` once added. No model weights are stored here.
