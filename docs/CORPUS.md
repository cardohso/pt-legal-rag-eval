# Corpus

**Código do Trabalho (Lei n.º 7/2009), consolidated text.** Public data only.

## Provenance

| | |
|---|---|
| Source | [diariodarepublica.pt, versão consolidada](https://diariodarepublica.pt/dr/legislacao-consolidada/lei/2009-34546475) (INCM) |
| Version shown | "Versão à data de" 2026-11-09, "Mostrar revogado" off |
| Saved | 2026-11-09, from the rendered page (the site needs a browser; plain HTTP returns an empty shell) |
| File | `data/raw/ct_consolidado_2026-11-09.txt`, 699,183 bytes, UTF-8, **local only** (git-ignored) |
| SHA-256 | `943a79e250b28e42c33c8707b1db88615cb69f14bee173235afd251bb784d7d7` |
| Latest amending act in the text | Lei n.º 32/2025, de 27 de março, which adds art. `252.º-B` (in force 2025-04-26) |

This snapshot is the corpus, even if the law changes later. The page itself says the consolidated
version "não tem valor legal e não substitui a consulta dos atos que deram origem a esta
consolidação". This project is not legal advice.

## Parsing (`src/ptlegal_eval/ingest.py`)

One chunk per article of the Código (the 14 articles of the approving law are excluded). Each chunk
keeps article number, epigraph and section path (Livro > Título > Capítulo > Secção > Subsecção >
Divisão). Articles over 4,000 characters are split at numbered items (`n - `), never inside one.

| Count | |
|---|---|
| Article headings in the snapshot | 601 (566 numbers plus 35 lettered, such as `10.º-A`) |
| Fully revoked, no legal text left, **not chunked** | 2: `208.º-A`, `366.º-A` |
| Articles chunked | 599 |
| Chunks (4 articles split in two) | 603; median 743 characters, longest 3,960 |
| Suspended by a later act, **text kept, flagged** | 2: `263.º` (Subsídio de Natal), `264.º` (Retribuição do período de férias e subsídio); both suspended from 2013-01-01 by Lei n.º 11/2013 |

Removed as page furniture: amendment notes ("Alterado/Aditado/Retificado/REVOGADO/Suspenso pelo/a
…"), "Ver alterações ao texto", and "Notas" blocks. "(Revogado)" paragraphs inside an article are
law text and stay.

`data/corpus_manifest.json` holds the provenance above plus a SHA-256 per chunk and no law text, so
anyone can check their own copy of the source against it.

## Reuse terms (open question for the author)

- The [Avisos Legais](https://diariodarepublica.pt/dr/geral/avisos-legais) grant "acesso universal e
  gratuito", including "impressão, arquivo, pesquisa e livre acesso ao conteúdo dos atos publicados".
  They grant no explicit reuse licence, and the site footer reads "INCM, SA - TODOS OS DIREITOS
  RESERVADOS".
- [CDADC art. 8.º](https://diariodarepublica.pt/dr/legislacao-consolidada/decreto-lei/1985-34475475)
  says compiled or annotated official texts "não beneficiam de protecção". This reading is the
  author's, not a lawyer's.
- **Current policy:** the law text is not committed. The raw file and parsed chunks are git-ignored;
  only the manifest is committed. Reproducing the pipeline needs the reader to save the same page and
  match the hashes. Revisit before publishing the report.

## Notes for later steps

- The text spells many numbers as words ("cento e setenta e cinco horas por ano", e.g. in art. 228.º).
  The number-fidelity check must understand Portuguese number words, not only digits.
- The text uses pre-1990-agreement spelling ("colectiva", "protecção"), with some modern forms
  ("coletiva") mixed in. These must never count as pt-BR markers (pre-registration, section 9).
- Question writers should avoid `263.º` and `264.º` (suspended).
- Only one act post-dates AMALIA's June 2024 knowledge cutoff (Lei n.º 32/2025, one new article,
  `252.º-B`), which confirms dropping the "amended after cutoff" category.
