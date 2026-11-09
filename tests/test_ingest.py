"""Parser tests on an invented text that mimics the structure of the Diário da República page.

Nothing here is real law. The structure (approving law, Anexo, Livro/Título/Capítulo headings,
amendment notes, "Notas" blocks) was observed on the real page.
"""

from pathlib import Path

import pytest

from ptlegal_eval.ingest import ParseReport, parse_code

FAKE = """Artigo 1.º
Aprovação do código fictício
Texto da lei de aprovação, que não faz parte do código.

Anexo
CÓDIGO DO TRABALHO
Livro I
Parte geral
Título I
Fontes
Capítulo I
Princípios
Artigo 1.º
Âmbito
O presente texto aplica-se aos trabalhadores.

Artigo 2.º
Prazos
1 - O prazo é de trinta dias.
2 - (Revogado).
Alterado pelo/a Artigo 2.º do/a Lei n.º 1/2020 - Diário da República n.º 1/2020, Série I de 2020-01-01, em vigor a partir de 2020-01-02
Ver alterações ao texto
Capítulo II
Outros
Artigo 2.º-A
Artigo aditado
Texto aditado.
Aditado pelo/a Artigo 3.º do/a Lei n.º 2/2021 - Diário da República n.º 2/2021, Série I de 2021-02-02
Notas
Artigo 9.º, Lei n.º 90/2019  - Diário da República n.º 169/2019, Série I de 2019-09-04 Nota fictícia.
Artigo 2.º-B
Artigo revogado
REVOGADO pelo/a Artigo 10.º do/a Lei n.º 3/2022 - Diário da República n.º 3/2022, Série I de 2022-03-03, em vigor a partir de 2022-03-04
Artigo 2.º-C
Artigo suspenso
Texto em suspenso.
Suspenso pelo/a Artigo 6.º do/a Lei n.º 4/2023 - Diário da República n.º 4/2023, Série I de 2023-04-04, em vigor a partir de 2023-04-05
Livro II
Penal
Capítulo I
Crimes
Artigo 3.º
Artigo longo
1 - {a}
2 - {b}
3 - {c}
"""
LONG = FAKE.format(a="a" * 400, b="b" * 400, c="c" * 400)


def chunks(max_chars: int = 4000):
    return {c.id: c for c in parse_code(LONG, max_chars=max_chars)}


def test_approving_law_is_excluded() -> None:
    texts = " ".join(c.text for c in chunks().values())
    assert "lei de aprovação" not in texts
    assert chunks()["CT-1"].epigraph == "Âmbito"


def test_ids_articles_and_epigraphs() -> None:
    c = chunks()
    assert list(c) == ["CT-1", "CT-2", "CT-2-A", "CT-2-C", "CT-3"]
    assert c["CT-2-A"].article == "2.º-A" and c["CT-2-A"].epigraph == "Artigo aditado"


def test_headings_do_not_leak_into_the_previous_article() -> None:
    c = chunks()
    assert "Capítulo II" not in c["CT-2"].text and "Outros" not in c["CT-2"].text
    assert "Livro II" not in c["CT-2-A"].text and "Penal" not in c["CT-2-A"].text


def test_section_is_the_current_heading_path() -> None:
    c = chunks()
    assert (
        c["CT-1"].section == "Livro I — Parte geral > Título I — Fontes > Capítulo I — Princípios"
    )
    assert c["CT-2-A"].section.endswith("Capítulo II — Outros")
    assert "Capítulo I —" not in c["CT-2-A"].section


def test_a_higher_heading_resets_the_lower_ones() -> None:
    section = chunks()["CT-3"].section
    assert section == "Livro II — Penal > Capítulo I — Crimes"


def test_amendment_notes_and_notas_blocks_are_removed() -> None:
    for c in chunks().values():
        assert "pelo/a" not in c.text
        assert "Ver alterações" not in c.text
        assert "Notas" not in c.text and "Nota fictícia" not in c.text


def test_revoked_paragraphs_stay_in_the_text() -> None:
    assert chunks()["CT-2"].text == "1 - O prazo é de trinta dias.\n2 - (Revogado)."


def test_long_article_is_split_at_numbered_items() -> None:
    c = chunks(max_chars=1000)
    assert [i for i in c if i.startswith("CT-3")] == ["CT-3#1", "CT-3#2"]
    assert all(c[i].article == "3.º" for i in ("CT-3#1", "CT-3#2"))
    assert c["CT-3#2"].text.startswith("3 - ")


def test_fully_revoked_article_is_skipped_and_reported() -> None:
    report = ParseReport()
    ids = [c.id for c in parse_code(LONG, report=report)]
    assert "CT-2-B" not in ids
    assert report.revoked == ["2.º-B"]


def test_suspended_article_keeps_its_text_and_is_reported() -> None:
    report = ParseReport()
    parsed = {c.id: c for c in parse_code(LONG, report=report)}
    assert parsed["CT-2-C"].text == "Texto em suspenso."
    assert report.suspended == ["2.º-C"]


def test_no_chunk_carries_source_citation_lines() -> None:
    for c in chunks().values():
        assert "pelo/a" not in c.text and " - Diário da República n.º" not in c.text


def test_missing_code_marker_is_an_error() -> None:
    with pytest.raises(ValueError, match="CÓDIGO DO TRABALHO"):
        parse_code("Artigo 1.º\nX\ntexto\n")


def test_duplicate_article_is_an_error() -> None:
    dup = LONG + "\nArtigo 3.º\nOutra vez\ntexto\n"
    with pytest.raises(ValueError, match="duplicate"):
        parse_code(dup)


def test_unrecognised_note_line_is_an_error() -> None:
    bad = LONG.replace(
        "Artigo 9.º, Lei n.º 90/2019  - Diário da República", "Algo inesperado sem fonte"
    )
    with pytest.raises(ValueError, match="Notas"):
        parse_code(bad)


REAL = Path(__file__).resolve().parents[1] / "data" / "raw" / "ct_consolidado_2026-11-09.txt"


@pytest.mark.skipif(not REAL.exists(), reason="real corpus snapshot is local only (data/raw/)")
def test_real_snapshot_structure() -> None:
    report = ParseReport()
    parsed = parse_code(REAL.read_text(encoding="utf-8"), report=report)
    articles = {c.article for c in parsed}
    # 601 article headings in the snapshot; 2 are fully revoked and carry no legal text
    assert report.revoked == ["208.º-A", "366.º-A"]
    assert report.suspended == ["263.º", "264.º"]
    assert len(articles) == 599
    assert len({c.id for c in parsed}) == len(parsed)
    for c in parsed:
        assert "pelo/a" not in c.text and " - Diário da República n.º" not in c.text
        assert "Ver alterações ao texto" not in c.text and "\nNotas" not in c.text
        assert c.text.strip()
    assert all(len(c.text) <= 4000 or "#" not in c.id for c in parsed)
    ct228 = next(c for c in parsed if c.id == "CT-228")
    assert "cento e setenta e cinco horas por ano" in ct228.text
    assert ct228.section.startswith("Livro I — Parte geral > Título II")
