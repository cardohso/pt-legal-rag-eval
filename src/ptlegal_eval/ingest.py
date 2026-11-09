"""Turn the saved consolidated text of the Código do Trabalho into one chunk per article.

Input is the page text as saved from diariodarepublica.pt. Its layout (observed, not documented):

    <approving law: Artigo 1.º ... Artigo 14.º>
    Anexo
    CÓDIGO DO TRABALHO
    Livro I / Parte geral / Título I / <title> / Capítulo I / <title> / Artigo 1.º / <epigraph> ...

Headings sit between articles (sometimes straight after an amendment note), so an article's body
ends at the next heading, not only at the next article. Amendment notes ("Alterado pelo/a ...",
"Aditado pelo/a ...", "Retificado pelo/a ...", "Ver alterações ao texto") and "Notas" blocks are
page furniture and are dropped. "(Revogado)" paragraphs are part of the law's text and stay.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ptlegal_eval.chunks import Chunk, split_long_text

MARKER = "CÓDIGO DO TRABALHO"
LEVELS = ("Livro", "Título", "Capítulo", "Secção", "Subsecção", "Divisão")
_HEADING = re.compile(
    r"^(Livro|Título|Capítulo|Secção|Subsecção|Divisão)\s+([IVXLCDM]+|[Úú]nic[oa])(-[A-Z])?$"
)
_ARTICLE = re.compile(r"^Artigo (\d+)\.º(-[A-Z]+)?$")
_NOTE_SOURCE = " - Diário da República n.º"
# "Alterado pelo/a ...", "Aditado ...", "Retificado ...", "REVOGADO ...", "Suspenso ...": any
# first word, then "pelo/a <act> - Diário da República n.º ...". Legal text never looks like this.
_ANNOTATION = re.compile(
    r"^\S+ pelo/a .+" + re.escape(_NOTE_SOURCE) + r"|^Ver alterações ao texto$"
)
_SUSPENDED = re.compile(r"^Suspenso pelo/a ")
_REVOKED = re.compile(r"^REVOGADO pelo/a ")


@dataclass
class ParseReport:
    """What the parser left out or flagged, by article label ("208.º-A")."""

    revoked: list[str] = field(default_factory=list)  # no legal text left: not chunked
    suspended: list[str] = field(default_factory=list)  # text kept, but suspended by a later act


def parse_code(raw: str, max_chars: int = 4000, report: ParseReport | None = None) -> list[Chunk]:
    """One chunk per article of the Código do Trabalho; articles over `max_chars` are split.

    Articles whose whole text was revoked have nothing to answer from and are skipped. Pass a
    `ParseReport` to learn which, and which articles are suspended.
    """
    lines = raw.split("\n")
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == MARKER)
    except StopIteration:
        raise ValueError(
            f"marker line {MARKER!r} not found: is this the consolidated text?"
        ) from None

    levels: dict[str, str] = {}
    chunks: list[Chunk] = []
    seen: set[str] = set()
    current: dict[str, object] | None = None

    def close() -> None:
        nonlocal current
        if current is None:
            return
        body = "\n".join(current["body"])  # type: ignore[arg-type]
        number, suffix = current["number"], current["suffix"]
        base = f"CT-{number}{('-' + suffix) if suffix else ''}"
        if base in seen:
            raise ValueError(f"duplicate article {base}")
        seen.add(base)
        if report is not None and current["suspended"]:
            report.suspended.append(str(current["label"]))
        if not body.strip():
            if report is not None and current["revoked"]:
                report.revoked.append(str(current["label"]))
            current = None
            return
        parts = split_long_text(body, max_chars)
        for k, part in enumerate(parts, start=1):
            cid = base if len(parts) == 1 else f"{base}#{k}"
            chunks.append(
                Chunk(
                    cid,
                    str(current["label"]),
                    str(current["epigraph"]),
                    str(current["section"]),
                    part,
                )
            )
        current = None

    rest = lines[start + 1 :]
    i = 0
    while i < len(rest):
        line = rest[i]
        if (m := _HEADING.match(line)) and i + 1 < len(rest):
            close()
            level = m.group(1)
            levels[level] = f"{line} — {rest[i + 1]}"
            for deeper in LEVELS[LEVELS.index(level) + 1 :]:
                levels.pop(deeper, None)
            i += 2
        elif m := _ARTICLE.match(line):
            close()
            suffix = (m.group(2) or "")[1:]
            current = {
                "number": m.group(1),
                "suffix": suffix,
                "label": f"{m.group(1)}.º" + (f"-{suffix}" if suffix else ""),
                "epigraph": rest[i + 1] if i + 1 < len(rest) else "",
                "section": " > ".join(levels[k] for k in LEVELS if k in levels),
                "body": [],
                "suspended": False,
                "revoked": False,
            }
            i += 2
        elif line == "Notas":
            if i + 1 >= len(rest) or _NOTE_SOURCE not in rest[i + 1]:
                raise ValueError(f"unrecognised 'Notas' block at line {start + 2 + i}")
            i += 2
        elif _ANNOTATION.match(line) or not line.strip():
            if current is not None:
                current["suspended"] = current["suspended"] or bool(_SUSPENDED.match(line))
                current["revoked"] = current["revoked"] or bool(_REVOKED.match(line))
            i += 1
        else:
            if current is not None:
                current["body"].append(line)  # type: ignore[union-attr]
            i += 1
    close()
    return chunks


def sha256_hex(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def article_hashes(chunks: Sequence[Chunk]) -> dict[str, str]:
    """SHA-256 of each chunk's text, so anyone can check their own copy without us shipping it."""
    return {c.id: sha256_hex(c.text) for c in chunks}


def build_manifest(
    raw: bytes,
    chunks: Sequence[Chunk],
    report: ParseReport,
    source_url: str,
    version_date: str,
    retrieved: str,
) -> dict[str, object]:
    """Provenance and per-chunk hashes. Holds no legal text, so it is safe to commit."""
    return {
        "source_url": source_url,
        "version_date": version_date,
        "retrieved": retrieved,
        "raw_sha256": sha256_hex(raw),
        "raw_bytes": len(raw),
        "parser": "ptlegal_eval.ingest.parse_code",
        "n_chunks": len(chunks),
        "n_articles": len({c.article for c in chunks}),
        "revoked_articles": report.revoked,
        "suspended_articles": report.suspended,
        "chunk_sha256": article_hashes(chunks),
    }


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import json
    from pathlib import Path

    from ptlegal_eval.chunks import write_chunks

    p = argparse.ArgumentParser(description="Parse the saved consolidated text into chunks.")
    p.add_argument("raw", type=Path)
    p.add_argument("--source-url", required=True)
    p.add_argument("--version-date", required=True, help="the page's 'Versão à data de' date")
    p.add_argument("--retrieved", required=True, help="date the text was saved, YYYY-MM-DD")
    p.add_argument("--chunks-out", type=Path, default=Path("data/chunks.jsonl"))
    p.add_argument("--manifest-out", type=Path, default=Path("data/corpus_manifest.json"))
    args = p.parse_args(argv)

    raw = args.raw.read_bytes()
    report = ParseReport()
    chunks = parse_code(raw.decode("utf-8"), report=report)
    write_chunks(chunks, args.chunks_out)
    manifest = build_manifest(
        raw, chunks, report, args.source_url, args.version_date, args.retrieved
    )
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{len(chunks)} chunks, {manifest['n_articles']} articles; "
        f"revoked {report.revoked}; suspended {report.suspended}"
    )


if __name__ == "__main__":
    main()
