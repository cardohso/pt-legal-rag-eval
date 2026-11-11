"""The prompt every model gets. Written in European Portuguese, identical across models.

Versioned: tuned on dev only, every version logged (pre-registration, section 7). `v0` is the first
draft, used by the smoke test; it has not been tuned on anything.
"""

from __future__ import annotations

from collections.abc import Sequence

from ptlegal_eval.chunks import Chunk

PROMPT_VERSION = "v0"

SYSTEM_OPEN_BOOK = (
    "És um assistente que responde a perguntas sobre o Código do Trabalho português, em português "
    "europeu.\n"
    "Responde apenas com base nos artigos fornecidos. Não uses conhecimento externo.\n"
    "Indica sempre os artigos em que te baseias, no formato (art. 123.º, n.º 2).\n"
    "Se os artigos fornecidos não contêm a resposta, diz claramente que a informação não consta "
    "dos artigos fornecidos e não inventes uma resposta.\n"
    "Se a pergunta assume algo que os artigos contradizem, corrige essa premissa em vez de a "
    "aceitar.\n"
    "Responde de forma concisa."
)

SYSTEM_CLOSED_BOOK = (
    "És um assistente que responde a perguntas sobre o Código do Trabalho português, em português "
    "europeu.\n"
    "Não tens acesso ao texto da lei: responde com o que sabes.\n"
    "Se souberes, indica os artigos em que te baseias, no formato (art. 123.º, n.º 2).\n"
    "Se não souberes ou não tiveres a certeza, diz-o claramente em vez de inventar.\n"
    "Se a pergunta assume algo que sabes ser errado, corrige essa premissa em vez de a aceitar.\n"
    "Responde de forma concisa."
)


def render_context(chunks: Sequence[Chunk]) -> str:
    """The articles, in the order given, each under a heading with its number and epigraph."""
    return "\n\n".join(f"[Artigo {c.article} — {c.epigraph}]\n{c.text}" for c in chunks)


def build_messages(question: str, chunks: Sequence[Chunk] | None) -> list[dict[str, str]]:
    """Chat messages for one question. `None` is closed book (condition C); an empty list is an
    open-book prompt with no articles, which is not the same thing."""
    if chunks is None:
        return [
            {"role": "system", "content": SYSTEM_CLOSED_BOOK},
            {"role": "user", "content": f"Pergunta: {question}"},
        ]
    user = f"Artigos do Código do Trabalho:\n\n{render_context(chunks)}\n\nPergunta: {question}"
    return [
        {"role": "system", "content": SYSTEM_OPEN_BOOK},
        {"role": "user", "content": user},
    ]
