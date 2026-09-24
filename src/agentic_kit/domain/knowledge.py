"""Reference material the agent can look things up in.

Deliberately kept apart from the records in `crm.py`. Those are the customer's
own data, fetched by id and true by definition. This is text somebody wrote and
somebody else uploaded, and the gap between the two decides how far either is
trusted once it reaches a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid4().hex


class Document(BaseModel):
    """One article, page or file, as it arrived and before it is split up."""

    document_id: str = Field(default_factory=_new_id)
    title: str
    source: str
    """Where it came from, verbatim, so an answer can say where it got that."""
    text: str


class Chunk(BaseModel):
    """A passage of a document, small enough to embed and short enough to quote.

    Carries its document's title and source rather than only the id. A search
    hit is then something the agent can cite on its own, with no second lookup
    to find out what it was reading.
    """

    chunk_id: str
    document_id: str
    title: str
    source: str
    position: int
    """Which passage of the document this is, counting from zero."""
    text: str

    @classmethod
    def of(cls, document: Document, position: int, text: str) -> Chunk:
        return cls(
            chunk_id=f"{document.document_id}-{position}",
            document_id=document.document_id,
            title=document.title,
            source=document.source,
            position=position,
            text=text,
        )


@dataclass(frozen=True, slots=True)
class Passage:
    """A chunk a search turned up, and how well it matched."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Shelved:
    """What the knowledge base holds of one document, for an operator to look at."""

    document_id: str
    title: str
    source: str
    passages: int
