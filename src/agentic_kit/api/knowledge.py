"""The policy library: what goes in, what is held, and what a question finds."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..domain.knowledge import Document
from ..errors import KnowledgeError
from .deps import Wired

router = APIRouter(prefix="/v1/knowledge", tags=["knowledge"])


class DocumentBody(BaseModel):
    title: str
    source: str
    """Where it came from, verbatim. Every answer this supports will cite it."""
    text: str


class Ingested(BaseModel):
    document_id: str
    passages: int


class ShelvedView(BaseModel):
    document_id: str
    title: str
    source: str
    passages: int


class Question(BaseModel):
    question: str


class PassageView(BaseModel):
    title: str
    source: str
    text: str
    score: float
    """Shown here and withheld from the model: an operator tuning a library can
    read a score, and an agent writing to a customer cannot."""


@router.post("", status_code=201, summary="Add a document the agent can look things up in")
async def add_document(body: DocumentBody, wired: Wired) -> Ingested:
    document = Document(title=body.title, source=body.source, text=body.text)
    try:
        chunks = await wired.knowledge.ingest(document)
    except KnowledgeError as error:
        # Caught rather than left to the engine handler, which answers 500.
        # A refused document is a problem with what was sent, not with the server.
        raise HTTPException(status_code=422, detail=str(error)) from error
    return Ingested(document_id=document.document_id, passages=len(chunks))


@router.get("", summary="List everything the agent can look up")
async def list_documents(wired: Wired) -> list[ShelvedView]:
    return [
        ShelvedView(
            document_id=item.document_id,
            title=item.title,
            source=item.source,
            passages=item.passages,
        )
        for item in await wired.knowledge.shelf()
    ]


@router.post("/search", summary="See the passages a question would find")
async def search(body: Question, wired: Wired) -> list[PassageView]:
    """The retrieval half of a turn on its own, so a thin answer can be explained."""
    return [
        PassageView(
            title=passage.chunk.title,
            source=passage.chunk.source,
            text=passage.chunk.text,
            score=passage.score,
        )
        for passage in await wired.knowledge.find(body.question)
    ]


@router.delete("/{document_id}", summary="Forget a document")
async def forget_document(document_id: str, wired: Wired) -> Ingested:
    passages = await wired.knowledge.forget(document_id)
    if not passages:
        raise HTTPException(status_code=404, detail=f"nothing stored for {document_id}")
    return Ingested(document_id=document_id, passages=passages)
