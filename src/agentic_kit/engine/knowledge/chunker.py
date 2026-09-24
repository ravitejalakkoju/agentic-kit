"""Splitting a document into pieces that are each about something.

Cutting every 1800 characters is the easy way and the worst-measured one: it
lands mid-sentence, so the piece that gets embedded is half of one idea and
half of the next, and it matches nothing well. A writer has already marked
where the subject changes, by leaving a blank line, so those marks are the cuts
and the budget only decides how many paragraphs travel together.

There is no overlap between chunks. Overlap exists to repair boundaries chosen
badly, and paragraph boundaries are not; paying for it would mean the same
sentences twice in one prompt.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

CHUNK_BUDGET = 1800
"""Characters, so roughly 450 tokens: large enough to hold an argument, small
enough that several fit in a prompt beside everything else."""

PARAGRAPH = "\n\n"
_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk(text: str) -> list[str]:
    """Whole paragraphs where they fit, whole sentences where they do not."""
    chunks: list[str] = []
    run: list[str] = []

    for paragraph in _paragraphs(text):
        if len(paragraph) <= CHUNK_BUDGET:
            run.append(paragraph)
            continue
        chunks.extend(_pack(run, PARAGRAPH))
        run = []
        chunks.extend(_pack(_sentences(paragraph), " "))

    return [*chunks, *_pack(run, PARAGRAPH)]


def _paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks. Single newlines stay put, so lists survive."""
    blocks = (block.strip() for block in _BREAK.split(text.strip()))
    return [block for block in blocks if block]


def _sentences(paragraph: str) -> list[str]:
    """One paragraph too long to be a chunk, broken where its sentences end.

    A sentence longer than the whole budget is someone's unpunctuated wall of
    text, and words are the last boundary left before characters.
    """
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        if len(sentence) <= CHUNK_BUDGET:
            pieces.append(sentence)
        else:
            pieces.extend(_pack(sentence.split(), " "))
    return pieces


def _pack(pieces: Iterable[str], separator: str) -> list[str]:
    """Fill each chunk with whole pieces, starting a new one rather than splitting."""
    chunks: list[str] = []
    batch: list[str] = []

    for piece in pieces:
        if batch and _width(batch, separator) + len(separator) + len(piece) > CHUNK_BUDGET:
            chunks.append(separator.join(batch))
            batch = []
        batch.append(piece)

    return [*chunks, separator.join(batch)] if batch else chunks


def _width(batch: Sequence[str], separator: str) -> int:
    return sum(len(piece) for piece in batch) + len(separator) * (len(batch) - 1)
