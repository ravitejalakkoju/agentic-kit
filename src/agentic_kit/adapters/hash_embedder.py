"""An embedder that needs no network, so the suite and a fresh clone both run."""

from __future__ import annotations

import re
from collections.abc import Sequence
from hashlib import blake2b

from ..domain.text import readable
from ..ports.knowledge import Vector

_WORD = re.compile(r"\w+")

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
        "how", "i", "if", "in", "is", "it", "its", "my", "no", "not",
        "of", "on", "or", "our", "so", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "to", "was", "we", "were", "what", "when",
        "where", "which", "who", "why", "will", "with", "would", "you", "your",
    }
)  # fmt: skip
"""Words every document has, which is exactly why they say nothing about any of
them. Left in, the longest document wins every question."""


class HashEmbedder:
    """Feature hashing over words: every word has a fixed direction.

    Text that shares words points the same way, which is enough to tell a
    refund policy from a shipping one. It knows nothing about meaning, so
    "refund" and "reimbursement" are strangers to it, and closing that gap is
    the whole of what a real embedding model is paid for.
    """

    name = "hash"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.min_score = 0.1
        """Lower than a real embedder's, because a short question and a long
        passage share few of the exact words that are all this can see."""

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> Vector:
        """Presence, not frequency. A word said twelve times is not twelve times the subject."""
        vector = [0.0] * self.dimensions
        for word in self._words(text):
            vector[self._bucket(word)] = 1.0
        return vector

    def _words(self, text: str) -> set[str]:
        found = _WORD.findall(readable(text).lower())
        return {_singular(word) for word in found if word not in STOPWORDS}

    def _bucket(self, word: str) -> int:
        """Stable across processes, unlike `hash`, so a stored vector stays meaningful."""
        return int.from_bytes(blake2b(word.encode(), digest_size=8).digest()) % self.dimensions


def _singular(word: str) -> str:
    """The crudest stemming there is, so a customer asking about refunds finds
    the paragraph about a refund.

    It also turns "address" into "addres". Both sides of a comparison are
    folded the same way, so a mangled word still matches itself.
    """
    return word[:-1] if len(word) > 3 and word.endswith("s") else word
