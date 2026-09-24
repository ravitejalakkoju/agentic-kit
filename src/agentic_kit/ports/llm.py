"""The model seam. Two implementations today: deterministic and OpenAI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..domain.models import Message


class LlmPort(Protocol):
    name: str

    async def complete(self, *, system: str, messages: Sequence[Message]) -> str:
        """Return the assistant reply, or raise ProviderError."""
        ...
