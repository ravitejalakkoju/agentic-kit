"""Embeddings from OpenAI."""

from __future__ import annotations

from collections.abc import Sequence

from openai import AsyncOpenAI, OpenAIError

from ..errors import ProviderError
from ..ports.knowledge import Vector

DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class OpenAiEmbedder:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self._model = model
        self.dimensions = DIMENSIONS.get(model, 1536)
        self.min_score = 0.4
        """Where the text-embedding-3 models put genuinely related text."""

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(model=self._model, input=list(texts))
        except OpenAIError as error:
            raise ProviderError(f"openai embedding failed: {error}") from error

        # The API may answer out of order; `index` is what puts a vector back on its text.
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
