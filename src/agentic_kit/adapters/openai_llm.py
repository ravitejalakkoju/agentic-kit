from __future__ import annotations

from collections.abc import Sequence

from openai import AsyncOpenAI, OpenAIError

from ..domain.models import Message, Role
from ..errors import ProviderError

_ROLES = {Role.USER: "user", Role.AGENT: "assistant"}


class OpenAiLlm:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self._model = model

    async def complete(self, *, system: str, messages: Sequence[Message]) -> str:
        payload = [
            {"role": "system", "content": system},
            *({"role": _ROLES[m.role], "content": m.text} for m in messages),
        ]
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=payload,
            )
        except OpenAIError as error:
            raise ProviderError(f"openai call failed: {error}") from error

        return response.choices[0].message.content or ""
