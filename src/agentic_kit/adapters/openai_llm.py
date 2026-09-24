from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from ..domain.models import Role
from ..domain.tools import ToolCall, ToolDefinition
from ..errors import ProviderError
from ..ports.llm import LlmReply, LlmRequest, ToolExchange

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

    async def complete(self, request: LlmRequest) -> LlmReply:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages(request),
        }
        if request.tools:
            payload["tools"] = [_as_function(tool) for tool in request.tools]

        try:
            response = await self._client.chat.completions.create(**payload)
        except OpenAIError as error:
            raise ProviderError(f"openai call failed: {error}") from error

        message = response.choices[0].message
        return LlmReply(
            text=message.content or "",
            tool_calls=tuple(_as_call(raw) for raw in message.tool_calls or ()),
        )

    def _messages(self, request: LlmRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.system},
            *({"role": _ROLES[m.role], "content": m.text} for m in request.messages),
        ]
        for exchange in request.exchanges:
            messages.extend(_replay(exchange))
        return messages


def _as_function(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _as_call(raw: Any) -> ToolCall:
    try:
        arguments = json.loads(raw.function.arguments or "{}")
    except json.JSONDecodeError as error:
        raise ProviderError(f"model sent unreadable arguments for {raw.function.name}") from error
    return ToolCall(id=raw.id, name=raw.function.name, arguments=arguments)


def _replay(exchange: ToolExchange) -> list[dict[str, Any]]:
    """A finished round as the API wants it back: the ask, then each answer."""
    ask = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in exchange.calls
        ],
    }
    answers = [
        {"role": "tool", "tool_call_id": call_id, "content": output}
        for call_id, output in exchange.results
    ]
    return [ask, *answers]
