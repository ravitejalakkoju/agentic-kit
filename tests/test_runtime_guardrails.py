"""The prompt and output checkpoints, which only the runtime can reach."""

from __future__ import annotations

import pytest

from agentic_kit.composition import build_prompts
from agentic_kit.domain.models import ConversationState, TurnRequest
from agentic_kit.engine.guardrails.findings import (
    Action,
    Checkpoint,
    Finding,
    GuardrailContext,
    Level,
    Mode,
)
from agentic_kit.engine.guardrails.guard import Guardrails
from agentic_kit.engine.guardrails.profile import CheckpointPolicy, GuardrailProfile
from agentic_kit.engine.guardrails.responder import HANDOFF_REPLY, GuardrailResponder
from agentic_kit.engine.runtime.text_runtime import TextRuntime
from agentic_kit.seed import SUPPORT_SOP

from .fakes import ScriptedLlm

REQUEST = TurnRequest(conversation_id="conv-1", customer_id="cust-1", text="Where is my order?")
CONVERSATION = ConversationState(conversation_id="conv-1", customer_id="cust-1")


class AlwaysFails:
    def __init__(self, checkpoint: Checkpoint, action: Action = Action.BLOCK) -> None:
        self.id = "always_fails"
        self.checkpoints = frozenset({checkpoint})
        self._action = action

    def evaluate(self, context: GuardrailContext) -> Finding:
        return Finding(
            detector_id=self.id,
            checkpoint=context.checkpoint,
            level=Level.FAIL,
            message="refused on purpose",
            action=self._action,
        )


def runtime_with(llm: ScriptedLlm, checkpoint: Checkpoint, action: Action) -> TextRuntime:
    profile = GuardrailProfile(
        checkpoints={checkpoint: CheckpointPolicy(Mode.ENFORCE, ("always_fails",))},
        rules=(),
    )
    guardrails = Guardrails([AlwaysFails(checkpoint, action)], profile)
    return TextRuntime(llm, build_prompts(), guardrails, GuardrailResponder())


async def run(runtime: TextRuntime):
    return await runtime.run(sop=SUPPORT_SOP, conversation=CONVERSATION, request=REQUEST)


async def test_a_failing_prompt_check_skips_the_model(llm: ScriptedLlm) -> None:
    reply = await run(runtime_with(llm, Checkpoint.PROMPT, Action.BLOCK))

    assert reply.blocked_by is not None
    assert reply.blocked_by.checkpoint is Checkpoint.PROMPT
    assert reply.text
    assert llm.calls == []


async def test_a_failing_output_check_replaces_the_reply(llm: ScriptedLlm) -> None:
    llm.queue("the unsafe answer")

    reply = await run(runtime_with(llm, Checkpoint.OUTPUT, Action.BLOCK))

    assert reply.blocked_by is not None
    assert reply.text != "the unsafe answer"
    assert len(llm.calls) == 1


async def test_an_output_failure_can_ask_for_a_human(llm: ScriptedLlm) -> None:
    llm.queue("the unsafe answer")

    reply = await run(runtime_with(llm, Checkpoint.OUTPUT, Action.HANDOFF))

    assert reply.text == HANDOFF_REPLY


@pytest.mark.parametrize("checkpoint", [Checkpoint.PROMPT, Checkpoint.OUTPUT])
async def test_a_clean_turn_reports_no_block(llm: ScriptedLlm, checkpoint: Checkpoint) -> None:
    llm.queue("Order 1001 has shipped.")
    profile = GuardrailProfile(checkpoints={checkpoint: CheckpointPolicy(Mode.DISABLED)})
    runtime = TextRuntime(llm, build_prompts(), Guardrails([], profile), GuardrailResponder())

    reply = await run(runtime)

    assert reply.blocked_by is None
    assert reply.text == "Order 1001 has shipped."
