"""What the customer hears when a guardrail stops the turn."""

from __future__ import annotations

from .findings import Action, Checkpoint, Finding

HANDOFF_REPLY = "I am bringing in a human teammate to help with this."

_BY_DETECTOR = {
    "prompt_injection": "I can only help with questions about your account and orders.",
    "input_length": "That message is a little too long for me. Could you shorten it?",
}

_BY_CHECKPOINT = {
    Checkpoint.INPUT: "I cannot help with that one. Is there something else I can look into?",
    Checkpoint.PROMPT: "I do not have enough reliable information to answer that safely.",
    Checkpoint.OUTPUT: "I could not put together a safe reply. Let me get a teammate instead.",
}
"""MEMORY and KNOWLEDGE are missing on purpose: both only ever cost a piece of text, which is
dropped while the turn carries on, so neither becomes something a customer hears."""


class GuardrailResponder:
    def reply_for(self, finding: Finding) -> str:
        if finding.action is Action.HANDOFF:
            return HANDOFF_REPLY
        return _BY_DETECTOR.get(finding.detector_id) or _BY_CHECKPOINT[finding.checkpoint]
