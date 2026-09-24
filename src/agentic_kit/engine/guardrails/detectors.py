"""The checks themselves.

A detector looks at one thing and says nothing when it is happy. It never
decides what the engine does next; that is the mode on the checkpoint.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Protocol

from .findings import Action, Checkpoint, Finding, GuardrailContext, Level

INPUT = frozenset({Checkpoint.INPUT})
OUTPUT = frozenset({Checkpoint.OUTPUT})


class Detector(Protocol):
    id: str
    checkpoints: frozenset[Checkpoint]

    def evaluate(self, context: GuardrailContext) -> Finding | None:
        """Describe what is wrong with this text, or return None."""
        ...


class PolicyPhraseDetector:
    """Subjects a human has to own, such as fraud reports."""

    id = "policy_phrase"
    checkpoints = INPUT

    def __init__(self, phrases: Sequence[str]) -> None:
        self._phrases = tuple(phrase.lower() for phrase in phrases)

    def evaluate(self, context: GuardrailContext) -> Finding | None:
        text = context.text.lower()
        phrase = next((p for p in self._phrases if p in text), None)
        if phrase is None:
            return None
        return Finding(
            detector_id=self.id,
            checkpoint=context.checkpoint,
            level=Level.FAIL,
            message=f"matched the policy phrase {phrase!r}",
            action=Action.HANDOFF,
        )


class PromptInjectionDetector:
    """Attempts to talk the agent out of its own instructions."""

    id = "prompt_injection"
    checkpoints = INPUT

    FAIL_PATTERNS = (
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.I),
        re.compile(
            r"(?:reveal|show|print|repeat)\s+(?:your|the)?\s*(?:system|developer)\s+prompt", re.I
        ),
        re.compile(r"(?:act|behave)\s+as\s+(?:the\s+)?(?:system|developer)", re.I),
        re.compile(r"bypass\s+(?:your\s+)?(?:rules|guardrails|restrictions)", re.I),
    )
    WARN_PATTERN = re.compile(r"forget\s+(?:everything|all|the|your)?\s*previous", re.I)

    def evaluate(self, context: GuardrailContext) -> Finding | None:
        if any(pattern.search(context.text) for pattern in self.FAIL_PATTERNS):
            return self._finding(context, Level.FAIL, "looks like a prompt injection attempt")
        if self.WARN_PATTERN.search(context.text):
            return self._finding(context, Level.WARN, "mentions forgetting previous instructions")
        return None

    def _finding(self, context: GuardrailContext, level: Level, message: str) -> Finding:
        return Finding(
            detector_id=self.id,
            checkpoint=context.checkpoint,
            level=level,
            message=message,
        )


class InputLengthDetector:
    id = "input_length"
    checkpoints = INPUT

    def __init__(self, max_chars: int = 8000) -> None:
        self._max_chars = max_chars

    def evaluate(self, context: GuardrailContext) -> Finding | None:
        length = len(context.text)
        if length <= self._max_chars:
            return None
        return Finding(
            detector_id=self.id,
            checkpoint=context.checkpoint,
            level=Level.FAIL,
            message=f"message is {length} characters, over the {self._max_chars} limit",
        )


class CodeFenceReplyDetector:
    """A reply that is one code block reads as a snippet, not an answer."""

    id = "code_fence_reply"
    checkpoints = OUTPUT

    PATTERN = re.compile(r"^```[\w-]*\s*[\s\S]*```$")

    def evaluate(self, context: GuardrailContext) -> Finding | None:
        if not self.PATTERN.match(context.text.strip()):
            return None
        return Finding(
            detector_id=self.id,
            checkpoint=context.checkpoint,
            level=Level.WARN,
            message="the whole reply is wrapped in a code fence",
        )


class MarkupReplyDetector:
    id = "markup_reply"
    checkpoints = OUTPUT

    PATTERN = re.compile(r"</?[a-z][a-z0-9-]*(?:\s+[^<>]*)?>", re.I)

    def evaluate(self, context: GuardrailContext) -> Finding | None:
        if not self.PATTERN.search(context.text):
            return None
        return Finding(
            detector_id=self.id,
            checkpoint=context.checkpoint,
            level=Level.WARN,
            message="the reply contains markup tags",
        )


class ResponseLengthDetector:
    id = "response_length"
    checkpoints = OUTPUT

    def __init__(self, max_chars: int = 4000) -> None:
        self._max_chars = max_chars

    def evaluate(self, context: GuardrailContext) -> Finding | None:
        length = len(context.text)
        if length <= self._max_chars:
            return None
        return Finding(
            detector_id=self.id,
            checkpoint=context.checkpoint,
            level=Level.WARN,
            message=f"reply is {length} characters, over the {self._max_chars} limit",
        )


POLICY_PHRASES = ("fraud", "hack account", "abuse")


def default_detectors() -> Iterable[Detector]:
    return (
        PolicyPhraseDetector(POLICY_PHRASES),
        PromptInjectionDetector(),
        InputLengthDetector(),
        CodeFenceReplyDetector(),
        MarkupReplyDetector(),
        ResponseLengthDetector(),
    )
