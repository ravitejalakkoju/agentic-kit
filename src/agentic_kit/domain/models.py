"""Domain models shared by the engine, the stores, and the HTTP layer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

HISTORY_LIMIT = 20
"""Turns kept per conversation, matching the cap the TypeScript engine applies on save."""


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex


class FlowKind(StrEnum):
    CONVERSATION = "conversation"
    EVENT = "event"
    CALLBACK = "callback"
    TASK = "task"


class Role(StrEnum):
    USER = "user"
    AGENT = "agent"


class Message(BaseModel):
    role: Role
    text: str
    at: datetime = Field(default_factory=_now)


class TurnStatus(StrEnum):
    RESPONDED = "responded"
    NOOP = "noop"
    HANDOFF = "handoff"
    ENDED = "ended"
    FAILED = "failed"


class TurnOutcome(StrEnum):
    RESPONDED = "RESPONDED"
    NO_MATCH = "NO_MATCH"
    RUN_ENDED = "RUN_ENDED"
    POLICY_BLOCK = "POLICY_BLOCK"
    USER_REQUESTED_HUMAN = "USER_REQUESTED_HUMAN"
    FAILED = "FAILED"


class TurnRequest(BaseModel):
    conversation_id: str
    customer_id: str
    text: str


class TurnResult(BaseModel):
    status: TurnStatus
    outcome: TurnOutcome
    reply: str | None = None
    reason: str | None = None
    run_id: str | None = None


class Personality(BaseModel):
    name: str
    identity: str
    tone: str


class SopDefinition(BaseModel):
    """A procedure an agent follows, plus the persona it speaks with."""

    sop_id: str
    agent_id: str
    description: str
    instructions: str
    examples: list[str] = Field(default_factory=list)
    personality: Personality


class ConversationState(BaseModel):
    conversation_id: str
    customer_id: str
    active_agent_id: str | None = None
    active_sop_id: str | None = None
    history: list[Message] = Field(default_factory=list)
    attempts: int = 0
    human_handoff_requested: bool = False
    closed: bool = False

    def append(self, *messages: Message) -> None:
        """Record turns, keeping only the most recent HISTORY_LIMIT."""
        self.history = [*self.history, *messages][-HISTORY_LIMIT:]


class RunRecord(BaseModel):
    run_id: str = Field(default_factory=_new_id)
    conversation_id: str
    agent_id: str | None = None
    sop_id: str | None = None
    status: TurnStatus
    outcome: TurnOutcome
    created_at: datetime = Field(default_factory=_now)
