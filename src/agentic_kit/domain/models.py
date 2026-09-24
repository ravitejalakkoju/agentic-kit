"""Domain models shared by the engine, the stores, and the HTTP layer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from .memory import WorkingMemory

HISTORY_LIMIT = 20
"""Turns kept per conversation, matching the cap the TypeScript engine applies on save."""

MAX_ATTEMPTS = 3
"""Turns in a row that went nowhere before a person is asked for."""


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
    BLOCKED = "blocked"
    HANDOFF = "handoff"
    ENDED = "ended"
    FAILED = "failed"


class TurnOutcome(StrEnum):
    RESPONDED = "RESPONDED"
    NO_MATCH = "NO_MATCH"
    RUN_ENDED = "RUN_ENDED"
    POLICY_BLOCK = "POLICY_BLOCK"
    USER_REQUESTED_HUMAN = "USER_REQUESTED_HUMAN"
    MAX_ATTEMPTS = "MAX_ATTEMPTS"
    FAILED = "FAILED"


class TurnContext(BaseModel):
    """What the caller already knows about this turn. Collectors look records up from it."""

    channel: str = "chat"
    email: str | None = None
    phone: str | None = None
    order_id: str | None = None
    ticket_id: str | None = None


class TurnRequest(BaseModel):
    conversation_id: str
    customer_id: str
    text: str
    context: TurnContext = Field(default_factory=TurnContext)


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
    """Customer utterances that should route here. Used for matching, never shown as replies."""
    catch_all: bool = False
    """Whether this procedure takes a turn that matched nothing.

    Somebody has to answer "hi". Without a desk that accepts the unclassified,
    the most ordinary opening line a customer writes gets no reply at all.
    """
    personality: Personality
    response_strategy: str | None = None
    example_responses: list[str] = Field(default_factory=list)
    """Replies that show the intended style. Rendered as style guidance only."""


class ConversationState(BaseModel):
    conversation_id: str
    customer_id: str
    active_agent_id: str | None = None
    active_sop_id: str | None = None
    history: list[Message] = Field(default_factory=list)
    memory: WorkingMemory = Field(default_factory=WorkingMemory)
    attempts: int = 0
    """Turns in a row that did not help. A model that errored and a reply that
    had to be blocked are the same thing from where the customer is sitting."""
    human_handoff_requested: bool = False
    closed: bool = False

    def append(self, *messages: Message) -> None:
        """Record turns, keeping only the most recent HISTORY_LIMIT."""
        self.history = [*self.history, *messages][-HISTORY_LIMIT:]

    def record_stall(self) -> None:
        """Another turn that got the customer no further."""
        self.attempts += 1

    @property
    def stalled(self) -> bool:
        """Enough of them in a row that trying again is not the answer."""
        return self.attempts >= MAX_ATTEMPTS


class RunRecord(BaseModel):
    run_id: str = Field(default_factory=_new_id)
    conversation_id: str
    agent_id: str | None = None
    sop_id: str | None = None
    status: TurnStatus
    outcome: TurnOutcome
    created_at: datetime = Field(default_factory=_now)
