"""Where the pieces are put together.

Plain constructors in one readable function. If this ever needs a container,
the engine has grown in a direction worth questioning first.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapters.dummy_llm import DummyLlm
from .adapters.memory import InMemoryConversationStore, InMemoryRunStore, InMemorySopCatalog
from .adapters.openai_llm import OpenAiLlm
from .adapters.sample_crm import (
    InMemoryContactReader,
    InMemoryRecordReader,
    InMemoryTicketNotes,
)
from .domain.crm import Order, Ticket
from .domain.models import FlowKind
from .engine.ai_engine import AiEngine
from .engine.flows.conversation import ConversationFlow
from .engine.graph.executor import GraphExecutor
from .engine.graph.node import Node
from .engine.graph.state import NodeKey
from .engine.guardrails import (
    DEFAULT_PROFILE,
    GuardrailProfile,
    GuardrailResponder,
    Guardrails,
    default_detectors,
)
from .engine.nodes.failed import FailedNode
from .engine.nodes.finalize import FinalizeNode
from .engine.nodes.guard_input import GuardInputNode
from .engine.nodes.handoff import HandoffNode
from .engine.nodes.load_state import LoadStateNode
from .engine.nodes.passthrough import PassThroughNode
from .engine.nodes.persist_state import PersistStateNode
from .engine.nodes.route import RouteNode
from .engine.nodes.run_runtime import RunRuntimeNode
from .engine.prompt.builder import PromptBuilder
from .engine.prompt.context import ContactCollector, ContextPipeline, ResourceCollector
from .engine.prompt.sections import DEFAULT_SECTIONS
from .engine.runtime.text_runtime import TextRuntime
from .engine.tools import AddTicketNote, LookupContact, LookupTicket, ToolRegistry, TrackOrder
from .ports.llm import LlmPort
from .seed import DEFAULT_SOPS, SAMPLE_CONTACTS, SAMPLE_ORDERS, SAMPLE_TICKETS
from .settings import Settings

STUBBED_NODES = (
    NodeKey.SELECT_SOP,
    NodeKey.DETECT_SOP_DRIFT,
    NodeKey.BUILD_RUNTIME_REQUEST,
    NodeKey.REVIEW_RESULT,
)


@dataclass(frozen=True, slots=True)
class Components:
    engine: AiEngine
    conversations: InMemoryConversationStore
    runs: InMemoryRunStore
    catalog: InMemorySopCatalog
    prompts: PromptBuilder
    tools: ToolRegistry
    crm: Crm


@dataclass(frozen=True, slots=True)
class Crm:
    """The customer's systems, seeded in memory. Collectors read them, tools use them."""

    contacts: InMemoryContactReader
    orders: InMemoryRecordReader[Order]
    tickets: InMemoryRecordReader[Ticket]
    ticket_notes: InMemoryTicketNotes


def build_crm() -> Crm:
    tickets = InMemoryRecordReader(SAMPLE_TICKETS, id_of=lambda ticket: ticket.ticket_id)
    return Crm(
        contacts=InMemoryContactReader(SAMPLE_CONTACTS),
        orders=InMemoryRecordReader(SAMPLE_ORDERS, id_of=lambda order: order.order_id),
        tickets=tickets,
        ticket_notes=InMemoryTicketNotes(tickets),
    )


def build_prompts(rules: tuple[str, ...] = (), crm: Crm | None = None) -> PromptBuilder:
    crm = crm or build_crm()
    context = ContextPipeline(
        [
            ContactCollector(crm.contacts),
            ResourceCollector(
                "order", crm.orders, id_of=lambda ctx: ctx.order_id, fact_key="order_id"
            ),
            ResourceCollector(
                "ticket", crm.tickets, id_of=lambda ctx: ctx.ticket_id, fact_key="ticket_id"
            ),
        ]
    )
    return PromptBuilder(context, DEFAULT_SECTIONS, rules)


def build_tools(crm: Crm) -> ToolRegistry:
    return ToolRegistry(
        [
            TrackOrder(crm.orders),
            LookupTicket(crm.tickets),
            LookupContact(crm.contacts),
            AddTicketNote(crm.ticket_notes),
        ]
    )


def build_llm(settings: Settings) -> LlmPort:
    if not settings.has_live_llm:
        return DummyLlm()
    return OpenAiLlm(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.openai_timeout_seconds,
    )


def build(
    settings: Settings,
    llm: LlmPort | None = None,
    profile: GuardrailProfile = DEFAULT_PROFILE,
) -> Components:
    conversations = InMemoryConversationStore()
    runs = InMemoryRunStore()
    catalog = InMemorySopCatalog(DEFAULT_SOPS)
    guardrails = Guardrails(default_detectors(), profile)
    responder = GuardrailResponder()
    crm = build_crm()
    prompts = build_prompts(profile.rules, crm)
    tools = build_tools(crm)
    runtime = TextRuntime(llm or build_llm(settings), prompts, guardrails, responder, tools)

    nodes: list[Node] = [
        LoadStateNode(conversations),
        GuardInputNode(guardrails, responder),
        RouteNode(catalog),
        RunRuntimeNode(runtime),
        PersistStateNode(conversations, runs),
        HandoffNode(),
        FailedNode(),
        FinalizeNode(),
        *(PassThroughNode(key) for key in STUBBED_NODES),
    ]

    executor = GraphExecutor({node.key: node for node in nodes})
    engine = AiEngine({FlowKind.CONVERSATION: ConversationFlow(executor)})
    return Components(
        engine=engine,
        conversations=conversations,
        runs=runs,
        catalog=catalog,
        prompts=prompts,
        tools=tools,
        crm=crm,
    )
