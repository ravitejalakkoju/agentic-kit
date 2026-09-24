"""Where the pieces are put together.

Plain constructors in one readable function. If this ever needs a container,
the engine has grown in a direction worth questioning first.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapters.dummy_llm import DummyLlm
from .adapters.memory import InMemoryConversationStore, InMemoryRunStore, InMemorySopCatalog
from .adapters.openai_llm import OpenAiLlm
from .domain.models import FlowKind
from .engine.ai_engine import AiEngine
from .engine.flows.conversation import ConversationFlow
from .engine.graph.executor import GraphExecutor
from .engine.graph.node import Node
from .engine.graph.state import NodeKey
from .engine.nodes.failed import FailedNode
from .engine.nodes.finalize import FinalizeNode
from .engine.nodes.handoff import HandoffNode
from .engine.nodes.load_state import LoadStateNode
from .engine.nodes.passthrough import PassThroughNode
from .engine.nodes.persist_state import PersistStateNode
from .engine.nodes.route import RouteNode
from .engine.nodes.run_runtime import RunRuntimeNode
from .engine.runtime.text_runtime import TextRuntime
from .ports.llm import LlmPort
from .seed import DEFAULT_SOPS
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


def build_llm(settings: Settings) -> LlmPort:
    if not settings.has_live_llm:
        return DummyLlm()
    return OpenAiLlm(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.openai_timeout_seconds,
    )


def build(settings: Settings, llm: LlmPort | None = None) -> Components:
    conversations = InMemoryConversationStore()
    runs = InMemoryRunStore()
    catalog = InMemorySopCatalog(DEFAULT_SOPS)
    runtime = TextRuntime(llm or build_llm(settings))

    nodes: list[Node] = [
        LoadStateNode(conversations),
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
    return Components(engine=engine, conversations=conversations, runs=runs, catalog=catalog)
