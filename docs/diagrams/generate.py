#!/usr/bin/env python3
"""Draw the agentic-kit architecture diagrams.

Run from the repository root:

    python docs/diagrams/generate.py

Node names, detector ids, section keys and tool names are read out of the
engine rather than typed here, so a diagram cannot quietly disagree with the
code. Positions are hand-placed, because a good layout is a judgement call.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from _excalidraw import Diagram

from agentic_kit.composition import STUBBED_NODES, build_crm, build_tools
from agentic_kit.engine.graph.edges import EDGES
from agentic_kit.engine.graph.state import NodeKey, Outcome
from agentic_kit.engine.guardrails import default_detectors
from agentic_kit.engine.nodes.detect_sop_drift import DetectSopDriftNode
from agentic_kit.engine.nodes.failed import FailedNode
from agentic_kit.engine.nodes.finalize import FinalizeNode
from agentic_kit.engine.nodes.guard_input import GuardInputNode
from agentic_kit.engine.nodes.handoff import HandoffNode
from agentic_kit.engine.nodes.load_state import LoadStateNode
from agentic_kit.engine.nodes.passthrough import PassThroughNode
from agentic_kit.engine.nodes.persist_state import PersistStateNode
from agentic_kit.engine.nodes.review_result import ReviewResultNode
from agentic_kit.engine.nodes.route import RouteNode
from agentic_kit.engine.nodes.run_runtime import RunRuntimeNode
from agentic_kit.engine.nodes.select_sop import SelectSopNode
from agentic_kit.engine.prompt.sections import DEFAULT_SECTIONS
from agentic_kit.engine.routing.matcher import AMBIGUITY, MARGIN
from agentic_kit.engine.runtime.text_runtime import MAX_TOOL_ROUNDS

HERE = Path(__file__).resolve().parent

CLIENT = "#fff3bf"
EDGE = "#ffd8a8"
NODE = "#a5d8ff"
ENGINE = "#b2f2bb"
INTERNAL = "#d0bfff"
STORE = "#c3fae8"
STOP = "#ffc9c9"
PENDING = "#e9ecef"
BACKDROP = "#f8f9fa"

REAL_NODES = (
    LoadStateNode,
    GuardInputNode,
    RouteNode,
    SelectSopNode,
    DetectSopDriftNode,
    RunRuntimeNode,
    ReviewResultNode,
    PersistStateNode,
    HandoffNode,
    FailedNode,
    FinalizeNode,
)


def node_classes() -> dict[NodeKey, str]:
    """Which class answers for each node, mirroring how composition builds them."""
    names = {node.key: node.__name__ for node in REAL_NODES}
    names.update({key: PassThroughNode.__name__ for key in STUBBED_NODES})
    missing = sorted(set(NodeKey) - set(names))
    if missing:
        raise SystemExit(f"no class mapped for {missing}; add it to REAL_NODES")
    return names


# --------------------------------------------------------------------------
# 1. Layers and dependency direction
# --------------------------------------------------------------------------


def layers() -> Diagram:
    d = Diagram("layers")
    d.title("title", "agentic-kit - layers and dependency direction", 60, 40)
    d.text(
        "lede",
        "An arrow points at what a thing depends on. Read it top to bottom: "
        "nothing lower knows anything about what sits above it.",
        60,
        86,
    )

    d.group_box("band-api", "api - the edge", 40, 150, 1620, 200, BACKDROP)
    d.box("main", "main.py\ncreate_app(settings, llm)", 75, 210, 300, 100, EDGE)
    d.box(
        "routers",
        "api/ routers\nPOST /v1/turns      POST /v1/events      POST /v1/prompts/preview\n"
        "GET /v1/tools      GET /v1/graph      GET /v1/sops\n"
        "GET /v1/conversations/{id}      POST /v1/knowledge      POST /v1/knowledge/search",
        415,
        210,
        640,
        100,
        EDGE,
    )
    d.box(
        "deps",
        "api/deps.py\nWired = Annotated[Components, Depends(...)]",
        1095,
        210,
        470,
        100,
        EDGE,
    )

    d.group_box(
        "band-wiring", "composition - the only place classes meet", 40, 380, 1620, 170, BACKDROP
    )
    d.box("settings", "settings.py\nSettings", 75, 420, 230, 100, CLIENT)
    d.box(
        "seed",
        "seed.py\nDEFAULT_SOPS\nSAMPLE_CONTACTS / ORDERS / TICKETS\nSAMPLE_DOCUMENTS",
        345,
        420,
        340,
        100,
        CLIENT,
    )
    d.box(
        "build",
        "composition.py\nbuild(settings, llm, profile)\nplain constructors, no container",
        725,
        420,
        460,
        100,
        ENGINE,
    )
    d.box(
        "components",
        "Components\nengine - conversations - runs - catalog\nprompts - tools - crm - knowledge",
        1225,
        420,
        380,
        100,
        ENGINE,
    )

    d.group_box(
        "band-engine", "engine - the parts that decide things", 40, 580, 1620, 440, BACKDROP
    )
    d.box(
        "aiengine",
        "AiEngine\nthe front door - every kind of turn\nruns the same graph",
        75,
        650,
        460,
        90,
        ENGINE,
    )
    d.box("executor", "GraphExecutor\n+ NodeObserver", 575, 650, 220, 90, INTERNAL)
    d.box(
        "edges",
        "graph/edges.py\nEDGES, validate_graph\nthe topology, as data",
        835,
        650,
        250,
        90,
        INTERNAL,
    )
    d.box("nodes", f"nodes/\n{len(NodeKey)} Node classes", 1125, 650, 230, 90, NODE)
    d.box(
        "graphstate",
        "GraphState\nrequest - conversation\nsop - matches - reply\nresult - memory",
        1395,
        650,
        240,
        90,
        INTERNAL,
    )

    d.box("runtime", "TextRuntime\nthe agent loop, bounded", 590, 790, 300, 90, INTERNAL)
    d.box(
        "matcher",
        "SopMatcher\nranks the catalog against\nthe turn, once",
        1320,
        790,
        300,
        90,
        INTERNAL,
    )
    fanout = {
        "prompts": (f"PromptBuilder\nContextPipeline\n{len(DEFAULT_SECTIONS)} Sections", 200),
        "guardrails": ("Guardrails\nDetectors, GuardrailProfile", 480),
        "responder": ("GuardrailResponder\nthe safe reply", 760),
        "registry": ("ToolRegistry\nschemas, gating, isolation", 1040),
    }
    for key, (label, x) in fanout.items():
        d.box(key, label, x, 900, 240, 100, INTERNAL)
    d.box(
        "knowledge",
        "KnowledgeBase\nchunk, embed, search,\nscreen both ways",
        1320,
        900,
        240,
        100,
        INTERNAL,
    )

    d.group_box(
        "band-ports", "ports - protocols the engine depends on", 40, 1050, 600, 550, BACKDROP
    )
    d.group_box(
        "band-adapters", "adapters - what satisfies them today", 680, 1050, 980, 550, BACKDROP
    )
    seams = (
        ("conv", "ConversationStore", "InMemoryConversationStore"),
        ("runs", "RunStore", "InMemoryRunStore"),
        ("sops", "SopCatalog", "InMemorySopCatalog"),
        ("llm", "LlmPort", "OpenAiLlm  -  DummyLlm  -  ScriptedLlm (tests)"),
        (
            "crm",
            "ContactReader  -  RecordReader[T]  -  TicketWriter",
            "InMemoryContactReader  -  InMemoryRecordReader  -  InMemoryTicketNotes",
        ),
        (
            "tool",
            "Tool[ArgsT]  -  ToolContext  -  ConfirmableArgs",
            "TrackOrder  -  LookupTicket  -  LookupContact  -  AddTicketNote  -  SearchKnowledge",
        ),
        (
            "knowledge",
            "Embedder  -  VectorStore",
            "OpenAiEmbedder  -  HashEmbedder  -  InMemoryVectorStore",
        ),
    )
    for row, (key, port, adapter) in enumerate(seams):
        y = 1115 + row * 70
        d.box(f"port-{key}", port, 95, y, 480, 58, STORE)
        d.box(f"adapter-{key}", adapter, 720, y, 900, 58, STORE)
        d.arrow(f"adapter-{key}", f"port-{key}", dashed=True)
    d.text("implements", "implements", 588, 1118, size=12)

    d.group_box(
        "band-domain", "domain - plain data, imports nothing", 40, 1630, 1620, 290, BACKDROP
    )
    d.box(
        "models",
        "domain/models.py\nTurnRequest - TurnResult - ConversationState\n"
        "Message - SopDefinition - RunRecord",
        75,
        1670,
        340,
        100,
        CLIENT,
    )
    d.box(
        "memory-models",
        "domain/memory.py\nWorkingMemory - Fact - PendingInput\nMemoryUpdate",
        455,
        1670,
        280,
        100,
        CLIENT,
    )
    d.box(
        "knowledge-models",
        "domain/knowledge.py\nDocument - Chunk - Passage - Shelved",
        775,
        1670,
        330,
        100,
        CLIENT,
    )
    d.box("crm-models", "domain/crm.py\nContact - Order - Ticket", 1125, 1670, 230, 100, CLIENT)
    d.box(
        "tool-models",
        "domain/tools.py\nToolDefinition - ToolCall - ToolResult\nSafety - ToolStatus",
        75,
        1790,
        300,
        100,
        CLIENT,
    )
    d.box(
        "text-models",
        "domain/text.py\nreadable() - fold lookalikes,\ndrop what renders as nothing",
        415,
        1790,
        290,
        100,
        CLIENT,
    )
    d.box("errors", "errors.py\nEngineError", 745, 1790, 170, 100, CLIENT)

    d.arrow("main", "routers")
    d.arrow("routers", "deps")
    d.arrow("main", "build", via=[(225, 358), (940, 358)])
    d.arrow("deps", "build", via=[(1330, 370), (975, 370)])
    d.arrow("settings", "build", via=[(190, 535), (955, 535)])
    d.arrow("seed", "build")
    d.arrow("build", "components")
    d.arrow("build", "aiengine", via=[(955, 565), (175, 565)])
    d.arrow("aiengine", "executor")
    d.arrow("executor", "edges")
    d.arrow("executor", "nodes", via=[(685, 625), (1240, 625)])
    d.arrow("nodes", "graphstate")
    d.arrow("nodes", "runtime", via=[(1240, 765), (740, 765)], label="RunRuntimeNode")
    d.arrow("nodes", "matcher", via=[(1300, 765)], label="the routing nodes")
    for key, (_, x) in fanout.items():
        d.arrow("runtime", key, via=[(x + 120, 888)])
    d.arrow("registry", "knowledge", label="search_knowledge")
    d.arrow("runtime", "port-llm", via=[(62, 835), (62, 1354)])
    d.arrow(
        "nodes",
        "port-conv",
        via=[(1240, 775), (1625, 775), (1625, 1095), (335, 1095)],
        label="state and runs",
    )

    return d


# --------------------------------------------------------------------------
# 2. One turn, end to end
# --------------------------------------------------------------------------

COLUMN_X, COLUMN_W, NODE_H = 460, 240, 76
FIRST_Y, ROW_STEP = 290, 115
SPINE = (
    NodeKey.LOAD_STATE,
    NodeKey.GUARD_INPUT,
    NodeKey.ROUTE,
    NodeKey.SELECT_SOP,
    NodeKey.DETECT_SOP_DRIFT,
    NodeKey.BUILD_RUNTIME_REQUEST,
    NodeKey.RUN_RUNTIME,
    NodeKey.REVIEW_RESULT,
    NodeKey.PERSIST_STATE,
    NodeKey.FINALIZE,
)
ASIDE = {NodeKey.HANDOFF: 980, NodeKey.FAILED: 1095}

# Edges that would otherwise cut straight through the nodes between their ends.
DETOURS: dict[tuple[NodeKey, NodeKey], list[tuple[float, float]]] = {
    (NodeKey.LOAD_STATE, NodeKey.FINALIZE): [(240, 328), (240, 1363)],
    (NodeKey.SELECT_SOP, NodeKey.FINALIZE): [(300, 673), (300, 1380)],
    (NodeKey.GUARD_INPUT, NodeKey.PERSIST_STATE): [(360, 443), (360, 1248)],
    (NodeKey.SELECT_SOP, NodeKey.PERSIST_STATE): [(410, 673), (410, 1225)],
    (NodeKey.GUARD_INPUT, NodeKey.HANDOFF): [(760, 443), (760, 1018)],
    (NodeKey.RUN_RUNTIME, NodeKey.FAILED): [(770, 1040), (770, 1133)],
    (NodeKey.REVIEW_RESULT, NodeKey.HANDOFF): [(750, 1133), (750, 1056)],
}


def node_style(key: NodeKey) -> str:
    if key in STUBBED_NODES:
        return PENDING
    if key in (NodeKey.HANDOFF, NodeKey.FAILED):
        return STOP
    if key is NodeKey.PERSIST_STATE:
        return STORE
    if key is NodeKey.FINALIZE:
        return ENGINE
    return NODE


def turn_flow() -> Diagram:
    d = Diagram("turn-flow")
    d.title("title", "one turn - the graph, and what run_runtime does", 60, 40)
    d.text(
        "lede",
        "Every node and every arrow below is read from EDGES in "
        "engine/graph/edges.py. Nodes pick an outcome; the edge table alone "
        "decides where that outcome goes.",
        60,
        86,
    )

    doors = (
        ("in-turns", "POST /v1/turns\nsomebody spoke", 150),
        ("in-events", "POST /v1/events\nsomething happened", 230),
    )
    for key, label, y in doors:
        d.box(key, label, 60, y, 205, 70, CLIENT)
    d.box("in-engine", "AiEngine\nhandle(request)", 290, 190, 205, 70, ENGINE)
    d.box("in-exec", "GraphExecutor\nwalks the table", 520, 190, 205, 70, INTERNAL)
    for key, _, _ in doors:
        d.arrow(key, "in-engine")
    d.arrow("in-engine", "in-exec")

    classes = node_classes()
    for row, key in enumerate(SPINE):
        d.box(
            key,
            f"{key}\n{classes[key]}",
            COLUMN_X,
            FIRST_Y + row * ROW_STEP,
            COLUMN_W,
            NODE_H,
            node_style(key),
        )
    for key, y in ASIDE.items():
        d.box(key, f"{key}\n{classes[key]}", 820, y, 220, NODE_H, node_style(key))

    d.arrow("in-exec", NodeKey.LOAD_STATE, label="entry")
    for (source, target), outcomes in merged_edges().items():
        d.arrow(source, target, label=" / ".join(outcomes), via=DETOURS.get((source, target), ()))

    d.box(
        "result",
        "TurnResult\nreply - status - outcome",
        COLUMN_X,
        1440,
        COLUMN_W,
        NODE_H,
        ENGINE,
    )
    d.arrow(NodeKey.FINALIZE, "result")

    _runtime_inset(d)
    d.arrow(
        NodeKey.RUN_RUNTIME,
        "rt-prompt",
        via=[(760, 940), (1130, 940), (1130, 728)],
        dashed=True,
        label="expanded",
    )

    d.text(
        "legend",
        "grey nodes are PassThroughNode stubs waiting on a later phase\n"
        "red nodes end the turn without an answer from the model\n"
        "dashed arrows are the same thing seen closer up, not a step\n\n"
        "select_sop scores every procedure once and both routing nodes read that one Ranking: "
        f"within {AMBIGUITY} of each other and it asks which,\n"
        f"and a procedure already under way is only abandoned for one {MARGIN} clear of it.",
        60,
        1560,
    )
    return d


def merged_edges() -> dict[tuple[NodeKey, NodeKey], list[str]]:
    """Parallel edges share one arrow, so `no_match / ended` is drawn once."""
    merged: dict[tuple[NodeKey, NodeKey], list[str]] = {}
    for (source, outcome), target in EDGES.items():
        if target is not None:
            merged.setdefault((source, target), []).append(Outcome(outcome).value)
    return merged


def _runtime_inset(d: Diagram) -> None:
    d.group_box(
        "band-runtime",
        "run_runtime, up close - TextRuntime.run()",
        1130,
        620,
        780,
        800,
        BACKDROP,
    )
    d.box(
        "rt-prompt",
        "PromptBuilder.build()\nsections, context, tool catalog",
        1170,
        690,
        290,
        76,
        INTERNAL,
    )
    d.box(
        "rt-prompt-check",
        "Guardrails.check(PROMPT)\nmode: disabled today",
        1520,
        690,
        290,
        76,
        PENDING,
    )
    d.box(
        "rt-call", "LlmPort.complete()\nsystem, history, tools, exchanges", 1170, 810, 290, 76, NODE
    )
    d.box("rt-decide", "reply.wants_tools?", 1520, 810, 290, 76, NODE)
    d.box(
        "rt-tools",
        "ToolRegistry.execute()\nall calls at once, asyncio.gather",
        1520,
        930,
        290,
        90,
        INTERNAL,
    )
    d.box(
        "rt-screen",
        "Guardrails.check(MEMORY)\ndrops a poisoned fact,\nkeeps the reply",
        1520,
        1050,
        290,
        76,
        STOP,
    )
    d.box("rt-exchange", "ToolExchange\ncalls + results, fed back in", 1520, 1160, 290, 76, STORE)
    d.box("rt-final", "_final_answer()\none more ask, no tools offered", 1170, 1060, 290, 76, STOP)
    d.box(
        "rt-output-check", "Guardrails.check(OUTPUT)\nmode: observe", 1170, 1180, 290, 76, INTERNAL
    )
    d.box(
        "rt-reply",
        "RuntimeReply(text, blocked_by, tool_calls, memory)\n"
        "run_runtime sets state.reply and state.memory",
        1170,
        1300,
        640,
        76,
        ENGINE,
    )

    d.arrow("rt-prompt", "rt-prompt-check")
    d.arrow("rt-prompt-check", "rt-call")
    d.arrow("rt-call", "rt-decide", label="reply")
    d.arrow("rt-decide", "rt-tools", label="yes")
    d.arrow("rt-tools", "rt-screen", label="learned")
    d.arrow("rt-screen", "rt-exchange")
    d.arrow(
        "rt-exchange",
        "rt-call",
        via=[(1850, 1198), (1850, 788), (1315, 788)],
        label=f"max {MAX_TOOL_ROUNDS}",
    )
    d.arrow("rt-decide", "rt-final", via=[(1490, 848), (1490, 1098)], label="rounds used")
    d.arrow("rt-call", "rt-output-check", via=[(1150, 848), (1150, 1218)], label="no")
    d.arrow("rt-final", "rt-output-check")
    d.arrow("rt-output-check", "rt-reply")


# --------------------------------------------------------------------------
# 3. The seams you extend
# --------------------------------------------------------------------------

PANEL_W = 440
PANEL_X = (60, 530, 1000, 1470)


class Panel(NamedTuple):
    key: str
    title: str
    protocol: str
    used_by: str
    colour: str
    items: list[str]


def seams() -> Diagram:
    d = Diagram("seams")
    d.title("title", "the seams - where new behaviour plugs in", 60, 40)
    d.text(
        "lede",
        "Each panel is one protocol the engine depends on, and everything that "
        "satisfies it today. Adding behaviour means a new class in the list plus "
        "one line in composition.py; no existing class has to change.",
        60,
        86,
    )

    detectors = [f"{det.id}   ({checkpoints(det)})" for det in default_detectors()]
    sections = [f"{sec.priority:>4}   {sec.key}" for sec in DEFAULT_SECTIONS]
    tools = [f"{tool.name}   ({tool.safety})" for tool in build_tools(build_crm()).catalog()]

    panels = (
        Panel(
            "llm",
            "the model",
            "LlmPort",
            "TextRuntime",
            NODE,
            ["OpenAiLlm", "DummyLlm", "ScriptedLlm (tests/fakes.py)"],
        ),
        Panel("detector", "what gets checked", "Detector", "Guardrails", STOP, detectors),
        Panel("section", "what the prompt says", "Section", "PromptBuilder", INTERNAL, sections),
        Panel("tool", "what the model can do", "Tool[ArgsT]", "ToolRegistry", ENGINE, tools),
        Panel(
            "collector",
            "what the prompt knows",
            "Collector",
            "ContextPipeline",
            INTERNAL,
            ["ContactCollector", 'ResourceCollector("order")', 'ResourceCollector("ticket")'],
        ),
        Panel(
            "reader",
            "the customer's systems",
            "ContactReader\nRecordReader[RecordT]\nTicketWriter",
            "collectors and tools",
            STORE,
            [
                "InMemoryContactReader",
                "InMemoryRecordReader[Order]",
                "InMemoryRecordReader[Ticket]",
                "InMemoryTicketNotes",
            ],
        ),
        Panel(
            "store",
            "what survives a turn",
            "ConversationStore\nRunStore\nSopCatalog",
            "the nodes",
            STORE,
            ["InMemoryConversationStore", "InMemoryRunStore", "InMemorySopCatalog"],
        ),
        Panel(
            "knowledge",
            "what it can look up",
            "Embedder\nVectorStore",
            "KnowledgeBase and SopMatcher",
            CLIENT,
            ["OpenAiEmbedder", "HashEmbedder", "InMemoryVectorStore"],
        ),
    )

    row_one, row_two = panels[:4], panels[4:]
    top = 150
    height = _panel(d, row_one, top)
    _panel(d, row_two, top + height + 60)
    return d


def checkpoints(detector: object) -> str:
    return ", ".join(sorted(str(point) for point in detector.checkpoints))


def _panel(d: Diagram, panels: tuple[Panel, ...], y: float) -> float:
    """Draw a row of seam panels, all the same height, and report that height."""
    height = 200 + max(len(panel.items) for panel in panels) * 54
    for column, panel in enumerate(panels):
        x = PANEL_X[column]
        d.group_box(f"panel-{panel.key}", panel.title, x, y, PANEL_W, height, BACKDROP)
        d.box(f"proto-{panel.key}", panel.protocol, x + 25, y + 55, PANEL_W - 50, 80, panel.colour)
        d.text(
            f"used-{panel.key}",
            f"used by {panel.used_by}   -   implemented by",
            x + 25,
            y + 148,
            size=12,
        )
        for row, item in enumerate(panel.items):
            y_item = y + 180 + row * 54
            d.box(f"impl-{panel.key}-{row}", item, x + 25, y_item, PANEL_W - 50, 44, panel.colour)
    return height


# --------------------------------------------------------------------------
# 4. Where this is going
# --------------------------------------------------------------------------


class Phase(NamedTuple):
    number: str
    name: str
    done: bool
    detail: tuple[str, ...]


PHASES = (
    Phase(
        "0",
        "Test harness and safety net",
        True,
        (
            "tests/ with fakes and a wired TestClient",
            "create_app(settings, llm) so tests inject a model",
            "ruff config",
        ),
    ),
    Phase(
        "1",
        "Prompt builder and context",
        True,
        (
            "engine/prompt/: ContextPipeline, Collectors, PromptPolicy",
            "9 priority-ordered Sections, PromptBuilder",
            "POST /v1/prompts/preview",
        ),
    ),
    Phase(
        "2",
        "Guardrails",
        True,
        (
            "engine/guardrails/: 6 Detectors, GuardrailProfile",
            "Guardrails, GuardrailResponder",
            "GuardInputNode, a new edge before route",
        ),
    ),
    Phase(
        "3",
        "Tools and the tool loop",
        True,
        (
            "domain/tools.py, ports/tools.py, ToolRegistry, 4 tools",
            "LlmRequest / LlmReply / ToolExchange",
            "bounded loop in TextRuntime, GET /v1/tools",
        ),
    ),
    Phase(
        "4",
        "Working memory",
        True,
        (
            "identifiers only, written only by tools",
            "provenance, supersession archive, caps",
            "a MEMORY checkpoint screens every write",
        ),
    ),
    Phase(
        "5",
        "Knowledge and RAG",
        True,
        (
            "paragraph-first chunking, no overlap to pay for",
            "Embedder and VectorStore ports, top 3 over a floor",
            "retrieval is a tool, screened going in and coming out",
        ),
    ),
    Phase(
        "6",
        "Classifier routing",
        True,
        (
            "SopMatcher over the Embedder port, one ranking a turn",
            "select_sop chooses, detect_sop_drift changes its mind",
            "review_result escalates a conversation going nowhere",
        ),
    ),
    Phase(
        "7",
        "Events",
        True,
        (
            "a turn nobody typed, on the same graph",
            "procedures declare the events they answer",
            "the flow seam collapsed - one pipeline, said once",
        ),
    ),
    Phase(
        "8",
        "Persistence and observability",
        False,
        (
            "SQLite or Postgres behind the store ports",
            "run records worth querying, tracing on NodeObserver",
        ),
    ),
)

DEFERRED = (
    "reply formatting, the generateReply idea - rewrite the answer to a house style",
    "guardrail findings recorded on run records, not just acted on",
    "per-agent guardrail profiles instead of one DEFAULT_PROFILE",
    "a RAG quality detector - turns a gap in the library into a blocked turn",
    "an approximate index behind VectorStore, once exact search stops being fast",
    "guardrail checkpoints around tool arguments and tool results",
    "a model-based prompt injection classifier behind the regex one",
    "memory confidence, TTL, and re-promotion from the superseded archive",
)

CARD_W, CARD_H = 520, 180
CARD_X = (60, 620, 1180)
CARD_Y = (180, 400, 620)


def _middle(column: int) -> float:
    return CARD_X[column] + CARD_W / 2


def roadmap() -> Diagram:
    d = Diagram("roadmap")
    d.title("title", "phases - what is built, and what extends it", 60, 40)
    d.text(
        "lede",
        "Green is in the repository and under test. Grey is planned. Each grey "
        "card names the seam it arrives through, so nothing below needs the "
        "engine reshaped to land.",
        60,
        86,
    )

    for index, phase in enumerate(PHASES):
        row, column = divmod(index, len(CARD_X))
        d.box(
            f"phase-{phase.number}",
            f"Phase {phase.number}   {phase.name}\n\n" + "\n".join(phase.detail),
            CARD_X[column],
            CARD_Y[row],
            CARD_W,
            CARD_H,
            ENGINE if phase.done else PENDING,
            size=13,
        )

    for index in range(len(PHASES) - 1):
        row, column = divmod(index, len(CARD_X))
        here = f"phase-{PHASES[index].number}"
        following = f"phase-{PHASES[index + 1].number}"
        if column < len(CARD_X) - 1:
            d.arrow(here, following)
            continue
        wrap = CARD_Y[row] + CARD_H + 20
        d.arrow(here, following, via=[(_middle(column), wrap), (_middle(0), wrap)])

    d.group_box(
        "band-deferred", "deferred - looked at, deliberately not now", 60, 870, 1640, 340, BACKDROP
    )
    for row, item in enumerate(DEFERRED):
        d.box(f"deferred-{row}", item, 90, 920 + row * 40, 1580, 34, PENDING, size=13)

    return d


DIAGRAMS = {
    "01-layers.excalidraw": layers,
    "02-turn-flow.excalidraw": turn_flow,
    "03-seams.excalidraw": seams,
    "04-roadmap.excalidraw": roadmap,
}


def write_all(target: Path = HERE) -> list[Path]:
    written = []
    for filename, draw in DIAGRAMS.items():
        path = target / filename
        draw().dump(path)
        written.append(path)
    return written


if __name__ == "__main__":
    for written in write_all():
        print(f"wrote {written}")
