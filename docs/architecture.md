# Architecture

These diagrams show the code wired by
[`build()`](../src/agentic_kit/composition.py). Arrows show startup wiring,
request flow, or data exchange as labeled; each section names its current
implementation.

## 1. Components and adapters

```mermaid
flowchart TB
    caller[HTTP client] --> api[FastAPI: turns, events, knowledge, inspection]
    startup[FastAPI lifespan] --> composition[Composition: build components at startup]
    composition --> engine
    api --> front

    subgraph engine[Conversation engine]
        direction TB
        front[AiEngine] --> executor[GraphExecutor: nodes and edge table]
        executor --> routing[Routing: SOP catalog and embedding matcher]
        executor --> runtime[TextRuntime: prompt, model, tools, reply]
        executor --> policy[Guardrails and handoff]
        runtime --> prompts[PromptBuilder: context collectors and sections]
        runtime --> registry[ToolRegistry: schemas, availability, confirmation]
        registry --> knowledge[KnowledgeBase: ingest and search passages]
    end

    composition --> model[Model adapter: OpenAiLlm or DummyLlm]
    composition --> embedder[Embedding adapter: OpenAiEmbedder or HashEmbedder]
    composition --> stores[Conversation and run stores: in memory or SQLite]
    composition --> catalog[Seeded InMemorySopCatalog]
    composition --> crm[Seeded in-memory CRM: contacts, orders, tickets]
    composition --> vectors[InMemoryVectorStore]

    runtime --> model
    routing --> embedder
    routing --> catalog
    prompts --> crm
    registry --> crm
    knowledge --> embedder
    knowledge --> vectors
    executor --> stores
```

`ports/` defines the interfaces between the engine and adapters. The model
and embedder switch together when a real `OPENAI_API_KEY` is configured.
`SQLITE_PATH` switches only the conversation and run stores.

## 2. One turn through the graph

Both a customer message and a named event enter the same graph. An event
selects the SOP that declares its name; a customer message is ranked against
SOP descriptions and examples. The graph's
[`EDGES`](../src/agentic_kit/engine/graph/edges.py) table owns the branches.

```mermaid
flowchart TB
    request[POST /v1/turns or /v1/events] --> load[Load or start conversation]
    load --> input[Guard input]
    input --> route[Load active SOP]
    route --> select[Select SOP: event name or ranked message]
    select --> drift[Detect SOP drift]
    drift --> pass[Build runtime request: pass-through node]
    pass --> prompt[Build prompt: SOP, CRM context, working memory, tools]
    prompt --> promptGuard[Prompt guardrail checkpoint]
    promptGuard --> model[Ask model]
    model --> toolChoice{Tool calls?}
    toolChoice -- yes, up to 3 rounds --> tools[Validate and run offered tools]
    tools --> memory[Screen learned facts and collect tool results]
    memory --> model
    toolChoice -- no --> output[Output guardrail checkpoint]
    toolChoice -- after 3 tool rounds --> finalAsk[Final ask without tools]
    finalAsk --> output
    output --> review[Review result]
    review --> persist[Save history, working facts, and run record]
    persist --> finish[Finalize TurnResult]

    load -- invalid, closed, or waiting for human --> finish
    input -- blocked --> persist
    input -- handoff --> handoff[Handoff]
    select -- ambiguous: ask for clarification --> persist
    select -- no matching SOP --> finish
    promptGuard -- blocked --> review
    promptGuard -- handoff --> handoff
    model -- provider failure --> failed[Failed]
    output -- handoff --> handoff
    review -- repeated stall --> handoff
    failed --> persist
    handoff --> persist
```

The runtime offers six tools: `track_order`, `lookup_ticket`,
`lookup_contact`, `add_ticket_note`, `search_knowledge`, and
`finish_procedure`. The registry checks arguments and requires confirmation
for write tools. Tool results can contribute identifier facts to working
memory; those facts are committed when the turn is persisted. The default
guardrail profile enforces input, memory, and knowledge checks; prompt checks
are disabled and output checks observe only. The guarded branches shown above
are available when a profile enables them.

## 3. Memory, knowledge, and persistence

```mermaid
flowchart TB
    turn[Current turn] --> state[GraphState: transient data for this turn]
    state --> persist[PersistStateNode]
    state --> runtime[TextRuntime]
    conversation[ConversationState] --> history[Recent conversation history: up to 20 messages]
    conversation --> facts[WorkingMemory: active tool-learned identifier facts and pending input]
    history --> runtime
    facts --> runtime
    facts --> archive[Superseded facts: traceable, omitted from prompts]
    runtime --> updates[Screened MemoryUpdate from tools]
    updates --> persist[PersistStateNode]
    persist --> conversation
    persist --> runs[RunRecord: one turn's status, outcome, SOP, and reason]

    conversation --> convStore{ConversationStore}
    runs --> runStore{RunStore}
    convStore --> inMem[InMemoryConversationStore]
    convStore --> sqlite[SqliteConversationStore]
    runStore --> inMemRuns[InMemoryRunStore]
    runStore --> sqliteRuns[SqliteRunStore]

    documents[Seeded and API-added documents] --> knowledge[KnowledgeBase]
    knowledge --> screening[Screen, chunk, embed]
    screening --> vector[InMemoryVectorStore: passage vectors]
    vector --> search[Top passages, screened again]
    search --> searchTool[search_knowledge tool]
    searchTool --> runtime

    sops[Seeded SOPs] --> sopCatalog[InMemorySopCatalog]
    sopCatalog --> routing[SOP lookup and routing]
    routing --> state
    crm[Seeded contacts, orders, tickets] --> crmReaders[In-memory CRM readers and ticket notes]
    crmReaders --> runtime
```

`ConversationState` holds recent dialogue, the active SOP, and `WorkingMemory`.
Only tools write working facts; current order and ticket data is read from CRM
again rather than cached as facts. `RunRecord` is an audit summary of a turn,
not a transcript. SQLite can retain conversations and run records across
restarts. The vector index, SOP catalog, and sample CRM remain in memory and
are recreated on startup. The knowledge base is a reference document search
facility, not a persistent customer memory store.
