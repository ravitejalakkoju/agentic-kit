# agentic-kit

A small customer support conversation engine behind FastAPI. A request enters a
typed graph, selects a standard operating procedure (SOP), builds a prompt,
runs a bounded model and tool loop, and saves the result. The project includes
sample customers, orders, tickets, procedures, and policy documents so it can
be explored immediately.

Start with the [architecture guide](docs/architecture.md) for Mermaid diagrams
of the components, turn flow, and memory. The [diagram index](docs/diagrams/README.md)
also links to the existing Excalidraw views.

## Run with OpenAI

You need Python 3.12 or newer and an OpenAI API key. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Edit `.env` and replace `OPENAI_API_KEY=dummy-openai-key` with your real key.
The default chat model is `gpt-4.1-mini` and the embedding model is
`text-embedding-3-small`; both are configurable in `src/agentic_kit/settings.py`.
The app reads `.env` from the working directory, so start it from the repo root:

```sh
python -m uvicorn agentic_kit.main:app --host 127.0.0.1 --port 3200
```

Open [http://127.0.0.1:3200/docs](http://127.0.0.1:3200/docs) to explore the
API. Try a turn with the seeded customer and order:

```sh
curl -sS http://127.0.0.1:3200/v1/turns \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"demo-1","customer_id":"cust-1","text":"Where is my order?","context":{"order_id":"1001"}}'
```

The response contains a status, outcome, reply, and run ID. Model wording can
vary. Reuse `demo-1` for another turn, then inspect what was saved:

```sh
curl -sS http://127.0.0.1:3200/v1/conversations/demo-1
curl -sS http://127.0.0.1:3200/v1/conversations/demo-1/runs
```

The `PORT` value in `.env` is a setting, but Uvicorn uses the explicit `--port`
argument above; changing `PORT` alone will not change the listening port.

## Run without an API key

For a local walkthrough, leave the example `dummy-openai-key` in `.env` (or
leave `OPENAI_API_KEY` empty) and use the same Uvicorn command. The app then
uses `DummyLlm` and `HashEmbedder`, so it needs no provider connection. The
dummy model echoes the latest user message; it does not call tools or produce
real support answers. The sample request and inspection commands above still
work.

## Run with Docker Compose

If Docker Compose is installed, the app can be built and started with one
command. Without a `.env` file, Compose uses the offline dummy model:

```sh
docker compose up --build --detach
docker compose ps
curl -sS http://127.0.0.1:3200/health
```

For live OpenAI replies, copy `.env.example` to `.env` and set a real
`OPENAI_API_KEY` before running `docker compose up --build --detach`. Compose
reads that file and passes the key and model settings into the container. The
sample turn and inspection commands above work against the Compose app too.
`PORT` changes the host port; the app listens on port 3200 inside the container.

Compose stores conversations and run records in a named SQLite volume at
`/data/agentic-kit.db`, even when `SQLITE_PATH` is blank in the host `.env`.
The SOP catalog, sample CRM, and vector knowledge index are still recreated
in memory at startup. To inspect logs or stop the app:

```sh
docker compose logs --tail=50 api
docker compose down
```

`docker compose down` retains the SQLite volume for the next start.

## Optional persistence

When running directly with Python, conversations and run records live only in
process memory by default. To retain those across restarts, set
`SQLITE_PATH=./agentic-kit.db` in `.env` before starting the server. SQLite
stores the conversation state, including
recent history and working facts, plus run records. The seeded SOP catalog,
sample CRM, and vector knowledge index are always rebuilt in memory when the
app starts. Keep the SQLite file outside version control.

## Navigate the code

| Area | What to read |
| --- | --- |
| `src/agentic_kit/main.py`, `api/` | App startup and HTTP endpoints for turns, events, runs, prompts, tools, graph, and knowledge. |
| `src/agentic_kit/composition.py` | The wiring point for graph nodes, model, stores, tools, guardrails, and seeded data. |
| `src/agentic_kit/engine/graph/`, `engine/nodes/` | The turn state, edge table, executor, and decisions at each graph step. |
| `src/agentic_kit/engine/runtime/`, `engine/prompt/` | Prompt construction and the model/tool exchange. |
| `src/agentic_kit/engine/routing/`, `engine/guardrails/` | SOP selection, drift detection, and policy checkpoints. |
| `src/agentic_kit/engine/tools/`, `engine/knowledge/` | Tool registry and document ingestion/search. |
| `src/agentic_kit/domain/`, `ports/`, `adapters/` | Shared models, interfaces, and in-memory, SQLite, and OpenAI implementations. |
| `src/agentic_kit/seed.py` | The three example SOPs and sample CRM and knowledge data. |

Conversation turns use `POST /v1/turns`. Named system events use
`POST /v1/events` and pass through the same graph. `GET /v1/graph`,
`GET /v1/sops`, `GET /v1/tools`, and `POST /v1/prompts/preview` are useful
inspection endpoints; prompt preview does not call the model.

## Tests

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
```
