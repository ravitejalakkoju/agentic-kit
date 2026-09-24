from __future__ import annotations

from fastapi.testclient import TestClient

from agentic_kit.engine.graph.edges import EDGES, ENTRY
from agentic_kit.errors import ProviderError
from agentic_kit.seed import SUPPORT_SOP

from .fakes import ScriptedLlm

TURN = {"conversation_id": "conv-1", "customer_id": "cust-1", "text": "Where is my order?"}


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_turn_then_inspect_the_conversation(client: TestClient, llm: ScriptedLlm) -> None:
    llm.queue("It ships tomorrow.")

    turn = client.post("/v1/turns", json=TURN)

    assert turn.status_code == 200
    body = turn.json()
    assert body["status"] == "responded"
    assert body["reply"] == "It ships tomorrow."

    conversation = client.get("/v1/conversations/conv-1")
    assert conversation.status_code == 200
    assert [m["text"] for m in conversation.json()["history"]] == [
        "Where is my order?",
        "It ships tomorrow.",
    ]


def test_provider_failure_is_a_handled_turn_not_an_http_error(
    client: TestClient, llm: ScriptedLlm
) -> None:
    llm.queue(ProviderError("timeout"))

    response = client.post("/v1/turns", json=TURN)

    assert response.status_code == 200
    assert response.json()["status"] == "failed"


def test_invalid_body_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/turns", json={"text": "missing ids"})

    assert response.status_code == 422


def test_unknown_conversation_is_404(client: TestClient) -> None:
    assert client.get("/v1/conversations/nope").status_code == 404


def test_sops_lists_the_seeded_procedure(client: TestClient) -> None:
    response = client.get("/v1/sops")

    assert response.status_code == 200
    assert [sop["sop_id"] for sop in response.json()] == [SUPPORT_SOP.sop_id]


def test_graph_mirrors_the_edge_table(client: TestClient) -> None:
    response = client.get("/v1/graph")

    assert response.status_code == 200
    body = response.json()
    assert body["entry"] == ENTRY
    assert len(body["edges"]) == len(EDGES)
