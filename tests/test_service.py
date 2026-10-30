import base64
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from aurel_ec import service
from aurel_ec.reorder_evidence import reorder_evidence
from m2a_cpu.aurel_adapter import AurelMemory
from m2a_cpu.answering import Answerer
from m2a_cpu.clients import APIClient, BudgetLedger
from m2a_cpu.config import APIConfig

TOKEN = "test-service-token-0123456789abcdef"


class Response(io.BytesIO):
    status = 200
    headers = {}
    def getcode(self):
        return 200


class FakeWire:
    def __init__(self):
        self.bodies = []
        self.fail = False
    def __call__(self, request, timeout):
        body = json.loads(request.data)
        self.bodies.append(body)
        if self.fail:
            raise OSError("SYNTHETIC_FAILURE")
        if "input" in body:
            data = {"data": [{"index": i, "embedding": [1.0] + [0.0] * 1023}
                             for i in range(len(body["input"]))]}
        else:
            schema = body.get("response_format", {}).get("json_schema", {}).get("schema")
            assert schema, "Extraction must use the frozen source-slot wire contract"
            frames = {}
            for name in schema["properties"]:
                i = int(name.removeprefix("record_"))
                frames[name] = {"index": i, "summary": "red cup", "facts": ["The cup is red."],
                    "entities": ["cup"], "objects": ["cup"], "ocr": [], "visual_facts": ["red cup"],
                    "relations": [], "event_time": [], "uncertainties": [], "state_updates": []}
            data = {"model": "gpt-4o-mini", "choices": [{"finish_reason": "stop",
                      "message": {"role": "assistant", "content": json.dumps(frames)}}]}
        return Response(json.dumps(data).encode())


@pytest.fixture
def stack(tmp_path):
    wire = FakeWire()
    ledger = BudgetLedger(tmp_path / "test.sqlite", {"approved": True,
        "authorization": "Offline synthetic transport only", "max_requests": 100,
        "max_output_tokens_per_request": 6000, "max_concurrency": 1,
        "max_retries": 0, "min_interval_seconds": 0})
    generation = APIClient(APIConfig("llm", "https://model.example.invalid/v1", "gpt-4o-mini", "synthetic-generation-key"), ledger, wire)
    embedding = APIClient(APIConfig("embedding", "https://embedding.example.invalid/v1", "text-embedding-v4", "synthetic-embedding-key", dimensions=1024), ledger, wire)
    memory = AurelMemory(tmp_path / "memory", SimpleNamespace(generation=generation, embedding=embedding),
        identity={"test": True}, variant="enhanced", package_root=service.ROOT / "vendor/aurel")
    memory.bridge.audit_path = None
    with TestClient(service.create_service(memory, TOKEN)) as client:
        yield client, memory, wire, ledger


def headers():
    return {"Authorization": "Token " + TOKEN}


def payload(content="The cup is red.", user="alice"):
    return {"request_id": "req1", "user_id": user, "session_id": "s1",
            "messages": [{"role": "user", "content": content, "timestamp": 1704067200000}]}


def test_add_search_source_slots_and_isolation(stack):
    client, memory, wire, ledger = stack
    p = payload()
    added = client.post("/add", json=p, headers=headers())
    assert added.status_code == 200, added.text
    assert added.json() == {"success": True, **{k: p[k] for k in ("request_id", "user_id", "session_id")}}
    assert ledger.count() == 2
    assert client.post("/add", json=p, headers=headers()).status_code == 200
    assert ledger.count() == 2
    result = client.post("/search", json={"user_id": "alice", "query": "cup", "top_k": 100}, headers=headers())
    assert result.status_code == 200, result.text
    assert len(result.json()["data"]) == 1
    assert "The cup is red." in result.json()["data"][0]["content"]
    assert "score" not in result.json()["data"][0]
    assert ledger.count() == 3
    other = client.post("/search", json={"user_id": "bob", "query": "cup", "top_k": 100}, headers=headers())
    assert other.json() == {"data": []}
    assert ledger.count() == 3


def test_auth_validation_and_conflict_do_not_leak(stack):
    client, memory, wire, ledger = stack
    assert client.post("/add", json=payload()).status_code == 401
    bad = payload("PRIVATE_SOURCE_SENTINEL"); bad["messages"][0]["role"] = "system"
    response = client.post("/add", json=bad, headers=headers())
    assert response.status_code == 422 and "PRIVATE_SOURCE_SENTINEL" not in response.text
    assert ledger.count() == 0
    assert client.post("/add", json=payload(), headers=headers()).status_code == 200
    assert client.post("/add", json=payload("different"), headers=headers()).status_code == 409
    assert ledger.count() == 2


def test_failure_is_atomic_without_automatic_retry(stack):
    client, memory, wire, ledger = stack
    wire.fail = True
    assert client.post("/add", json=payload(), headers=headers()).status_code == 503
    assert len(wire.bodies) == 1 and ledger.count() == 1
    assert memory.store.list_records("alice") == []
    wire.fail = False
    assert client.post("/add", json=payload(), headers=headers()).status_code == 200
    assert ledger.count() == 3


def test_images_return_original_bytes(stack):
    client, memory, wire, ledger = stack
    out = io.BytesIO(); Image.new("RGB", (16, 16), "red").save(out, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()
    content = [{"type": "text", "text": "red cup"}, {"type": "image_url", "image_url": {"url": uri}}]
    assert client.post("/add", json=payload(content), headers=headers()).status_code == 200
    data = client.post("/search", json={"user_id": "alice", "query": "red cup", "top_k": 100}, headers=headers()).json()["data"]
    assert data[0]["content"][-1] == content[-1]


def test_completion_keeps_real_answer_input_identical_to_frozen_selector():
    evidence = [{"id": str(i), "content": "x" * size, "score": 4-i}
                for i, size in enumerate([70000, 60000, 20000, 20000])]
    before = copy.deepcopy(evidence)
    frozen, audit = reorder_evidence("question", [], evidence, service.ANSWER_CONFIG)
    delivered, new_audit = service.complete_evidence("question", [], evidence)
    assert audit["added_items"] == 2 and audit == new_audit and evidence == before
    assert [x["id"] for x in delivered["data"]] == ["0", "2", "3", "1"]
    class Capture:
        model = "gpt-4o-mini"
        def chat(self, messages, **kwargs):
            self.messages = messages
            return {"message": {"content": "synthetic"}}
    c1, c2 = Capture(), Capture()
    Answerer(c1, service.ANSWER_CONFIG)("question", [], frozen)
    Answerer(c2, service.ANSWER_CONFIG)("question", [], delivered["data"])
    assert c1.messages == c2.messages


def test_source_lock_and_disallowed_token():
    assert len(service.verify_sources()) == 64
    with pytest.raises(ValueError):
        service.create_service(None, "short")


def test_response_media_cap_keeps_whole_records(monkeypatch):
    out = io.BytesIO(); Image.new("RGB", (8, 8), "red").save(out, "PNG")
    raw = out.getvalue()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}}
    items = [{"id": str(i), "content": [{"type": "text", "text": "record " + str(i)}, image]}
             for i in range(2)] + [{"id": "text", "content": "small text"}]
    monkeypatch.setattr(service, "MEDIA_LIMIT", len(raw))
    result, audit = service.complete_evidence("red", [], items)
    assert result["data"] == [items[0], items[2]]
    assert result["data"][0]["content"] == items[0]["content"]


def test_formal_top100_keeps_frozen_candidate_policy(stack, monkeypatch):
    client, memory, wire, ledger = stack
    seen = []
    def search(request):
        seen.append(request["top_k"])
        return {"data": []}
    monkeypatch.setattr(memory, "search", search)
    for top_k in (100, 5):
        assert client.post("/search", json={"user_id": "a", "query": "q", "top_k": top_k}, headers=headers()).json() == {"data": []}
    assert seen == [10, 5] and ledger.count() == 0


def test_disabled_budget_cannot_call_models(stack):
    client, memory, wire, ledger = stack
    with TestClient(service.create_service(memory, TOKEN, lambda: False)) as disabled:
        assert disabled.get("/healthz").status_code == 503
        assert disabled.post("/add", json=payload(), headers=headers()).status_code == 503
        assert disabled.get("/version").json()["version"] == service.VERSION
    assert ledger.count() == 0


def test_real_factory_disabled_allowance(tmp_path, monkeypatch):
    (tmp_path / ".env.local").write_text("OPENAI_API_KEY=synthetic-generation-key\nOPENAI_API_BASE=https://model.example.invalid/v1\nOPENAI_EMBEDDING_API_KEY=synthetic-embedding-key\nOPENAI_EMBEDDING_API_BASE=https://embedding.example.invalid/v1\n")
    (tmp_path / ".env.leaderboard.local").write_text("MEMORY_API_TOKEN=" + TOKEN + "\n")
    monkeypatch.setenv("AUREL_EC_CONFIG_ROOT", str(tmp_path))
    monkeypatch.setenv("AUREL_EC_STATE", str(tmp_path / "state"))
    with TestClient(service.app_factory()) as client:
        assert client.get("/healthz").status_code == 503
        assert client.post("/search", json={"user_id": "new", "query": "q", "top_k": 100}, headers=headers()).status_code == 503
    import sqlite3
    with sqlite3.connect(next((tmp_path / "state").glob("http-*.sqlite"))) as db:
        assert db.execute("SELECT COUNT(*) FROM budget_attempts").fetchone()[0] == 0
