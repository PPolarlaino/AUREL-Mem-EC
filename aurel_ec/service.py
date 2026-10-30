"""Deploy the frozen v3 + evidence-completion method as authenticated Add/Search."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from m2a_cpu.aurel_adapter import AurelMemory
from m2a_cpu.clients import APIClient, APIError, BudgetError, BudgetLedger
from m2a_cpu.config import load_config, read_env_file
from m2a_cpu.telemetry import api_context
from .reorder_evidence import reorder_evidence

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_NAME = "AUREL-Mem-EC"
VERSION = "v3-ec-20261030"
# This is one public, dataset-independent delivery profile. No routing by IDs.
# memeye selects the frozen generic local packing prompt; no Answer is executed.
ANSWER_CONFIG = {"dataset": "memeye", "model": "gpt-4o-mini", "context_window": 128000,
                 "reserved_output_tokens": 8192, "safety_tokens": 2048, "max_output_tokens": 1024}
CANDIDATE_LIMIT = 10
MEDIA_LIMIT = 30 * 1024 * 1024


def verify_sources():
    lock = json.loads((ROOT / "source-lock.json").read_text())
    for name, expected in lock["files"].items():
        path = ROOT / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Frozen method source differs: " + name)
    manifest = ROOT / "release-manifest.json"
    if manifest.exists():
        for name, expected in json.loads(manifest.read_text())["files"].items():
            path = ROOT / name
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Release source differs: " + name)
    return hashlib.sha256((ROOT / "source-lock.json").read_bytes()).hexdigest()


def complete_evidence(query, options, evidence):
    """Frozen completion first; apply the public media cap to whole records only."""
    ordered, audit = reorder_evidence(query, options, evidence, ANSWER_CONFIG)
    result, total = [], 0
    for item in ordered:
        parts = item["content"] if isinstance(item["content"], list) else []
        size = 0
        for part in parts:
            if part["type"] == "image_url":
                encoded = part["image_url"]["url"].split(",", 1)[1]
                size += len(encoded) * 3 // 4 - len(encoded) + len(encoded.rstrip("="))
        if total + size > MEDIA_LIMIT:
            continue
        total += size
        # Optional original relevance scores do not express the new delivery
        # order. Omitting score does not change the local Answer input bytes.
        result.append({k: v for k, v in item.items() if k != "score"})
    return {"data": result}, audit


def create_service(memory, token, ready=lambda: True):
    """An injectable boundary for offline HTTP tests; the real factory is below."""
    if not isinstance(token, str) or len(token) < 32 or any(c.isspace() for c in token):
        raise ValueError("A dedicated service token of at least32 characters is required")
    serial = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        memory.close()

    app = FastAPI(title=SYSTEM_NAME, version=VERSION, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(memory.package.app.BodyLimitMiddleware, limit=48 * 1024 * 1024)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path in {"/healthz", "/version"}:
            return await call_next(request)
        headers = request.headers.getlist("Authorization")
        value = headers[0] if len(headers) == 1 else ""
        scheme, _, supplied = value.partition(" ")
        valid = scheme.lower() in {"bearer", "token"} and hmac.compare_digest(
            supplied.encode(), token.encode())
        if not valid:
            return JSONResponse({"detail": "invalid_service_authorization"}, status_code=401)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(request, exc):
        return JSONResponse({"detail": "invalid_request"}, status_code=422)

    @app.get("/healthz")
    def health():
        available = bool(ready())
        return JSONResponse({"status": "ok" if available else "awaiting_model_allowance",
                             "version": VERSION}, status_code=200 if available else 503)

    @app.get("/version")
    def version():
        return {"system": SYSTEM_NAME, "version": VERSION, "candidate_limit": CANDIDATE_LIMIT,
                "method": "aurel-v3-source-slots-whole-records-evidence-completion",
                "generation_model": "gpt-4o-mini", "embedding_model": "text-embedding-v4",
                "embedding_dimensions": 1024, "official_score": None}

    def invoke(operation, payload):
        try:
            schema = memory.package.schemas.AddRequest if operation == "add" else memory.package.schemas.SearchRequest
            validated = schema.model_validate(payload).model_dump(exclude_none=True)
        except ValidationError:
            return JSONResponse({"detail": "invalid_request"}, status_code=422)
        with serial:
            if not ready():
                return JSONResponse({"detail": "model_allowance_unavailable"}, status_code=503)
            scope = hashlib.sha256(validated["user_id"].encode()).hexdigest()
            try:
                with api_context(run_id="aml-v3-ec-20261030", stage=operation, scope_hash=scope):
                    if operation == "add":
                        return memory.add(validated)
                    validated["top_k"] = min(validated["top_k"], CANDIDATE_LIMIT)
                    result = memory.search(validated)
                    return complete_evidence(validated["query"], validated.get("options"), result["data"])[0]
            except memory.package.storage.ConflictError:
                return JSONResponse({"detail": "request_id_conflict"}, status_code=409)
            except (APIError, BudgetError, memory.package.provider.UpstreamError):
                return JSONResponse({"detail": "model_or_allowance_unavailable"}, status_code=503)
            except Exception:
                # No provider responses, credentials, or source text in public errors.
                return JSONResponse({"detail": "memory_operation_failed"}, status_code=503)

    @app.post("/add")
    def add(payload: dict):
        return invoke("add", payload)

    @app.post("/search")
    def search(payload: dict):
        return invoke("search", payload)

    app.state.memory = memory
    return app


def app_factory():
    verify_sources()
    os.umask(0o077)
    config_root = Path(os.environ.get("AUREL_EC_CONFIG_ROOT", ROOT)).resolve()
    state = Path(os.environ.get("AUREL_EC_STATE", ROOT / "state")).resolve()
    control = read_env_file(config_root / ".env.leaderboard.local")
    token = control.get("MEMORY_API_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Dedicated MEMORY_API_TOKEN is missing from private leaderboard configuration")
    config = load_config(config_root)
    if (config.llm.model != "gpt-4o-mini" or config.embedding.model != "text-embedding-v4"
            or config.embedding.dimensions != 1024):
        raise RuntimeError("Frozen model configuration mismatch")
    policy_path = Path(os.environ.get("AUREL_EC_BUDGET_POLICY", ROOT / "deploy/budget.disabled.json"))
    policy = json.loads(policy_path.read_text())
    if policy.get("max_concurrency") != 1 or policy.get("max_retries") != 0:
        raise RuntimeError("The release requires serial transport and no automatic network retries")
    # A new budget creates a distinct immutable ledger; never reuse experiment ledgers.
    policy_id = hashlib.sha256(policy_path.read_bytes()).hexdigest()[:16]
    ledger = BudgetLedger(state / ("http-" + policy_id + ".sqlite"), policy)
    generation, embedding = APIClient(config.llm, ledger), APIClient(config.embedding, ledger)
    identity = {"generation": generation.identity, "embedding": embedding.identity,
                "service_version": VERSION, "candidate_limit": CANDIDATE_LIMIT}
    memory = AurelMemory(state / "memory", SimpleNamespace(generation=generation, embedding=embedding),
                         identity=identity, variant="enhanced", package_root=ROOT / "vendor/aurel")
    # Match the actual full29 transport timeout; constructors do not send HTTP.
    generation.TIMEOUT_SECONDS = embedding.TIMEOUT_SECONDS = 120
    memory.bridge.audit_path = None  # Official source data stays in private DB, not prompt logs.
    def ready():
        if not policy.get("approved"):
            return False
        # Keep health/search readiness O(1) in returned rows on large campaigns.
        with ledger._db() as db:
            ledger._check(db)
            used = db.execute("SELECT COUNT(*) FROM budget_attempts").fetchone()[0]
        return used < policy["max_requests"]
    return create_service(memory, token, ready)
