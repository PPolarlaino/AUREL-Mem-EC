"""Opt-in OpenAI-compatible clients with a process-safe, immutable request budget.

No environment is read here. Transport injection supports completely offline testing.
"""
from __future__ import annotations

import base64
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from typing import Optional

from .config import APIConfig, valid_base_url
from .telemetry import (aggregate_attempt_rows, current_api_context, validate_context,
                        validate_elapsed)


_ERROR_CODES = {
    "client_error", "configuration_incomplete", "invalid_configuration", "invalid_embedding_request",
    "invalid_extra_body", "invalid_json", "invalid_request", "invalid_schema", "request_too_large",
    "response_too_large", "secret_in_request", "secret_in_response", "transport_error", "http_error",
    "authentication_failed", "access_forbidden", "endpoint_not_found", "rate_limited", "quota_exhausted",
    "model_not_found", "thinking_unsupported", "budget_client_identity_mismatch", "budget_not_approved",
    "budget_policy_or_host_mismatch", "concurrency_or_rate_wait_timeout", "incompatible_budget_ledger",
    "invalid_budget_attempt", "invalid_budget_operation", "invalid_budget_policy", "invalid_budget_result",
    "output_token_budget_exhausted", "output_token_limit_exceeded", "request_limit_reached",
}


class APIError(RuntimeError):
    """Only fixed local diagnostic codes are exposed to callers."""
    def __init__(self, code: str, *, http_status: Optional[int] = None):
        self.code = code if isinstance(code, str) and code in _ERROR_CODES else "client_error"
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        super().__init__(self.code)


class BudgetError(APIError):
    pass


_DEFAULTS = {
    "approved": False,
    "authorization": "",
    "max_requests": 0,
    "max_output_tokens_per_request": 2048,
    "max_reserved_output_tokens": None,
    "max_concurrency": 2,
    "max_retries": 1,
    "min_interval_seconds": 0.25,
    "retry_base_seconds": 1.0,
    "max_retry_delay_seconds": 30.0,
    "acquire_timeout_seconds": 60.0,
    "max_request_bytes": 64 * 1024 * 1024,
    "max_response_bytes": 8 * 1024 * 1024,
}
_USAGE_KEYS = {"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"}
_OUTCOMES = {"ok", "http_error", "transport_error", "invalid_json", "invalid_schema",
             "response_too_large", "secret_in_response", "abandoned", "client_error"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _int(value, minimum=0, maximum=10**9):
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _policy(policy):
    incoming = dict(policy or {})
    if "requests" in incoming:
        if "max_requests" in incoming:
            raise BudgetError("invalid_budget_policy")
        incoming["max_requests"] = incoming.pop("requests")
    if set(incoming) - set(_DEFAULTS):
        raise BudgetError("invalid_budget_policy")
    result = {**_DEFAULTS, **incoming}
    if (type(result["approved"]) is not bool or not isinstance(result["authorization"], str)
            or len(result["authorization"]) > 2000
            or (result["approved"] and not result["authorization"].strip())):
        raise BudgetError("invalid_budget_policy")
    for key, low, high in (("max_requests", 0, 10**7), ("max_output_tokens_per_request", 1, 65536),
                           ("max_concurrency", 1, 32), ("max_retries", 0, 3),
                           ("max_request_bytes", 1, 64 * 1024 * 1024),
                           ("max_response_bytes", 1, 64 * 1024 * 1024)):
        if not _int(result[key], low, high):
            raise BudgetError("invalid_budget_policy")
    if result["max_reserved_output_tokens"] is not None and not _int(result["max_reserved_output_tokens"]):
        raise BudgetError("invalid_budget_policy")
    for key in ("min_interval_seconds", "retry_base_seconds", "max_retry_delay_seconds", "acquire_timeout_seconds"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 300:
            raise BudgetError("invalid_budget_policy")
        result[key] = float(value)
    return result


def safe_usage(data):
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    if not isinstance(usage, dict):
        return {}
    return {key: value for key, value in usage.items() if key in _USAGE_KEYS and _int(value)}


class BudgetLedger:
    """SQLite reservations are shared across threads/processes on one host.

    Reservations are never refunded. A process crash consumes the request; dead
    process slots are released without removing their history. Keep this database
    on a local filesystem (not NFS), and preserve it across restarts.
    """
    def __init__(self, path, policy=None):
        self.path = Path(path)
        self._policy = _policy(policy)
        self.fingerprint = _hash(self._policy)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        # Reject unrelated databases, including the separate exhausted probe ledger.
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if names and "budget_metadata" not in names:
                raise BudgetError("incompatible_budget_ledger")
            db.execute("CREATE TABLE IF NOT EXISTS budget_metadata (singleton INTEGER PRIMARY KEY CHECK(singleton=1), fingerprint TEXT NOT NULL, policy_json TEXT NOT NULL, host TEXT NOT NULL)")
            row = db.execute("SELECT fingerprint,host FROM budget_metadata WHERE singleton=1").fetchone()
            if row and (row[0] != self.fingerprint or row[1] != socket.gethostname()):
                raise BudgetError("budget_policy_or_host_mismatch")
            if not row:
                db.execute("INSERT INTO budget_metadata VALUES(1,?,?,?)", (self.fingerprint, _canonical(self._policy), socket.gethostname()))
            db.execute("CREATE TABLE IF NOT EXISTS budget_clients (role TEXT PRIMARY KEY, identity_hash TEXT NOT NULL, identity_json TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS budget_attempts (id INTEGER PRIMARY KEY, started REAL NOT NULL, owner_pid INTEGER NOT NULL, role TEXT NOT NULL, operation TEXT NOT NULL, identity_hash TEXT NOT NULL, reserved_output_tokens INTEGER NOT NULL, active INTEGER NOT NULL, outcome TEXT NOT NULL, http_status INTEGER, usage_json TEXT NOT NULL DEFAULT '{}')")
            # Append-only attribution schema; old attempts/policy stay unchanged.
            # Missing rows explicitly identify pre-attribution legacy attempts.
            db.execute("CREATE TABLE IF NOT EXISTS budget_attempt_context (attempt_id INTEGER PRIMARY KEY, run_id TEXT, stage TEXT, scope_hash TEXT, qid TEXT, call_id TEXT, retry_index INTEGER, finished REAL, elapsed_seconds REAL, http_elapsed_seconds REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS budget_attempt_context_run_stage ON budget_attempt_context(run_id,stage)")
        self.path.chmod(0o600)

    @property
    def policy(self):
        return copy.deepcopy(self._policy)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _check(self, db):
        row = db.execute("SELECT fingerprint FROM budget_metadata WHERE singleton=1").fetchone()
        if not row or row[0] != self.fingerprint:
            raise BudgetError("budget_policy_or_host_mismatch")

    def bind_client(self, identity):
        digest = _hash(identity)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check(db)
            row = db.execute("SELECT identity_hash FROM budget_clients WHERE role=?", (identity["role"],)).fetchone()
            if row and row[0] != digest:
                raise BudgetError("budget_client_identity_mismatch")
            if not row:
                db.execute("INSERT INTO budget_clients VALUES(?,?,?)", (identity["role"], digest, _canonical(identity)))
        return digest

    def reserve(self, role, operation, identity_hash, output_tokens=0, *, context=None, call_id=None, retry_index=None):
        policy = self._policy
        attribution = validate_context(current_api_context() if context is None else context)
        if ((call_id is None) != (retry_index is None) or
                (call_id is not None and (not isinstance(call_id, str) or not re.fullmatch(r'[0-9a-f]{32}', call_id))) or
                (retry_index is not None and not _int(retry_index, 0, policy['max_retries']))):
            raise BudgetError('invalid_budget_operation')
        if not policy["approved"]:
            raise BudgetError("budget_not_approved")
        if not _int(output_tokens, 0, policy["max_output_tokens_per_request"]):
            raise BudgetError("output_token_limit_exceeded")
        if operation not in {"chat", "embed"}:
            raise BudgetError("invalid_budget_operation")
        deadline = time.monotonic() + policy["acquire_timeout_seconds"]
        while True:
            wait = 0.01
            with self._db() as db:
                db.execute("BEGIN IMMEDIATE")
                self._check(db)
                identity = db.execute("SELECT identity_hash FROM budget_clients WHERE role=?", (role,)).fetchone()
                if not identity or identity[0] != identity_hash:
                    raise BudgetError("budget_client_identity_mismatch")
                count, reserved = db.execute("SELECT COUNT(*), COALESCE(SUM(reserved_output_tokens),0) FROM budget_attempts").fetchone()
                if count >= policy["max_requests"]:
                    raise BudgetError("request_limit_reached")
                cap = policy["max_reserved_output_tokens"]
                if cap is not None and reserved + output_tokens > cap:
                    raise BudgetError("output_token_budget_exhausted")
                # Fail conservatively if a PID was reused; only proven dead PIDs free slots.
                for (pid,) in db.execute("SELECT DISTINCT owner_pid FROM budget_attempts WHERE active=1").fetchall():
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        db.execute("UPDATE budget_attempts SET active=0,outcome='abandoned' WHERE owner_pid=? AND active=1", (pid,))
                    except PermissionError:
                        pass
                active, latest = db.execute("SELECT COALESCE(SUM(active),0),COALESCE(MAX(started),0) FROM budget_attempts").fetchone()
                delay = latest + policy["min_interval_seconds"] - time.time()
                if active < policy["max_concurrency"] and delay <= 0:
                    cur = db.execute("INSERT INTO budget_attempts(started,owner_pid,role,operation,identity_hash,reserved_output_tokens,active,outcome) VALUES(?,?,?,?,?,?,1,'started')",
                                     (time.time(), os.getpid(), role, operation, identity_hash, output_tokens))
                    db.execute("INSERT INTO budget_attempt_context(attempt_id,run_id,stage,scope_hash,qid,call_id,retry_index) VALUES(?,?,?,?,?,?,?)",
                               (cur.lastrowid, attribution.get('run_id'), attribution.get('stage'), attribution.get('scope_hash'), attribution.get('qid'), call_id, retry_index))
                    return cur.lastrowid
                if delay > 0:
                    wait = min(delay, 0.05)
            if time.monotonic() >= deadline:
                raise BudgetError("concurrency_or_rate_wait_timeout")
            time.sleep(wait)

    def finish(self, attempt_id, outcome, http_status=None, usage=None, *, elapsed_seconds=None, http_elapsed_seconds=None):
        if outcome not in _OUTCOMES or (http_status is not None and not _int(http_status, 100, 599)):
            raise BudgetError("invalid_budget_result")
        if not validate_elapsed(elapsed_seconds) or not validate_elapsed(http_elapsed_seconds):
            raise BudgetError('invalid_budget_result')
        if elapsed_seconds is not None and http_elapsed_seconds is not None and http_elapsed_seconds > elapsed_seconds:
            raise BudgetError('invalid_budget_result')
        clean_usage = safe_usage({"usage": usage or {}})
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check(db)
            row = db.execute("SELECT active FROM budget_attempts WHERE id=?", (attempt_id,)).fetchone()
            if not row or not row[0]:
                raise BudgetError("invalid_budget_attempt")
            db.execute("UPDATE budget_attempts SET active=0,outcome=?,http_status=?,usage_json=? WHERE id=?", (outcome, http_status, _canonical(clean_usage), attempt_id))
            db.execute("UPDATE budget_attempt_context SET finished=?,elapsed_seconds=?,http_elapsed_seconds=? WHERE attempt_id=?",
                       (time.time(), elapsed_seconds, http_elapsed_seconds, attempt_id))

    def count(self):
        return self.summary()["requests_used"]

    def summary(self, run_id=None, stage=None):
        filters = validate_context({k: v for k, v in {'run_id': run_id, 'stage': stage}.items() if v is not None})
        with self._db() as db:
            self._check(db)
            db.row_factory = sqlite3.Row
            all_rows = [dict(r) for r in db.execute("SELECT a.*,c.attempt_id AS context_attempt_id,c.run_id,c.stage,c.scope_hash,c.qid,c.call_id,c.retry_index,c.finished,c.elapsed_seconds,c.http_elapsed_seconds FROM budget_attempts a LEFT JOIN budget_attempt_context c ON c.attempt_id=a.id ORDER BY a.id")]
        rows = [r for r in all_rows if all(r[key] == value for key, value in filters.items())]
        result = aggregate_attempt_rows(rows)
        remaining = max(0, self._policy['max_requests'] - len(all_rows))
        result.update({'policy_fingerprint': self.fingerprint, 'attribution_filter': filters,
                       'requests_remaining': remaining, 'global_requests_remaining': remaining,
                       'global_requests_used': len(all_rows),
                       'global_unattributed_requests': sum(r.get('run_id') is None for r in all_rows)})
        for key in ('role', 'stage'):
            groups = {}
            for row in rows:
                groups.setdefault(row.get(key) or 'unattributed', []).append(row)
            result['by_' + key] = {name: aggregate_attempt_rows(group) for name, group in groups.items()}
        return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _has_secret(value, secret):
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        return any(_has_secret(key, secret) or _has_secret(item, secret) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_secret(item, secret) for item in value)
    return False


def _diagnostic(status, data):
    error = data.get("error", {}) if isinstance(data, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str):
        if code in {"insufficient_quota", "quota_exceeded", "quota_exhausted", "billing_hard_limit_reached"}:
            return "quota_exhausted"
        if code in {"model_not_found", "ModelNotFound", "InvalidModel.NotFound"}:
            return "model_not_found"
        if code in {"thinking_not_supported", "unsupported_thinking", "enable_thinking_not_supported"}:
            return "thinking_unsupported"
    return {401: "authentication_failed", 403: "access_forbidden", 404: "endpoint_not_found", 429: "rate_limited"}.get(status, "http_error")


def _parse_json(raw):
    def nonfinite(_):
        raise ValueError()
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError()
        return result
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError()
            result[key] = value
        return result
    return json.loads(raw, parse_constant=nonfinite, parse_float=finite_float, object_pairs_hook=unique_object)


def _tool_calls(calls):
    if not isinstance(calls, list) or not calls:
        raise APIError("invalid_schema")
    seen = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise APIError("invalid_schema")
        ident, function = call.get("id"), call.get("function")
        if not isinstance(ident, str) or not ident or ident in seen or not isinstance(function, dict):
            raise APIError("invalid_schema")
        seen.add(ident)
        if not isinstance(function.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", function["name"]):
            raise APIError("invalid_schema")
        if not isinstance(function.get("arguments"), str):
            raise APIError("invalid_schema")
        try:
            arguments = _parse_json(function["arguments"])
        except (ValueError, TypeError, UnicodeError):
            raise APIError("invalid_schema") from None
        if not isinstance(arguments, dict):
            raise APIError("invalid_schema")


def _messages(messages):
    if not isinstance(messages, list) or not messages:
        raise APIError("invalid_request")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise APIError("invalid_request")
        content = message.get("content")
        if content is None:
            if message["role"] != "assistant" or not message.get("tool_calls"):
                raise APIError("invalid_request")
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise APIError("invalid_request")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    continue
                if part.get("type") != "image_url" or not isinstance(part.get("image_url"), dict):
                    raise APIError("invalid_request")
                image = part["image_url"]
                url = image.get("url")
                if not isinstance(url, str) or image.get("detail", "auto") not in {"auto", "low", "high"}:
                    raise APIError("invalid_request")
                if url.startswith("data:"):
                    if not re.match(r"^data:image/(png|jpeg|webp|gif);base64,", url):
                        raise APIError("invalid_request")
                    try:
                        if not base64.b64decode(url.split(",", 1)[1], validate=True):
                            raise ValueError()
                    except (ValueError, TypeError):
                        raise APIError("invalid_request") from None
                elif not valid_base_url(url):
                    raise APIError("invalid_request")
        elif not isinstance(content, str):
            raise APIError("invalid_request")
        if "tool_calls" in message:
            _tool_calls(message["tool_calls"])
        if message["role"] == "tool" and not isinstance(message.get("tool_call_id"), str):
            raise APIError("invalid_request")


class APIClient:
    TIMEOUT_SECONDS = 30

    def __init__(self, config: APIConfig, budget: BudgetLedger, transport=None):
        self._config, self.budget = config, budget
        if not config.ready or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", config.role):
            raise APIError("configuration_incomplete")
        identity = {"schema_version": 1, "role": config.role, "base_url": config.base_url.rstrip("/"),
                    "model": config.model, "dimensions": config.dimensions, "protocol": "openai-compatible"}
        if _has_secret(identity, config.api_key):
            raise APIError("invalid_configuration")
        self._identity = identity
        self.identity_hash = budget.bind_client(identity)
        self.transport = transport or urllib.request.build_opener(_NoRedirect()).open

    @property
    def config(self):
        return self._config

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def _call(self, operation, payload, output_tokens, validate):
        attribution = current_api_context()
        if _has_secret(payload, self.config.api_key) or _has_secret(attribution, self.config.api_key):
            raise APIError("secret_in_request")
        try:
            body = _canonical(payload).encode("utf-8")
        except (ValueError, TypeError, OverflowError):
            raise APIError("invalid_request") from None
        policy = self.budget.policy
        if len(body) > policy["max_request_bytes"]:
            raise APIError("request_too_large")
        path = "/chat/completions" if operation == "chat" else "/embeddings"
        call_id = uuid.uuid4().hex
        for retry in range(policy["max_retries"] + 1):
            attempt_id = self.budget.reserve(self.config.role, operation, self.identity_hash, output_tokens,
                                            context=attribution, call_id=call_id, retry_index=retry)
            attempt_started = time.monotonic()
            http_elapsed = None
            outcome, status, usage, failure, result = "client_error", None, {}, None, None
            retry_delay = min(policy["max_retry_delay_seconds"], policy["retry_base_seconds"] * 2**retry)
            try:
                request = urllib.request.Request(self.config.base_url.rstrip("/") + path, data=body,
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.config.api_key,
                             "User-Agent": "2w-M2A-CPU/1.0"}, method="POST")
                http_started = time.monotonic()
                try:
                    try:
                        response = self.transport(request, timeout=self.TIMEOUT_SECONDS)
                    except urllib.error.HTTPError as exc:
                        response = exc
                    with response:
                        status = int(response.status)
                        if not 100 <= status <= 599:
                            raise APIError("invalid_schema")
                        raw = response.read(policy["max_response_bytes"] + 1)
                        headers = getattr(response, "headers", {})
                        delay_header = headers.get("Retry-After") if headers else None
                        if delay_header and isinstance(delay_header, str) and re.fullmatch(r"\d{1,5}", delay_header):
                            retry_delay = min(policy["max_retry_delay_seconds"], max(retry_delay, int(delay_header)))
                finally:
                    http_elapsed = time.monotonic() - http_started
                if not isinstance(raw, bytes) or len(raw) > policy["max_response_bytes"]:
                    raise APIError("response_too_large")
                if self.config.api_key.encode("utf-8") in raw:
                    raise APIError("secret_in_response")
                try:
                    data = _parse_json(raw)
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    if not 200 <= status < 300:
                        raise APIError(_diagnostic(status, {}), http_status=status) from None
                    raise APIError("invalid_json") from None
                if _has_secret(data, self.config.api_key):
                    raise APIError("secret_in_response")
                usage = safe_usage(data)
                if not 200 <= status < 300:
                    raise APIError(_diagnostic(status, data), http_status=status)
                if not isinstance(data, dict):
                    raise APIError("invalid_schema")
                result = validate(data)
                outcome = "ok"
            except APIError as exc:
                outcome = exc.code if exc.code in _OUTCOMES else "http_error"
                failure = APIError(exc.code, http_status=status)
            except (urllib.error.URLError, TimeoutError, OSError):
                outcome, failure = "transport_error", APIError("transport_error")
            except Exception:
                outcome, failure = "client_error", APIError("client_error")
            finally:
                self.budget.finish(attempt_id, outcome, status if _int(status, 100, 599) else None, usage,
                                   elapsed_seconds=time.monotonic() - attempt_started,
                                   http_elapsed_seconds=http_elapsed)
            if failure is None:
                return result
            retriable = outcome == "http_error" and (status == 429 or (status is not None and 500 <= status <= 599))
            if not retriable or retry == policy["max_retries"] or failure.code in {"quota_exhausted", "model_not_found"}:
                raise failure from None
            time.sleep(retry_delay)
        raise APIError("client_error")  # Unreachable, keeps the failure contract explicit.

    def chat(self, messages, *, tools=None, tool_choice="auto", max_output_tokens=1024, temperature=0, extra_body=None):
        if not _int(max_output_tokens, 1, self.budget.policy["max_output_tokens_per_request"]):
            raise BudgetError("output_token_limit_exceeded")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise APIError("invalid_request")
        _messages(messages)
        payload = {"model": self.config.model, "messages": messages, "max_tokens": max_output_tokens, "temperature": temperature}
        if tools is not None:
            if not isinstance(tools, list) or not tools:
                raise APIError("invalid_request")
            for tool in tools:
                if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict) or not isinstance(tool["function"].get("name"), str):
                    raise APIError("invalid_request")
            payload.update({"tools": tools, "tool_choice": tool_choice})
        if extra_body is not None:
            # Additional provider flags cannot replace model, messages or cost controls.
            allowed = {"enable_thinking", "response_format", "parallel_tool_calls", "seed"}
            if not isinstance(extra_body, dict) or set(extra_body) - allowed:
                raise APIError("invalid_extra_body")
            payload.update(extra_body)

        def validate(data):
            choices, model = data.get("choices"), data.get("model")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise APIError("invalid_schema")
            if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_./:@+-]{1,256}", model):
                raise APIError("invalid_schema")
            message = choices[0].get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise APIError("invalid_schema")
            content, calls = message.get("content"), message.get("tool_calls")
            if content is not None and not isinstance(content, str):
                raise APIError("invalid_schema")
            if calls is not None:
                _tool_calls(calls)
            if content is None and not calls:
                raise APIError("invalid_schema")
            # Preserve all message fields for lossless tool-call conversation replay.
            return {"message": copy.deepcopy(message), "usage": safe_usage(data), "model": model,
                    "finish_reason": choices[0].get("finish_reason")}
        return self._call("chat", payload, max_output_tokens, validate)

    def embed(self, texts):
        if (not isinstance(texts, list) or not texts or any(not isinstance(text, str) or not text for text in texts)
                or not _int(self.config.dimensions, 1, 65536)):
            raise APIError("invalid_embedding_request")
        payload = {"model": self.config.model, "input": texts, "dimensions": self.config.dimensions, "encoding_format": "float"}

        def validate(data):
            rows = data.get("data")
            if not isinstance(rows, list) or len(rows) != len(texts):
                raise APIError("invalid_schema")
            vectors = [None] * len(texts)
            for row in rows:
                if not isinstance(row, dict) or not _int(row.get("index"), 0, len(texts) - 1):
                    raise APIError("invalid_schema")
                index, vector = row["index"], row.get("embedding")
                if (vectors[index] is not None or not isinstance(vector, list) or len(vector) != self.config.dimensions
                        or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector)):
                    raise APIError("invalid_schema")
                vectors[index] = vector
            return vectors
        return self._call("embed", payload, 0, validate)
