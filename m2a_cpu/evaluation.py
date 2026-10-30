"""Recoverable evaluation with gold-isolated prediction and complete denominators.

No model clients are constructed here. Callbacks own API access and its budget.
Storage, validation, aggregation and bootstrap use only the standard library.
Mem-Gallery Porter F1 additionally requires NLTK; there is no approximate fallback.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import string
import time
import threading
from typing import Any, Callable, Iterable, Mapping
from .clients import APIError, BudgetError


def _global_api_failure(exc):
    return isinstance(exc, BudgetError) or (isinstance(exc, APIError) and
        (exc.http_status in (401, 403) or exc.code in {'authentication_failed', 'access_forbidden', 'quota_exhausted', 'configuration_incomplete', 'endpoint_not_found', 'model_not_found'}))


class EvaluationError(ValueError):
    pass


class RunBusy(EvaluationError):
    pass


class MetricUnavailable(EvaluationError):
    pass


PUBLIC_QUERY_KEYS = {"qid", "scope_id", "query", "options", "question_date"}
NORMALIZER_VERSION = "memgallery-universal-2026-10-29-porter"
METRIC_SOURCE = "https://github.com/YuanchenBei/Mem-Gallery/blob/main/benchmark/memengine/evaluate/evaluation.py"
MEMEYE_EVALUATOR_COMMIT = "0358e70d714980980dfd8c87903384db474f0b16"
MEMEYE_EVALUATOR_SHA256 = "28f1c2ce1f1d7c7d0907577dd4a81080ab01e1ecf56395af23af6aaf20abf439"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_public_config(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if re.search(r"(^|_)(api_key|access_token|secret|password|authorization)($|_)", str(key).lower()):
                raise EvaluationError("Run config must not contain credentials")
            _check_public_config(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_public_config(child)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise EvaluationError("JSONL row must be an object at line " + str(number))
            records.append(value)
    return records


def _unique_rows(rows: Iterable[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        qid = row.get("qid")
        if not isinstance(qid, str) or not qid:
            raise EvaluationError(label + " requires a nonempty string qid")
        if qid in result:
            raise EvaluationError("Duplicate qid in " + label)
        result[qid] = copy.deepcopy(row)
    if not result:
        raise EvaluationError(label + " cannot be empty")
    return result


def validate_queries(rows: Iterable[dict]) -> list[dict]:
    result = list(_unique_rows(rows, "queries").values())
    for row in result:
        if set(row) - PUBLIC_QUERY_KEYS:
            raise EvaluationError("Query contains a non-whitelisted field")
        if not isinstance(row.get("scope_id"), str) or not row["scope_id"]:
            raise EvaluationError("Query requires scope_id")
        if not isinstance(row.get("query"), (str, list)):
            raise EvaluationError("Query must contain text or ordered content parts")
        if isinstance(row["query"], list):
            for part in row["query"]:
                if not isinstance(part, dict):
                    raise EvaluationError("Query content parts must be objects")
                kind = part.get("type")
                if kind == "text":
                    valid = set(part) == {"type", "text"} and isinstance(part["text"], str)
                elif kind == "image_ref":
                    valid = set(part) == {"type", "asset_id"} and isinstance(part["asset_id"], str) and bool(part["asset_id"])
                elif kind == "image_url":
                    image = part.get("image_url")
                    valid = (set(part) == {"type", "image_url"} and isinstance(image, dict)
                             and set(image) <= {"url", "detail"} and isinstance(image.get("url"), str)
                             and image["url"].startswith("data:image/")
                             and image.get("detail", "auto") in {"auto", "low", "high"})
                else:
                    valid = False
                if not valid:
                    raise EvaluationError("Query content contains unsupported or non-whitelisted fields")
        if "options" in row and (not isinstance(row["options"], list) or any(not isinstance(option, str) for option in row["options"])):
            raise EvaluationError("Query options must be a list of strings")
        canonical(row)
    return result


def bundle_prediction_inputs(bundle: Path) -> tuple[list[dict], str]:
    """Read no gold file, including when the gold file is unreadable or absent."""
    bundle = Path(bundle)
    hashes = {}
    for name in ("history.jsonl", "queries.jsonl", "assets.jsonl", "manifest.json"):
        path = bundle / name
        if not path.is_file():
            raise EvaluationError("Missing prediction input: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    queries = validate_queries(read_jsonl(bundle / "queries.jsonl"))
    manifest = json.loads((bundle / "manifest.json").read_text())
    expected = manifest.get("expected_qids")
    if expected is not None:
        if not isinstance(expected, list) or len(expected) != len(set(expected)) or set(expected) != {q["qid"] for q in queries}:
            raise EvaluationError("Bundle manifest and query qid set differ")
    for name in ("history.jsonl", "queries.jsonl", "assets.jsonl"):
        declared = manifest.get("files", {}).get(name)
        if isinstance(declared, dict):
            declared = declared.get("sha256")
        if declared is not None and declared != hashes[name]:
            raise EvaluationError("Bundle file differs from declared content hash: " + name)
    return queries, fingerprint(hashes)


def bundle_scoring_inputs(bundle: Path) -> list[dict]:
    """Scoring-only gold loader: bind the answers to the prepared manifest hash."""
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    declared = manifest.get('files', {}).get('gold.jsonl')
    if isinstance(declared, dict):
        declared = declared.get('sha256')
    if not isinstance(declared, str) or not re.fullmatch(r'[0-9a-f]{64}', declared):
        raise EvaluationError('Scoring requires a manifest-locked gold file')
    raw = (bundle / 'gold.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != declared:
        raise EvaluationError('Gold differs from the prepared manifest')
    # Parse the same checked bytes, avoiding a second read of a changing file.
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    gold = _unique_rows(rows, 'gold')
    if set(gold) != set(manifest.get('expected_qids', [])):
        raise EvaluationError('Gold and bundle manifest qid sets differ')
    return list(gold.values())


def _error_name(exc: Exception) -> str:
    # Provider messages can echo credentials; never persist str(exc) or a traceback.
    value = type(exc).__name__
    return value if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value) else "CallbackError"


def _safe_usage(value: Any) -> dict:
    if not isinstance(value, Mapping):
        return {}
    names = {"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"}
    return {k: v for k, v in value.items() if k in names and isinstance(v, int) and not isinstance(v, bool) and v >= 0}


def _atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    temporary.replace(path)


class EvaluationRun:
    """One immutable prediction configuration and expected query set per directory.

    A run-wide flock prevents simultaneous executors from repeating paid requests.
    A crashed executor releases the OS lock; its 'running' rows are retried only
    after a new executor actually acquires the lock. Predictions never load gold.
    """

    def __init__(self, run_dir: Path, config: Mapping[str, Any], expected_queries: Iterable[dict],
                 source_fingerprint: str):
        self.root = Path(run_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "run.sqlite"
        if not config or not isinstance(config, Mapping) or not source_fingerprint:
            raise EvaluationError("Explicit nonempty run config and source fingerprint required")
        _check_public_config(config)
        self.config = json.loads(canonical(config))
        self.config_hash = fingerprint(self.config)
        self.queries = validate_queries(expected_queries)
        self.by_qid = {q["qid"]: q for q in self.queries}
        self.source_fingerprint = str(source_fingerprint)
        signature = {"config_hash": self.config_hash, "source_fingerprint": self.source_fingerprint,
                     "queries_hash": fingerprint(self.queries), "schema_version": 1}
        with self._connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS predictions (qid TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, input_hash TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, prediction TEXT, raw_prediction TEXT, prediction_hash TEXT, evidence_json TEXT, usage_json TEXT, error_type TEXT, elapsed_seconds REAL, updated_at TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS scores (qid TEXT NOT NULL, scorer_hash TEXT NOT NULL, prediction_hash TEXT NOT NULL, gold_hash TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, score REAL, error_type TEXT, updated_at TEXT, PRIMARY KEY(qid,scorer_hash))")
            db.execute("CREATE TABLE IF NOT EXISTS score_details (qid TEXT NOT NULL, scorer_hash TEXT NOT NULL, details_json TEXT NOT NULL, PRIMARY KEY(qid,scorer_hash))")
            db.execute("CREATE TABLE IF NOT EXISTS answer_details (qid TEXT PRIMARY KEY, details_json TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, time TEXT NOT NULL, stage TEXT NOT NULL, qid TEXT NOT NULL, status TEXT NOT NULL, error_type TEXT)")
            previous = db.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
            if previous and json.loads(previous[0]) != signature:
                raise EvaluationError("Run configuration or input changed; use a new run directory")
            if previous is None:
                db.execute("INSERT INTO meta VALUES('signature',?)", (canonical(signature),))
                db.execute("INSERT INTO meta VALUES('config',?)", (canonical(self.config),))
                for ordinal, query in enumerate(self.queries):
                    db.execute("INSERT INTO predictions(qid,ordinal,input_hash,status) VALUES(?,?,?,'pending')",
                               (query["qid"], ordinal, fingerprint(query)))
            present = {row[0] for row in db.execute("SELECT qid FROM predictions")}
            if present != set(self.by_qid):
                raise EvaluationError("Stored qid set differs from the expected manifest")
        self.db_path.chmod(0o600)
        (self.root / "declared_config.json").write_text(canonical(self.config)+"\n", encoding="utf-8")

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def _exclusive(self):
        with (self.root / "executor.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunBusy("An executor already owns this run") from None
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _event(self, db, stage, qid, status, error=None):
        db.execute("INSERT INTO events(time,stage,qid,status,error_type) VALUES(?,?,?,?,?)",
                   (utcnow(), stage, qid, status, error))

    def run_predictions(self, retriever: Callable[[dict], Any], answerer: Callable[..., Any],
                        *, workers: int = 1, max_questions: int | None = None,
                        retry_failed: bool = True, query_materializer: Callable[[dict], dict] | None = None,
                        request_context: Callable[[dict], Any] | None = None,
                        qids: list[str] | tuple[str, ...] | None = None) -> dict:
        if not 1 <= workers <= 64 or (max_questions is not None and max_questions < 0):
            raise EvaluationError("Invalid worker or question limit")
        # Select execution only; the registered queries and scoring denominator
        # stay complete. Validate before any state transition or callback.
        selected = None
        if qids is not None:
            if (not isinstance(qids, (list, tuple)) or not qids or
                    any(not isinstance(qid, str) or qid not in self.by_qid for qid in qids) or
                    len(set(qids)) != len(qids)):
                raise EvaluationError("Question selection must contain unique declared qids")
            selected = tuple(qids)
        with self._exclusive():
            with self._connection() as db:
                # OS ownership, rather than a stale status file, proves no previous executor lives.
                selection_sql = " AND qid IN (%s)" % ",".join("?" for _ in selected) if selected else ""
                db.execute("UPDATE predictions SET status='failed',error_type='InterruptedExecutor' WHERE status='running'" + selection_sql,
                           selected or ())
                statuses = ("pending", "failed") if retry_failed else ("pending",)
                pending_qids = [r[0] for r in db.execute(
                    "SELECT qid FROM predictions WHERE status IN (%s)" % ",".join("?" for _ in statuses) +
                    selection_sql + " ORDER BY ordinal", statuses + (selected or ()))]
            if max_questions is not None:
                pending_qids = pending_qids[:max_questions]
            abort = threading.Event()
            fatal = []

            def execute(qid):
                if abort.is_set():
                    return
                query = copy.deepcopy(self.by_qid[qid])
                with self._connection() as db:
                    db.execute("UPDATE predictions SET status='running',attempts=attempts+1,error_type=NULL,updated_at=? WHERE qid=?", (utcnow(), qid))
                    self._event(db, "prediction", qid, "running")
                began = time.monotonic()
                try:
                    with (request_context(copy.deepcopy(query)) if request_context else nullcontext()):
                        evidence = retriever(copy.deepcopy(query))
                        if isinstance(evidence, dict):
                            evidence = evidence.get("data")
                        if not isinstance(evidence, list):
                            raise EvaluationError("Retriever must return evidence list or data object")
                        evidence_json = canonical(evidence)
                        # The data layer resolves images/date, then Answer sees only content/options.
                        if query_materializer is not None:
                            prepared = query_materializer(copy.deepcopy(query))
                            if not isinstance(prepared, dict) or set(prepared) != {"query", "options"}:
                                raise EvaluationError("Materializer must return only query and options")
                        else:
                            parts = copy.deepcopy(query["query"])
                            if query.get("question_date"):
                                if isinstance(parts, str):
                                    parts = [{"type": "text", "text": parts}]
                                parts.insert(0, {"type": "text", "text": "Question date: " + str(query["question_date"])})
                            prepared = {"query": parts, "options": copy.deepcopy(query.get("options", []))}
                        result = answerer(prepared["query"], prepared["options"], copy.deepcopy(evidence))
                        details = {}
                        if isinstance(result, str):
                            prediction, raw, usage = result, result, {}
                        elif isinstance(result, Mapping):
                            prediction, raw = result.get("prediction"), result.get("raw_prediction", result.get("prediction"))
                            usage = _safe_usage(result.get("usage"))
                            details = result.get("packing", {})
                            _check_public_config(details)
                        else:
                            raise EvaluationError("Answerer returned an unsupported response")
                        if not isinstance(prediction, str) or not prediction.strip() or not isinstance(raw, str):
                            raise EvaluationError("Answerer returned an empty or invalid prediction")
                        phash = fingerprint({"prediction": prediction, "raw_prediction": raw})
                        with self._connection() as db:
                            db.execute("UPDATE predictions SET status='succeeded',prediction=?,raw_prediction=?,prediction_hash=?,evidence_json=?,usage_json=?,error_type=NULL,elapsed_seconds=?,updated_at=? WHERE qid=?",
                                       (prediction, raw, phash, evidence_json, canonical(usage), time.monotonic()-began, utcnow(), qid))
                            self._event(db, "prediction", qid, "succeeded")
                            db.execute("INSERT OR REPLACE INTO answer_details VALUES(?,?)", (qid, canonical(details)))
                except Exception as exc:
                    with self._connection() as db:
                        db.execute("UPDATE predictions SET status='failed',error_type=?,elapsed_seconds=?,updated_at=? WHERE qid=?", (_error_name(exc), time.monotonic()-began, utcnow(), qid))
                        self._event(db, "prediction", qid, "failed", _error_name(exc))
                    if _global_api_failure(exc):
                        fatal.append(exc)
                        abort.set()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(execute, pending_qids))
            self.export_predictions()
            if fatal:
                raise fatal[0]
        return self.summary()

    def prediction_rows(self) -> list[dict]:
        with self._connection() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM predictions ORDER BY ordinal")]
            details = {r[0]: json.loads(r[1]) for r in db.execute("SELECT qid,details_json FROM answer_details")}
        for row in rows:
            row["config_hash"] = self.config_hash
            row["evidence"] = json.loads(row.pop("evidence_json") or "[]")
            row["usage"] = json.loads(row.pop("usage_json") or "{}")
            row["packing"] = details.get(row["qid"], {})
        return rows

    def export_predictions(self) -> Path:
        path = self.root / "predictions.jsonl"
        _atomic_jsonl(path, self.prediction_rows())
        return path

    def score(self, gold_rows: Iterable[dict], scorer: Callable[[dict, dict, str], float],
              scorer_config: Mapping[str, Any], *, retry_failed: bool = True) -> dict:
        gold = _unique_rows(gold_rows, "gold")
        if set(gold) != set(self.by_qid):
            raise EvaluationError("Gold qid set must equal the complete expected query set")
        if not scorer_config.get("name") or not scorer_config.get("version"):
            raise EvaluationError("Scorer config requires name and version")
        scorer_identity = {"config": dict(scorer_config), "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        shash = fingerprint(scorer_identity)
        with self._exclusive():
            with self._connection() as db:
                db.execute("INSERT OR IGNORE INTO meta VALUES(?,?)", ("scorer:"+shash, canonical(scorer_identity)))
                db.execute("UPDATE scores SET status='failed',error_type='InterruptedExecutor' WHERE scorer_hash=? AND status='running'", (shash,))
            for row in self.prediction_rows():
                if row["status"] != "succeeded":
                    continue
                qid, phash, ghash = row["qid"], row["prediction_hash"], fingerprint(gold[row["qid"]])
                with self._connection() as db:
                    old = db.execute("SELECT * FROM scores WHERE qid=? AND scorer_hash=?", (qid, shash)).fetchone()
                    if old and old["prediction_hash"] == phash and old["gold_hash"] == ghash:
                        if old["status"] == "succeeded" or (old["status"] == "failed" and not retry_failed):
                            continue
                    attempts = (old["attempts"] if old else 0) + 1
                    db.execute("INSERT OR REPLACE INTO scores VALUES(?,?,?,?,?,?,?,?,?)", (qid, shash, phash, ghash, "running", attempts, None, None, utcnow()))
                    self._event(db, "score", qid, "running")
                try:
                    value = scorer(copy.deepcopy(self.by_qid[qid]), copy.deepcopy(gold[qid]), row["prediction"])
                    details = {}
                    if isinstance(value, Mapping):
                        details = value.get("details", {})
                        _check_public_config(details)
                        value = value.get("score")
                    if isinstance(value, bool):
                        value = float(value)
                    if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise EvaluationError("Scorer must return a finite score in [0,1]")
                    with self._connection() as db:
                        db.execute("UPDATE scores SET status='succeeded',score=?,error_type=NULL,updated_at=? WHERE qid=? AND scorer_hash=?", (float(value), utcnow(), qid, shash))
                        self._event(db, "score", qid, "succeeded")
                        db.execute("INSERT OR REPLACE INTO score_details VALUES(?,?,?)", (qid, shash, canonical(details)))
                except Exception as exc:
                    with self._connection() as db:
                        db.execute("UPDATE scores SET status='failed',error_type=?,updated_at=? WHERE qid=? AND scorer_hash=?", (_error_name(exc), utcnow(), qid, shash))
                        self._event(db, "score", qid, "failed", _error_name(exc))
                    if _global_api_failure(exc):
                        _atomic_jsonl(self.root / ("scores-"+shash+".jsonl"), self.score_rows(shash))
                        raise
            _atomic_jsonl(self.root / ("scores-"+shash+".jsonl"), self.score_rows(shash))
        return self.summary(shash)

    def score_rows(self, scorer_hash: str) -> list[dict]:
        with self._connection() as db:
            present = {row[0] for row in db.execute("SELECT qid FROM scores WHERE scorer_hash=?", (scorer_hash,))}
            if present - set(self.by_qid):
                raise EvaluationError("Unexpected qid in stored scores")
            rows = [dict(r) for r in db.execute("SELECT s.* FROM scores s JOIN predictions p USING(qid) WHERE scorer_hash=? ORDER BY p.ordinal", (scorer_hash,))]
            details = {r[0]: json.loads(r[1]) for r in db.execute("SELECT qid,details_json FROM score_details WHERE scorer_hash=?", (scorer_hash,))}
            for row in rows:
                row["details"] = details.get(row["qid"], {})
            return rows

    def summary(self, scorer_hash: str | None = None) -> dict:
        rows = self.prediction_rows()
        expected = len(self.queries)
        counts = Counter(r["status"] for r in rows)
        result = {"expected": expected, "prediction_counts": dict(counts), "prediction_completion": counts["succeeded"]/expected,
                  "prediction_complete": counts["succeeded"] == expected, "config_hash": self.config_hash,
                  "dataset_score": None, "score_protocol": "not_scored"}
        if scorer_hash is not None:
            scores = self.score_rows(scorer_hash)
            succeeded = [r for r in scores if r["status"] == "succeeded"]
            complete = len(succeeded) == expected and result["prediction_complete"]
            result.update({"scorer_hash": scorer_hash, "score_counts": dict(Counter(r["status"] for r in scores)),
                           "scored": len(succeeded), "score_complete": complete,
                           "dataset_score": sum(r["score"] for r in succeeded)/expected if complete else None,
                           "failure_as_zero_lower_bound": sum(r["score"] for r in succeeded)/expected,
                           "score_protocol": "complete" if complete else "incomplete_no_official_aggregate"})
        return result


def memgallery_normalize(text: str) -> str:
    """Official universal normalization formula, reviewed 2026-10-29.

    Python re implements the same fixed-width lookarounds used by upstream regex.
    Unlike the older normalizer, decimals and image-ID underscores are protected.
    """
    value = str(text).lower()
    value = re.sub(r"(?<=\d)\.(?=\d)", "DOTPLACEHOLDER", value).replace("_", "UNDERSCOREPLACEHOLDER")
    value = re.sub(r"\b(a|an|the|and)\b", " ", value)
    value = value.translate(str.maketrans({p: " " for p in string.punctuation}))
    value = value.replace("DOTPLACEHOLDER", ".").replace("UNDERSCOREPLACEHOLDER", "_")
    return " ".join(value.split())


def memgallery_f1(prediction: str, answer: str) -> float:
    try:
        from nltk.stem import PorterStemmer
    except ImportError:
        raise MetricUnavailable("Mem-Gallery F1 requires NLTK PorterStemmer; no fallback is permitted") from None
    stem = PorterStemmer().stem
    predicted = [stem(t) for t in memgallery_normalize(prediction).split()]
    gold = [stem(t) for t in memgallery_normalize(answer).split()]
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    # Upstream assigns zero when both normalized strings are empty.
    return 2.0 * overlap / (len(predicted)+len(gold)) if overlap else 0.0


def extract_mcq_choice(prediction: str, valid_keys: Iterable[str] = ("A", "B", "C", "D")) -> str:
    """Match the pinned MemEye extract_choice on its four-option MCQ path.

    Retain upstream's case-sensitivity quirk: text is uppercased, but its third
    answer/choice pattern is lowercase with no IGNORECASE. The final fallback
    may therefore select an earlier standalone A in descriptive prose. This is
    benchmark compatibility, not a claim that the heuristic is unambiguous.
    """
    keys = set(valid_keys)
    if not keys or any(not re.fullmatch(r"[A-Z]", key) for key in keys):
        raise EvaluationError("MCQ keys must be uppercase single letters")
    text = prediction.strip().upper()
    if text in keys:
        return text
    choices = "|".join(sorted(keys))
    for pattern in (rf"^({choices})[.\):\s,]", rf"(?:answer|choice)\s*(?:is|:)\s*({choices})\b", rf"\b({choices})\b"):
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return "INVALID"


def memeye_mcq_score(query: dict, gold: dict, prediction: str) -> float:
    answer = gold.get("answer")
    if not isinstance(answer, str) or not re.fullmatch(r"[A-D]", answer):
        raise EvaluationError("MCQ gold answer must be the key for this exact rotation")
    return float(extract_mcq_choice(prediction) == answer)


def aggregate_mcq_rotations(score_rows: Iterable[dict], gold_rows: Iterable[dict]) -> dict:
    gold = _unique_rows(gold_rows, "MCQ gold")
    scores = _unique_rows(score_rows, "MCQ scores")
    if set(scores)-set(gold):
        raise EvaluationError("Unexpected MCQ score qid")
    groups = defaultdict(list)
    for qid, row in gold.items():
        if row.get("variant") != "mcq":
            raise EvaluationError("MCQ aggregation cannot include Open questions")
        if not row.get("group_id") or not row.get("base_qid"):
            raise EvaluationError("MCQ aggregation requires scenario group and base question identities")
        groups[(row.get("group_id"), row.get("base_qid"))].append((qid, row))
    values = []
    incomplete = []
    for group, items in groups.items():
        expected = {r.get("expected_rotations") for _, r in items}
        indices = [r.get("rotation_index") for _, r in items]
        if expected != {4} or sorted(indices) != [0,1,2,3]:
            raise EvaluationError("Each MCQ base question requires four unique rotations")
        successful = [scores[qid]["score"] for qid, _ in items if qid in scores and scores[qid].get("status") == "succeeded"]
        if any(not isinstance(v, (float,int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in successful):
            raise EvaluationError("MCQ scores must be finite numbers in [0,1]")
        if len(successful) != 4:
            incomplete.append(group)
        else:
            values.append(sum(successful)/4)
    complete = not incomplete
    return {"base_questions": len(groups), "completed_base_questions": len(values), "complete": complete,
            "debiased_em": sum(values)/len(groups) if complete else None,
            "failure_as_zero_lower_bound": sum(values)/len(groups), "incomplete_groups": incomplete}


def paired_bootstrap(baseline: Iterable[dict], candidate: Iterable[dict], groups: Mapping[str, str],
                     *, repeats: int = 2000, seed: int = 42) -> dict:
    """Cluster bootstrap of paired per-condition scores; grouping is mandatory.

    All qids must match and be scored. For lengths/rotations use their underlying
    qid as a minimum grouping unit; shared histories require larger groups.
    """
    left, right = _unique_rows(baseline, "baseline scores"), _unique_rows(candidate, "candidate scores")
    if set(left) != set(right) or set(left) != set(groups):
        raise EvaluationError("Paired comparison requires identical complete qid and group sets")
    scorer_ids = {row.get("scorer_hash") for row in list(left.values())+list(right.values())}
    if len(scorer_ids) != 1 or None in scorer_ids:
        raise EvaluationError("Paired scores must use the same explicit scorer identity")
    if repeats < 100:
        raise EvaluationError("At least 100 bootstrap replicates required")
    clusters = defaultdict(list)
    for qid in sorted(left):
        if not left[qid].get("gold_hash") or left[qid]["gold_hash"] != right[qid].get("gold_hash"):
            raise EvaluationError("Paired scores must use the same gold reference for each qid")
        for row in (left[qid], right[qid]):
            if row.get("status") != "succeeded" or not isinstance(row.get("score"), (int,float)) or not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
                raise EvaluationError("Paired comparison requires finite successful scores")
        if not groups[qid]:
            raise EvaluationError("Missing independence group")
        clusters[groups[qid]].append(right[qid]["score"]-left[qid]["score"])
    values = list(clusters.values())
    delta = sum(sum(v) for v in values)/sum(map(len, values))
    if len(values) < 2:
        return {"pairs": len(left), "groups": len(values), "mean_delta": delta, "ci95": None,
                "ci_excludes_zero_positive": False, "reason": "fewer_than_two_independence_groups"}
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        sampled = [rng.choice(values) for _ in values]
        draws.append(sum(sum(v) for v in sampled)/sum(map(len, sampled)))
    draws.sort()
    interval = [draws[int((repeats-1)*p)] for p in (.025,.975)]
    return {"pairs": len(left), "groups": len(values), "mean_delta": delta, "ci95": interval,
            "bootstrap_repeats": repeats, "seed": seed, "ci_excludes_zero_positive": interval[0] > 0,
            "grouping_supplied_by_caller": True}


def compare_runs(baseline: EvaluationRun, candidate: EvaluationRun, scorer_hash: str,
                 groups: Mapping[str, str], **kwargs) -> dict:
    """Require declared fixed controls before comparing two method runs.

    evaluation_controls must include all held-fixed answer model/prompt, dataset,
    judge and budget settings. Method-specific retrieval knobs stay outside it.
    """
    left = baseline.config.get("evaluation_controls")
    right = candidate.config.get("evaluation_controls")
    required = {"answer_model", "answer_prompt_hash", "dataset_revision", "judge_protocol", "input_budget"}
    if not isinstance(left, dict) or not required <= set(left) or left != right:
        raise EvaluationError("Runs need identical declared answer/data/judge/budget controls")
    if baseline.source_fingerprint != candidate.source_fingerprint:
        raise EvaluationError("Runs use different source inputs")
    result = paired_bootstrap(baseline.score_rows(scorer_hash), candidate.score_rows(scorer_hash), groups, **kwargs)
    result["evaluation_controls_hash"] = fingerprint(left)
    return result
