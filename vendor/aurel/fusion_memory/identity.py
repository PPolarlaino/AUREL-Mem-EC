"""Public, reproducible index identity; no environment or credential loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from . import provider
from .config import Settings
from .retrieval import RETRIEVAL_STRATEGY_VERSION
from .storage import IndexIdentityError


INDEX_IDENTITY_VERSION = "aurel-public-index-v1"
PRIOR_CONTEXT_VERSION = "aurel-prior-context-v1-source-labels-derived-excerpts"


def _hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _public_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError()
        parsed.port  # Validate malformed ports without exposing the supplied URL.
    except (TypeError, ValueError):
        raise IndexIdentityError("public API base must be a valid URL without credentials or a query") from None
    return value.rstrip("/")


def build_index_identity(settings: Settings) -> dict:
    """Return indexing identity from explicit settings and local code, never Keys/DB paths.

    Output budgets for the retrieval response are intentionally not index fields:
    changing delivery limits does not change stored raw/semantic evidence. The
    experiment controller separately pins all evaluation and response controls.
    """
    directory = Path(__file__).resolve().parent
    files = ("provider.py", "schemas.py", "retrieval.py", "storage.py", "app.py", "identity.py")
    return {
        "identity_version": INDEX_IDENTITY_VERSION,
        "generation": {"model": settings.model, "base_url": _public_base_url(settings.api_base),
                       "temperature": 0, "max_output_tokens": provider.MAX_OUTPUT_TOKENS,
                       "response_format": "json_object"},
        "embedding": {"enabled": settings.embedding_enabled, "model": settings.embedding_model,
                      "base_url": _public_base_url(settings.embedding_base or settings.api_base),
                      "dimensions": settings.embedding_dimensions},
        "versions": {"extraction": provider.EXTRACTION_VERSION,
                     "embedding_text": provider.EMBEDDING_TEXT_VERSION,
                     "prior_context": PRIOR_CONTEXT_VERSION,
                     "retrieval": RETRIEVAL_STRATEGY_VERSION},
        "prompt_hashes": {"extract": _hash(provider.EXTRACT_SYSTEM),
                          "format_correction": _hash(provider.EXTRACT_FORMAT_CORRECTION),
                          "query": _hash(provider.QUERY_SYSTEM)},
        "frame_schema_sha256": _hash(provider.Frame.model_json_schema()),
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files},
    }
