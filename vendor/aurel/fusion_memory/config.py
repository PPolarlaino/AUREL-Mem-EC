"""Environment-only configuration; never log secrets."""
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class Settings:
    db_path: str = "data/memory.sqlite3"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    embedding_key: str = ""
    embedding_base: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024
    embedding_enabled: bool = False
    service_token: str = ""
    llm_concurrency: int = 4
    upstream_timeout: float = 60
    max_response_images: int = 12
    max_response_image_bytes: int = 30 * 1024 * 1024
    max_response_text_chars: int = 100000
    max_body_bytes: int = 48 * 1024 * 1024
    retrieval_limit: int = 32
    neighbor_hops: int = 1
    graph_expansion: bool = True
    diversity: bool = True

    def __post_init__(self):
        if self.embedding_dimensions <= 0:
            raise ValueError("OPENAI_EMBEDDING_DIMENSIONS must be positive")

    @classmethod
    def from_env(cls):
        load_dotenv(os.environ.get("ENV_FILE", ".env"), override=False)
        def flag(name, default):
            return os.environ.get(name, str(default)).lower() in ("true", "1", "yes")
        key = os.environ.get("OPENAI_API_KEY", "")
        base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
        return cls(
            db_path=os.environ.get("MEMORY_DB", "data/memory.sqlite3"),
            api_key=key, api_base=base,
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            embedding_key=os.environ.get("EMBEDDING_API_KEY", "") or key,
            embedding_base=os.environ.get("EMBEDDING_API_BASE", "") or base,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-v4"),
            embedding_dimensions=int(os.environ.get("OPENAI_EMBEDDING_DIMENSIONS", "1024")),
            embedding_enabled=flag("EMBEDDING_ENABLED", False),
            service_token=os.environ.get("MEMORY_API_TOKEN", ""),
            llm_concurrency=max(1, int(os.environ.get("LLM_CONCURRENCY", "4"))),
            upstream_timeout=float(os.environ.get("UPSTREAM_TIMEOUT", "60")),
            max_response_images=max(0, int(os.environ.get("MAX_RESPONSE_IMAGES", "12"))),
            max_response_text_chars=max(1000, int(os.environ.get("MAX_RESPONSE_TEXT_CHARS", "100000"))),
            retrieval_limit=max(1, min(100, int(os.environ.get("RETRIEVAL_LIMIT", "32")))),
            neighbor_hops=max(0, min(2, int(os.environ.get("NEIGHBOR_HOPS", "1")))),
            graph_expansion=flag("GRAPH_EXPANSION", True),
            diversity=flag("DIVERSITY", True),
        )
