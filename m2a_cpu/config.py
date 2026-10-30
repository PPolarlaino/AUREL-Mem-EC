"""Small, secret-safe configuration reader; .env.local is data, never shell code."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Mapping, Optional
from urllib.parse import urlsplit


def read_env_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError("Invalid environment assignment at line " + str(number))
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("Invalid environment name at line " + str(number))
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].strip().startswith("#")):
                raise ValueError("Invalid quoted value at line " + str(number))
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[name] = value  # No interpolation, unescaping, eval, or subprocess.
    return values


def key_state(value: str) -> str:
    if not value.strip():
        return "missing"
    lowered = value.strip().lower()
    if (lowered in {"empty", "none", "null", "changeme", "replace_me", "your_api_key"}
            or re.fullmatch(r"(?:sk-)?(?:x{3,}|\.{3,}|your[_-].*|placeholder.*)", lowered)
            or "<" in value or ">" in value or "xxxxx" in lowered
            or "$" in value or "`" in value):
        return "placeholder"
    if any(ch.isspace() for ch in value):
        return "invalid"
    return "present"


def valid_base_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return bool(parsed.scheme == "https" and parsed.hostname and not parsed.username
                    and not parsed.password and not parsed.query and not parsed.fragment)
    except ValueError:
        return False


@dataclass(frozen=True)
class APIConfig:
    role: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    sources: dict[str, str] = field(default_factory=dict)
    dimensions: Optional[int] = None

    @property
    def ready(self) -> bool:
        return key_state(self.api_key) == "present" and valid_base_url(self.base_url) and bool(self.model)

    def public_status(self) -> dict:
        result = {
            "role": self.role, "ready": self.ready,
            "key_state": key_state(self.api_key), "base_url_valid": valid_base_url(self.base_url),
            "model_configured": bool(self.model), "sources": self.sources,
        }
        if self.dimensions is not None:
            result["dimensions"] = self.dimensions
        if self.role == "embedding":
            result["endpoint_path_needs_probe"] = valid_base_url(self.base_url) and urlsplit(self.base_url).path.rstrip("/") == ""
        return result


@dataclass(frozen=True)
class Config:
    llm: APIConfig
    embedding: APIConfig
    evaluator: APIConfig
    env_file_exists: bool


def load_config(project_root: Path, environ: Optional[Mapping[str, str]] = None) -> Config:
    env = os.environ if environ is None else environ
    env_path = Path(project_root) / ".env.local"
    local = read_env_file(env_path)

    def get(names, default=""):
        # Preferred names win over aliases; process environment wins for that name.
        for name in names:
            if name in env:
                return env[name], "environment:" + name
            if name in local:
                return local[name], ".env.local:" + name
        return default, "default"

    def client(role, fields, defaults, dimensions=None):
        values, sources = {}, {}
        for name, aliases in fields.items():
            values[name], sources[name] = get(aliases, defaults.get(name, ""))
        return APIConfig(role=role, sources=sources, dimensions=dimensions, **values)

    dimension_raw, _ = get(("OPENAI_EMBEDDING_DIMENSIONS", "M2A_TEXT_EMBEDDING_DIMENSIONS"), "1024")
    try:
        dimensions = int(dimension_raw)
        if dimensions <= 0 or dimensions > 65536:
            raise ValueError()
    except ValueError:
        raise ValueError("Embedding dimensions must be a positive integer <=65536") from None
    return Config(
        llm=client("llm", {
            "api_key": ("OPENAI_API_KEY", "M2A_LLM_API_KEY"),
            "base_url": ("OPENAI_API_BASE", "OPENAI_BASE_URL", "M2A_LLM_BASE_URL"),
            "model": ("OPENAI_MODEL", "M2A_LLM_MODEL"),
        }, {"model": "gpt-4o-mini", "base_url": "https://api.gptplus5.com/v1"}),
        embedding=client("embedding", {
            "api_key": ("OPENAI_EMBEDDING_API_KEY", "M2A_TEXT_EMBEDDING_API_KEY", "M2A_EMBEDDING_API_KEY"),
            "base_url": ("OPENAI_EMBEDDING_API_BASE", "M2A_TEXT_EMBEDDING_BASE_URL", "M2A_EMBEDDING_BASE_URL"),
            "model": ("OPENAI_EMBEDDING_MODEL", "M2A_TEXT_EMBEDDING_MODEL", "M2A_EMBEDDING_MODEL"),
        }, {"model": "text-embedding-v4", "base_url": "https://api.gpt.ge/v1"}, dimensions),
        evaluator=client("evaluator", {
            "api_key": ("EVALUATOR_API_KEY", "M2A_EVALUATOR_API_KEY"),
            "base_url": ("EVALUATOR_API_BASE", "M2A_EVALUATOR_BASE_URL"),
            "model": ("EVALUATOR_MODEL", "M2A_EVALUATOR_MODEL"),
        }, {"model": "qwen3-14b", "base_url": "https://ws-mbz5kjxxbzx810g2.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"}),
        env_file_exists=env_path.is_file(),
    )
