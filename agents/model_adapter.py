"""One model adapter for every free-tier crew model: the OpenAI chat-completions dialect.

Gemini (Google AI Studio), Groq and Ollama all serve it, so the crew talks to each the
same way and only the base URL, key and model name change. Keys come from the
environment or a .env file in the repo root (gitignored); none is ever logged.

Claude is not called through here: it runs through the Claude Code CLI on the
user's own plan (see run_claude_task in evals/runners.py), so there is no paid API spend.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str | None
    model_env: str
    default_model: str
    simple_schemas: bool  # strip JSON Schema features the provider rejects


PROVIDERS = {
    "gemini": Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
                       "GEMINI_MODEL", "gemini-flash-latest", True),
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL",
                     "openai/gpt-oss-120b", False),
    "ollama": Provider("ollama", "http://127.0.0.1:11434/v1", None, "OLLAMA_MODEL", "qwen3:8b", False),
}


def load_dotenv(path: Path = REPO / ".env") -> None:
    """Minimal KEY=VALUE reader; existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class ModelError(RuntimeError):
    pass


def simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make an MCP input schema acceptable to stricter function-calling dialects (Gemini).

    Drops titles, folds `X | None` (anyOf with null) into X, and moves defaults into the
    description so the information survives.
    """
    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        node = dict(node)
        node.pop("title", None)
        if "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            if len(options) == 1:
                merged = {**options[0], **{k: v for k, v in node.items() if k != "anyOf"}}
                node = merged
        if "default" in node:
            d = node.pop("default")
            if d is not None:
                desc = node.get("description", "")
                node["description"] = (desc + " " if desc else "") + f"Default: {json.dumps(d)}."
        return {k: walk(v) for k, v in node.items()}
    return walk(copy.deepcopy(schema))


def mcp_tools_to_openai(tools, simple: bool) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        params = getattr(t, "input_schema", None) or {"type": "object", "properties": {}}  # mcp 2.x: snake_case
        if simple:
            params = simplify_schema(params)
        out.append({"type": "function",
                    "function": {"name": t.name, "description": (t.description or "").strip(), "parameters": params}})
    return out


class ChatModel:
    """Chat completions with tools, retrying rate limits (free tiers hit them often)."""

    def __init__(self, provider: str, model: str | None = None, *, transport: httpx.BaseTransport | None = None,
                 timeout_s: float = 120.0, max_retries: int = 6):
        if provider not in PROVIDERS:
            raise ModelError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
        load_dotenv()
        self.provider = PROVIDERS[provider]
        self.model = model or os.environ.get(self.provider.model_env) or self.provider.default_model
        headers = {"Content-Type": "application/json"}
        if self.provider.key_env:
            key = os.environ.get(self.provider.key_env)
            if not key and transport is None:
                raise ModelError(f"{self.provider.key_env} is not set; put it in {REPO / '.env'}")
            headers["Authorization"] = f"Bearer {key or 'test'}"
        self.client = httpx.Client(base_url=self.provider.base_url, headers=headers, timeout=timeout_s,
                                   transport=transport)
        self.max_retries = max_retries

    @property
    def label(self) -> str:
        return f"{self.provider.name}:{self.model}"

    def list_models(self) -> list[str]:
        r = self.client.get("/models")
        if r.status_code != 200:
            raise ModelError(f"listing models failed: HTTP {r.status_code}: {r.text[:300]}")
        return sorted(m["id"] for m in r.json().get("data", []))

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages, "tools": tools, "temperature": 0}
        delay = 5.0
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client.post("/chat/completions", json=body)
            except httpx.TransportError as exc:  # timeouts, dropped connections
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 90.0)
                    continue
                raise ModelError(f"{self.provider.name} unreachable: {type(exc).__name__}: {exc}") from exc
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError as exc:
                    raise ModelError(f"non-JSON response: {r.text[:300]}") from exc
                if not data.get("choices"):
                    raise ModelError(f"no choices in response: {json.dumps(data)[:300]}")
                return data["choices"][0]["message"]
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                try:
                    wait = float(r.headers.get("retry-after", delay))
                except ValueError:  # HTTP-date form
                    wait = delay
                time.sleep(min(max(wait, 1.0), 90.0))
                delay = min(delay * 2, 90.0)
                continue
            raise ModelError(f"HTTP {r.status_code} from {self.provider.name}: {r.text[:500]}")
        raise ModelError("unreachable")
