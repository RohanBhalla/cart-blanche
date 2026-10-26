"""Shared LLM wrapper.

Every model call goes through LLM.complete, which
  * caches responses on disk, so reruns of an unchanged config are free,
  * logs tokens, latency and cost for every call (cached calls cost 0).

LiteLLM does the provider routing (e.g. "gemini/gemini-2.5-flash-lite",
"openai/gpt-5.4-nano"). Tests pass a fake completion_fn instead.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CompletionFn = Callable[..., Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, as the model produced it


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool
    message: dict[str, Any]  # assistant message, ready to append to the conversation

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.cached_calls += r.cached
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cost_usd += r.cost_usd
        self.latency_s += r.latency_s


def _litellm_completion(**kwargs: Any) -> Any:
    import litellm  # imported lazily: it is slow to import and tests don't need it
    from dotenv import load_dotenv

    load_dotenv()  # API keys from .env; real environment variables take precedence
    return litellm.completion(**kwargs)


def _litellm_cost(raw: Any) -> float:
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=raw))
    except Exception:  # unknown model pricing, fake responses, etc.
        return 0.0


def _to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return raw.model_dump()


def _parse(data: dict[str, Any]) -> tuple[dict[str, Any], list[ToolCall], int, int]:
    msg = data["choices"][0]["message"]
    calls = [
        ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"] or "{}")
        for tc in (msg.get("tool_calls") or [])
    ]
    message: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
    if calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in calls
        ]
    usage = data.get("usage") or {}
    return message, calls, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


@dataclass
class LLM:
    model: str
    temperature: float = 0.0
    cache_dir: Path | None = Path("cache/llm")
    log_path: Path | None = None
    completion_fn: CompletionFn = _litellm_completion
    cost_fn: Callable[[Any], float] = _litellm_cost
    usage: Usage = field(default_factory=Usage)

    def _key(self, request: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tag: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        """One chat completion. `tag` (e.g. "t042/agent") only labels the log line."""
        request: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature, **kwargs}
        if tools:
            request["tools"] = tools
        key = self._key(request)
        cache_file = self.cache_dir / f"{key}.json" if self.cache_dir else None

        if cache_file and cache_file.exists():
            data = json.loads(cache_file.read_text())
            cached, cost, latency = True, 0.0, 0.0
        else:
            start = time.perf_counter()
            raw = self.completion_fn(**request)
            latency = time.perf_counter() - start
            data = _to_dict(raw)
            cost = self.cost_fn(raw)
            cached = False
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(data, default=str))

        message, calls, p_tok, c_tok = _parse(data)
        resp = LLMResponse(
            content=message["content"],
            tool_calls=calls,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=cost,
            latency_s=latency,
            cached=cached,
            message=message,
        )
        self.usage.add(resp)
        self._log(tag, key, resp)
        return resp

    def _log(self, tag: str, key: str, r: LLMResponse) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "ts": time.time(),
            "tag": tag,
            "model": self.model,
            "key": key,
            "cached": r.cached,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "cost_usd": r.cost_usd,
            "latency_s": round(r.latency_s, 4),
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(line) + "\n")
