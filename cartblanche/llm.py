"""Shared LLM wrapper.

Every model call goes through LLM.complete, which
  * caches responses on disk, so reruns of an unchanged config are free,
  * logs tokens, latency and cost for every call,
  * retries rate limits and timeouts with backoff, and can keep a minimum
    gap between calls (for the Gemini free tier).

A cached response keeps the cost and latency of the original call, so a
rerun scores exactly like the first run; Usage.spent_usd counts only what
was actually paid this time. When a daily quota runs out, QuotaExhausted is
raised so the run stops and can be resumed later.

LiteLLM does the provider routing (e.g. "gemini/gemini-3.5-flash-lite",
"openai/gpt-5.4-nano"). Tests pass a fake completion_fn instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CompletionFn = Callable[..., Any]

# LiteLLM exception names worth retrying (matched by name so this module doesn't import litellm)
RETRYABLE = {"RateLimitError", "Timeout", "APIConnectionError", "ServiceUnavailableError", "InternalServerError"}


# Errors no retry fixes for any task: credits ran out, or the key is missing or rejected
STOP_RUN = {"PaymentRequiredError", "AuthenticationError", "PermissionDeniedError"}


class QuotaExhausted(RuntimeError):
    """A daily quota or the prepaid credits ran out, or the key was rejected: the run should stop."""


def provider_failed(error: Exception) -> bool:
    """A transient provider failure (rate limit, timeout, outage) that outlasted the retries.

    Not the agent's fault, so the harness leaves the task unfinished for the next run instead of scoring it."""
    return type(error).__name__ in RETRYABLE


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
    cost_usd: float = 0.0  # cost of every call, cached or not (what the scorecard reports)
    spent_usd: float = 0.0  # cost of calls actually made this time
    latency_s: float = 0.0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.cached_calls += r.cached
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cost_usd += r.cost_usd
        self.spent_usd += 0.0 if r.cached else r.cost_usd
        self.latency_s += r.latency_s


def _litellm_completion(**kwargs: Any) -> Any:
    import litellm  # imported lazily: it is slow to import and tests don't need it
    from dotenv import load_dotenv

    load_dotenv()  # API keys from .env; real environment variables take precedence
    # drop_params: newer models (e.g. gemini-3.5-flash-lite) reject temperature; the cache keeps reruns stable
    return litellm.completion(timeout=60, drop_params=True, **kwargs)


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


def _read_cache(path: Path | None) -> dict[str, Any] | None:
    """A cached reply, or None. A damaged file (e.g. from a crash mid-write) counts as a miss."""
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    """Write then rename, so runs in parallel never read a half-written reply."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, default=str))
    os.replace(tmp, path)


@dataclass
class LLM:
    model: str
    temperature: float = 0.0
    cache_dir: Path | None = Path("cache/llm")
    log_path: Path | None = None
    completion_fn: CompletionFn = _litellm_completion
    cost_fn: Callable[[Any], float] = _litellm_cost
    min_gap_s: float = 0.0  # minimum seconds between real calls (free-tier rate limits)
    max_retries: int = 5
    retry_wait_s: float = 10.0  # doubles on every retry: 10, 20, 40, 80, 160 s
    sleep: Callable[[float], None] = time.sleep
    usage: Usage = field(default_factory=Usage)
    _last_call: float = field(default=0.0, repr=False)

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

        data = _read_cache(cache_file)
        cached = data is not None
        if data is None:
            raw, latency = self._call(request)
            data = {**_to_dict(raw), "_cost_usd": self.cost_fn(raw), "_latency_s": latency}
            if cache_file:
                _write_cache(cache_file, data)
        cost, latency = float(data.get("_cost_usd", 0.0)), float(data.get("_latency_s", 0.0))

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

    def _call(self, request: dict[str, Any]) -> tuple[Any, float]:
        """One real call, with the minimum gap and retries. Returns (raw response, latency)."""
        for attempt in range(self.max_retries + 1):
            wait = self._last_call + self.min_gap_s - time.monotonic()
            if wait > 0:
                self.sleep(wait)
            self._last_call = time.monotonic()
            start = time.perf_counter()
            try:
                raw = self.completion_fn(**request)
                return raw, time.perf_counter() - start
            except Exception as e:
                if type(e).__name__ in STOP_RUN:
                    raise QuotaExhausted(str(e)) from e
                if type(e).__name__ not in RETRYABLE:
                    raise
                if "perday" in str(e).lower().replace(" ", "").replace("_", ""):
                    raise QuotaExhausted(str(e)) from e
                if attempt == self.max_retries:
                    raise
                self.sleep(self.retry_wait_s * 2**attempt)
        raise AssertionError("unreachable")

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
