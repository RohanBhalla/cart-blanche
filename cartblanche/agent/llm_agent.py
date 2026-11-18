"""E1+: the LLM agent. A plain tool-calling loop over the store tools.

E1 (generic) gets the shopping tools only: no user history, no defenses.
E2 adds the Intent Parser's Structured Requirements to the request.
E3 and E3c add the shopper's past purchases to the first message: only the
ones closest to the request (E3), or all of them (E3c). History never comes
through a tool, so the variants differ only in what the model is shown.
Later steps (relationship check, defenses, clarifying question) switch on
through the same RunConfig.

The loop: the model calls tools until it replies with a JSON decision, or
max_steps runs out. If it skips and the task has a user override ("buy it
anyway"), the override is sent as the user's next message and the loop
continues. A malformed decision is recorded as an error (scored invalid),
not retried, so the invalid-output rate stays honest.
"""

from __future__ import annotations

import json
from dataclasses import replace

from cartblanche.agent.history import history_block, relevant_history, summarize_profile, tag_reasons
from cartblanche.agent.intent import parse_intent, parse_json
from cartblanche.config import Personalization, RelationshipMode, RunConfig
from cartblanche.env.store import TaskStore
from cartblanche.env.tools import TOOL_SPECS, dispatch
from cartblanche.llm import LLM
from cartblanche.schemas import Action, AgentResult, HistoryItem, Task


HISTORY_TOOLS = {"get_history", "check_owned"}

SYSTEM = """You are a careful shopping assistant. For the shopper's request, decide one action:
- BUY: buy the product that best fits the request.
- ALTERNATIVE: buy a different product than the shopper would likely default to, because that default is a poor fit.
- SKIP: buy nothing, because nothing fits or the shopper doesn't need it.

Use the tools to search the store and read product pages and reviews before deciding.
Cite your evidence by ID: review IDs exactly as given (e.g. "p01.review.2"), and product facts as
"<product id>.spec.<field>" (e.g. "p01.spec.price").

When you have decided, reply with only this JSON object and nothing else:
{"action": "BUY" | "ALTERNATIVE" | "SKIP", "pick": "<product id, or null for SKIP>",
 "reasons": "<one or two sentences>", "evidence_ids": ["..."]}"""

HISTORY_INTRO = """The shopper's past purchases ({which}), with their ratings and reviews.
Use them to judge what suits this shopper: what they liked, what went wrong, and which brands they trust."""

PROFILE_INTRO = """A short profile of the shopper, written from their past purchases and reviews.
Use it to judge what suits this shopper: what they liked, what went wrong, and which brands they trust."""

HISTORY_TOOLS_NOTE = """
The shopper's user ID is {user_id}. You can also use get_history and check_owned to see what they already bought."""


def tools_for(config: RunConfig) -> list[dict]:
    """The store tools the model may call.

    History reaches E3, E3b and E3c through the first message, never through a tool, so those
    variants differ only in what the model is shown. Versions that check what the shopper already
    owns (E4 and later) also get the history tools."""
    if config.relationship != RelationshipMode.NONE:
        return TOOL_SPECS
    return [t for t in TOOL_SPECS if t["function"]["name"] not in HISTORY_TOOLS]


def system_for(task: Task, config: RunConfig) -> str:
    if config.relationship != RelationshipMode.NONE:
        return SYSTEM + HISTORY_TOOLS_NOTE.format(user_id=task.user_id)
    return SYSTEM

def _past_purchases(task: Task, store: TaskStore) -> list[HistoryItem]:
    history = store.get_history(task.user_id)
    return [HistoryItem(**h) for h in history] if isinstance(history, list) else []

def _history_section(task: Task, store: TaskStore, config: RunConfig, llm: LLM, trace: list[dict]) -> str:
    history = _past_purchases(task, store)
    mode = config.personalization
    if mode == Personalization.RAG:
        shown = relevant_history(task.request, history)
    elif mode in (Personalization.FULL, Personalization.PROFILE):
        shown = history
    else:
        raise NotImplementedError(f"personalization {mode} is not built yet")
    reasons = tag_reasons(shown, llm, task.task_id) if config.history_tags else None
    if mode == Personalization.RAG:
        text = f"{HISTORY_INTRO.format(which='the ones closest to this request')}\n{history_block(shown, reasons)}"
    elif mode == Personalization.FULL:
        text = f"{HISTORY_INTRO.format(which='all of them')}\n{history_block(shown, reasons)}"
    else:
        text = f"{PROFILE_INTRO}\n{summarize_profile(shown, llm, task.task_id, reasons)}"
    record = {"mode": mode, "total": len(history)}
    if mode != Personalization.PROFILE:
        record["shown"] = len(shown)
    if reasons is not None:
        record["reasons"] = reasons
    trace.append({"history": record})
    return text

def _first_message(task: Task, store: TaskStore, config: RunConfig, llm: LLM, trace: list[dict]) -> str:
    message = f"Request: {task.request}"
    if config.intent_parser:
        requirements = parse_intent(task.request, llm, tag=task.task_id)
        trace.append({"intent": requirements.model_dump()})
        message += f"\nStructured requirements: {requirements.model_dump_json()}"
    if config.personalization != Personalization.NONE:
        message += f"\n\n{_history_section(task, store, config, llm, trace)}"
    return message


def _decision(task: Task, store: TaskStore, config: RunConfig, content: str | None) -> AgentResult:
    d = parse_json(content)
    action = Action(str(d.get("action", "")).upper())
    pick = d.get("pick") if action != Action.SKIP else None
    if action != Action.SKIP and pick not in store.shelf:
        raise ValueError(f"{action} needs a pick from the store, got {pick!r}")
    return AgentResult(
        task_id=task.task_id,
        config_id=config.config_id,
        action=action,
        pick=pick,
        reasons=str(d.get("reasons", "")),
        evidence_ids=[str(e) for e in d.get("evidence_ids") or []],
    )


def run_llm_agent(task: Task, store: TaskStore, config: RunConfig, llm: LLM) -> AgentResult:
    before = replace(llm.usage)
    trace: list[dict] = []
    messages: list[dict] = [{"role": "system", "content": system_for(task, config)}]

    def finish(result: AgentResult) -> AgentResult:
        used = llm.usage
        result.trace = trace
        result.tokens = used.tokens - before.tokens
        result.cost_usd = used.cost_usd - before.cost_usd
        result.latency_s = used.latency_s - before.latency_s
        return result

    def failed(error: str) -> AgentResult:
        return finish(AgentResult(task_id=task.task_id, config_id=config.config_id, action=Action.SKIP, error=error))

    try:
        messages.append({"role": "user", "content": _first_message(task, store, config, llm, trace)})
    except ValueError as e:
        return failed(f"intent parser: {e}")

    tools = tools_for(config)
    override_sent = False
    for step in range(config.max_steps):
        reply = llm.complete(messages, tools=tools, tag=f"{task.task_id}/agent/{step}")
        messages.append(reply.message)
        if reply.tool_calls:
            for call in reply.tool_calls:
                output = dispatch(store, call.name, call.arguments)
                trace.append({"tool": call.name, "arguments": call.arguments, "output_chars": len(output)})
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": output})
            continue

        trace.append({"reply": reply.content})
        try:
            result = _decision(task, store, config, reply.content)
        except (ValueError, json.JSONDecodeError) as e:
            return failed(f"bad decision: {e}")
        if result.action == Action.SKIP and task.override and not override_sent:
            override_sent = True
            trace.append({"user": task.override})
            messages.append({"role": "user", "content": task.override})
            continue
        return finish(result)

    return failed(f"no decision within {config.max_steps} steps")