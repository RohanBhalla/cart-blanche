"""The simulated shopper who answers the agent's one clarifying question (E8; tested early as E4q).

Environment side, like the store: it reads the task's hidden profile, which agent code never sees.
The answer may only use the profile's facts (what kind of product, who it is for, the budget), never
a brand or product name, so a question can resolve vagueness but cannot leak the gold. A task with
no hidden profile has nothing to add, and answering it costs no model call.
"""

from __future__ import annotations

from typing import Any

from cartblanche.llm import LLM

NOTHING_TO_ADD = "I don't have anything to add, just what I asked for."

PROMPT = """You are a shopper talking to a shopping assistant. You asked: "{request}"
What you actually have in mind (private; the assistant can't see this): {facts}
Answer the assistant's question in one short sentence, as this shopper would, using only these facts.
Never name a brand or a specific product. If the question asks about something not covered here, say you
have no preference."""


def _facts(profile: dict[str, Any]) -> str:
    """The profile as plain facts, e.g. "looking for: body lotion; for: myself; budget: $20"."""
    parts = []
    for key, value in profile.items():
        if value is None or value == "":
            continue
        shown = f"${value:g}" if key == "budget" and isinstance(value, (int, float)) else value
        parts.append(f"{key.replace('_', ' ')}: {shown}")
    return "; ".join(parts) or "nothing beyond the request"


def answer(request: str, profile: dict[str, Any] | None, question: str, llm: LLM, tag: str = "") -> str:
    """The shopper's reply to one question."""
    if not profile:
        return NOTHING_TO_ADD
    reply = llm.complete(
        [{"role": "system", "content": PROMPT.format(request=request, facts=_facts(profile))},
         {"role": "user", "content": question}],
        tag=f"{tag}/shopper",
    )
    return (reply.content or "").strip() or NOTHING_TO_ADD
