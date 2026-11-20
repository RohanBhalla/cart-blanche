"""Turn a shopper's past purchases into text for the agent."""

from __future__ import annotations

from cartblanche.agent.intent import parse_json
from cartblanche.agent.retrieval import rank_by_similarity
from cartblanche.llm import LLM
from cartblanche.schemas import HistoryItem

REASONS = ("reaction", "quality", "scent or shade", "other")

TAG_PROMPT = """You label past purchases of a beauty shopper. For each numbered purchase, pick the main reason behind the rating, from what the review says:
- reaction: only a BAD reaction: irritation, breakout, rash, allergy, stinging, a bad feel on skin, scalp or hair. If the review only praises the product for being gentle or not irritating, use other
- quality: how well it works or how well it is made (effective, weak, broke, leaked, ran out fast)
- scent or shade: the smell, the color or the shade match
- other: price, shipping, packaging, praise for being gentle or not irritating, or no clear reason in the review
Reply with only this JSON object, with one label per purchase, in the same order:
{"reasons": ["reaction" | "quality" | "scent or shade" | "other", ...]}"""

PROFILE_PROMPT = """Write a short profile of this shopper from their past purchases and reviews, in at most 120 words.
Cover only what the ratings and reviews support: what they liked and why, what went wrong (reactions, bad fit, poor quality), brands they trust or avoid, and any skin type, hair type, shade or scent preferences they mention.
Do not guess and do not invent details. Reply with the profile only."""


def describe(item: HistoryItem, reason: str | None = None) -> str:
    """One line for one past purchase: when, what, which brand, how it was rated, why, what they said."""
    brand = item.brand or "unknown brand"
    review = " ".join(part for part in (item.review_title, item.review_text) if part)
    why = f" | reason: {reason}" if reason else ""
    return f"{item.timestamp[:10]} | {item.title} | {brand} | rated {item.rating:g} of 5{why} | review: {review[:300]}"


def _match_text(item: HistoryItem) -> str:
    return " ".join([item.title, *item.categories, item.brand or "", item.review_title, item.review_text])


def relevant_history(request: str, items: list[HistoryItem], k: int = 8) -> list[HistoryItem]:
    """The k past purchases closest in meaning to the request, best first."""
    order = rank_by_similarity(request, [_match_text(i) for i in items], k)
    return [items[i] for i in order]


def history_block(items: list[HistoryItem], reasons: list[str | None] | None = None) -> str:
    """A numbered list of past purchases, ready to put in the agent's message.

    reasons, if given, has one entry per item: a label, or None for no label (see tag_reasons)."""
    if not items:
        return "The shopper has no past purchases on record."
    labels = reasons or [None] * len(items)
    return "\n".join(f"{n}. {describe(item, why)}" for n, (item, why) in enumerate(zip(items, labels), start=1))


def tag_reasons(items: list[HistoryItem], llm: LLM, tag: str = "") -> list[str | None]:
    """One reason per past purchase (reaction, quality, scent or shade), from one model call.

    "other" is too vague to act on, and a label we do not recognise cannot be trusted, so both
    become None: that purchase is shown without a reason. If the reply is unusable (no JSON, or the
    wrong number of labels) every purchase is shown without a reason."""
    if not items:
        return []
    reply = llm.complete(
        [{"role": "system", "content": TAG_PROMPT}, {"role": "user", "content": history_block(items)}],
        tag=f"{tag}/tags",
    )
    try:
        labels = parse_json(reply.content).get("reasons")
    except ValueError:
        return [None] * len(items)
    if not isinstance(labels, list) or len(labels) != len(items):
        return [None] * len(items)
    cleaned = [str(label).strip().lower() for label in labels]
    return [label if label in REASONS and label != "other" else None for label in cleaned]


def summarize_profile(
    items: list[HistoryItem], llm: LLM, tag: str = "", reasons: list[str | None] | None = None
) -> str:
    """A short written profile of the shopper (E3b). One model call, unless there is no history."""
    if not items:
        return history_block(items)
    reply = llm.complete(
        [{"role": "system", "content": PROFILE_PROMPT}, {"role": "user", "content": history_block(items, reasons)}],
        tag=f"{tag}/profile",
    )
    profile = (reply.content or "").strip()
    if not profile:
        raise ValueError("the profile reply was empty")
    return profile