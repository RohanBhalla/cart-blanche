"""Turn a shopper's past purchases into text for the agent."""

from __future__ import annotations

from cartblanche.agent.retrieval import rank_by_similarity
from cartblanche.schemas import HistoryItem


def describe(item: HistoryItem) -> str:
    """One line for one past purchase: when, what, which brand, how it was rated, what they said."""
    brand = item.brand or "unknown brand"
    review = " ".join(part for part in (item.review_title, item.review_text) if part)
    return f"{item.timestamp[:10]} | {item.title} | {brand} | rated {item.rating:g} of 5 | review: {review[:300]}"


def _match_text(item: HistoryItem) -> str:
    return " ".join([item.title, *item.categories, item.brand or "", item.review_title, item.review_text])


def relevant_history(request: str, items: list[HistoryItem], k: int = 8) -> list[HistoryItem]:
    """The k past purchases closest in meaning to the request, best first."""
    order = rank_by_similarity(request, [_match_text(i) for i in items], k)
    return [items[i] for i in order]


def history_block(items: list[HistoryItem]) -> str:
    """A numbered list of past purchases, ready to put in the agent's message."""
    if not items:
        return "The shopper has no past purchases on record."
    return "\n".join(f"{n}. {describe(item)}" for n, item in enumerate(items, start=1))