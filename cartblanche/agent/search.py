"""Personal product search (E3 and later): rank the shelf for this shopper, not just for the query.

The store's search_products ranks by keyword overlap and returns 10 hits. Each shelf is one narrow
category, so on dev the bought product was in those 10 only 56% of the time. This search keeps the
same filters but scores every product that passes them:

    score = meaning(query, product) + 0.3 * keyword rank + 1.0 * closeness to a product the shopper
            rated 4-5 + 0.2 if the brand is one they rated 4-5

and returns the top 15. Weights were chosen on dev tasks offline (no model calls): the bought
product ranked first 11% -> 29% and in the top 15 67% -> 79%; checked once on test (15% -> 28%,
64% -> 76%). Meaning alone did not beat keywords, so E1 and E2 keep the store's search.

Data comes only through the store tools (search_products, get_product, get_history), so the
history cut at as_of and the shelf limit still hold.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cartblanche.agent.retrieval import embed_items, embed_query
from cartblanche.env.store import TaskStore, _summary
from cartblanche.schemas import HistoryItem, Product, Task

LIMIT = 15
KEYWORD_W, LIKED_W, BRAND_W = 0.3, 1.0, 0.2
LIKED = 4  # ratings at or above this count as liked


def product_text(p: Product) -> str:
    return " | ".join([p.title, p.brand or "", " > ".join(p.categories[1:]), " ".join(p.features[:3])])[:1000]


def history_text(h: HistoryItem) -> str:
    return " | ".join([h.title, h.brand or "", " > ".join(h.categories[1:])])


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-9, None)


def personal_search(task: Task, store: TaskStore, query: str, filters: dict[str, Any] | None = None) -> list[dict]:
    """Shelf products that pass the filters, best for this shopper and query first."""
    passing = store.search_products("", filters, limit=len(store.shelf))  # empty query: filters only
    if not passing:
        return []
    products = [Product(**store.get_product(h["id"])) for h in passing]

    keyword_hits = [h["id"] for h in store.search_products(query, filters, limit=len(store.shelf))]
    keyword = np.array([len(keyword_hits) - keyword_hits.index(p.id) if p.id in keyword_hits else 0 for p in products],
                       dtype=float)
    keyword /= keyword.max() or 1.0

    vectors = _unit(embed_items([product_text(p) for p in products]))
    score = vectors @ _unit(embed_query(query)) + KEYWORD_W * keyword

    history = store.get_history(task.user_id)
    liked = [HistoryItem(**h) for h in history if h["rating"] >= LIKED] if isinstance(history, list) else []
    if liked:
        score += LIKED_W * (vectors @ _unit(embed_items([history_text(h) for h in liked])).T).max(axis=1)
        brands = {(h.brand or "").lower() for h in liked} - {""}
        score += BRAND_W * np.array([(p.brand or "").lower() in brands for p in products], dtype=float)

    order = np.argsort(-score, kind="stable")[:LIMIT]
    return [_summary(products[i]) for i in order]
