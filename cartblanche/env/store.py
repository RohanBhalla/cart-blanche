"""The store the agent shops in.

A Catalog holds every product, review and user. A TaskStore is the view one
task sees: search is limited to the task's shelf, history is cut at the
task's as_of time (so the held-out gold purchase never leaks), and the
task's attack, if any, is applied to the target product's page.

Tool methods return plain JSON-able dicts so they can go straight back to
the LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cartblanche.env.attacks import AttackedPage, apply_attack
from cartblanche.schemas import HistoryItem, Product, Review, Task, User

DATA_DIR = Path(__file__).parent / "data"
_STOPWORDS = {"a", "an", "the", "for", "and", "or", "with", "my", "to", "of", "in", "on", "i", "need", "want", "under"}


def _stem(token: str) -> str:
    """Crude plural folding so 'earbud' matches 'earbuds' and 'case' matches 'cases'."""
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class Catalog:
    def __init__(self, products: list[Product], reviews: list[Review], users: list[User]):
        self.products = {p.id: p for p in products}
        self.reviews: dict[str, list[Review]] = {}
        for r in reviews:
            self.reviews.setdefault(r.product_id, []).append(r)
        self.users = {u.id: u for u in users}

    @classmethod
    def from_json(cls, path: Path) -> Catalog:
        raw = json.loads(Path(path).read_text())
        return cls(
            [Product(**p) for p in raw["products"]],
            [Review(**r) for r in raw["reviews"]],
            [User(**u) for u in raw["users"]],
        )


def load_tasks(path: Path) -> list[Task]:
    return [Task(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def load_fake_store() -> tuple[Catalog, list[Task]]:
    """The hand-written Week 0 store and its 10 tasks."""
    return Catalog.from_json(DATA_DIR / "fake_store.json"), load_tasks(DATA_DIR / "fake_tasks.jsonl")


def _summary(p: Product) -> dict[str, Any]:
    return {
        "id": p.id,
        "title": p.title,
        "brand": p.brand,
        "price": p.price,
        "average_rating": p.average_rating,
        "rating_number": p.rating_number,
    }


class TaskStore:
    def __init__(self, catalog: Catalog, task: Task):
        missing = [pid for pid in task.shelf if pid not in catalog.products]
        if missing:
            raise ValueError(f"task {task.task_id} shelf has unknown products: {missing}")
        self.catalog = catalog
        self.task = task
        self.shelf = list(task.shelf)

    def page(self, product_id: str) -> AttackedPage:
        """The product page as this task sees it, with the task's attack applied."""
        return apply_attack(
            self.catalog.products[product_id], self.catalog.reviews.get(product_id, []), self.task.attack
        )

    @property
    def injected_ids(self) -> set[str]:
        """IDs of every injected item on this task's pages (for the evaluator only)."""
        if self.task.attack is None:
            return set()
        return self.page(self.task.attack.target).injected_ids

    def _visible_history(self, user_id: str) -> list[HistoryItem] | None:
        user = self.catalog.users.get(user_id)
        if user is None or user_id != self.task.user_id:
            return None
        return [h for h in user.history if h.timestamp < self.task.as_of]

    # ---- tools -----------------------------------------------------------

    def search_products(self, query: str, filters: dict[str, Any] | None = None, limit: int = 10) -> list[dict]:
        filters = filters or {}
        q = set(_tokens(query))
        scored = []
        for pid in self.shelf:
            p = self.page(pid).product
            if filters.get("max_price") is not None and (p.price is None or p.price > filters["max_price"]):
                continue
            if filters.get("min_rating") is not None and (p.average_rating or 0) < filters["min_rating"]:
                continue
            if filters.get("brand") and (p.brand or "").lower() != filters["brand"].lower():
                continue
            if filters.get("category") and filters["category"].lower() not in " ".join(p.categories).lower():
                continue
            title = set(_tokens(p.title))
            body = set(_tokens(" ".join([*p.features, *p.categories, p.description])))
            score = 2 * len(q & title) + len(q & body)
            if q and score == 0:
                continue
            scored.append((score, p.average_rating or 0, p))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [_summary(p) for _, _, p in scored[:limit]]

    def get_product(self, product_id: str) -> dict:
        if product_id not in self.shelf:
            return {"error": f"product {product_id} not found"}
        return self.page(product_id).product.model_dump()

    def get_reviews(self, product_id: str) -> list[dict] | dict:
        if product_id not in self.shelf:
            return {"error": f"product {product_id} not found"}
        return [r.model_dump(exclude={"product_id"}) for r in self.page(product_id).reviews]

    def get_history(self, user_id: str) -> list[dict] | dict:
        history = self._visible_history(user_id)
        if history is None:
            return {"error": f"user {user_id} not found"}
        return [h.model_dump() for h in history]

    def check_owned(self, user_id: str, category: str) -> list[dict] | dict:
        history = self._visible_history(user_id)
        if history is None:
            return {"error": f"user {user_id} not found"}
        terms = _tokens(category)
        owned = []
        for h in history:
            haystack = set(_tokens(" ".join([h.title, *h.categories])))
            if terms and all(t in haystack for t in terms):
                owned.append(h.model_dump(include={"product_id", "title", "brand", "rating", "timestamp"}))
        return owned
