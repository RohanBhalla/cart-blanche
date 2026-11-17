"""Task generator: builds tasks with ground truth from real user histories.

    python -m cartblanche.env.taskgen --buy 150     # buy tasks -> tasks/beauty.jsonl
    python -m cartblanche.env.taskgen --other       # all other types -> tasks/beauty_other.jsonl (env/task_types.py)

Buy tasks (the plan's "hold out a later purchase"):
  * Gold: one of the user's 4-5 star reviews, on a priced product, with at least
    `min_history` earlier reviews. The task's as_of is that review's time, so the
    agent only sees what came before. Products the user had already reviewed are
    skipped (re-buying is a restock task, built separately).
  * Request: an LLM writes what the shopper might type: only the general kind of
    product, never its brand, title or specific details, so the request fits many
    products and choosing well needs the user's history. A budget above the gold's price is
    added, and at most one must-have feature, kept only if the gold really has it.
  * Shelf (~50): the gold, 9 hard distractors from the same narrow category, and
    same-category fillers. Distractors are popular products near the gold's price;
    brands the user rated 1-2 stars come first, so knowing the history matters, and
    3 of the 9 break the request's constraints (most often the budget).
  * Acceptable: other candidates that meet every hard constraint and are from a brand
    the user rated 4-5 stars before. A first pass in code; the team checks it by hand
    on the gold subset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from cartblanche.agent.intent import parse_json
from cartblanche.env.store import TaskStore, has_features
from cartblanche.llm import LLM
from cartblanche.schemas import Action, Constraints, HistoryItem, Product, Split, Task, User

TASKS_DIR = Path("tasks")
SHELF_SIZE = 50
N_DISTRACTORS = 9
N_BREAK_CONSTRAINTS = 3
MIN_CATEGORY_SIZE = 50

REQUEST_PROMPT = """You write realistic shopping requests for a test set.

Given a product, write the short request a shopper might type to an assistant when looking for this
kind of product. The request must fit many products of this kind, not just this one, because the
assistant is meant to choose using what it knows about the shopper. Rules:
- Name only the general kind of product, in plain words (e.g. "a face moisturizer", "a hair dryer").
- Don't mention ingredients, scents, colors, sizes, pack counts, materials or other details of this
  product, and never name the brand or copy the title.
- Optionally add one must-have feature that the product really has and that many similar products
  also have (e.g. "fragrance-free", "sulfate-free", "for curly hair"). If you do, also list it in
  required_features, using words that appear in the product's title, features or details.
- End with the budget exactly as given, e.g. "under $25".
- Keep it to one short sentence.

Reply with only JSON: {"request": "...", "required_features": ["..."]}"""


@dataclass
class Index:
    """Priced products that have a review page, grouped by their full category path."""

    by_category: dict[tuple[str, ...], list[Product]]

    @classmethod
    def build(cls, catalog) -> Index:
        with_reviews = set(catalog.reviews)
        groups: dict[tuple[str, ...], list[Product]] = defaultdict(list)
        for pid in catalog.products:
            if pid not in with_reviews:
                continue
            p = catalog.products[pid]
            if p.price is not None and len(p.categories) >= 3:
                groups[tuple(p.categories)].append(p)
        for products in groups.values():
            products.sort(key=lambda p: p.id)
        return cls(dict(groups))

    def category(self, product: Product) -> list[Product]:
        return self.by_category.get(tuple(product.categories), [])


def budget_for(price: float, rng: random.Random) -> float:
    """A round budget 10-50% above the gold's price."""
    raw = price * rng.uniform(1.1, 1.5)
    step = 5 if raw < 100 else 10
    return float(math.ceil(raw / step) * step)


def pick_gold(user: User, index: Index, min_history: int) -> HistoryItem | None:
    """The user's latest 4-5 star review that can be a buy task's gold."""
    for i in range(len(user.history) - 1, min_history - 1, -1):
        h = user.history[i]
        earlier = {e.product_id for e in user.history[:i]}
        if h.rating < 4 or h.product_id in earlier or h.timestamp <= user.history[i - 1].timestamp:
            continue
        same_category = index.by_category.get(tuple(h.categories), [])
        if len(same_category) >= MIN_CATEGORY_SIZE and any(p.id == h.product_id for p in same_category):
            return h
    return None


def write_request(gold: Product, max_price: float, llm: LLM, tag: str) -> tuple[str, list[str]] | None:
    """LLM-written request plus validated required features, or None if it named the product."""
    product = {"title": gold.title, "category": " > ".join(gold.categories[1:]), "features": gold.features[:5],
               "details": gold.details, "budget": f"under ${max_price:g}"}
    reply = llm.complete([{"role": "system", "content": REQUEST_PROMPT},
                          {"role": "user", "content": json.dumps(product)}], tag=f"{tag}/request")
    try:
        d = parse_json(reply.content)
    except ValueError:
        return None
    request = str(d.get("request", "")).strip()
    if not request or (gold.brand and gold.brand.lower() in request.lower()):
        return None
    if f"${max_price:g}" not in request:
        request = f"{request.rstrip('.')} under ${max_price:g}"
    features = [f for f in d.get("required_features") or [] if isinstance(f, str) and has_features(gold, [f])]
    return request, features[:1]


def meets(product: Product, c: Constraints) -> bool:
    return (c.max_price is None or (product.price is not None and product.price <= c.max_price)) and has_features(
        product, c.required_features
    )


def build_shelf(gold: Product, constraints: Constraints, history: list[HistoryItem], index: Index,
                rng: random.Random, first: tuple[Product, ...] = (),
                filler_pool: list[Product] | None = None) -> tuple[list[str], list[str]]:
    """(shelf, distractors) around a product (the gold, or for a SKIP task the item in question).

    `first` are distractors that must be included (e.g. the product the user rated badly);
    `filler_pool` replaces the same-category fillers (ambiguous tasks use sibling categories)."""
    owned = {h.product_id for h in history}
    forced = {p.id for p in first}
    pool = [p for p in index.category(gold) if p.id != gold.id and p.id not in owned and p.id not in forced]
    disliked = {(h.brand or "").lower() for h in history if h.rating <= 2 and h.brand}

    def rank(p: Product):  # disliked brands first, then popular, then near the gold's price
        return ((p.brand or "").lower() not in disliked, -p.rating_number, abs(p.price - gold.price))

    near = [p for p in pool if 0.5 * gold.price <= p.price <= 2 * gold.price]
    fits = sorted((p for p in near if meets(p, constraints)), key=rank)
    breaks = sorted((p for p in near if not meets(p, constraints)), key=rank)
    distractors = list(first)
    distractors += breaks[: max(0, N_BREAK_CONSTRAINTS - len(distractors))]
    distractors += fits[: N_DISTRACTORS - len(distractors)]
    distractors += breaks[N_BREAK_CONSTRAINTS:][: N_DISTRACTORS - len(distractors)]  # if too few fit
    chosen = {gold.id, *(p.id for p in distractors)}
    rest = [p.id for p in (pool if filler_pool is None else filler_pool) if p.id not in chosen and p.id not in owned]
    fillers = rng.sample(rest, min(SHELF_SIZE - len(chosen), len(rest)))
    shelf = [gold.id, *(p.id for p in distractors), *fillers]
    rng.shuffle(shelf)
    return shelf, [p.id for p in distractors]


def make_buy_task(task_id: str, user: User, catalog, index: Index, llm: LLM, rng: random.Random,
                  min_history: int) -> Task | None:
    h = pick_gold(user, index, min_history)
    if h is None:
        return None
    gold = catalog.products[h.product_id]
    max_price = budget_for(gold.price, rng)
    written = write_request(gold, max_price, llm, tag=task_id)
    if written is None:
        return None
    request, features = written
    constraints = Constraints(max_price=max_price, required_features=features)
    history = [x for x in user.history if x.timestamp < h.timestamp]
    shelf, distractors = build_shelf(gold, constraints, history, index, rng)
    if len(distractors) < N_DISTRACTORS:
        return None
    liked = {(x.brand or "").lower() for x in history if x.rating >= 4 and x.brand}
    acceptable = [d for d in distractors if meets(catalog.products[d], constraints)
                  and (catalog.products[d].brand or "").lower() in liked]
    return Task(task_id=task_id, user_id=user.id, request=request, type="buy", as_of=h.timestamp, shelf=shelf,
                candidates=[gold.id, *distractors], gold_action=Action.BUY, gold_pick=gold.id,
                acceptable=acceptable, constraints=constraints, relationship="unrelated")


def generate_buy_tasks(catalog, llm: LLM, n: int, seed: int = 0, min_history: int = 5, n_dev: int = 40) -> list[Task]:
    rng = random.Random(seed)
    index = Index.build(catalog)
    users = sorted(catalog.users)
    rng.shuffle(users)
    tasks: list[Task] = []
    for uid in users:
        if len(tasks) == n:
            break
        task = make_buy_task(f"b{len(tasks) + 1:04d}", catalog.users[uid], catalog, index, llm, rng, min_history)
        if task:
            tasks.append(task)
            print(f"  {task.task_id}: {task.request}", flush=True)
    for i in rng.sample(range(len(tasks)), len(tasks)):  # random dev / test split
        tasks[i].split = Split.DEV if n_dev > 0 else Split.TEST
        n_dev -= 1
    return tasks


def report(tasks: list[Task], catalog) -> dict:
    """How guessable each task is from the request alone: the gold's (or SKIP task's item's) rank in a plain keyword search
    of the request. If it's often first, the requests are too specific and history won't matter."""
    ranks = []
    for t in tasks:
        hits = [h["id"] for h in TaskStore(catalog, t).search_products(t.request, limit=len(t.shelf))]
        target = t.gold_pick or t.candidates[0]  # SKIP tasks: the product in question
        ranks.append(hits.index(target) + 1 if target in hits else None)
    n = len(tasks) or 1
    return {
        "tasks": len(tasks),
        "gold_rank_1_by_keywords": sum(r == 1 for r in ranks) / n,
        "gold_top_10_by_keywords": sum(r is not None and r <= 10 for r in ranks) / n,
        "with_required_feature": sum(bool(t.constraints.required_features) for t in tasks) / n,
        "with_acceptable_alternatives": sum(len(t.acceptable) > 1 for t in tasks) / n,
        "dev": sum(t.split == Split.DEV for t in tasks),
    }


def save_tasks(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(t.model_dump_json(exclude_defaults=True) + "\n" for t in tasks))


def main() -> None:
    from cartblanche.config import RunConfig
    from cartblanche.env.db import SqliteCatalog

    parser = argparse.ArgumentParser(description="Generate tasks from the real store")
    parser.add_argument("--buy", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--other", action="store_true", help="generate the non-buy task types instead")
    parser.add_argument("--out", type=Path, help="default: tasks/beauty.jsonl, or tasks/beauty_other.jsonl with --other")
    parser.add_argument("--call-gap", type=float, default=0.0)
    args = parser.parse_args()
    llm = LLM(model=RunConfig(config_id="taskgen").agent_model, log_path=Path("runs/taskgen/calls.jsonl"),
              min_gap_s=args.call_gap)
    catalog = SqliteCatalog()
    if args.other:
        from cartblanche.env.task_types import generate_more_tasks

        tasks, out = generate_more_tasks(catalog, llm, args.seed), args.out or TASKS_DIR / "beauty_other.jsonl"
    else:
        tasks, out = generate_buy_tasks(catalog, llm, args.buy, args.seed), args.out or TASKS_DIR / "beauty.jsonl"
    save_tasks(tasks, out)
    print(json.dumps(report(tasks, catalog), indent=2))
    print(f"Wrote {len(tasks)} tasks to {out} ({llm.usage.calls} model calls, ${llm.usage.spent_usd:.3f})")


if __name__ == "__main__":
    main()
