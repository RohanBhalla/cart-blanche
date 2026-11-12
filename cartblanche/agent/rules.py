"""E0: the non-LLM baseline.

Search the shelf with the request text, drop products that break a stated
constraint (budget, size, hard brand), then buy the highest-rated of the top
few hits. It never skips and never reads history, so E1 minus E0 is what
GenAI adds.
"""

from __future__ import annotations

from cartblanche.config import RunConfig
from cartblanche.env.store import TaskStore
from cartblanche.schemas import Action, AgentResult, BrandStrength, Task, spec_evidence_id

TOP_HITS = 5


def run_rules_agent(task: Task, store: TaskStore, config: RunConfig) -> AgentResult:
    c = task.constraints
    filters = {"max_price": c.max_price, "min_rating": c.min_rating}
    if c.brand.strength == BrandStrength.HARD:
        filters["brand"] = c.brand.name
    filters = {k: v for k, v in filters.items() if v is not None}

    hits = store.search_products(task.request, filters, limit=len(store.shelf))
    if c.size:
        hits = [h for h in hits if c.size in store.get_product(h["id"])["details"].get("sizes", [])]
    trace = [{"tool": "search_products", "query": task.request, "filters": filters, "hits": [h["id"] for h in hits]}]

    if not hits:
        return AgentResult(
            task_id=task.task_id, config_id=config.config_id, action=Action.SKIP,
            reasons="No product matches the request's constraints.", trace=trace,
        )
    best = max(hits[:TOP_HITS], key=lambda h: (h["average_rating"] or 0, h["rating_number"]))
    return AgentResult(
        task_id=task.task_id,
        config_id=config.config_id,
        action=Action.BUY,
        pick=best["id"],
        reasons=f"Highest-rated of the top {TOP_HITS} matches ({best['average_rating']} stars).",
        evidence_ids=[spec_evidence_id(best["id"], "average_rating"), spec_evidence_id(best["id"], "price")],
        trace=trace,
    )
