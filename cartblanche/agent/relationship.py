"""E4 and E4b: what to do when the shopper already owns something like the product the agent picked.

Both run at the same point, after the agent proposes a purchase. Code looks up the shopper's
visible past purchases of the same kind (the same product, or the same finest category, e.g.
"Hair Dryers") through the store tools. If there are none, the purchase goes ahead unchanged.
If there are some:

  E4  (rule):  owns something similar -> SKIP. No model call.
  E4b (check): one model call labels the relationship (duplicate, replacement, upgrade,
               complement, second unit, unrelated), and code maps it to an action:
                 duplicate                              -> SKIP
                 replacement, pick repeats a product or
                 brand the shopper rated 1-2            -> ask the agent once to choose again
                 replacement after a 1-2 rating         -> ALTERNATIVE
                 anything else                          -> keep the agent's decision

So E4 and E4b differ only in how they treat "already owns one", and the check costs a call only
when there is something to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cartblanche.agent.history import describe
from cartblanche.agent.intent import parse_json
from cartblanche.config import RelationshipMode
from cartblanche.env.store import TaskStore
from cartblanche.llm import LLM
from cartblanche.schemas import Action, HistoryItem, Product, Relationship, Task

BAD_RATING = 2  # a rating at or below this means it didn't work for the shopper

CHECK_PROMPT = """You check a purchase before it is made. The shopper already owns products of the same kind as the
one about to be bought. Decide how the new purchase relates to what they own:
- duplicate: only for a lasting tool or device (hair dryer, straightener, curling iron, shaver, trimmer,
  nail lamp, lighted mirror) of the same kind as one they bought recently, still like, and give no reason
  to replace. Never for products that get used up or that people usually own several of (skin care,
  serums, moisturizers, makeup, lashes, fragrance, aftershave, shampoo, polish): owning one of those is not
  a reason to skip.
- replacement: the new one takes the place of an owned one: a product that gets used up running low or
  simply bought again, or an owned one that broke, wore out or didn't work for them.
- upgrade: they want a better version of something they own.
- complement: a different kind of product used together with one they own (conditioner for a shampoo, top
  coat for a nail polish). Two tools or devices of the same kind are never complements.
- second_unit: they want another one on purpose, for another place or another person (a spare, a gift).
- unrelated: what they own doesn't matter for this purchase.
Use the request, today's date, and the owned products' dates, ratings and reviews. Compare the products by
name: a different product of the same kind is not "the same one".
Reply with only this JSON object:
{"relationship": "duplicate" | "replacement" | "upgrade" | "complement" | "second_unit" | "unrelated",
 "reason": "<one sentence>"}"""


@dataclass
class Verdict:
    """What the relationship step decided for one proposed purchase."""

    action: Action | None = None  # None: keep the agent's action
    relationship: str | None = None
    reason: str = ""
    feedback: str | None = None  # a note asking the agent to choose again
    owned: list[HistoryItem] = field(default_factory=list)

    def record(self, mode: RelationshipMode) -> dict:
        return {"mode": mode, "owned": [h.product_id for h in self.owned], "relationship": self.relationship,
                "action": self.action, "chose_again": self.feedback is not None, "reason": self.reason}


def leaf(categories: list[str]) -> str:
    """The finest category, when the path is specific enough to mean one kind of product (as in task_types)."""
    return categories[-1] if len(categories) >= 3 else ""


def owned_similar(task: Task, store: TaskStore, product: Product) -> list[HistoryItem]:
    """Past purchases of the same product or the same finest category, oldest first."""
    history = store.get_history(task.user_id)
    if not isinstance(history, list):
        return []
    kind = leaf(product.categories)
    items = [HistoryItem(**h) for h in history]
    return [h for h in items if h.product_id == product.id or (kind and leaf(h.categories) == kind)]


def _repeats_a_bad_one(product: Product, owned: list[HistoryItem]) -> HistoryItem | None:
    """The owned item rated 1-2 that this pick repeats (same product, or same brand), if any."""
    brand = (product.brand or "").lower()
    for h in reversed(owned):
        if h.rating <= BAD_RATING and (h.product_id == product.id or (brand and (h.brand or "").lower() == brand)):
            return h
    return None


def _check_message(task: Task, product: Product, owned: list[HistoryItem]) -> str:
    lines = "\n".join(f"{n}. {describe(h)}" for n, h in enumerate(owned, start=1))
    return (f"Today: {task.as_of[:10]}\nRequest: {task.request}\n"
            f"About to buy: {product.title} | {product.brand or 'unknown brand'} | {' > '.join(product.categories)}\n"
            f"Owned products of the same kind:\n{lines}")


def rule(owned: list[HistoryItem]) -> Verdict:
    """E4: owns something similar -> skip."""
    return Verdict(action=Action.SKIP, relationship=Relationship.DUPLICATE, owned=owned,
                   reason="The shopper already owns a product of this kind.")


def check(task: Task, product: Product, owned: list[HistoryItem], llm: LLM, may_choose_again: bool) -> Verdict:
    """E4b: label the relationship with one model call, then map it to an action."""
    reply = llm.complete(
        [{"role": "system", "content": CHECK_PROMPT}, {"role": "user", "content": _check_message(task, product, owned)}],
        tag=f"{task.task_id}/relationship",
    )
    found = parse_json(reply.content)
    label = str(found.get("relationship", "")).strip().lower()
    if label not in set(Relationship):
        raise ValueError(f"unknown relationship {label!r}")
    verdict = Verdict(relationship=label, reason=str(found.get("reason", "")), owned=owned)
    if label == Relationship.DUPLICATE:
        verdict.action = Action.SKIP
    elif label == Relationship.REPLACEMENT:
        bad = _repeats_a_bad_one(product, owned)
        if bad and may_choose_again:
            verdict.feedback = (
                f"Purchase check: the shopper rated {bad.title} ({bad.brand or 'unknown brand'}) {bad.rating:g} of 5"
                f" on {bad.timestamp[:10]}: \"{bad.review_text[:200]}\". {product.title} repeats that product or"
                " brand. Choose a different product that avoids the problem, and answer ALTERNATIVE."
            )
        elif any(h.rating <= BAD_RATING for h in owned) and not bad:
            verdict.action = Action.ALTERNATIVE
    return verdict


def review_purchase(task: Task, store: TaskStore, mode: RelationshipMode, pick: str, llm: LLM,
                    may_choose_again: bool) -> Verdict | None:
    """The relationship step for one proposed purchase; None when the shopper owns nothing similar."""
    product = Product(**store.get_product(pick))
    owned = owned_similar(task, store, product)
    if not owned:
        return None
    if mode == RelationshipMode.RULE:
        return rule(owned)
    return check(task, product, owned, llm, may_choose_again)
