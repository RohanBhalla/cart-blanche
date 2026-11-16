"""Relationship, override, ambiguous, brand and edge-case tasks, found in real histories.

Each finder looks for one real event in a user's history and returns where it
happened; `assemble` turns it into a Task with the same kind of shelf as buy tasks.

| subtype         | real event in the history                                   | gold        |
|-----------------|-------------------------------------------------------------|-------------|
| restock         | reviewed the same product twice, liked it both times        | BUY it      |
| duplicate       | bought a lasting item (e.g. a hair dryer), liked it         | SKIP        |
| override        | a duplicate, but the user says "buy it anyway"              | BUY it      |
| second_unit     | bought a second lasting item of the same kind, liked both   | BUY         |
| replacement_bad | rated a product 1-2, later liked another brand of that kind | ALTERNATIVE |
| complement      | liked A, later bought its companion B (shampoo -> conditioner) | BUY      |
| ambiguous       | a later purchase, asked for only vaguely ("something for my hair") | BUY  |
| brand_hard      | a later purchase, with the brand named as a must            | BUY         |
| brand_soft      | a later purchase from a brand they liked twice before, brand unstated | BUY |
| brand_conflict  | names a brand they rated 1-2 before, and did buy and like it | BUY        |
| nothing_fits    | a later purchase, but the budget is below every product     | SKIP        |
| tiny_history    | a later purchase with only 1-2 earlier reviews              | BUY         |

Requests come from templates around a readable name for each kind of product.
The names come from one batched LLM call, so this costs a single request.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from cartblanche.agent.intent import parse_json
from cartblanche.env.taskgen import Index, budget_for, build_shelf, meets
from cartblanche.llm import LLM
from cartblanche.schemas import (Action, Brand, BrandStrength, Constraints, HistoryItem, Product, Relationship,
                                 Split, Task, TaskType, User)

# Lasting items: owning a recent, well-liked one means you don't need another.
DURABLE = {"Hair Dryers", "Straighteners", "Curling Wands", "Curling Tongs", "Hot-Air Brushes", "Makeup Mirrors",
           "Rotary Shavers", "Foil Shavers", "Epilators", "Hair Clippers", "Beard & Mustache Trimmers",
           "Nose & Ear Hair Trimmers", "Curing Lamps", "Nail Dryers"}
# Owning the first makes the second a natural next purchase.
COMPLEMENTS = {("Shampoos", "Conditioners"), ("Conditioners", "Shampoos"), ("Nail Polish", "Nail Polish Remover"),
               ("Nail Polish", "Combination Base & Top Coats"), ("Nail Polish", "Top Coat"),
               ("Foundation", "Foundation Primers"), ("Foundation", "Blenders & Sponges"),
               ("Acrylic Nail Tools", "Nail Tips"), ("Hair Dryers", "Hair Brushes"), ("Curing Lamps", "Nail Polish"),
               ("Safety Razors", "Shaving Creams"), ("Cartridge Razors", "Shaving Creams")}
RECENT_DAYS = 365  # a duplicate is only a duplicate if the owned item is this recent

OVERRIDE = "I know I already have one. Buy the same one again anyway, I want a spare."
TEMPLATES = {
    "restock": ["I'm running low on my {kind}, I need more, under {budget}",
                "Time to restock my {kind}, under {budget}"],
    "duplicate": ["I need a {kind}, under {budget}", "Can you find me a good {kind}? Under {budget}"],
    "override": ["I need a {kind}, under {budget}", "Can you find me a good {kind}? Under {budget}"],  # then OVERRIDE
    "second_unit": ["I'd like a second {kind} to keep at my partner's place, under {budget}",
                    "I want another {kind} as a gift for my sister, under {budget}"],
    "replacement_bad": ["I need a new {kind}, the last one didn't work for me, under {budget}",
                        "Looking for a different {kind}, my last one was a letdown, under {budget}"],
    "complement": ["I need a {kind} to go with my {owned}, under {budget}"],
    "ambiguous": ["Can you get me something for my {broad}? Under {budget}", "I need a {broad} product, under {budget}"],
    "brand_hard": ["I want a {kind} from {brand}, nothing else, under {budget}"],
    "brand_soft": ["I need a {kind}, under {budget}"],
    "brand_conflict": ["I want a {kind} from {brand}, under {budget}"],
    "nothing_fits": ["I need a {kind}, under {budget}"],
    "tiny_history": ["I need a {kind}, under {budget}"],
}
KIND_PROMPT = """For each Amazon category path below, give the short everyday name a shopper would use for that kind of
product, singular and lowercase (e.g. "Hair Care > Styling Products > Gels" -> "hair gel",
"Skin Care > Face > Creams & Moisturizers > Face Moisturizers" -> "face moisturizer", "Hair Care" -> "hair").
Reply with only a JSON object mapping each path exactly as given to its name."""


@dataclass
class Event:
    """Where a task comes from in one user's history."""

    anchor: str  # the gold product, or for a SKIP task the product in question
    as_of: str
    first: tuple[str, ...] = ()  # products that must be distractors (e.g. the one they rated badly)
    keep: tuple[str, ...] = ()  # history products allowed on the shelf (e.g. the one they own)
    extra: dict = field(default_factory=dict)  # template fields such as "owned" or "brand"


def leaf(item: HistoryItem | Product) -> str:
    return item.categories[-1] if len(item.categories) >= 3 else ""


def _later(ts: str, days: int) -> str:
    return (datetime.fromisoformat(ts) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")


def _good_targets(user: User, index: Index, min_history: int) -> Iterator[tuple[int, HistoryItem]]:
    """Later 4-5 star purchases of a new product in a well-stocked category, latest first."""
    for i in range(len(user.history) - 1, min_history - 1, -1):
        h = user.history[i]
        earlier = user.history[:i]
        if h.rating < 4 or h.product_id in {e.product_id for e in earlier} or h.timestamp <= earlier[-1].timestamp:
            continue
        if any(p.id == h.product_id for p in index.by_category.get(tuple(h.categories), [])):
            yield i, h


# ---- finders: one per subtype; each returns an Event or None -------------------


def find_restock(user: User, index: Index, rng: random.Random) -> Event | None:
    for i in range(len(user.history) - 1, 2, -1):
        h = user.history[i]
        before = [e for e in user.history[:i] if e.product_id == h.product_id and e.timestamp < h.timestamp]
        if h.rating >= 4 and before and before[-1].rating >= 4 and _indexed(index, h):
            return Event(h.product_id, h.timestamp, keep=(h.product_id,))
    return None


def find_duplicate(user: User, index: Index, rng: random.Random) -> Event | None:
    for h in reversed(user.history[2:]):
        if leaf(h) in DURABLE and h.rating >= 4 and _indexed(index, h):
            return Event(h.product_id, _later(h.timestamp, rng.randint(30, RECENT_DAYS // 2)), keep=(h.product_id,))
    return None


def find_second_unit(user: User, index: Index, rng: random.Random) -> Event | None:
    for i in range(len(user.history) - 1, 2, -1):
        h = user.history[i]
        same = [e for e in user.history[:i] if leaf(e) == leaf(h) and e.product_id != h.product_id and e.rating >= 4]
        if leaf(h) in DURABLE and h.rating >= 4 and same and _indexed(index, h) and h.timestamp > same[-1].timestamp:
            return Event(h.product_id, h.timestamp, keep=(same[-1].product_id,), extra={"also_ok": same[-1].product_id})
    return None


def find_replacement_bad(user: User, index: Index, rng: random.Random) -> Event | None:
    for i in range(len(user.history) - 1, 2, -1):
        h = user.history[i]
        bad = [e for e in user.history[:i] if leaf(e) == leaf(h) and e.rating <= 2 and e.brand and e.brand != h.brand
               and e.timestamp < h.timestamp]
        if h.rating >= 4 and bad and _indexed(index, h):
            shown = tuple(b.product_id for b in bad[-1:] if _indexed(index, b))
            return Event(h.product_id, h.timestamp, first=shown, keep=shown, extra={"bad_brand": bad[-1].brand})
    return None


def find_complement(user: User, index: Index, rng: random.Random) -> Event | None:
    for i in range(len(user.history) - 1, 2, -1):
        h = user.history[i]
        earlier = [e for e in user.history[:i] if e.timestamp < h.timestamp]
        if h.rating < 4 or not _indexed(index, h) or any(leaf(e) == leaf(h) for e in earlier):
            continue
        owned = [e for e in earlier if (leaf(e), leaf(h)) in COMPLEMENTS and e.rating >= 4]
        if owned:
            return Event(h.product_id, h.timestamp, extra={"owned_category": owned[-1].categories})
    return None


def find_later_purchase(min_history: int = 5, max_history: int | None = None):
    def find(user: User, index: Index, rng: random.Random) -> Event | None:
        for i, h in _good_targets(user, index, min_history):
            if max_history is None or i <= max_history:
                return Event(h.product_id, h.timestamp)
        return None
    return find


def find_brand_soft(user: User, index: Index, rng: random.Random) -> Event | None:
    for i, h in _good_targets(user, index, 5):
        liked = [e for e in user.history[:i] if h.brand and e.brand == h.brand and e.rating >= 4]
        if len(liked) >= 2:
            return Event(h.product_id, h.timestamp)
    return None


def find_brand_conflict(user: User, index: Index, rng: random.Random) -> Event | None:
    for i, h in _good_targets(user, index, 3):
        if h.brand and any(e.brand == h.brand and e.rating <= 2 for e in user.history[:i]):
            return Event(h.product_id, h.timestamp, extra={"brand": h.brand})
    return None


def _indexed(index: Index, h: HistoryItem) -> bool:
    return len(h.categories) >= 3 and any(p.id == h.product_id for p in index.by_category.get(tuple(h.categories), []))


@dataclass
class Spec:
    subtype: str
    type: TaskType
    relationship: Relationship
    action: Action
    find: Callable[[User, Index, random.Random], Event | None]
    count: int


SPECS = [
    Spec("restock", TaskType.RELATIONSHIP, Relationship.REPLACEMENT, Action.BUY, find_restock, 20),
    Spec("duplicate", TaskType.RELATIONSHIP, Relationship.DUPLICATE, Action.SKIP, find_duplicate, 20),
    Spec("second_unit", TaskType.RELATIONSHIP, Relationship.SECOND_UNIT, Action.BUY, find_second_unit, 10),
    Spec("replacement_bad", TaskType.RELATIONSHIP, Relationship.REPLACEMENT, Action.ALTERNATIVE, find_replacement_bad, 15),
    Spec("complement", TaskType.RELATIONSHIP, Relationship.COMPLEMENT, Action.BUY, find_complement, 15),
    Spec("override", TaskType.OVERRIDE, Relationship.DUPLICATE, Action.BUY, find_duplicate, 20),
    Spec("ambiguous", TaskType.AMBIGUOUS, Relationship.UNRELATED, Action.BUY, find_later_purchase(), 30),
    Spec("brand_hard", TaskType.BRAND, Relationship.UNRELATED, Action.BUY, find_later_purchase(), 10),
    Spec("brand_soft", TaskType.BRAND, Relationship.UNRELATED, Action.BUY, find_brand_soft, 10),
    Spec("brand_conflict", TaskType.BRAND, Relationship.UNRELATED, Action.BUY, find_brand_conflict, 10),
    Spec("nothing_fits", TaskType.BUY, Relationship.UNRELATED, Action.SKIP, find_later_purchase(), 10),
    Spec("tiny_history", TaskType.BUY, Relationship.UNRELATED, Action.BUY, find_later_purchase(1, 2), 10),
]


# ---- assembling a task ----------------------------------------------------------


def kind_names(paths: list[tuple[str, ...]], llm: LLM) -> dict[tuple[str, ...], str]:
    """Readable names for category paths, from one LLM call. Falls back to the lowercased leaf."""
    keys = {" > ".join(p[1:]): p for p in paths}
    reply = llm.complete([{"role": "system", "content": KIND_PROMPT},
                          {"role": "user", "content": json.dumps(sorted(keys))}], tag="taskgen/kinds")
    try:
        names = parse_json(reply.content)
    except ValueError:
        names = {}
    return {p: str(names.get(k) or p[-1]).lower() for k, p in keys.items()}


def assemble(spec: Spec, task_id: str, user: User, ev: Event, catalog, index: Index, kinds: dict,
             rng: random.Random) -> Task | None:
    anchor = catalog.products[ev.anchor]
    history = [h for h in user.history if h.timestamp < ev.as_of]
    budget = budget_for(anchor.price, rng)
    brand = Brand()
    if spec.subtype in ("brand_hard", "brand_conflict"):
        brand = Brand(name=anchor.brand, strength=BrandStrength.HARD)
    if spec.subtype == "nothing_fits":
        budget = None  # set below, once the shelf is known
    constraints = Constraints(max_price=budget, brand=brand)

    first = tuple(catalog.products[p] for p in ev.first)
    filler_pool = None
    if spec.subtype == "ambiguous":  # fillers from sibling categories, so the vague request really is vague
        broad = tuple(anchor.categories[:2])
        filler_pool = [p for cat, ps in index.by_category.items() if cat[:2] == broad and cat != tuple(anchor.categories)
                       for p in ps]
    shelf_history = [h for h in history if h.product_id not in ev.keep]
    shelf, distractors = build_shelf(anchor, constraints, shelf_history, index, rng, first=first, filler_pool=filler_pool)
    if len(distractors) < 5:
        return None
    fillers = [i for i, pid in enumerate(shelf) if pid != anchor.id and pid not in distractors]
    for pid in ev.keep:  # e.g. the owned hair dryer, shown again in place of a filler
        if pid not in shelf and fillers:
            shelf[fillers.pop(rng.randrange(len(fillers)))] = pid
    candidates = [anchor.id, *distractors, *(p for p in ev.keep if p != anchor.id and p not in distractors)]
    candidates = [c for c in dict.fromkeys(candidates) if c in shelf]
    if anchor.id not in shelf:
        return None

    if spec.subtype == "nothing_fits":
        cheapest = min(catalog.products[p].price for p in shelf)
        constraints.max_price = float(max(1, math.floor(cheapest * 0.6)))
    fields = {"kind": kinds[tuple(anchor.categories)], "budget": f"${constraints.max_price:g}",
              "brand": anchor.brand, "broad": kinds.get(tuple(anchor.categories[:2]), anchor.categories[1].lower())}
    if "owned_category" in ev.extra:
        fields["owned"] = kinds[tuple(ev.extra["owned_category"])]
    request = rng.choice(TEMPLATES[spec.subtype]).format(**fields)

    gold_pick = None if spec.action == Action.SKIP else anchor.id
    acceptable = _acceptable(spec, anchor, candidates, constraints, history, catalog, ev)
    hidden = None
    if spec.subtype == "ambiguous":
        hidden = {"looking_for": fields["kind"], "for": "myself", "budget": constraints.max_price}
    return Task(task_id=task_id, user_id=user.id, request=request, type=spec.type, as_of=ev.as_of, shelf=shelf,
                candidates=candidates, gold_action=spec.action, gold_pick=gold_pick,
                acceptable=[a for a in acceptable if a != gold_pick] if gold_pick else [],
                constraints=constraints, relationship=spec.relationship, subtype=spec.subtype,
                override=OVERRIDE if spec.subtype == "override" else None, hidden_profile=hidden)


def _acceptable(spec: Spec, anchor: Product, candidates: list[str], c: Constraints, history: list[HistoryItem],
                catalog, ev: Event) -> list[str]:
    if spec.action == Action.SKIP:
        return []
    if spec.subtype in ("restock", "override"):
        return [anchor.id]
    if spec.subtype == "second_unit":
        return [anchor.id, ev.extra["also_ok"]]
    others = [catalog.products[x] for x in candidates if x != anchor.id and meets(catalog.products[x], c)]
    if spec.subtype in ("brand_hard", "brand_conflict"):
        return [p.id for p in others if p.brand == anchor.brand]
    liked = {(h.brand or "").lower() for h in history if h.rating >= 4 and h.brand}
    bad = (ev.extra.get("bad_brand") or "").lower()
    return [p.id for p in others if (p.brand or "").lower() in liked and (p.brand or "").lower() != bad]


def generate_more_tasks(catalog, llm: LLM, seed: int = 0, specs: list[Spec] = SPECS,
                        dev_share: float = 0.2) -> list[Task]:
    """All non-buy tasks: for each spec, scan users in a seeded random order until `count` tasks are made."""
    rng = random.Random(seed)
    index = Index.build(catalog)
    users = sorted(catalog.users)
    rng.shuffle(users)
    found: list[tuple[Spec, str, Event]] = []
    used: set[str] = set()  # each user gives at most one task, so types don't share users
    for spec in specs:
        made = 0
        for uid in users:
            if made == spec.count:
                break
            ev = None if uid in used else spec.find(catalog.users[uid], index, rng)
            if ev:
                found.append((spec, uid, ev))
                used.add(uid)
                made += 1
    paths = {tuple(catalog.products[ev.anchor].categories) for _, _, ev in found}
    paths |= {tuple(catalog.products[ev.anchor].categories[:2]) for _, _, ev in found}
    paths |= {tuple(ev.extra["owned_category"]) for _, _, ev in found if "owned_category" in ev.extra}
    kinds = kind_names(sorted(paths), llm)

    tasks: list[Task] = []
    for spec in specs:
        group = [assemble(spec, f"{spec.subtype}_{n:03d}", catalog.users[uid], ev, catalog, index, kinds, rng)
                 for n, (s, uid, ev) in enumerate((f for f in found if f[0] is spec), 1)]
        group = [t for t in group if t]
        dev = set(rng.sample(range(len(group)), round(len(group) * dev_share)))  # same dev share in every subtype
        tasks += [t.model_copy(update={"split": Split.DEV if i in dev else Split.TEST}) for i, t in enumerate(group)]
    return tasks
