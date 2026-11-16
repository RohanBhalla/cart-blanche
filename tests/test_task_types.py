"""Non-buy task types on a synthetic catalog: one hand-built user per real event."""

import json
import random

import pytest

from cartblanche.env.store import Catalog, TaskStore
from cartblanche.env.task_types import (SPECS, find_brand_conflict, find_brand_soft, find_complement, find_duplicate,
                                        find_replacement_bad, find_restock, find_second_unit, generate_more_tasks,
                                        kind_names)
from cartblanche.env.taskgen import Index
from cartblanche.llm import LLM
from cartblanche.schemas import HistoryItem, Product, Review, User

B = "Beauty & Personal Care"
CATS = {
    "dryer": [B, "Hair Care", "Styling Tools & Appliances", "Hair Dryers"],
    "shampoo": [B, "Hair Care", "Shampoo & Conditioner", "Shampoos"],
    "conditioner": [B, "Hair Care", "Shampoo & Conditioner", "Conditioners"],
    "moisturizer": [B, "Skin Care", "Face", "Face Moisturizers"],
}


def make_catalog():
    products = []
    for kind, cats in CATS.items():
        for i in range(60):
            products.append(Product(id=f"{kind}{i:02d}", title=f"{kind} {i}", main_category="All Beauty",
                                    categories=cats, brand=f"{kind}brand{i % 6}", price=20.0 + i % 7,
                                    average_rating=4.4, rating_number=100 + i))
    by_id = {p.id: p for p in products}

    def h(pid, rating, day):
        p = by_id[pid]
        return HistoryItem(product_id=pid, title=p.title, main_category="All Beauty", categories=p.categories,
                           brand=p.brand, price=p.price, rating=rating, timestamp=f"2022-{1 + day // 28:02d}-{1 + day % 28:02d}T00:00:00")

    filler = [h("moisturizer50", 4, 0), h("moisturizer51", 4, 1), h("moisturizer52", 4, 2)]
    users = [
        User(id="restocker", history=filler + [h("shampoo01", 5, 10), h("shampoo01", 5, 60)]),
        User(id="dryer_owner", history=filler + [h("dryer01", 5, 10)]),
        User(id="two_dryers", history=filler + [h("dryer02", 5, 10), h("dryer03", 4, 80)]),
        User(id="switcher", history=filler + [h("moisturizer01", 1, 10), h("moisturizer08", 5, 60)]),
        User(id="pairer", history=filler + [h("shampoo04", 5, 10), h("conditioner05", 5, 60)]),
        User(id="loyal", history=filler + [h("shampoo06", 5, 5), h("shampoo12", 4, 9), h("shampoo18", 5, 40)]),
        User(id="forgiver", history=filler + [h("conditioner07", 2, 10), h("conditioner13", 5, 60)]),
    ] + [User(id=f"plain{n}", history=filler + [h(f"moisturizer{20 + n}", 4, 10), h(f"moisturizer{30 + n}", 4, 20),
                                                  h(f"shampoo{40 + n}", 5, 60)]) for n in range(3)]
    reviews = [Review(id=f"{p.id}.review.1", product_id=p.id, user_id="r", rating=4, timestamp="2021-01-01T00:00:00")
               for p in products]
    return Catalog(products, reviews, users)


@pytest.fixture(scope="module")
def catalog():
    return make_catalog()


@pytest.fixture(scope="module")
def index(catalog):
    return Index.build(catalog)


@pytest.mark.parametrize("finder, user, anchor", [
    (find_restock, "restocker", "shampoo01"),
    (find_duplicate, "dryer_owner", "dryer01"),
    (find_second_unit, "two_dryers", "dryer03"),
    (find_replacement_bad, "switcher", "moisturizer08"),
    (find_complement, "pairer", "conditioner05"),
    (find_brand_soft, "loyal", "shampoo18"),
    (find_brand_conflict, "forgiver", "conditioner13"),
])
def test_finders(catalog, index, finder, user, anchor):
    ev = finder(catalog.users[user], index, random.Random(0))
    assert ev is not None and ev.anchor == anchor


def test_finders_ignore_other_histories(catalog, index):
    rng = random.Random(0)
    assert find_duplicate(catalog.users["restocker"], index, rng) is None  # shampoo isn't a lasting item
    assert find_restock(catalog.users["dryer_owner"], index, rng) is None
    assert find_replacement_bad(catalog.users["pairer"], index, rng) is None
    assert find_complement(catalog.users["switcher"], index, rng) is None


GENERIC = {"override", "ambiguous", "brand_hard", "nothing_fits", "tiny_history"}


class Kinds:
    def __call__(self, **request):
        paths = json.loads(request["messages"][1]["content"])
        names = {p: p.split(" > ")[-1].lower().rstrip("s") for p in paths}
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps(names)}}], "usage": {}}


@pytest.fixture(scope="module")
def tasks(catalog):
    llm = LLM(model="fake/model", cache_dir=None, completion_fn=Kinds(), cost_fn=lambda raw: 0.0)
    specific = [s for s in SPECS if s.subtype not in GENERIC]  # generic types would use up these users first
    specs = [s.__class__(**{**s.__dict__, "count": 1}) for s in specific]
    return {t.subtype: t for t in generate_more_tasks(catalog, llm, specs=specs)}


def test_each_user_gives_one_task(tasks):
    users = [t.user_id for t in tasks.values()]
    assert len(users) == len(set(users))


def test_gold_actions(tasks):
    expected = {"restock": "BUY", "duplicate": "SKIP", "second_unit": "BUY", "replacement_bad": "ALTERNATIVE",
                "complement": "BUY", "brand_soft": "BUY", "brand_conflict": "BUY"}
    for subtype, action in expected.items():
        assert tasks[subtype].gold_action == action, subtype


def test_restock(tasks):
    t = tasks["restock"]
    assert t.gold_pick == "shampoo01" and t.acceptable == ["shampoo01"]
    assert t.request.startswith(("I'm running low on my shampoo", "Time to restock my shampoo"))


def test_duplicate_shows_the_owned_item(catalog, tasks):
    t = tasks["duplicate"]
    assert t.gold_pick is None and "dryer01" in t.shelf and t.candidates[0] == "dryer01"
    assert t.as_of > "2022-01-11"  # some time after buying it
    owned = TaskStore(catalog, t).check_owned("dryer_owner", "hair dryer")
    assert [o["product_id"] for o in owned] == ["dryer01"]


def test_second_unit_also_accepts_the_same_again(tasks):
    t = tasks["second_unit"]
    assert t.gold_pick == "dryer03" and set(t.acceptable) == {"dryer03", "dryer02"} and "dryer02" in t.shelf


def test_replacement_shows_the_bad_product(tasks):
    t = tasks["replacement_bad"]
    assert t.gold_pick == "moisturizer08" and "moisturizer01" in t.candidates
    assert "moisturizer01" not in t.acceptable


def test_complement_names_what_they_own(tasks):
    t = tasks["complement"]
    assert t.request.startswith("I need a conditioner to go with my shampoo")


def test_brand_tasks(tasks):
    assert tasks["brand_conflict"].constraints.brand.strength == "hard"
    assert tasks["brand_conflict"].constraints.brand.name == "conditionerbrand1"
    assert tasks["brand_soft"].constraints.brand.strength == "none"


def test_shelves_have_no_repeats(tasks):
    for t in tasks.values():
        assert len(t.shelf) == len(set(t.shelf)) <= 50, t.subtype


def test_split_is_dev_or_test(tasks):
    assert {t.split for t in tasks.values()} <= {"dev", "test"}


def test_nothing_fits(catalog):
    llm = LLM(model="fake/model", cache_dir=None, completion_fn=Kinds(), cost_fn=lambda raw: 0.0)
    spec = next(s for s in SPECS if s.subtype == "nothing_fits")
    [t] = generate_more_tasks(catalog, llm, specs=[spec.__class__(**{**spec.__dict__, "count": 1})])
    assert t.gold_action == "SKIP"
    assert all(catalog.products[p].price > t.constraints.max_price for p in t.shelf)


def test_kind_names_fall_back_to_the_leaf():
    def broken(**request):
        return {"choices": [{"message": {"role": "assistant", "content": "sorry"}}], "usage": {}}

    llm = LLM(model="fake/model", cache_dir=None, completion_fn=broken, cost_fn=lambda raw: 0.0)
    assert kind_names([tuple(CATS["dryer"])], llm) == {tuple(CATS["dryer"]): "hair dryers"}


def test_generic_types(catalog):
    llm = LLM(model="fake/model", cache_dir=None, completion_fn=Kinds(), cost_fn=lambda raw: 0.0)
    specs = [s.__class__(**{**s.__dict__, "count": 1}) for s in SPECS if s.subtype in GENERIC - {"nothing_fits"}]
    tasks = {t.subtype: t for t in generate_more_tasks(catalog, llm, specs=specs)}
    assert tasks["override"].override and tasks["override"].gold_pick == tasks["override"].candidates[0]
    amb = tasks["ambiguous"]
    assert amb.hidden_profile["looking_for"] and amb.request.count("hair") + amb.request.count("skin") >= 1
    kinds = {tuple(catalog.products[p].categories) for p in amb.shelf}
    assert len(kinds) > 1  # fillers from sibling categories, so the request really is vague
    hard = tasks["brand_hard"]
    assert hard.constraints.brand.strength == "hard" and hard.constraints.brand.name in hard.request
    assert all(catalog.products[a].brand == hard.constraints.brand.name for a in hard.acceptable)
    assert len(TaskStore(catalog, tasks["tiny_history"]).get_history(tasks["tiny_history"].user_id)) <= 2
