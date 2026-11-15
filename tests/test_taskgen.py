"""Buy-task generation on a small synthetic catalog, with a scripted request writer (no API calls)."""

import json

import pytest

from cartblanche.env.store import Catalog, TaskStore, load_tasks
from cartblanche.env.taskgen import Index, budget_for, generate_buy_tasks, meets, pick_gold, report, save_tasks
from cartblanche.llm import LLM
from cartblanche.schemas import HistoryItem, Product, Review, User

CATS = ["Beauty & Personal Care", "Skin Care", "Face", "Moisturizers"]


def product(i, price=20.0, brand=None, rating_number=100, features=("Lightweight",)):
    return Product(id=f"P{i:02d}", title=f"Moisturizer {i}", main_category="All Beauty", categories=CATS,
                   brand=brand or f"Brand{i}", price=price, average_rating=4.3, rating_number=rating_number,
                   features=list(features))


def item(p, rating, day):
    return HistoryItem(product_id=p.id, title=p.title, main_category="All Beauty", categories=p.categories,
                       brand=p.brand, price=p.price, rating=rating, timestamp=f"2022-01-{day:02d}T00:00:00")


@pytest.fixture
def catalog():
    products = [product(i, price=18 + i % 5) for i in range(1, 61)]
    products[0] = product(1, price=20.0, features=["Fragrance-free", "For sensitive skin"])  # the gold
    products[1] = product(2, price=21.0, brand="Badco", rating_number=5)  # disliked brand, unpopular
    products[2] = product(3, price=19.0, brand="Goodco", features=["Fragrance-free"])  # liked brand, fits
    products[3] = product(4, price=99.0, rating_number=9999)  # popular, way over budget (outside 2x band)
    p = {x.id: x for x in products}
    old = [product(70 + i, brand=b) for i, b in enumerate(["Badco", "Goodco", "Other", "Other2", "Other3"])]
    history = [item(old[0], 1, 1), item(old[1], 5, 2), item(old[2], 4, 3), item(old[3], 4, 4), item(old[4], 3, 5),
               item(p["P01"], 5, 20)]
    shopper = User(id="shopper", history=history)
    newbie = User(id="newbie", history=[item(p["P05"], 5, 1)])  # too little history
    reviews = [Review(id=f"{x.id}.review.1", product_id=x.id, user_id="r", rating=4, timestamp="2021-01-01T00:00:00")
               for x in products]
    return Catalog(products + old, reviews, [shopper, newbie])


class Writer:
    """Scripted request writer: plays back replies in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, **request):
        self.calls += 1
        content = self.replies.pop(0) if self.replies else self.default(request)
        return {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": {}}

    @staticmethod
    def default(request):
        budget = json.loads(request["messages"][1]["content"])["budget"]
        return json.dumps({"request": f"a gentle face moisturizer, {budget}", "required_features": ["fragrance-free", "spf 50"]})


def llm(writer):
    return LLM(model="fake/model", cache_dir=None, completion_fn=writer, cost_fn=lambda raw: 0.0)


def test_budget_is_round_and_above_price():
    import random

    rng = random.Random(0)
    for price in [7.99, 20.0, 64.5, 180.0]:
        b = budget_for(price, rng)
        assert price * 1.1 <= b <= price * 1.5 + 10 and b % 5 == 0


def test_gold_is_latest_good_new_purchase(catalog):
    index = Index.build(catalog)
    assert pick_gold(catalog.users["shopper"], index, min_history=5).product_id == "P01"
    assert pick_gold(catalog.users["newbie"], index, min_history=5) is None


def test_buy_task(catalog):
    [task] = generate_buy_tasks(catalog, llm(Writer()), n=5, min_history=5, n_dev=1)
    assert task.user_id == "shopper" and task.gold_pick == "P01" and task.as_of == "2022-01-20T00:00:00"
    assert task.request.startswith("a gentle face moisturizer, under $")
    assert task.constraints.required_features == ["fragrance-free"]  # "spf 50" dropped: the gold doesn't have it
    assert len(task.shelf) == 50 and len(set(task.shelf)) == 50 and "P01" in task.shelf
    assert len(task.candidates) == 10 and task.candidates[0] == "P01"
    assert "P02" in task.candidates  # the user's disliked brand is a distractor even though it's unpopular
    assert task.split == "dev"
    assert task.acceptable == ["P01", "P03"]  # Goodco: liked brand, fits the constraints
    store = TaskStore(catalog, task)
    assert [h["product_id"] for h in store.get_history("shopper")][-1] == "P74"  # the gold purchase is hidden


def test_three_distractors_break_constraints(catalog):
    [task] = generate_buy_tasks(catalog, llm(Writer()), n=1, min_history=5)
    breaking = [p for p in task.candidates[1:] if not meets(catalog.products[p], task.constraints)]
    assert len(breaking) == 8  # 3 by design, plus 5 more because only P03 also fits (fragrance-free)
    assert "P03" in task.candidates


def test_request_naming_the_brand_is_rejected(catalog):
    named = json.dumps({"request": "Brand1 Moisturizer 1 under $25", "required_features": []})
    tasks = generate_buy_tasks(catalog, llm(Writer(named)), n=5, min_history=5)
    assert tasks == []  # the only eligible user's request leaked the product, so no task


def test_budget_appended_when_missing(catalog):
    tasks = generate_buy_tasks(catalog, llm(Writer('{"request": "a light moisturizer", "required_features": []}')),
                               n=1, min_history=5)
    assert tasks[0].request.startswith("a light moisturizer under $")


def test_save_and_load(catalog, tmp_path):
    tasks = generate_buy_tasks(catalog, llm(Writer()), n=1, min_history=5)
    save_tasks(tasks, tmp_path / "t.jsonl")
    assert load_tasks(tmp_path / "t.jsonl") == tasks


def test_report(catalog):
    tasks = generate_buy_tasks(catalog, llm(Writer()), n=1, min_history=5)
    r = report(tasks, catalog)
    assert r["tasks"] == 1 and r["with_required_feature"] == 1.0 and r["with_acceptable_alternatives"] == 1.0
    assert 0 <= r["gold_rank_1_by_keywords"] <= r["gold_top_10_by_keywords"] <= 1
