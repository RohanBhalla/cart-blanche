"""Build the SQLite store from tiny hand-made source files and read it back through TaskStore."""

import json
import shutil
import sqlite3

import pytest

from cartblanche.env.build_db import build, spot_check
from cartblanche.env.db import SqliteCatalog
from cartblanche.env.store import TaskStore
from cartblanche.schemas import Task

DAY = 86_400_000  # ms


def meta(pid, title, price=10.0, brand="Acme", **extra):
    return {"parent_asin": pid, "title": title, "main_category": "All Beauty", "price": price, "store": brand,
            "categories": ["Beauty & Personal Care", "Skin Care"], "average_rating": 4.2, "rating_number": 50,
            "features": ["Fragrance-free"], "description": ["A cream."], "images": [{"large": "x.jpg"}],
            "details": {"Brand": brand, "UPC": "123", "Skin Type": "Sensitive"}, **extra}


def review(uid, pid, day, rating=5.0, helpful=0, text="fine"):
    return {"user_id": uid, "parent_asin": pid, "asin": pid, "rating": rating, "title": "t", "text": text,
            "timestamp": 1_600_000_000_000 + day * DAY, "helpful_vote": helpful, "verified_purchase": True,
            "images": []}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("src")
    # 5-core: alice has 8 core reviews, bob has 3 (too few); products P1..P8 are core
    core = ["user_id,parent_asin,rating,timestamp"]
    core += [f"alice,P{i},5.0,{i}" for i in range(1, 9)]
    core += [f"bob,P{i},4.0,{i}" for i in range(1, 4)]
    (d / "core.csv").write_text("\n".join(core) + "\n")

    reviews = [review("alice", f"P{i}", i) for i in range(1, 9)]
    reviews.append(review("alice", "X1", 20, rating=2.0, text="broke me out"))  # not in the 5-core: history only
    reviews += [review(f"r{n}", "P1", n, helpful=n) for n in range(1, 6)]  # 5 more reviews on P1
    reviews.append(review("carol", "X2", 3))  # not sampled, not core: dropped
    (d / "reviews.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reviews))

    metas = [meta(f"P{i}", f"Product {i}") for i in range(1, 9)]
    metas.append(meta("X1", "Scented Cream", brand="Smellco"))
    metas.append(meta("X2", "Unrelated Thing"))
    (d / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas))

    src = {"core": str(d / "core.csv"), "reviews": str(d / "reviews.jsonl"), "meta": str(d / "meta.jsonl")}
    out = d / "store.sqlite"
    stats = build(src, out, n_users=10, min_reviews=8, per_product=3)
    return src, out, stats


def test_only_users_with_enough_reviews(built):
    _, out, stats = built
    catalog = SqliteCatalog(out)
    assert list(catalog.users) == ["alice"] and stats["users"] == 1
    assert "bob" not in catalog.users


def test_history_is_complete_and_ordered(built):
    catalog = SqliteCatalog(built[1])
    history = catalog.users["alice"].history
    assert [h.product_id for h in history] == [f"P{i}" for i in range(1, 9)] + ["X1"]
    x1 = history[-1]
    assert x1.title == "Scented Cream" and x1.brand == "Smellco" and x1.rating == 2.0
    assert x1.review_text == "broke me out"


def test_products_kept_and_trimmed(built):
    catalog = SqliteCatalog(built[1])
    assert set(catalog.products) == {f"P{i}" for i in range(1, 9)} | {"X1"}  # X2 is neither core nor history
    p = catalog.products["P1"]
    assert p.brand == "Acme" and p.price == 10.0 and p.description == "A cream."
    assert p.details == {"Brand": "Acme", "Skin Type": "Sensitive"}  # UPC dropped


def test_top_reviews_by_helpful_votes(built):
    catalog = SqliteCatalog(built[1])
    shown = catalog.reviews["P1"]
    assert [r.id for r in shown] == ["P1.review.1", "P1.review.2", "P1.review.3"]
    assert [r.helpful_vote for r in shown] == [5, 4, 3]  # alice's 0-vote review was evicted from the page...
    assert "P1" in [h.product_id for h in catalog.users["alice"].history]  # ...but stays in her history
    assert "X1" not in catalog.reviews  # history-only products don't get a review page


def test_works_through_task_store(built):
    catalog = SqliteCatalog(built[1])
    task = Task(task_id="t", user_id="alice", request="a cream", type="buy", as_of="2020-09-20T00:00:00",
                shelf=["P1", "P2"], candidates=["P1", "P2"], gold_action="BUY", gold_pick="P2")
    store = TaskStore(catalog, task)
    assert store.get_product("P1")["title"] == "Product 1"
    assert len(store.get_history("alice")) < 9  # later reviews hidden by as_of
    assert all(r["timestamp"] < task.as_of for r in store.get_reviews("P1"))
    assert [h["title"] for h in store.search_products("product")] == ["Product 1", "Product 2"]


def test_spot_check(built, tmp_path):
    src, out, _ = built
    assert spot_check(out, src["meta"], n=5) == []
    edited = tmp_path / "edited.sqlite"
    shutil.copy(out, edited)
    with sqlite3.connect(edited) as conn:
        conn.execute("UPDATE products SET price = 99 WHERE id = 'P3'")
    assert spot_check(edited, src["meta"], n=9) == ["P3: price differs"]


def test_missing_db_has_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_db"):
        SqliteCatalog(tmp_path / "nope.sqlite")
