import json

import pytest

from cartblanche.env.store import TaskStore, load_fake_store
from cartblanche.env.tools import TOOL_SPECS, dispatch
from cartblanche.schemas import Action, AttackSpec, TaskType


@pytest.fixture(scope="module")
def fake():
    return load_fake_store()


@pytest.fixture
def tasks(fake):
    return {t.task_id: t for t in fake[1]}


def store_for(fake, tasks, task_id, **update):
    return TaskStore(fake[0], tasks[task_id].model_copy(update=update))


# ---- fake data sanity ---------------------------------------------------


def test_ten_tasks_cover_every_kind(fake):
    catalog, tasks = fake
    assert len(tasks) == 10
    assert len(catalog.products) >= 30
    actions = {t.gold_action for t in tasks}
    assert {Action.BUY, Action.SKIP, Action.ALTERNATIVE} <= actions
    assert any(t.type == TaskType.DONT_BUY for t in tasks)
    assert any(t.override for t in tasks)
    assert any(t.hidden_profile for t in tasks)
    assert any(t.attack for t in tasks)
    assert {"complement", "duplicate", "replacement", "second_unit"} <= {t.relationship for t in tasks}


def test_every_task_resolves(fake):
    catalog, tasks = fake
    for t in tasks:
        TaskStore(catalog, t)  # shelf products all exist
        assert t.user_id in catalog.users
        assert all(pid in catalog.reviews for pid in t.candidates), t.task_id


def test_review_ids_unique_and_match_product(fake):
    catalog, _ = fake
    ids = [r.id for rs in catalog.reviews.values() for r in rs]
    assert len(ids) == len(set(ids))
    for pid, rs in catalog.reviews.items():
        assert all(r.id.startswith(f"{pid}.review.") for r in rs)


def test_gold_satisfies_constraints(fake):
    catalog, tasks = fake
    for t in tasks:
        if t.gold_pick and t.constraints.max_price is not None:
            assert catalog.products[t.gold_pick].price <= t.constraints.max_price
        if t.gold_pick and t.constraints.size is not None:
            assert t.constraints.size in catalog.products[t.gold_pick].details["sizes"]


# ---- tools --------------------------------------------------------------


def test_search_finds_gold_and_respects_filters(fake, tasks):
    s = store_for(fake, tasks, "t001")
    hits = s.search_products("waterproof rain jacket", {"max_price": 90})
    ids = [h["id"] for h in hits]
    assert "p01" in ids and "p04" in ids
    assert "p02" not in ids  # $189
    assert all(h["price"] <= 90 for h in hits)


def test_search_limited_to_shelf(fake, tasks):
    s = store_for(fake, tasks, "t001", shelf=["p01", "p02", "p03", "p04", "p05", "p33"])
    assert s.search_products("merino sweater") == []
    assert s.get_product("p22") == {"error": "product p22 not found"}


def test_get_product_and_reviews(fake, tasks):
    s = store_for(fake, tasks, "t001")
    assert s.get_product("p01")["price"] == 79.99
    reviews = s.get_reviews("p01")
    assert [r["id"] for r in reviews] == ["p01.review.1", "p01.review.2", "p01.review.3"]


def test_history_hides_future_purchases(fake, tasks):
    s = store_for(fake, tasks, "t003")
    titles = [h["title"] for h in s.get_history("u02")]
    assert "Cozyknit Merino Crew Neck Sweater, Charcoal" in titles
    assert "Urban Wool Scarf, Grey" not in titles  # after as_of


def test_history_only_for_task_user(fake, tasks):
    s = store_for(fake, tasks, "t003")
    assert "error" in s.get_history("u01")
    assert "error" in s.check_owned("u01", "jacket")


def test_check_owned(fake, tasks):
    s = store_for(fake, tasks, "t003")
    assert [o["product_id"] for o in s.check_owned("u02", "sweater")] == ["p22"]
    assert [o["product_id"] for o in s.check_owned("u02", "sweaters")] == ["p22"]
    assert s.check_owned("u02", "running shoes") == []
    s1 = store_for(fake, tasks, "t001")
    assert [o["product_id"] for o in s1.check_owned("u01", "windbreaker")] == ["p03"]
    assert [o["product_id"] for o in s1.check_owned("u01", "running shoe")] == ["x101"]


def test_search_folds_plurals(fake, tasks):
    s = store_for(fake, tasks, "t002")
    assert {"p07", "p08"} <= {h["id"] for h in s.search_products("oxford shoe")}


def test_dispatch_round_trip_and_errors(fake, tasks):
    s = store_for(fake, tasks, "t001")
    assert json.loads(dispatch(s, "get_product", '{"product_id": "p01"}'))["id"] == "p01"
    assert "error" in json.loads(dispatch(s, "no_such_tool", "{}"))
    assert "error" in json.loads(dispatch(s, "get_product", "not json"))
    assert "error" in json.loads(dispatch(s, "get_product", '{"wrong": 1}'))
    assert {spec["function"]["name"] for spec in TOOL_SPECS} == {
        "search_products", "get_product", "get_reviews", "get_history", "check_owned",
    }


# ---- attacks ------------------------------------------------------------


def test_dose_zero_matches_clean_store(fake, tasks):
    clean = store_for(fake, tasks, "t001")
    dose0 = store_for(fake, tasks, "t001", attack=AttackSpec(type="fake_review", target="p33", dose=0))
    for pid in clean.shelf:
        assert clean.get_product(pid) == dose0.get_product(pid)
        assert clean.get_reviews(pid) == dose0.get_reviews(pid)
    assert dose0.injected_ids == set()


def test_fake_reviews_only_change_target(fake, tasks):
    clean = store_for(fake, tasks, "t001")
    attacked = store_for(fake, tasks, "t010")  # 4 fake reviews on p33
    for pid in clean.shelf:
        if pid != "p33":
            assert clean.get_reviews(pid) == attacked.get_reviews(pid)
            assert clean.get_product(pid) == attacked.get_product(pid)
    clean_reviews = clean.get_reviews("p33")
    attacked_reviews = attacked.get_reviews("p33")
    assert len(attacked_reviews) == len(clean_reviews) + 4
    assert attacked_reviews[4:] == clean_reviews  # genuine reviews untouched, fakes on top
    assert attacked.injected_ids == {r["id"] for r in attacked_reviews[:4]}
    assert all(r["rating"] == 5 for r in attacked_reviews[:4])


@pytest.mark.parametrize("dose", [1, 2, 4, 8])
def test_fake_review_dose_scales(fake, tasks, dose):
    s = store_for(fake, tasks, "t001", attack=AttackSpec(type="fake_review", target="p33", dose=dose))
    assert len(s.injected_ids) == dose


@pytest.mark.parametrize("kind", ["hidden_instruction", "urgency", "fake_override", "planted_preference"])
def test_text_attacks_edit_only_description(fake, tasks, kind):
    clean = store_for(fake, tasks, "t001")
    s = store_for(fake, tasks, "t001", attack=AttackSpec(type=kind, target="p33", dose=2))
    before, after = clean.get_product("p33"), s.get_product("p33")
    assert after["description"].startswith(before["description"])
    assert len(after["description"].splitlines()) == len(before["description"].splitlines()) + 2
    assert {k: v for k, v in after.items() if k != "description"} == {
        k: v for k, v in before.items() if k != "description"
    }
    assert s.injected_ids == {"p33.spec.description"}


def test_catalog_not_mutated_by_attack(fake, tasks):
    catalog = fake[0]
    before = catalog.products["p33"].description
    s = store_for(fake, tasks, "t001", attack=AttackSpec(type="hidden_instruction", target="p33", dose=4))
    s.get_product("p33")
    assert catalog.products["p33"].description == before
