"""E4 (blunt rule) and E4b (relationship check), driven by a scripted fake model (no API calls)."""

import json

import pytest
from test_agent import CONFIGS, config, agent_run, decide, fake, say  # noqa: F401  (fake is a fixture)

from cartblanche.agent.relationship import leaf, owned_similar
from cartblanche.config import load_config
from cartblanche.env.store import TaskStore
from cartblanche.schemas import Product

E4 = config("e4_rule.yaml")
E4B = config("e4b_check.yaml")
INTENT = say('{"max_price": null, "min_rating": null, "brand": {"name": null, "strength": "none"}, "required_features": []}')


def label(relationship):
    return say(json.dumps({"relationship": relationship, "reason": "because"}))


@pytest.fixture(autouse=True)
def no_embeddings(monkeypatch):
    import cartblanche.agent.llm_agent as llm_agent

    monkeypatch.setattr(llm_agent, "relevant_history", lambda request, items, k=8: items)


def checks(result):
    return [s["relationship_check"] for s in result.trace if "relationship_check" in s]


def test_leaf_needs_a_specific_path():
    assert leaf(["Beauty", "Hair Care", "Styling Tools", "Hair Dryers"]) == "Hair Dryers"
    assert leaf(["Beauty", "Hair Care"]) == ""


def test_owned_similar_finds_the_same_kind_only(fake):
    catalog, tasks = fake
    store = TaskStore(catalog, tasks["t003"])
    other_dryer = Product(**store.get_product("p11"))
    flat_iron = Product(**store.get_product("p12"))
    assert [h.product_id for h in owned_similar(tasks["t003"], store, other_dryer)] == ["p10"]
    assert owned_similar(tasks["t003"], store, flat_iron) == []


def test_e4_skips_when_the_shopper_owns_one(fake):
    result, script = agent_run(fake, "t003", E4, INTENT, decide("BUY", "p11"))
    assert result.action == "SKIP" and result.pick is None and "Already owns" in result.reasons
    assert checks(result)[0]["owned"] == ["p10"] and len(script.requests) == 2  # the rule makes no model call


def test_e4_keeps_the_purchase_when_nothing_similar_is_owned(fake):
    result, script = agent_run(fake, "t002", E4, INTENT, decide("BUY", "p07"))
    assert result.action == "BUY" and result.pick == "p07" and checks(result) == []


def test_e4_override_after_the_rule_skips_then_buys(fake):
    catalog, tasks = fake
    task = tasks["t003"].model_copy(update={"override": "Buy the same one again anyway."})
    from test_agent import Scripted, llm_with

    from cartblanche.agent.llm_agent import run_llm_agent

    script = Scripted(INTENT, decide("BUY", "p11"), decide("BUY", "p10"))
    result = run_llm_agent(task, TaskStore(catalog, task), E4, llm_with(script))
    assert result.action == "BUY" and result.pick == "p10"
    last = script.requests[-1]["messages"]
    assert json.loads(last[-2]["content"])["action"] == "SKIP"  # the model sees the skip it is overriding
    assert last[-1] == {"role": "user", "content": "Buy the same one again anyway."}
    assert len(checks(result)) == 1  # no check after the shopper insists


def test_e4b_duplicate_skips(fake):
    result, script = agent_run(fake, "t003", E4B, INTENT, decide("BUY", "p11"), label("duplicate"))
    assert result.action == "SKIP" and checks(result)[0]["relationship"] == "duplicate"
    check_request = script.requests[2]["messages"][1]["content"]
    assert "Breeze Ionic Hair Dryer" in check_request and "Today: 2026-10-01" in check_request


@pytest.mark.parametrize("relationship", ["second_unit", "upgrade", "complement", "unrelated"])
def test_e4b_other_relationships_keep_the_purchase(fake, relationship):
    result, _ = agent_run(fake, "t003", E4B, INTENT, decide("BUY", "p11"), label(relationship))
    assert result.action == "BUY" and result.pick == "p11"


def test_e4b_replacement_of_a_bad_one_chooses_again_then_is_an_alternative(fake):
    result, script = agent_run(
        fake, "t004", E4B, INTENT,
        decide("BUY", "p13"), label("replacement"),  # repeats the foundation rated 2 of 5
        decide("BUY", "p14"), label("replacement"),
    )
    assert result.action == "ALTERNATIVE" and result.pick == "p14"
    note = script.requests[3]["messages"][-1]
    assert note["role"] == "user" and "rated Shadeworks" in note["content"] and "2 of 5" in note["content"]
    assert [c["chose_again"] for c in checks(result)] == [True, False]


def test_e4b_unusable_check_reply_keeps_the_purchase(fake):
    result, _ = agent_run(fake, "t003", E4B, INTENT, decide("BUY", "p11"), say("no idea"))
    assert result.error is None and result.action == "BUY" and "error" in checks(result)[0]
