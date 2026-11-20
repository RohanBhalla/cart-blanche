"""Personal search (E3 and later): ranks the shelf for the shopper, through the store tools only."""

import json

import pytest
from test_agent import CONFIGS, config, agent_run, decide, fake, say, tool  # noqa: F401  (fake is a fixture)

from cartblanche.agent.search import LIMIT, personal_search
from cartblanche.config import RunConfig, load_config
from cartblanche.env.store import TaskStore

E3 = config("e3_rag.yaml")
INTENT = say('{"max_price": null, "min_rating": null, "brand": {"name": null, "strength": "none"}, "required_features": []}')


def search(fake, task_id, query, filters=None):
    catalog, tasks = fake
    return personal_search(tasks[task_id], TaskStore(catalog, tasks[task_id]), query, filters)


def test_a_brand_the_shopper_loved_comes_before_one_they_rated_badly(fake):
    # u02 rated Silkstrand shampoo 5 and Volumax shampoo 1; both conditioners are rated 4.6 in the store
    ids = [h["id"] for h in search(fake, "t002", "conditioner")]
    assert ids[0] == "p07" and ids.index("p07") < ids.index("p08")


def test_filters_still_apply_and_results_are_capped(fake):
    hits = search(fake, "t002", "conditioner", {"max_price": 10})
    assert hits and all(h["price"] <= 10 for h in hits)
    assert len(search(fake, "t002", "anything")) == LIMIT


def test_products_with_no_shared_words_can_still_be_found(fake):
    ids = [h["id"] for h in search(fake, "t002", "something to soften my curls after washing")]
    assert "p07" in ids[:5]


def test_personal_search_needs_history():
    with pytest.raises(ValueError):
        RunConfig(config_id="bad", personal_search=True)


def test_e3_agent_gets_personal_results_and_e1_gets_the_store_search(fake, monkeypatch):
    import cartblanche.agent.llm_agent as llm_agent

    monkeypatch.setattr(llm_agent, "relevant_history", lambda request, items, k=8: items)
    result, script = agent_run(fake, "t002", E3, INTENT, tool("search_products", query="conditioner"), decide("BUY", "p07"))
    hits = json.loads(script.requests[-1]["messages"][-1]["content"])
    assert hits[0]["id"] == "p07" and len(hits) == LIMIT

    e1 = config("e1_generic.yaml")
    result, script = agent_run(fake, "t002", e1, tool("search_products", query="conditioner"), decide("BUY", "p07"))
    assert len(json.loads(script.requests[-1]["messages"][-1]["content"])) <= 10  # the store's own search


def test_bad_search_arguments_come_back_as_an_error(fake, monkeypatch):
    import cartblanche.agent.llm_agent as llm_agent

    monkeypatch.setattr(llm_agent, "relevant_history", lambda request, items, k=8: items)
    _, script = agent_run(fake, "t002", E3, INTENT, tool("search_products", query="x", filters={"brand": {"n": 1}}),
                          decide("BUY", "p07"))
    assert "bad arguments" in json.loads(script.requests[-1]["messages"][-1]["content"])["error"]
