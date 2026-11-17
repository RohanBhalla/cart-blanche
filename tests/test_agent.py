"""The E1/E2 LLM agent loop and intent parser, driven by a scripted fake model (no API calls)."""

import json
from pathlib import Path

import pytest

from cartblanche.agent.intent import parse_intent, parse_json
from cartblanche.agent.llm_agent import run_llm_agent, tools_for
from cartblanche.config import RunConfig, load_config
from cartblanche.env.store import TaskStore, load_fake_store
from cartblanche.eval.harness import run
from cartblanche.llm import LLM
from cartblanche.schemas import BrandStrength

CONFIGS = Path(__file__).parent.parent / "configs"
E1 = load_config(CONFIGS / "e1_generic.yaml")
E2 = load_config(CONFIGS / "e2_intent.yaml")
E3 = load_config(CONFIGS / "e3_rag.yaml")
E3B = load_config(CONFIGS / "e3b_profile.yaml")
E3C = load_config(CONFIGS / "e3c_full.yaml")


def tool(name, **args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def say(content):
    return {"role": "assistant", "content": content}


def decide(action, pick=None, evidence=()):
    return say(json.dumps({"action": action, "pick": pick, "reasons": "fits", "evidence_ids": list(evidence)}))


class Scripted:
    """Plays back assistant messages in order and records every request."""

    def __init__(self, *messages):
        self.messages = list(messages)
        self.requests = []

    def __call__(self, **request):
        self.requests.append(json.loads(json.dumps(request)))  # snapshot: the agent keeps appending
        return {"choices": [{"message": self.messages.pop(0)}], "usage": {"prompt_tokens": 50, "completion_tokens": 10}}


def llm_with(script):
    return LLM(model="fake/model", cache_dir=None, completion_fn=script, cost_fn=lambda raw: 0.001)


@pytest.fixture(scope="module")
def fake():
    catalog, tasks = load_fake_store()
    return catalog, {t.task_id: t for t in tasks}


def agent_run(fake, task_id, config, *messages):
    catalog, tasks = fake
    script = Scripted(*messages)
    result = run_llm_agent(tasks[task_id], TaskStore(catalog, tasks[task_id]), config, llm_with(script))
    return result, script


def test_tool_loop_then_decision(fake):
    result, script = agent_run(
        fake, "t001", E1,
        tool("search_products", query="fragrance-free moisturizer", filters={"max_price": 25}),
        tool("get_reviews", product_id="p01"),
        decide("BUY", "p01", ["p01.review.1", "p01.spec.price"]),
    )
    assert result.error is None and result.action == "BUY" and result.pick == "p01"
    assert result.evidence_ids == ["p01.review.1", "p01.spec.price"]
    assert [s.get("tool") for s in result.trace[:2]] == ["search_products", "get_reviews"]
    assert result.tokens == 3 * 60 and result.cost_usd == pytest.approx(0.003)
    # the tool output went back to the model as a tool message
    last = script.requests[-1]["messages"]
    assert last[-1]["role"] == "tool" and json.loads(last[-1]["content"])[0]["id"] == "p01.review.1"


def test_e1_has_no_history_tools(fake):
    names = {t["function"]["name"] for t in tools_for()}
    assert names == {"search_products", "get_product", "get_reviews"}
    _, script = agent_run(fake, "t001", E1, decide("BUY", "p01"))
    assert "u01" not in json.dumps(script.requests[0]["messages"])  # no user ID either


def test_json_in_fences_is_accepted(fake):
    reply = '```json\n{"action": "skip", "pick": "p10", "reasons": "already owns one", "evidence_ids": []}\n```'
    result, _ = agent_run(fake, "t003", E1, say(reply))
    assert result.error is None and result.action == "SKIP" and result.pick is None


@pytest.mark.parametrize("reply", [
    "I would buy the Dewlab one.",  # no JSON
    '{"action": "MAYBE", "pick": "p01"}',  # unknown action
    '{"action": "BUY", "pick": "p99"}',  # not in the store
    '{"action": "BUY", "pick": null}',  # buying nothing
])
def test_bad_decisions_are_invalid(fake, reply):
    result, _ = agent_run(fake, "t001", E1, say(reply))
    assert result.error and result.error.startswith("bad decision")


def test_max_steps(fake):
    config = E1.model_copy(update={"max_steps": 2})
    result, _ = agent_run(fake, "t001", config, tool("get_product", product_id="p01"), tool("get_product", product_id="p02"))
    assert result.error == "no decision within 2 steps"


def test_override_sent_after_skip(fake):
    result, script = agent_run(fake, "t008", E1, decide("SKIP"), decide("BUY", "p10"))
    assert result.action == "BUY" and result.pick == "p10"
    assert script.requests[-1]["messages"][-1] == {"role": "user", "content": fake[1]["t008"].override}
    assert {"user": fake[1]["t008"].override} in result.trace


def test_no_override_means_skip_stands(fake):
    result, script = agent_run(fake, "t003", E1, decide("SKIP"))
    assert result.action == "SKIP" and len(script.requests) == 1


def test_e2_passes_structured_requirements(fake):
    intent = say('{"max_price": 25, "min_rating": null, "brand": {"name": null, "strength": "none"}, '
                 '"required_features": ["fragrance-free"]}')
    result, script = agent_run(fake, "t001", E2, intent, decide("BUY", "p01"))
    assert result.trace[0] == {"intent": {"max_price": 25.0, "brand": {"name": None, "strength": "none"},
                                          "min_rating": None, "required_features": ["fragrance-free"]}}
    first_user = script.requests[1]["messages"][1]["content"]
    assert "Structured requirements" in first_user and "fragrance-free" in first_user


def test_intent_parser_brand_and_nulls():
    script = Scripted(say('Sure: {"max_price": null, "brand": {"name": "Dewlab", "strength": "hard"}, '
                          '"required_features": null}'))
    c = parse_intent("a Dewlab vitamin C serum", llm_with(script))
    assert c.brand.name == "Dewlab" and c.brand.strength == BrandStrength.HARD
    assert c.max_price is None and c.required_features == []


def test_intent_parser_failure_is_invalid(fake):
    result, _ = agent_run(fake, "t001", E2, say("no idea"))
    assert result.error.startswith("intent parser")


def test_parse_json_rejects_prose():
    with pytest.raises(ValueError):
        parse_json("nothing here")


def test_e1_through_harness_resumes(tmp_path, fake):
    catalog, tasks = load_fake_store()
    tasks = tasks[:3]  # t001 BUY, t002 BUY, t003 SKIP
    script = Scripted(decide("BUY", "p01"), say("garbled"), decide("SKIP"))
    card = run(E1, catalog, tasks, tmp_path, llm_factory=lambda config, run_dir: llm_with(script))
    assert card["invalid_rate"] == pytest.approx(1 / 3)  # t002 failed
    # second run retries only the failed task
    script2 = Scripted(decide("BUY", "p07"))
    card = run(E1, catalog, tasks, tmp_path, llm_factory=lambda config, run_dir: llm_with(script2))
    assert len(script2.requests) == 1 and card["invalid_rate"] == 0.0 and card["accuracy"] == 1.0
    lines = (tmp_path / "E1" / "results.jsonl").read_text().splitlines()
    assert sorted(json.loads(l)["task_id"] for l in lines) == ["t001", "t002", "t003"]


def test_fresh_reruns_everything(tmp_path):
    catalog, tasks = load_fake_store()
    script = Scripted(decide("BUY", "p01"))
    run(E1, catalog, tasks[:1], tmp_path, llm_factory=lambda c, d: llm_with(script))
    script2 = Scripted(decide("BUY", "p04"))
    run(E1, catalog, tasks[:1], tmp_path, llm_factory=lambda c, d: llm_with(script2), fresh=True)
    assert len(script2.requests) == 1


def test_e2_config_is_e1_plus_parser():
    assert RunConfig(**{**E1.model_dump(), "config_id": "E2", "intent_parser": True,
                        "description": E2.description}) == E2

def test_e3c_shows_the_whole_history_and_no_history_tools(fake):
    intent = say('{"max_price": null, "min_rating": null, "brand": {"name": null, "strength": "none"}, "required_features": []}')
    result, script = agent_run(fake, "t003", E3C, intent, decide("SKIP"))
    message = script.requests[1]["messages"][1]["content"]
    assert "all of them" in message and "Breeze Ionic Hair Dryer" in message and "Edgeline 5-Blade" in message
    assert {t["function"]["name"] for t in script.requests[1]["tools"]} == {"search_products", "get_product", "get_reviews"}
    assert {"history": {"mode": "full", "shown": 2, "total": 2}} in result.trace


def test_e3_shows_only_the_relevant_history(fake, monkeypatch):
    import cartblanche.agent.llm_agent as llm_agent

    asked = {}

    def closest_only(request, items, k=8):
        asked["request"] = request
        return [i for i in items if "Dryer" in i.title]

    monkeypatch.setattr(llm_agent, "relevant_history", closest_only)
    intent = say('{"max_price": null, "min_rating": null, "brand": {"name": null, "strength": "none"}, "required_features": []}')
    result, script = agent_run(fake, "t003", E3, intent, decide("SKIP"))
    message = script.requests[1]["messages"][1]["content"]
    assert asked["request"] == fake[1]["t003"].request
    assert "the ones closest to this request" in message and "Breeze Ionic Hair Dryer" in message
    assert "Edgeline" not in message
    assert {"history": {"mode": "rag", "shown": 1, "total": 2}} in result.trace


def test_history_stays_hidden_after_as_of(fake):
    # the store cuts history at the task's as_of time, so the held-out purchase can never reach the prompt
    catalog, tasks = fake
    task = tasks["t003"].model_copy(update={"as_of": "2000-01-01T00:00:00"})
    script = Scripted(say('{"max_price": null, "min_rating": null, "brand": {"name": null, "strength": "none"}, "required_features": []}'), decide("BUY", "p10"))
    run_llm_agent(task, TaskStore(catalog, task), E3C, llm_with(script))
    assert "no past purchases on record" in script.requests[1]["messages"][1]["content"]

def test_e3b_shows_a_written_profile_instead_of_the_purchases(fake):
    intent = say('{"max_price": null, "min_rating": null, "brand": {"name": null, "strength": "none"}, "required_features": []}')
    profile = say("Thick hair. Loves ionic dryers and trusts Breeze.")
    result, script = agent_run(fake, "t003", E3B, intent, profile, decide("SKIP"))
    asked_for_profile = script.requests[1]["messages"]
    assert "Breeze Ionic Hair Dryer" in asked_for_profile[1]["content"]  # the profile writer sees the history
    message = script.requests[2]["messages"][1]["content"]
    assert "Thick hair. Loves ionic dryers and trusts Breeze." in message
    assert "Edgeline" not in message and "rated" not in message  # the agent never sees the raw purchases
    assert {"history": {"mode": "profile", "total": 2}} in result.trace


def test_e3b_empty_profile_is_invalid(fake):
    intent = say('{"max_price": null, "min_rating": null, "brand": {"name": null, "strength": "none"}, "required_features": []}')
    result, _ = agent_run(fake, "t003", E3B, intent, say("  "))
    assert result.error and "profile" in result.error