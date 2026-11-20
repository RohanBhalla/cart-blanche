"""The clarifying question (a fixed first step when `clarify` is on) and the simulated shopper who answers it
(environment side)."""

import json

from test_agent import Scripted, agent_run, config, decide, fake, llm_with, say  # noqa: F401

from cartblanche.config import RelationshipMode
from cartblanche.env.simulated_user import NOTHING_TO_ADD, _facts, answer

# E4q without the relationship check, so each script below covers only what it tests
E4Q = config("e4q_clarify.yaml", relationship=RelationshipMode.NONE)
INTENT = say('{"max_price": null, "min_rating": null, "brand": {"name": null, "strength": "none"}, "required_features": []}')


def no_embeddings(monkeypatch):
    import cartblanche.agent.llm_agent as llm_agent

    monkeypatch.setattr(llm_agent, "relevant_history", lambda request, items, k=8: items)


def test_facts_come_from_the_profile_only():
    assert _facts({"looking_for": "body lotion", "for": "myself", "budget": 20.0}) == \
        "looking for: body lotion; for: myself; budget: $20"


def test_no_profile_means_nothing_to_add_and_no_model_call():
    script = Scripted()  # would fail if called
    assert answer("a perfume", None, "Which scent?", llm_with(script)) == NOTHING_TO_ADD and not script.requests


def test_a_vague_request_gets_one_question_answered_from_the_hidden_profile(fake, monkeypatch):
    no_embeddings(monkeypatch)
    result, script = agent_run(
        fake, "t009", E4Q,
        say('{"question": "Who is it for, and what scents do they like?"}'),  # the clarify step
        say("It's a gift for my mom; she likes light floral scents."),  # the simulated shopper
        INTENT, decide("BUY", "p26"),
    )
    assert result.action == "BUY" and result.question == "Who is it for, and what scents do they like?"
    shopper = script.requests[1]["messages"]
    assert "light floral scents" in shopper[0]["content"] and shopper[1]["content"].startswith("Who is it for")
    first = script.requests[3]["messages"][1]["content"]
    assert "You asked the shopper: Who is it for" in first and "The shopper answered: It's a gift" in first


def test_a_specific_request_gets_no_question(fake, monkeypatch):
    no_embeddings(monkeypatch)
    result, script = agent_run(fake, "t009", E4Q, say('{"question": null}'), INTENT, decide("BUY", "p26"))
    assert result.question is None and len(script.requests) == 3  # no shopper call
    assert "You asked the shopper" not in script.requests[2]["messages"][1]["content"]


def test_an_unusable_clarify_reply_means_no_question(fake, monkeypatch):
    no_embeddings(monkeypatch)
    result, _ = agent_run(fake, "t009", E4Q, say("hmm"), INTENT, decide("BUY", "p26"))
    assert result.error is None and result.question is None
