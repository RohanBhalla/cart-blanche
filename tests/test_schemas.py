import pytest
from pydantic import ValidationError

from cartblanche.schemas import Action, AgentResult, Task

BASE = dict(
    task_id="t", user_id="u", request="r", type="buy", as_of="2026-10-01T00:00:00",
    shelf=["p1", "p2", "p3"], candidates=["p1", "p2"],
)


def test_gold_pick_added_to_acceptable():
    t = Task(**BASE, gold_action="BUY", gold_pick="p1", acceptable=["p2"])
    assert t.acceptable == ["p1", "p2"]


@pytest.mark.parametrize(
    "extra",
    [
        dict(gold_action="BUY"),  # missing pick
        dict(gold_action="SKIP", gold_pick="p1"),  # pick on a skip
        dict(gold_action="BUY", gold_pick="p3"),  # pick not a candidate
        dict(gold_action="BUY", gold_pick="p1", acceptable=["p3"]),  # acceptable not a candidate
        dict(gold_action="BUY", gold_pick="p1", attack={"type": "fake_review", "target": "p1", "dose": 1}),
        dict(gold_action="BUY", gold_pick="p1", attack={"type": "fake_review", "target": "p9", "dose": 1}),
    ],
)
def test_inconsistent_tasks_rejected(extra):
    with pytest.raises(ValidationError):
        Task(**BASE, **extra)


def test_candidates_must_be_on_shelf():
    with pytest.raises(ValidationError):
        Task(**{**BASE, "candidates": ["p1", "p9"]}, gold_action="SKIP")


def test_agent_result_roundtrip():
    r = AgentResult(task_id="t", config_id="E1", action="BUY", pick="p1", evidence_ids=["p1.spec.price"])
    assert AgentResult.model_validate_json(r.model_dump_json()) == r
    assert r.action is Action.BUY
