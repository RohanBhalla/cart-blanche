import json
from pathlib import Path

import pytest

from cartblanche.config import RunConfig, load_config
from cartblanche.env.store import load_fake_store
from cartblanche.eval.harness import run

E0 = load_config(Path(__file__).parent.parent / "configs" / "e0_rules.yaml")


def test_e0_run_writes_all_outputs(tmp_path):
    catalog, tasks = load_fake_store()
    card = run(E0, catalog, tasks, tmp_path)
    run_dir = tmp_path / "E0"
    assert len((run_dir / "results.jsonl").read_text().splitlines()) == len(tasks)
    assert len((run_dir / "scores.jsonl").read_text().splitlines()) == len(tasks)
    assert json.loads((run_dir / "scorecard.json").read_text()) == card
    assert card["invalid_rate"] == 0.0


def test_e0_behaves_like_a_rules_baseline(tmp_path):
    catalog, tasks = load_fake_store()
    card = run(E0, catalog, tasks, tmp_path)
    results = {json.loads(l)["task_id"]: json.loads(l) for l in (tmp_path / "E0" / "results.jsonl").read_text().splitlines()}
    assert all(r["action"] == "BUY" for r in results.values())  # never skips...
    assert card["missed_skip_rate"] == 1.0  # ...so it buys the duplicate hair dryer
    assert card["constraint_satisfaction"] == 1.0  # but always respects stated constraints
    assert all(catalog.products[r["pick"]].brand == "Dewlab" for r in [results["t011"]])  # hard brand filter
    assert results["t001"]["pick"] == "p23"  # keyword match + rating: a sunscreen for a moisturizer request
    assert card["attack_success_rate"] == 0.0  # it never reads reviews, so fake reviews can't move it


def test_crashing_agent_becomes_invalid_result(tmp_path, monkeypatch):
    import cartblanche.eval.harness as harness

    def boom(task, store, config, llm):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(harness, "get_agent", lambda config: boom)
    catalog, tasks = load_fake_store()
    card = run(RunConfig(config_id="crash"), catalog, tasks[:3], tmp_path)
    assert card["invalid_rate"] == 1.0 and card["accuracy"] == 0.0


def test_configs_with_unbuilt_components_fail_fast(tmp_path):
    catalog, tasks = load_fake_store()
    e7 = load_config(Path(__file__).parent.parent / "configs" / "e7_full.yaml")
    with pytest.raises(NotImplementedError, match="verifier"):
        run(e7, catalog, tasks, tmp_path)


def test_quota_stop_still_scores_finished_tasks(tmp_path, monkeypatch):
    import cartblanche.eval.harness as harness
    from cartblanche.llm import QuotaExhausted
    from cartblanche.schemas import AgentResult

    calls = []

    def agent(task, store, config, llm):
        calls.append(task.task_id)
        if len(calls) == 3:
            raise QuotaExhausted("per day")
        return AgentResult(task_id=task.task_id, config_id=config.config_id, action="SKIP")

    monkeypatch.setattr(harness, "get_agent", lambda config: agent)
    catalog, tasks = load_fake_store()
    with pytest.raises(QuotaExhausted):
        run(RunConfig(config_id="Q"), catalog, tasks, tmp_path, llm_factory=lambda c, d: None)
    card = json.loads((tmp_path / "Q" / "scorecard.json").read_text())
    assert card["n_tasks"] == 2 and len((tmp_path / "Q" / "scores.jsonl").read_text().splitlines()) == 2


def test_provider_failures_are_left_unfinished_and_retried(tmp_path, monkeypatch):
    import cartblanche.eval.harness as harness
    from cartblanche.schemas import AgentResult

    class RateLimitError(Exception):  # the name LiteLLM uses
        pass

    failing = {"t002"}

    def agent(task, store, config, llm):
        if task.task_id in failing:
            raise RateLimitError("429 after every retry")
        return AgentResult(task_id=task.task_id, config_id=config.config_id, action="SKIP")

    monkeypatch.setattr(harness, "get_agent", lambda config: agent)
    catalog, tasks = load_fake_store()
    card = run(RunConfig(config_id="P"), catalog, tasks[:3], tmp_path, llm_factory=lambda c, d: None)
    assert card["n_tasks"] == 2 and card["invalid_rate"] == 0.0  # not scored as the agent's fault
    assert "t002" not in (tmp_path / "P" / "results.jsonl").read_text()

    failing.clear()  # the provider is back: the next run finishes it
    card = run(RunConfig(config_id="P"), catalog, tasks[:3], tmp_path, llm_factory=lambda c, d: None)
    assert card["n_tasks"] == 3
