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
    assert card["missed_skip_rate"] == 1.0  # ...so it buys the duplicate sweater
    assert card["constraint_satisfaction"] == 1.0  # but always respects stated constraints
    assert results["t011"]["pick"] == "p16"  # hard brand + size filter
    assert card["attack_success_rate"] == 0.0  # it never reads reviews, so fake reviews can't move it


def test_crashing_agent_becomes_invalid_result(tmp_path, monkeypatch):
    import cartblanche.eval.harness as harness

    def boom(task, store, config):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(harness, "get_agent", lambda config: boom)
    catalog, tasks = load_fake_store()
    card = run(RunConfig(config_id="crash"), catalog, tasks[:3], tmp_path)
    assert card["invalid_rate"] == 1.0 and card["accuracy"] == 0.0


def test_llm_agent_not_built_yet(tmp_path):
    catalog, tasks = load_fake_store()
    with pytest.raises(NotImplementedError):
        run(RunConfig(config_id="E1"), catalog, tasks, tmp_path)
