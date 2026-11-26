"""Dose sweeps with a scripted agent that falls for the attack once it sees enough fakes."""

import json

import pytest

import cartblanche.eval.harness as harness
from cartblanche.config import RunConfig
from cartblanche.env.attackgen import generate_attacked_tasks
from cartblanche.env.store import load_fake_store
from cartblanche.eval.sweep import break_point, doses_needed, robustness_table, sweep
from cartblanche.llm import LLM
from cartblanche.schemas import AgentResult
from test_attackgen import Writer


@pytest.fixture(scope="module")
def attacked():
    catalog, tasks = load_fake_store()
    llm = LLM(model="fake/model", cache_dir=None, completion_fn=Writer(), cost_fn=lambda raw: 0.0)
    out, _ = generate_attacked_tasks(tasks, catalog, llm)  # 8 copies of t001, target p33, gold p01
    return catalog, out


def gullible(threshold, calls):
    """Buys the target once the dose reaches `threshold`; otherwise buys the gold."""
    def agent(task, store, config, llm):
        calls.append(task.task_id)
        dose = task.attack.dose
        pick = task.attack.target if dose >= threshold else task.gold_pick
        cited = sorted(store.injected_ids)[:1] if pick == task.attack.target else []
        return AgentResult(task_id=task.task_id, config_id=config.config_id, action="BUY", pick=pick, evidence_ids=cited)
    return agent


@pytest.mark.parametrize("picked, expected", [
    ({0: False, 8: False}, []), ({0: True, 8: True}, []), ({0: False}, [8]), ({}, [0, 8]),
    ({0: False, 8: True}, [2]), ({0: False, 8: True, 2: True}, [1]), ({0: False, 8: True, 2: False}, [4]),
    ({0: False, 8: True, 2: False, 4: True}, []),
])
def test_doses_needed(attacked, picked, expected):
    assert doses_needed(attacked[1][0], picked) == expected


@pytest.mark.parametrize("picked, bp", [
    ({0: False, 8: False}, None), ({0: True, 8: True}, 0), ({0: False, 8: True, 2: True, 1: True}, 1),
    ({0: False, 8: True, 2: True, 1: False}, 2), ({0: False, 8: True, 2: False, 4: True}, 4),
    ({0: False, 8: True, 2: False, 4: False}, 8),
    ({0: False, 1: False, 2: True, 4: True, 8: True}, 2),  # every dose run
])
def test_break_point(picked, bp):
    assert break_point(picked) == bp


@pytest.mark.parametrize("threshold, bp, runs_per_task", [(1, 1, 4), (2, 2, 4), (3, 4, 4), (5, 8, 4), (99, None, 2), (0, 0, 2)])
def test_sweep_finds_the_break_point(tmp_path, monkeypatch, attacked, threshold, bp, runs_per_task):
    catalog, tasks = attacked
    calls = []
    monkeypatch.setattr(harness, "get_agent", lambda config: gullible(threshold, calls))
    summary = sweep(RunConfig(config_id="G"), catalog, tasks, tmp_path, llm_factory=lambda c, d: None)
    rows = [json.loads(line) for line in (tmp_path / "G" / "sweep.jsonl").read_text().splitlines()]
    assert {r["break_point"] for r in rows} == {bp}
    assert len(calls) == runs_per_task * len(tasks)  # binary search: at most 4 runs per task
    o = summary["overall"]
    assert o["finished"] == len(tasks)
    if bp == 0:
        assert o["bought_target_without_attack"] == 1.0 and o["never_broke"] is None
    elif bp is None:
        assert o["never_broke"] == 1.0 and o["attack_success_by_dose"]["8"] == 0.0 and o["pick_unchanged_dose_8"] == 1.0
    else:
        assert o["attack_success_by_dose"] == {str(d): float(d >= bp) for d in (1, 2, 4, 8)}
        assert o["accuracy_dose_0"] == 1.0 and o["accuracy_dose_8"] == 0.0 and o["fakes_cited_per_task_dose_8"] == 1
    assert set(summary["by_kind"]) == {"fake_review", "copy_paste", "hidden_instruction", "urgency",
                                       "fake_override", "planted_preference"}
    assert set(summary["fake_review_by_seed"]) == {"0", "1", "2"}


def test_sweep_resumes_without_rerunning(tmp_path, monkeypatch, attacked):
    catalog, tasks = attacked
    first, second = [], []
    monkeypatch.setattr(harness, "get_agent", lambda config: gullible(3, first))
    sweep(RunConfig(config_id="G"), catalog, tasks, tmp_path, llm_factory=lambda c, d: None)
    monkeypatch.setattr(harness, "get_agent", lambda config: gullible(3, second))
    sweep(RunConfig(config_id="G"), catalog, tasks, tmp_path, llm_factory=lambda c, d: None)
    assert len(first) == 4 * len(tasks) and second == []


def test_all_doses_and_monotonic_check(tmp_path, monkeypatch, attacked):
    catalog, tasks = attacked
    calls = []
    monkeypatch.setattr(harness, "get_agent", lambda config: gullible(3, calls))
    summary = sweep(RunConfig(config_id="G"), catalog, tasks[:1], tmp_path, llm_factory=lambda c, d: None,
                    all_doses=True)
    assert len(calls) == 5 and summary["overall"]["non_monotonic"] == 0
    assert summary["overall"]["median_break_point"] == 4


def test_invalid_answers_count_as_not_fooled(tmp_path, monkeypatch, attacked):
    catalog, tasks = attacked

    def broken(task, store, config, llm):
        return AgentResult(task_id=task.task_id, config_id=config.config_id, action="SKIP", error="bad decision")

    monkeypatch.setattr(harness, "get_agent", lambda config: broken)
    summary = sweep(RunConfig(config_id="G"), catalog, tasks[:2], tmp_path, llm_factory=lambda c, d: None)
    assert summary["overall"]["finished"] == 2 and summary["overall"]["never_broke"] == 1.0


def test_robustness_table(tmp_path, monkeypatch, attacked):
    catalog, tasks = attacked
    monkeypatch.setattr(harness, "get_agent", lambda config: gullible(3, []))
    summary = sweep(RunConfig(config_id="G"), catalog, tasks, tmp_path, llm_factory=lambda c, d: None)
    table = robustness_table([summary])
    assert "| G | all | 8 | 0% | 0% | 0% | 0% | 100% | 100% | 4 |" in table
    assert "| G | copy_paste | 1 |" in table


def test_compare_attacks_and_clean_tables_stay_apart(tmp_path, monkeypatch, attacked):
    from cartblanche.eval.compare import attacks, compare

    catalog, tasks = attacked
    monkeypatch.setattr(harness, "get_agent", lambda config: gullible(3, []))
    sweep(RunConfig(config_id="E1"), catalog, tasks, tmp_path / "beauty_attacked" / "dev", llm_factory=lambda c, d: None)
    assert "| E1 | all | 8 |" in attacks([tmp_path], split="dev")
    assert "use --attacks" in compare([tmp_path], split="dev")  # sweep runs never mix into the clean table
    assert "No dose sweeps" in attacks([tmp_path], split="test")
