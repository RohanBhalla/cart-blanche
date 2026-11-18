"""The compare script, on score files written by the real harness over the fake store."""

import json

from cartblanche.config import RunConfig
from cartblanche.env.store import load_fake_store
from cartblanche.eval.compare import compare, config_order, find_runs
from cartblanche.eval.harness import run
from cartblanche.schemas import AgentResult


def write_run(out, config_id, tasks, action="SKIP"):
    import cartblanche.eval.harness as harness

    def agent(task, store, config, llm):
        pick = task.gold_pick if action == "GOLD" and task.gold_pick else None
        return AgentResult(task_id=task.task_id, config_id=config_id, action="BUY" if pick else "SKIP", pick=pick)

    catalog, _ = load_fake_store()
    original = harness.get_agent
    harness.get_agent = lambda config: agent
    try:
        run(RunConfig(config_id=config_id), catalog, tasks, out, llm_factory=lambda c, d: None)
    finally:
        harness.get_agent = original


def test_config_order():
    ids = ["E4b", "E10", "E3c", "E0", "E3", "E3b", "E1"]
    assert sorted(ids, key=config_order) == ["E0", "E1", "E3", "E3b", "E3c", "E4b", "E10"]


def test_table_and_breakdown(tmp_path):
    _, tasks = load_fake_store()
    write_run(tmp_path / "fake" / "test", "E1", tasks, action="SKIP")
    write_run(tmp_path / "fake" / "test", "E0", tasks, action="GOLD")
    out = compare([tmp_path], split="test", by="subtype")
    lines = out.splitlines()
    rows = [line for line in lines if line.startswith("| E")]
    assert [r.split(" | ")[0] for r in rows] == ["| E0", "| E1"]  # sorted, not in write order
    assert "| E0 | 11 | 100.0% | 90.9% |" in out  # all right; one exact miss: BUY where the gold is ALTERNATIVE
    assert "| E1 | 11 | 9.1% |" in out and "| restock | 1 | 100% | 0% |" in out  # E1 skips everything
    assert "| duplicate | 1 | 100% | 100% |" in out  # skipping is right there, for both
    assert "left out" not in out


def test_only_common_tasks_by_default(tmp_path):
    _, tasks = load_fake_store()
    write_run(tmp_path / "s" / "test", "E0", tasks)
    write_run(tmp_path / "s" / "test", "E1", tasks[:4])  # still running
    out = compare([tmp_path], split="test")
    assert "| E0 | 4 |" in out and "| E1 | 4 |" in out and "left out: E0 7" in out
    assert "| E0 | 11 |" in compare([tmp_path], split="test", all_tasks=True)


def test_split_filter_and_no_runs(tmp_path):
    _, tasks = load_fake_store()
    write_run(tmp_path / "s" / "dev", "E0", tasks)
    assert find_runs([tmp_path], "test") == []
    assert "No runs found" in compare([tmp_path], split="test")
    assert "| E0 | 11 |" in compare([tmp_path], split="dev")
    assert json.loads((tmp_path / "s" / "dev" / "E0" / "scorecard.json").read_text())["n_tasks"] == 11
