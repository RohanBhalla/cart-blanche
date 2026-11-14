"""Run one configuration over a task set and write results, scores and the scorecard.

Output, under runs/<config_id>/:
  results.jsonl   one AgentResult per task (what the agent did, with its trace)
  scores.jsonl    one TaskScore per task
  scorecard.json  the aggregated row for this configuration
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from cartblanche.agent.rules import run_rules_agent
from cartblanche.config import AgentKind, RunConfig
from cartblanche.env.store import Catalog, TaskStore
from cartblanche.eval.metrics import TaskScore, score, scorecard
from cartblanche.schemas import Action, AgentResult, Task

Agent = Callable[[Task, TaskStore, RunConfig], AgentResult]


def get_agent(config: RunConfig) -> Agent:
    if config.agent == AgentKind.RULES:
        return run_rules_agent
    raise NotImplementedError("the LLM agent (E1+) is built in the next step")


def run_task(agent: Agent, task: Task, store: TaskStore, config: RunConfig) -> AgentResult:
    """Run one task. A crash becomes an invalid result instead of stopping the run."""
    start = time.perf_counter()
    try:
        result = agent(task, store, config)
    except Exception as e:
        result = AgentResult(task_id=task.task_id, config_id=config.config_id, action=Action.SKIP, error=repr(e))
    if not result.latency_s:
        result.latency_s = time.perf_counter() - start
    return result


def run(config: RunConfig, catalog: Catalog, tasks: list[Task], out_dir: Path = Path("runs")) -> dict:
    agent = get_agent(config)
    results: list[AgentResult] = []
    scores: list[TaskScore] = []
    for task in tasks:
        store = TaskStore(catalog, task)
        result = run_task(agent, task, store, config)
        results.append(result)
        scores.append(score(task, result, catalog, store.injected_ids))

    card = scorecard(scores)
    run_dir = out_dir / config.config_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in results))
    (run_dir / "scores.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in scores))
    (run_dir / "scorecard.json").write_text(json.dumps(card, indent=2))
    return card
