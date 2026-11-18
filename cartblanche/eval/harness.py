"""Run one configuration over a task set and write results, scores and the scorecard.

Output, under runs/<config_id>/:
  results.jsonl   one AgentResult per task (what the agent did, with its trace),
                  appended as each task finishes
  calls.jsonl     one line per model call (tokens, cost, latency, cached)
  scores.jsonl    one TaskScore per task
  scorecard.json  the aggregated row for this configuration

Runs resume: tasks that already have a valid result in results.jsonl are
skipped, and tasks that errored are retried. If a daily quota runs out, the
run stops with QuotaExhausted; run it again later to continue. Scores and the
scorecard always cover the tasks finished so far, even after a stop.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from cartblanche.agent.llm_agent import run_llm_agent
from cartblanche.agent.rules import run_rules_agent
from cartblanche.config import AgentKind, Labeling, Personalization, RelationshipMode, RunConfig
from cartblanche.env.store import Catalog, TaskStore
from cartblanche.eval.metrics import score, scorecard
from cartblanche.llm import LLM, QuotaExhausted
from cartblanche.schemas import Action, AgentResult, Task

Agent = Callable[[Task, TaskStore, RunConfig, LLM], AgentResult]
LLMFactory = Callable[[RunConfig, Path], LLM]


def not_built_yet(config: RunConfig) -> list[str]:
    """Components switched on in the config that later steps will build."""
    parts = {
        "personalization": config.personalization != Personalization.NONE,
        "relationship": config.relationship != RelationshipMode.NONE,
        "extractor": config.extractor,
        "labeling": config.labeling != Labeling.NONE,
        "verifier": config.verifier,
        "clarify": config.clarify,
    }
    return [name for name, on in parts.items() if on]


def get_agent(config: RunConfig) -> Agent:
    if config.agent == AgentKind.RULES:
        return run_rules_agent
    missing = not_built_yet(config)
    if missing:
        raise NotImplementedError(f"{config.config_id} needs components not built yet: {', '.join(missing)}")
    return run_llm_agent


def default_llm(config: RunConfig, run_dir: Path, call_gap_s: float = 0.0) -> LLM:
    return LLM(model=config.agent_model, temperature=config.temperature,
               log_path=run_dir / "calls.jsonl", min_gap_s=call_gap_s)


def run_task(agent: Agent, task: Task, store: TaskStore, config: RunConfig, llm: LLM) -> AgentResult:
    """Run one task. A crash becomes an invalid result instead of stopping the run."""
    start = time.perf_counter()
    try:
        result = agent(task, store, config, llm)
    except QuotaExhausted:
        raise
    except Exception as e:
        result = AgentResult(task_id=task.task_id, config_id=config.config_id, action=Action.SKIP, error=repr(e))
    if not result.latency_s:
        result.latency_s = time.perf_counter() - start
    return result


def _load_valid(path: Path) -> dict[str, AgentResult]:
    if not path.exists():
        return {}
    results = [AgentResult.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    return {r.task_id: r for r in results if r.error is None}


def run(
    config: RunConfig,
    catalog: Catalog,
    tasks: list[Task],
    out_dir: Path = Path("runs"),
    llm_factory: LLMFactory = default_llm,
    fresh: bool = False,
) -> dict:
    agent = get_agent(config)
    run_dir = out_dir / config.config_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    done = {} if fresh else _load_valid(results_path)
    done = {tid: r for tid, r in done.items() if tid in {t.task_id for t in tasks}}
    results_path.write_text("".join(r.model_dump_json() + "\n" for r in done.values()))  # drop errored lines

    llm = llm_factory(config, run_dir)
    try:
        for task in tasks:
            if task.task_id in done:
                continue
            result = run_task(agent, task, TaskStore(catalog, task), config, llm)
            done[task.task_id] = result
            with results_path.open("a") as f:
                f.write(result.model_dump_json() + "\n")
    finally:  # also when a quota stops the run: score what finished, so it can be compared
        finished = [t for t in tasks if t.task_id in done]
        scores = [score(t, done[t.task_id], catalog, TaskStore(catalog, t).injected_ids) for t in finished]
        card = scorecard(scores)
        (run_dir / "scores.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in scores))
        (run_dir / "scorecard.json").write_text(json.dumps(card, indent=2))
    return card
