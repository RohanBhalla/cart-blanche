"""Compare configurations side by side, as Markdown tables (ready to paste into docs or the report).

    python -m cartblanche.eval.compare                          # every test run under runs/
    python -m cartblanche.eval.compare runs/beauty/dev          # one task set and split
    python -m cartblanche.eval.compare --split dev --by subtype # accuracy per subtype too

Scores are pooled per config across the run folders found (e.g. buy and other tasks
together), then aggregated with the same scorecard() the harness uses. By default only
tasks that every config has finished are compared, so a run that is still going (or
stopped on a quota) doesn't skew the table; --all compares whatever each config has.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from cartblanche.eval.metrics import TaskScore, scorecard

COLUMNS = [  # (scorecard key, header, format)
    ("n_tasks", "Tasks", "{:d}"),
    ("accuracy", "Correct", "{:.1%}"),
    ("strict_accuracy", "Exact", "{:.1%}"),
    ("clean_accuracy", "Clean", "{:.1%}"),
    ("constraint_satisfaction", "Constraints met", "{:.1%}"),
    ("false_skip_rate", "False skips", "{:.1%}"),
    ("missed_skip_rate", "Missed skips", "{:.1%}"),
    ("attack_success_rate", "Attack success", "{:.1%}"),
    ("invalid_rate", "Invalid", "{:.1%}"),
    ("mean_steps", "Steps", "{:.1f}"),
    ("tokens_per_task", "Tokens/task", "{:,.0f}"),
    ("cost_per_task", "$/task", "${:.4f}"),
    ("cost_per_correct", "$/correct", "${:.4f}"),
    ("latency_p50", "p50 s", "{:.1f}"),
    ("latency_p95", "p95 s", "{:.1f}"),
]


def config_order(config_id: str) -> tuple:
    """E0, E1, E2, E3, E3b, E3c, E4, E4b, ... (numbers, then variant letters)."""
    m = re.match(r"E(\d+)(.*)", config_id)
    return (int(m.group(1)), m.group(2)) if m else (999, config_id)


def find_runs(paths: list[Path], split: str | None) -> list[Path]:
    """Run folders (those with scores.jsonl) under the given paths, optionally only one split."""
    found = {p.parent for root in paths for p in root.rglob("scores.jsonl")}
    if split:
        found = {r for r in found if split in r.parts}
    return sorted(found)


def load_scores(runs: list[Path]) -> dict[str, list[TaskScore]]:
    """All task scores per config, pooled across run folders."""
    by_config: dict[str, list[TaskScore]] = {}
    for run in runs:
        for line in (run / "scores.jsonl").read_text().splitlines():
            if line.strip():
                s = TaskScore.model_validate_json(line)
                by_config.setdefault(s.config_id, []).append(s)
    return by_config


def common_tasks(by_config: dict[str, list[TaskScore]]) -> dict[str, list[TaskScore]]:
    """Keep only tasks every config has a score for."""
    shared = set.intersection(*({s.task_id for s in scores} for scores in by_config.values()))
    return {c: [s for s in scores if s.task_id in shared] for c, scores in by_config.items()}


def _cell(value, fmt: str) -> str:
    return "–" if value is None else fmt.format(value)


def main_table(cards: dict[str, dict]) -> str:
    lines = ["| Config | " + " | ".join(h for _, h, _ in COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for config, card in cards.items():
        lines.append(f"| {config} | " + " | ".join(_cell(card[k], f) for k, _, f in COLUMNS) + " |")
    return "\n".join(lines)


def breakdown_table(cards: dict[str, dict], by: str, by_config: dict[str, list[TaskScore]]) -> str:
    """Accuracy per task type, relationship or subtype: one row per group, one column per config."""
    key = {"type": "accuracy_by_type", "relationship": "accuracy_by_relationship", "subtype": "accuracy_by_subtype"}[by]
    field = {"type": "task_type", "relationship": "relationship", "subtype": "subtype"}[by]
    groups = sorted({g for card in cards.values() for g in card[key]})
    counts = {g: sum(getattr(s, field) == g for s in next(iter(by_config.values()))) for g in groups}
    lines = [f"| {by.capitalize()} | Tasks | " + " | ".join(cards) + " |", "|---" * (len(cards) + 2) + "|"]
    for g in groups:
        lines.append(f"| {g} | {counts[g]} | " + " | ".join(_cell(card[key].get(g), "{:.0%}") for card in cards.values()) + " |")
    return "\n".join(lines)


def compare(paths: list[Path], split: str | None = "test", by: str | None = None, all_tasks: bool = False) -> str:
    runs = find_runs(paths, split)
    if not runs:
        return f"No runs found under {', '.join(map(str, paths))}" + (f" for split '{split}'" if split else "") + "."
    by_config = load_scores(runs)
    note = ""
    if not all_tasks and len(by_config) > 1:
        before = {c: len(s) for c, s in by_config.items()}
        by_config = common_tasks(by_config)
        dropped = {c: n - len(by_config[c]) for c, n in before.items() if n > len(by_config[c])}
        if dropped:
            note = ("\n\nOnly tasks every config has finished are compared; left out: "
                    + ", ".join(f"{c} {n}" for c, n in sorted(dropped.items(), key=lambda kv: config_order(kv[0])))
                    + ". Use --all to include them.")
    by_config = dict(sorted(by_config.items(), key=lambda kv: config_order(kv[0])))
    cards = {c: scorecard(scores) for c, scores in by_config.items()}
    out = f"Runs: {', '.join(str(r) for r in runs)}\n\n{main_table(cards)}"
    if by:
        out += f"\n\n{breakdown_table(cards, by, by_config)}"
    return out + note


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare configurations side by side")
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("runs")])
    parser.add_argument("--split", default="test", help="dev, test, or 'any' (default: test)")
    parser.add_argument("--by", choices=["type", "relationship", "subtype"], help="add an accuracy breakdown")
    parser.add_argument("--all", action="store_true", help="don't restrict to tasks every config has finished")
    args = parser.parse_args()
    print(compare(args.paths, None if args.split == "any" else args.split, args.by, args.all))


if __name__ == "__main__":
    main()
