"""Command line: run one configuration.

    python -m cartblanche.eval.run configs/e1_generic.yaml                       # fake store
    python -m cartblanche.eval.run configs/e1_generic.yaml --tasks tasks/beauty.jsonl --split dev --call-gap 4

With --tasks, tasks come from that file and products from the real store (data/cartblanche.sqlite).
Results go to <out>/<config_id>/ (fake store) or <out>/<tasks name>/<config_id>/ (real tasks).
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

from cartblanche.config import load_config
from cartblanche.env.db import SqliteCatalog
from cartblanche.env.store import load_fake_store, load_tasks
from cartblanche.eval.harness import default_llm, run
from cartblanche.llm import QuotaExhausted


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one configuration")
    parser.add_argument("config", type=Path)
    parser.add_argument("--tasks", type=Path, help="task file for the real store (default: the fake store)")
    parser.add_argument("--split", choices=["dev", "test"], help="only run tasks from this split")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--call-gap", type=float, default=0.0,
                        help="minimum seconds between model calls, e.g. 4 for a 15-requests-per-minute free tier")
    parser.add_argument("--fresh", action="store_true", help="ignore saved results and rerun every task")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.tasks:
        catalog, tasks, out = SqliteCatalog(), load_tasks(args.tasks), args.out / args.tasks.stem
    else:
        (catalog, tasks), out = load_fake_store(), args.out
    if args.split:
        tasks = [t for t in tasks if t.split == args.split]
        out = out / args.split
    try:
        card = run(config, catalog, tasks, out, partial(default_llm, call_gap_s=args.call_gap), fresh=args.fresh)
    except QuotaExhausted as e:
        sys.exit(f"Daily quota used up; finished tasks are saved. Run the same command later to resume.\n{e}")
    print(json.dumps(card, indent=2))
    print(f"\nWrote {out / config.config_id}/")


if __name__ == "__main__":
    main()
