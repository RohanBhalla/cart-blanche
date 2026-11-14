"""Command line: python -m cartblanche.eval.run configs/e0_rules.yaml [--out runs]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cartblanche.config import load_config
from cartblanche.env.store import load_fake_store
from cartblanche.eval.harness import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one configuration on the fake store")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    config = load_config(args.config)
    catalog, tasks = load_fake_store()
    card = run(config, catalog, tasks, args.out)
    print(json.dumps(card, indent=2))
    print(f"\nWrote {args.out / config.config_id}/")


if __name__ == "__main__":
    main()
