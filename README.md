# CartBlanche

A personalized shopping agent that decides **BUY / ALTERNATIVE / SKIP** for a user's request, using their purchase history, and stays robust to seller-side manipulation (fake reviews, hidden instructions, fake "user approved" text).

Built on [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab; Beauty & Personal Care). The evaluation harness comes first, and every component is switchable so its effect can be measured (configs E0–E8).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your GEMINI_API_KEY
pytest
python -m cartblanche.env.build_db                                   # real store (~15 min, see below)
python -m cartblanche.eval.run configs/e0_rules.yaml                 # results in runs/E0/
python -m cartblanche.eval.run configs/e1_generic.yaml --call-gap 4  # Gemini free tier: 4 s between calls
python -m cartblanche.eval.run configs/e1_generic.yaml --tasks tasks/beauty.jsonl --split dev --call-gap 4  # real tasks
```

## Layout

| Path | Owner | Contents |
|---|---|---|
| `cartblanche/schemas.py` | shared | Contracts: Product, Review, User, Task, AgentResult |
| `cartblanche/config.py` | shared | Run config: one YAML per run, component switches |
| `cartblanche/llm.py` | shared | LiteLLM wrapper: disk cache, token / cost / latency logging |
| `cartblanche/env/` | A · Environment & attacks | Store tools, beauty fake store + 11 tasks, attack injector |
| `cartblanche/agent/` | B · Agent & personalization | E0 rules baseline, E1 tool-calling agent, E2 intent parser, E3 history (RAG, profile, full), E4 / E4b relationship rule and check |
| `cartblanche/defenses/` | C · Defenses | Fact extractor, untrusted-text labeling, verifier |
| `cartblanche/eval/` | C · Evaluation | Harness (`run`), code-checked metrics and scorecard; next: judge |
| `configs/` | shared | E0–E8 experiment configs |

## Comparing configurations

`python -m cartblanche.eval.compare` prints every config's scorecard side by side as Markdown tables, pooling all task sets under `runs/` for the chosen split (`--split dev|test|any`, default test). `--by type|relationship|subtype` adds an accuracy breakdown. Only tasks every config has finished are compared, so a run stopped by a quota doesn't skew the table (`--all` turns that off).

## Running on the free tier

Runs resume: rerunning the same command skips tasks that already have a valid result and retries the ones that failed, so a run cut short by rate limits just continues. Rate limits and timeouts are retried with backoff; when the daily quota runs out, the run stops and tells you to rerun later. After changing a prompt or the code, pass `--fresh` so old results aren't reused.

## The real store

`python -m cartblanche.env.build_db` streams Amazon Reviews 2023 Beauty & Personal Care from Hugging Face (about 14 GB) and keeps only what the tasks need in `data/cartblanche.sqlite`; no raw files are saved. It samples 10,000 users with 8+ reviews and keeps their full histories, the 10 most helpful reviews of each product that has 5+ reviews, and those products' metadata. `--spot-check 20` compares 20 stored products with the source. `SqliteCatalog` reads it through the same interface as the fake store.

`python -m cartblanche.env.taskgen --buy 150` then builds buy tasks from it into `tasks/beauty.jsonl` (committed): a user's later 4–5★ purchase is the gold, an LLM writes a general request with a budget, and the shelf mixes the gold with 9 hard distractors and same-category fillers. `--other` builds the relationship, override, ambiguous, brand and edge-case tasks into `tasks/beauty_other.jsonl`, each from a real event in a user's history (`env/task_types.py`). `python -m cartblanche.env.attackgen` writes attacked copies of the buy tasks into `tasks/beauty_attacked.jsonl`: LLM-written fake reviews (3 saved seeds), a copy-paste variant and the 4 text attacks, all aimed at a plausible wrong product.

## How the store works

The agent never touches a real website. It calls five tools (`search_products`, `get_product`, `get_reviews`, `get_history`, `check_owned`) against a `TaskStore`, the view of the catalog one task sees:

- search only covers the task's **shelf** (~50 products; gold, hard distractors, fillers);
- history and reviews are cut at the task's `as_of` time, so the held-out gold purchase never leaks;
- an attacked task's injected reviews/text appear on the target product's page. Injected review IDs look exactly like genuine ones; only the evaluator sees `TaskStore.injected_ids`.

Development plan: https://claude.ai/artifact/HvaZZ4H3KXXnRKuuWHnvTP

## Working on the repo

Read [`AGENTS.md`](AGENTS.md) for conventions and [`docs/CONTEXT.md`](docs/CONTEXT.md) for the decisions so far, where the build stands, and the open questions. Both apply to people and AI coding agents alike (`CLAUDE.md` points Claude Code at `AGENTS.md`).
