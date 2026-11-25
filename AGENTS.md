# AGENTS.md

Instructions for anyone working in this repo, whether a person or an AI coding agent (Claude Code, Codex, Cursor and so on).

**Before you start, read [`docs/CONTEXT.md`](docs/CONTEXT.md).** It holds the decisions made so far and why, where the build stands, and the open questions. **Update it in the same PR** whenever you finish a step, make a decision, or learn something about the data.

## The project in one paragraph

CartBlanche is a shopping agent. It answers a user's request with **BUY / ALTERNATIVE / SKIP**, using the user's purchase history, and stays robust to seller manipulation: fake reviews, hidden instructions, and fake "the user approved this" text. It runs on Amazon Reviews 2023 (Beauty & Personal Care), where reviewers' review histories stand in for purchases. It is a course study: every component can be switched on or off in a config (E0–E8), so its effect can be measured. The full plan is the [development plan](https://claude.ai/artifact/HvaZZ4H3KXXnRKuuWHnvTP).

## Setup and commands

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add GEMINI_API_KEY (agent) and OPENAI_API_KEY (judge)
pytest                      # must pass before every commit
python -m cartblanche.env.build_db                    # real store: streams ~14 GB into data/cartblanche.sqlite (~15 min)
python -m cartblanche.env.taskgen --buy 150 --call-gap 4   # regenerate tasks/beauty.jsonl (needs the store)
python -m cartblanche.env.taskgen --other                   # relationship/override/ambiguous/brand/edge tasks -> tasks/beauty_other.jsonl
python -m cartblanche.env.attackgen --call-gap 4            # attacked copies of the buy tasks -> tasks/beauty_attacked.jsonl
python -m cartblanche.eval.run configs/e0_rules.yaml   # writes runs/E0/ (fake store)
python -m cartblanche.eval.run configs/e1_generic.yaml --tasks tasks/beauty.jsonl --split dev --call-gap 4   # real tasks
python -m cartblanche.eval.compare --split dev --by subtype   # all configs side by side, as Markdown tables
python -m cartblanche.eval.run configs/e1_generic.yaml --call-gap 4   # real Gemini calls; resumable, --fresh to redo
```

Use Python 3.12 with pip and a venv, not uv or poetry.

## Where things live

| Path | Owner | What |
|---|---|---|
| `cartblanche/schemas.py` | shared | The contracts: Product, Review, User, Task, AgentResult. Everything else depends on these |
| `cartblanche/config.py` | shared | `RunConfig`, one YAML per run in `configs/` |
| `cartblanche/llm.py` | shared | The only path to a model: LiteLLM, a disk cache, and token, cost and latency logging |
| `cartblanche/env/` | A · Rohan | Store and `TaskStore`, the five agent tools, the attack injector, the fake store (`env/data/`), the real SQLite store (`db.py`, built by `build_db.py`) |
| `cartblanche/agent/` | B · Megha | E0 rules baseline, E1 agent loop (`llm_agent.py`), E2 intent parser (`intent.py`), E3 history (`history.py`, `retrieval.py`), E4 / E4b relationship step (`relationship.py`) |
| `cartblanche/defenses/` | C · Riddhi | Fact extractor, untrusted-text labeling, copied-phrasing check, verifier |
| `cartblanche/eval/` | C · Riddhi | Harness, code-checked metrics, scorecard; then the judge |
| `tasks/` | A · Rohan | Generated task files (committed), built by `env/taskgen.py` |
| `tests/` | everyone | One test file per module |

## Rules that keep the experiment valid

- **The agent sees only what a shopper would see.** Agent code gets data only through the five tools (`search_products`, `get_product`, `get_reviews`, `get_history`, `check_owned`) on a `TaskStore`. It must never read `Task.gold_*`, `acceptable`, `hidden_profile`, `attack`, or `TaskStore.injected_ids`. Those exist for the evaluator and the simulated user only.
- **Every model call goes through `llm.LLM`.** That keeps cost and latency logged and makes reruns free from the cache. Use temperature 0.
- **The agent and the judge come from different model families.** The agent is Gemini and the judge is OpenAI.
- **Prompts are tuned on the `dev` split only.** Reported numbers come from `test`.
- **Every component is a config switch.** A new component gets a field in `RunConfig`, is off by default, and is turned on in the config that tests it. Until it's built, `harness.not_built_yet` makes configs that use it fail fast.
- **After changing a prompt or agent code, rerun with `--fresh`.** Runs resume from saved results, which would otherwise be reused.
- **Attacks are saved and reused.** Every config must face the same injected text (3 seeds per task and dose). Dose 0 must equal the clean page exactly.
- **Histories are real reviews only.** Never invent items for a user's history. Choose real users whose history fits the test case.

## How we write code

- **Keep it simple.** Write plain functions and pydantic models, and avoid frameworks, plugin systems and speculative options. If a feature isn't in the plan, don't add it. Delete code that isn't used.
- **Don't break the contracts.** If you change `schemas.py`, update `env/data/fake_store.json`, `fake_tasks.jsonl` and the tests in the same PR, and say so in the PR description, because every role depends on it.
- **Each change comes with a test.** Metric tests use hand-computed expected numbers. Tests never call a real API: use the fake completion in `tests/test_llm.py` or LiteLLM's `mock_response`.
- **Don't commit generated files.** `.env`, `data/`, `cache/` and `runs/` are git-ignored. Keep it that way.

## Git workflow

- Never push to `main` directly. Create a fresh branch from an up-to-date `main` for each step (`git checkout main && git pull && git checkout -b <step-name>`), then open a PR. The repo owner reviews and merges.
- Before pushing to an existing branch, check that its PR is still open (`gh pr view --json state`). After a merge, start a new branch rather than reusing the old one, and delete merged branches.
- PR descriptions are short and plainly worded, with only two sections: **What** (what changed) and **Why** (why it changed). Test details and results go in `docs/CONTEXT.md`.
- **No AI attribution:** no `Co-Authored-By` lines for AI tools and no "Generated with" footers, in commits or PRs.
- Never commit API keys. Keys live only in `.env`.
