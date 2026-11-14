# CartBlanche

A personalized shopping agent that decides **BUY / ALTERNATIVE / SKIP** for a user's request, using their purchase history, and stays robust to seller-side manipulation (fake reviews, hidden instructions, fake "user approved" text).

Built on [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab; Clothing, Shoes & Jewelry). The evaluation harness comes first, and every component is switchable so its effect can be measured (configs E0–E8).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your GEMINI_API_KEY
pytest
python -m cartblanche.eval.run configs/e0_rules.yaml   # results in runs/E0/
```

## Layout

| Path | Owner | Contents |
|---|---|---|
| `cartblanche/schemas.py` | shared | Contracts: Product, Review, User, Task, AgentResult |
| `cartblanche/config.py` | shared | Run config: one YAML per run, component switches |
| `cartblanche/llm.py` | shared | LiteLLM wrapper: disk cache, token / cost / latency logging |
| `cartblanche/env/` | A · Environment & attacks | Store tools, fake store + 10 tasks, attack injector |
| `cartblanche/agent/` | B · Agent & personalization | E0 rules baseline; next: intent parser, retrieval, history RAG, recommendation agent |
| `cartblanche/defenses/` | C · Defenses | Fact extractor, untrusted-text labeling, verifier |
| `cartblanche/eval/` | C · Evaluation | Harness (`run`), code-checked metrics and scorecard; next: judge |
| `configs/` | shared | E0–E8 experiment configs |

## How the store works

The agent never touches a real website. It calls five tools (`search_products`, `get_product`, `get_reviews`, `get_history`, `check_owned`) against a `TaskStore`, the view of the catalog one task sees:

- search only covers the task's **shelf** (~50 products; gold, hard distractors, fillers);
- history is cut at the task's `as_of` time, so the held-out gold purchase never leaks;
- an attacked task's injected reviews/text appear on the target product's page. Injected review IDs look exactly like genuine ones; only the evaluator sees `TaskStore.injected_ids`.

Development plan: https://claude.ai/artifact/HvaZZ4H3KXXnRKuuWHnvTP
