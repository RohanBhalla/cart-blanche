# CartBlanche

A personalized shopping agent that decides **BUY / SKIP / WAIT / ALTERNATIVE** for a user's request, using their purchase history, and stays robust to seller-side manipulation (fake reviews, hidden instructions, fake "user approved" text).

Built on [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab), with an evaluation harness built first and every component switchable so its effect can be measured (configs E0–E8).

## Layout

| Dir | Owner | Contents |
|---|---|---|
| `env/` | A · Environment & attacks | Store API, user histories, task set + ground truth, attack injector |
| `agent/` | B · Agent & personalization | Intent parser, retrieval, history RAG, recommendation agent |
| `defenses/` | C · Defenses | Fact extractor, untrusted-text labeling, verifier |
| `eval/` | C · Evaluation | Harness, metrics, judge, results tables |
| `llm.py` | shared | LLM wrapper: logs tokens/latency/cost, caches responses |

Development plan: https://claude.ai/artifact/HvaZZ4H3KXXnRKuuWHnvTP
