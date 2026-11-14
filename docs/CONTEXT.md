# Project context

This file is the shared memory for anyone working on the repo, people and AI agents alike. It records what has been decided and why, where the build stands, and what is still open. Keep it short and current: when you finish a step, make a decision, or learn something about the data, update it in the same PR.

- [Development plan](https://claude.ai/artifact/HvaZZ4H3KXXnRKuuWHnvTP): the source of truth for scope, configs E0–E8, metrics and the schedule.
- [Design document](https://claude.ai/artifact/4Tfv4GF3NxuznimRyyo7uF): the team's design.
- Team: Rohan (A · environment & attacks), Megha (B · agent & personalization), Riddhi (C · defenses & evaluation).

## Status

| Step | State | PR |
|---|---|---|
| Week 0: contracts (`schemas.py`), config, `llm.py`, fake store + tasks, store tools, attack injector | done | #1, #2 |
| Eval harness, code-checked metrics and scorecard, E0 rules baseline | done | #3 |
| Real data: stream Clothing into SQLite, real `Catalog` | **next** | |
| E1 generic LLM agent (tool-calling loop) and E2 intent parser (with brand field) | **next** | |
| Task generator (buy, relationship, override, ambiguous, brand, edge cases) | to do | |
| Product retrieval (filter → embeddings → rerank) | to do | |
| LLM judge and 50-task gold subset | to do | |
| E3 / E3b / E3c personalization and history reason tagging | to do | |
| E4 / E4b relationship rule vs check, override handling | to do | |
| Saved attacks (3 seeds per task and dose) and copy-paste variant | to do | |
| Copied-phrasing check, fact extractor, labeling, verifier (E5–E7) | to do | |
| E8 clarifying question and simulated user | to do | |
| Dose sweeps, full matrix, variance, scorecard, failure analysis | to do | |

E0 on the 11 fake tasks gets 2 of 11 right. It always buys, so it misses every skip; it always meets the stated constraints, and fake reviews never move it, because it doesn't read reviews.

## Decisions

| Date | Decision | Why |
|---|---|---|
| Oct 22 | Build the whole plan. **No scope cuts**: all configs, attacks and metrics (reconfirmed Nov 12) | The study needs every comparison |
| Oct 22 | `search_products` searches a per-task **shelf** of about 50 products (the gold, ~9 hard distractors, fillers) | Keeps tasks controlled and cheap; the gold can't be buried in 7M items |
| Oct 22 | Importing a user's real Amazon order export is an optional Week 6 demo, not core | Not needed for the evaluation |
| Oct 26 | Stack: Python 3.12, pip + venv, pydantic, LiteLLM behind `llm.py`, a plain tool-calling loop | Simple, and switching models is one string |
| Oct 26 | Agent `gemini/gemini-2.5-flash-lite`; judge `openai/gpt-5.4-nano` | Cheap tier, and the judge is a different family from the agent |
| Oct 27 | Category: **Clothing_Shoes_and_Jewelry** | Matches the design doc |
| Oct 27 | Actions are **BUY / ALTERNATIVE / SKIP**; WAIT dropped | The data can't justify WAIT |
| Nov 4 | E0 is scored with the code metrics only, no judge | It gives no explanation to judge |
| Nov 4 | Brand: hard (filter), soft (preference) or empty (filled from history as a soft preference) | Milestone 2 feedback |
| Nov 4 | Histories are real reviews only; choose real users who fit each case | Milestone 2 feedback |
| Nov 4 | Fake reviews have varied text, 3 seeds per task and dose, and are saved and reused; plus one copy-paste variant | Every config faces the same attacks |
| Nov 4 | History agent tags each past review with a reason (fit, quality, style, other), size if mentioned, and date | Tell a wrong size apart from bad quality |
| Nov 4 | Scoring: lenient accuracy (pick is in the acceptable set; BUY and ALTERNATIVE both count as buying) and strict accuracy (exact gold) | A held-out purchase isn't the only right answer |

## Data

- Source: Hugging Face `McAuley-Lab/Amazon-Reviews-2023`. The UCSD download links on the dataset page return 404.
  - `raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl`: 27.8 GB, 66M reviews
  - `raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl`: 18.0 GB, 7.2M products
  - `benchmark/5core/rating_only/Clothing_Shoes_and_Jewelry.csv`: 1.3 GB, user / item / rating / timestamp only
- The files are uncompressed, and there is no Parquet version. The plan is to **stream and filter** into `data/cartblanche.sqlite` (git-ignored) rather than store the raw files: use the 5-core index to pick users with 8+ reviews and products with 5+ reviews, then one pass over metadata and one over reviews. That is about 40 minutes at roughly 20 MB/s.
- **Reviews have no size field.** Size and fit come only from review text ("ordered a medium"), so fit tasks are built only from reviews that mention a size.
- Review fields: rating, title, text, images, asin, parent_asin, user_id, timestamp, helpful_vote, verified_purchase.

## Open questions

- Is Nov 15 the real experiment freeze? The build is behind the plan's schedule.
- Approval to start the data streaming step.
