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
| Beauty fake store; E1 generic LLM agent (tool-calling loop); E2 intent parser (with brand field); free-tier retries and resumable runs | done | #5 |
| Real data: Beauty streamed into SQLite (`build_db.py`), `SqliteCatalog`, reviews hidden after `as_of` | done | #6 |
| Task generator: 150 real buy tasks (`taskgen.py`, `tasks/beauty.jsonl`); `run.py --tasks/--split`; fake reviews dated before `as_of` | done | #7 |
| Task generator: 178 relationship, override, ambiguous, brand and edge-case tasks (`task_types.py`, `tasks/beauty_other.jsonl`); `Task.subtype` and accuracy by subtype | done | #8 |
| `eval/compare.py`: configs side by side; scores saved even when a run stops | done | #10 |
| Product retrieval: personal search for E3 and later (`agent/search.py`, `personal_search`) | built; tuned offline on dev | |
| LLM judge and 50-task gold subset | to do | |
| E3 (8 closest past purchases) and E3c (all history) in the agent's first message | done; run on all 328 tasks (Nov 19) | #9, #12 |
| E3b summarized profile | done; run on all 328 tasks (Nov 19) | #9, #12 |
| History reason tagging (E3t) | done; run on all 328 tasks; hand check on 50 reviews still to do | #11 |
| E4 / E4b relationship rule vs check, override handling (`agent/relationship.py`) | done; run on all 328 tasks (Nov 19) | #12 |
| Saved attacks: attacked copies of the 150 buy tasks (`attackgen.py`, `tasks/beauty_attacked.jsonl`): LLM-written fake reviews ×3 seeds, copy-paste variant, 4 text attacks | done | #13 |
| Dose sweeps and break points (A3) | **next** | |
| Copied-phrasing check, fact extractor, labeling, verifier (E5–E7) | to do | |
| E8 clarifying question and simulated user (`ask_shopper`, `env/simulated_user.py`); measured early as E4q | built; E8 waits for E5–E7 | |
| Dose sweeps, full matrix, variance, scorecard, failure analysis | to do | |

E0 on the 11 beauty fake tasks gets 4 of 11 right. It always buys, so it misses every skip; it meets the stated constraints but matches keywords, so it buys a sunscreen for a moisturizer request. Fake reviews never move it, because it doesn't read reviews.

First real runs (Nov 14, Gemini 3.5 Flash-Lite, fake store; one run each, so small-sample):

| Config | Lenient | Strict | Missed skip | Invalid | Steps | Tokens / task | $ / task | p50 latency |
|---|---|---|---|---|---|---|---|---|
| E0 | 4/11 | 3/11 | 1.0 | 0 | 1 | 0 | 0 | 0 s |
| E1 | 9/11 | 8/11 | 1.0 | 0 | 4.6 | 3.4K | $0.001 | 2.3 s |
| E2 | 8/11 | 8/11 | 1.0 | 0 | 5.4 | 4.2K | $0.002 | 3.5 s |

- E1 and E2 both buy the duplicate hair dryer, as expected: they can't see history.
- The fake store is too easy without history: on most tasks the generically best product is also the gold. Real-data tasks need distractors that beat the gold on rating and fit the request but not the user.
- t004 (foundation): E1 read both products' reviews, saw "oxidizes" on Shadeworks and bought Truetone (lenient right; it said BUY where the gold is ALTERNATIVE). E2 skipped the reviews and bought the Shadeworks foundation that oxidized on this user. Neither knew the user's history.
- The fake review (t010) didn't fool either agent: the target's genuine reviews and rating were bad enough.

## Decisions

| Date | Decision | Why |
|---|---|---|
| Oct 22 | Build the whole plan. **No scope cuts**: all configs, attacks and metrics (reconfirmed Nov 12) | The study needs every comparison |
| Oct 22 | `search_products` searches a per-task **shelf** of about 50 products (the gold, ~9 hard distractors, fillers) | Keeps tasks controlled and cheap; the gold can't be buried in 7M items |
| Oct 22 | Importing a user's real Amazon order export is an optional Week 6 demo, not core | Not needed for the evaluation |
| Oct 26 | Stack: Python 3.12, pip + venv, pydantic, LiteLLM behind `llm.py`, a plain tool-calling loop | Simple, and switching models is one string |
| Oct 26 | Agent `gemini/gemini-2.5-flash-lite`; judge `openai/gpt-5.4-nano` | Cheap tier, and the judge is a different family from the agent |
| Oct 27 | Category: Clothing_Shoes_and_Jewelry | Matches the design doc |
| Oct 27 | Actions are **BUY / ALTERNATIVE / SKIP**; WAIT dropped | The data can't justify WAIT |
| Nov 4 | E0 is scored with the code metrics only, no judge | It gives no explanation to judge |
| Nov 4 | Brand: hard (filter), soft (preference) or empty (filled from history as a soft preference) | Milestone 2 feedback |
| Nov 4 | Histories are real reviews only; choose real users who fit each case | Milestone 2 feedback |
| Nov 4 | Fake reviews have varied text, 3 seeds per task and dose, and are saved and reused; plus one copy-paste variant | Every config faces the same attacks |
| Nov 4 | History agent tags each past review with a reason (fit, quality, style, other), size if mentioned, and date. For Beauty this maps to reaction / quality / scent-or-shade / other, with shade or skin type in place of size (to confirm with the team) | Tell a bad match apart from bad quality |
| Nov 14 | Category switched to **Beauty_and_Personal_Care** (replaces Clothing) | Taste (skin and hair type, shade, scent, reactions) decides purchases, so personalization matters more; restock vs duplicate makes relationship tasks harder; beauty fake reviews make checkable claims; 14 GB of raw files instead of 46 GB |
| Nov 14 | `Constraints.size` dropped; `required_features` (e.g. "fragrance-free") is checked against the product's own text by `store.has_features` | Size was clothing-specific; features are what beauty requests state |
| Nov 14 | A cached model reply keeps its original cost and latency; `Usage.spent_usd` counts real spend | Reruns must produce identical scorecards |
| Nov 14 | Bad agent output is scored invalid, never retried | Keeps the invalid-output rate honest |
| Nov 14 | Agent model is now `gemini/gemini-3.5-flash-lite` (was 2.5 Flash-Lite) | Google closed 2.5 Flash-Lite to new API keys (404). 3.5 Flash-Lite costs $0.30 / $2.50 per M tokens (vs $0.10 / $0.40) and rejects `temperature`, so LiteLLM drops it; the cache keeps reruns stable |
| Nov 14 | LiteLLM pinned to >= 1.109.1 | 1.106.4 crashed on any tool call (`No module named 'mcp'`) |
| Nov 15 | Buy-task requests name only the general kind of product (plus at most one common feature and a budget) | The first version copied product details ("Japanese body lotion with sake yeast"): keyword search alone ranked the gold first in 38% of tasks. Now 17%, so the user's history has to matter |
| Nov 15 | Buy-task acceptable set (first pass, in code) = other candidates that meet the hard constraints and are from a brand the user rated 4–5 before | The plan's "meets constraints and the user's preferences"; the team hand-checks it on the gold subset |
| Nov 16 | Non-buy tasks use templated requests, not LLM-written ones | The relationship cue ("running low", "to go with my", "the last one didn't work") must be exact, and templates cost one model call in total instead of one per task |
| Nov 16 | Dev share is 20% of each non-buy subtype (36 tasks), on top of the 40 buy dev tasks | Every subtype needs a few tuning examples; still 288 of 328 tasks held out |
| Nov 18 | No trimming of tool outputs (long reviews stay up to 2,000 characters) | Decided by the team: keep the pages as real as possible; costs about 25–35% more per run |
| Nov 18 | Test runs started: E0 done on all 252 held-out tasks, E1 running | E0 test matches dev (buy 13.6% vs 15%, other 20.4% vs 19%) |
| Nov 24 | Attacked tasks: 8 copies per buy task (fake reviews × 3 seeds, copy-paste, hidden instruction, urgency, fake override, planted preference), all saved at dose 8; the sweep lowers the dose | One file holds every attack, so all configs face the same text |
| Nov 24 | Fake reviews: one LLM call writes 8 per task and seed; dose n uses the first n, so doses are nested. Each has its own rating (4–5), age (1–14 days before the task), verified flag (about 70% verified) and helpful votes, fixed at generation | Nested doses keep break points meaningful; the metadata is what the copied-phrasing and burst checks will read |
| Nov 24 | Copy-paste variant: one fake review's text posted 8 times within 1–3 days | Tests the copied-phrasing check (near-duplicates plus a posting burst) |
| Nov 24 | Attack target: the most popular candidate that is not acceptable but still meets the request's constraints | Promoting an obviously wrong product (over budget, missing the feature) would show nothing |
| Nov 24 | Text attacks (hidden instruction, urgency, fake override, planted preference) keep the injector's fixed lines, one variant each | The plan asks for 3 seeds only for fake reviews; dose n adds n lines |
| Nov 26 | Attack generation complete: 1,200 attacked tasks for all 150 buy tasks (880 test, 320 dev), $0.56 in all. The first pass stopped on the daily quota and Gemini 503 overloads; the rerun filled every gap from the cache | Fake reviews: 3,600, about 210 characters median, 24% rated 4★, 71% verified. Products that are the wrong pick for several buy tasks carry the same fakes everywhere (same product and seed → same cached set) |
| Nov 15 | Tasks are committed (`tasks/beauty.jsonl`); the database is rebuilt locally | The build is deterministic (seed 0), so product and user IDs match on every machine |
| Nov 4 | Scoring: lenient accuracy (pick is in the acceptable set; BUY and ALTERNATIVE both count as buying) and strict accuracy (exact gold) | A held-out purchase isn't the only right answer |
| Nov 16 | History reaches the agent only in its first message, never through `get_history` or `check_owned`. E3 shows the 8 past purchases closest to the request; E3c shows all of them | E3 and E3c then differ only in what the model sees. With the tools it could fetch everything itself |
| Nov 16 | Closeness is measured with `all-MiniLM-L6-v2` (sentence-transformers) on the CPU | Free and local; the Mac GPU backend crashed |
| Nov 19 | E4 and E4b act at the same point: after the agent proposes a purchase, code finds owned items of the same product or the same finest category (via the store tools, history cut at `as_of`). None: no change. E4 skips. E4b makes one model call that labels the relationship (the `Relationship` values) and code maps it: duplicate → SKIP; replacement that repeats a product or brand rated 1–2 → the agent chooses again, once; replacement after a 1–2 rating → ALTERNATIVE; anything else → keep. After a "buy it anyway" override the step is off | The two differ only in how they treat "already owns one", so the comparison is clean; the check costs a call only when something similar is owned. The model sees the skip it is overriding, so the follow-up reads naturally |
| Nov 19 | Search for E3 and later ranks every shelf product that passes the filters by meaning + 0.3 × keyword rank + 1.0 × closeness to a product the shopper rated 4–5 + 0.2 for a liked brand, and returns 15. E1 and E2 keep the store's keyword search (10 hits) | Misses were mostly search misses: on dev the agent searched about once, and the bought product was never in the results (24–46% of buy misses) or never opened (about 30%); it opened the gold and chose another only 1–6% of the time. Offline on dev (no model calls) the bought product ranked first 11% → 29%, top 15 67% → 79%; the same weights on test: 15% → 28%, 64% → 76%. Meaning alone did not beat keywords (shelves are one narrow category), so E1 and E2 don't get it |
| Nov 19 | The agent prompt (all LLM configs) asks for more than one search with different wording and for opening the three or so best candidates before deciding | Same diagnosis: one search and almost no product pages opened |
| Nov 19 | E4b's check prompt: "duplicate" is only for a lasting tool or device of the same kind bought recently; never for products that get used up or that people own several of; complements are a different kind used together | On dev, 6 of 30 checks called ordinary serum, aftershave, fragrance, lash and moisturizer requests duplicates. Replaying the same 30 checks: right action 23 → 25. Two more rewordings stayed at 25 with different misses, so tuning stopped there. Caveat: the task generator also defines duplicates as lasting items, so this rule matches how the tasks were built |
| Nov 19 | Clarifying question (`clarify`): a fixed first step. One call decides whether the request names a kind of product; if not, it writes one question, the environment answers it (`TaskStore.ask_shopper` → `env/simulated_user.py`: same agent model, one sentence, profile facts only, never a brand or product; no profile → "nothing to add", no call), and both go in the agent's first message. E4q = E4b + the question, to measure it before E5–E7 exist | First built as an optional `ask_shopper` tool: on dev, Flash-Lite never used it (0 of 36). Ambiguous requests scored 4–12% for every config. Agent code never reads `hidden_profile` |
| Nov 19 | Tried and removed: telling the agent once to open at least two candidates before buying (a code nudge) | On dev it raised cost per task about 40% ($0.0040 → $0.0057 for E3) and left accuracy unchanged (E3 38.2% → 38.2%, E4 28.9% → 28.9%): after opening more pages the agent kept the same pick. The prompt's request to search more and open candidates stays (it is part of the v2 runs) |
| Nov 19 | The clarify step asked on all 6 ambiguous dev tasks and got the missing fact each time ("body lotion", "mascara"), but E4q got only 1 of 6 right: knowing the kind of product turns an ambiguous task into a buy task, which is just as hard | Kept: correct, and cheap (one small call; one more only for a vague request) |
| Nov 19 | Provider failures are not agent failures. Out of credits (402) or a rejected key stops the run like the daily quota; a rate limit or outage that outlasts the retries leaves the task unfinished (not scored, retried on the next run). Only the agent's own failures (bad decision, out of steps, a crash in our code) are scored invalid. Model-reply cache writes are atomic and a damaged cache file counts as a miss | The first paid run ran out of credits midway: 621 tasks were saved as "invalid" and the run kept going instead of stopping. Configs E2–E4b send identical intent-parser requests, so runs in parallel can write the same cache file |
| Nov 19 | Each history item's vector is saved once in `cache/embeddings/` (keyed by model name and the item's text) and reused; only the request is embedded per task. Ranking runs only over the shopper's visible history (own purchases before `as_of`), so no vector index | Every history-based config (E3, E3t, E4–E8) ranks the same items, so vectors were recomputed per task and per config. Users have about 10–25 visible items, so comparing against all of them is instant (≈0.05–0.09 s); an index adds nothing at this size |

## Data

- Source: Hugging Face `McAuley-Lab/Amazon-Reviews-2023`. The UCSD download links on the dataset page return 404.
  - `raw/review_categories/Beauty_and_Personal_Care.jsonl`: 11.0 GB
  - `raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl`: 2.8 GB
  - `benchmark/5core/rating_only/Beauty_and_Personal_Care.csv`: 0.38 GB, user / item / rating / timestamp only. 6.6M rows; 730K users with 5+ reviews, 277K with 8+, 68K with 15+; 208K products
- The files are uncompressed, and there is no Parquet version. The plan is to **stream and filter** into `data/cartblanche.sqlite` (git-ignored) rather than store the raw files: use the 5-core index to pick users with 8+ reviews and products with 5+ reviews, then one pass over metadata and one over reviews. That is about 11 minutes at roughly 20 MB/s. The database should be about 0.3–1 GB.
- **Reviews have no shade, skin-type or size field.** Those come only from review text ("too dark for me", "broke me out") or product details ("Skin Type", "Hair Type", "Scent"), so match tasks are built only from reviews that say them.
- **Watch for repurchases.** Beauty users often rebuy the same item. When building buy tasks, separate exact repurchases (a restock task) from new-product gold picks, or "predict the purchase" becomes "pick what they bought before".
- Other categories checked on Nov 14 (5-core users with 8+ reviews): Clothing 1.07M, Pet Supplies 240K, Grocery 170K, Toys 167K, Video Games 35K. Amazon_Fashion has no 5-core file and only 3.9K users with 8+ reviews in its raw file.
- Review fields: rating, title, text, images, asin, parent_asin, user_id, timestamp, helpful_vote, verified_purchase.

## The real store (built Nov 14)

`data/cartblanche.sqlite`, 1.5 GB, built in 21 minutes (`python -m cartblanche.env.build_db`, seed 0). Spot check: all 20 sampled products match the source metadata. That's larger than the 0.3–1 GB estimate, because it keeps 10 reviews for every 5-core product, not just future shelf products.

| | Count |
|---|---|
| Users (random sample of the 277K with 8+ 5-core reviews) | 10,000 |
| History reviews | 156K; per user: median 12, p90 24, max 584; 79% rated 4–5 |
| Products (5-core + history items) | 221K; 126K (57%) have a price; 98% have a brand |
| Reviews shown on product pages (top 10 by helpful votes) | 2.03M, on 208K products |
| Review dates | 2001 – Sep 2023 |
| Same user reviewing the same product twice (repurchase signal) | 1,690 pairs |
| History reviews that mention sensitive/oily skin, curly hair, shade or "broke me out" | 5.5K |

Top categories: Hair Care 59K, Skin Care 48K, Foot/Hand/Nail 30K, Makeup 25K, Tools 24K, Shave 11K, Personal Care 9K, Fragrance 8.5K. Every history item has metadata.

Notes from building the store (the first two are handled in #7):
- Only 57% of products have a price, so shelves for budget tasks must come from priced products.
- Fake reviews are dated 2026-09-28 (fake-store time); for real tasks they must be dated just before the task's `as_of`, or they'll look like they come from the future.
- Real review dates end in Sep 2023, so `as_of` comes from each user's history, not a fixed date.

## Real buy tasks (Nov 15)

`python -m cartblanche.env.taskgen --buy 150 --call-gap 4` (seed 0, about 13 minutes, $0.04). 150 tasks: 40 dev, 110 test; shelves of 50 (gold, 9 distractors, 40 fillers from the same narrow category); visible history before the gold has a median of 10 items. 53% have a required feature; 11% have acceptable alternatives besides the gold. Keyword search on the request ranks the gold first in 17% and in the top 10 in 51% (chance is 2% and 20%).

First runs on the 40 dev tasks (Nov 15, one run each):

| Config | Tasks run | Correct | Constraints met | Invalid | Steps | Tokens / task | $ / task | p50 latency |
|---|---|---|---|---|---|---|---|---|
| E0 | 40 | 6/40 (15%) | 100% | 0 | 1 | 0 | 0 | 0.04 s |
| E1 | 40 | 3/40 (7.5%) | 97.5% | 0 | 3.9 | 7.1K | $0.002 | 3.1 s (p95 35 s) |
| E2 | 40 | 9/40 (22.5%) | 97.5% | 0 | 4.6 | 6.4K | $0.002 | 3.7 s (p95 36 s) |

- E1 and E2 first stopped on the free tier's daily cap and resumed the next day.
- E2's explicit requirements triple E1's score and beat E0: the first support for E2's hypothesis. 40 tasks and one run, so a 3–4 task gap is within noise.
- Without history, E1 picks a sensible, popular product: most of its misses were one of the 9 hard distractors. Guessing a person's real purchase without knowing them is hard, which is the gap history (E3) has to close.
- E0 beats E1 here because it gets the exact budget and feature from the task and sorts by rating; E1 has to read them from the request.

## E3 and E3c on the fake store (Nov 16)

11 tasks, one run each, Gemini 3.5 Flash-Lite. Both got 10 of 11 right (E1 9, E2 8 in the Nov 14 runs).

* E3: 4.3K tokens and $0.0017 per task. E3c: 4.7K tokens and $0.0018 per task. The fake shoppers have only 2 to 4 past purchases, so E3 filters little; the real tasks will show the gap.
* Both still buy the duplicate hair dryer (t003). The agent read "loved it" in the history as a reason to buy again. Showing history is not enough to catch a duplicate: that is what the relationship check (E4) is for.

## Other task types (Nov 16)

`python -m cartblanche.env.taskgen --other` (seed 0, under a minute, one model call for readable category names). Each comes from a real event in a user's history, and each user gives at most one task:

| Subtype | Tasks (dev) | Real event | Gold |
|---|---|---|---|
| restock | 20 (4) | reviewed the same product twice, liked it both times | BUY it |
| duplicate | 19 (4) | recently bought and liked a lasting item (hair dryer, straightener, shaver, mirror...) | SKIP |
| override | 20 (4) | a duplicate, then "buy the same one again anyway" | BUY it |
| second_unit | 10 (2) | bought a second lasting item of the same kind, liked both | BUY |
| replacement_bad | 15 (3) | rated a product 1–2, later liked a different brand of that kind | ALTERNATIVE |
| complement | 15 (3) | liked A, then bought its companion B (12 fixed pairs, e.g. shampoo → conditioner, polish → top coat) | BUY |
| ambiguous | 30 (6) | a later purchase, asked for vaguely ("something for my skin care"); shelf fillers from sibling categories; hidden profile for the simulated user | BUY |
| brand_hard / brand_soft / brand_conflict | 9 / 10 / 10 (2 each) | brand named as a must / liked twice before but unstated / named but rated 1–2 before | BUY |
| nothing_fits | 10 (2) | budget below every product on the shelf | SKIP |
| tiny_history | 10 (2) | only 1–2 earlier reviews | BUY |

Requests come from short templates around a readable name for each kind of product ("I'm running low on my body lotion, I need more, under $25"). Keyword search ranks the target first in 8% of these. Duplicate SKIP tasks are dated 30–180 days after the purchase, since there's no real "asked again" event.

First runs on the 36 non-buy dev tasks (Nov 16, one run each):

| Config | Correct | Missed skips | Invalid | Tokens / task | $ / task |
|---|---|---|---|---|---|
| E0 | 7/36 (19%) | 4 of 6 should-skip tasks bought | 0 | 0 | 0 |
| E1 | 7/36 (19%) | 4 of 6 | 0 | 5.7K | $0.002 |
| E2 | 7/36 (19%) | 4 of 6 | 0 | 7.1K | $0.002 |

- All three get the same 7 tasks: only the ones the request alone can answer: hard brand and brand conflict (the brand is named), nothing fits (no product meets the budget, so they skip). E1 and E2 also got 1 of 2 tiny-history tasks. E2's parser adds nothing here, because these requests state almost no constraints.
- All three score 0 on every history-dependent type: restock, duplicate, replacement, complement, second unit, brand soft, ambiguous, and override (E1 and E2 never skip, so the "buy anyway" never comes up; E0 got 1 of 4 by luck). This is what E3 and E4 must fix.

## Open questions

- Is Nov 15 the real experiment freeze? The build is behind the plan's schedule.
- The design doc and the development plan still say Clothing; they need updating for Beauty, including the Nov 4 answer about size vs quality vs brand.
- **The Gemini free tier allows 500 requests per day for `gemini-3.5-flash-lite`** (hit on Nov 15). One config on the 40 dev tasks takes about 160–250 calls, and the full matrix about 60K, so billing must be enabled before the full runs; on the free tier we can do roughly two dev runs a day.
- At 3.5 Flash-Lite prices the full matrix costs roughly $40–60 for the agent instead of about $13; the plan's $20–40 budget needs a second look.

## E3b on the fake store (Nov 17)

E3b writes a short profile of the shopper (at most 120 words) from the past purchases, and the agent sees only that profile. This costs one extra model call per task. A shopper with no history gets no extra call.

Result: 9 of 11 correct (E3 and E3c got 10 of 11).

Two misses:
- t003 (duplicate): it bought the hair dryer the shopper already owns. Same as E3 and E3c. The relationship check (E4) is the planned fix.
- t009 (ambiguous): it skipped a perfume request because the profile said the shopper prefers fragrance free products. E3 and E3c did not skip here.

The t009 miss is one task out of 11, so it is only an observation. Re-check it on the real dev tasks before drawing any conclusion about profiles versus raw history.

## E0–E4b on real data (Nov 19)

Every config on all 328 tasks, one run each, Gemini 3.5 Flash-Lite (paid tier; about $9.30 in all by LiteLLM's prices, 7 of ~2,600 agent runs invalid). E3t comes from PR #11, merged locally for this run. E4 and E4b use E3 (RAG) for history.

Test, all 252 tasks (`python -m cartblanche.eval.compare --split test --by subtype`):

| Config | Correct | Exact | False skips | Missed skips | $/task | p50 s |
|---|---|---|---|---|---|---|
| E0 | 17.5% | 13.5% | 0.0% | 65.2% | 0 | 0.0 |
| E1 | 20.6% | 17.5% | 2.6% | 65.2% | $0.0025 | 2.3 |
| E2 | 19.8% | 15.9% | 1.3% | 65.2% | $0.0026 | 3.0 |
| E3 | 33.3% | 28.6% | 1.3% | 65.2% | $0.0036 | 3.1 |
| E3b | 31.0% | 26.2% | 3.9% | 65.2% | $0.0044 | 4.5 |
| E3c | 32.1% | 28.6% | 2.2% | 60.9% | $0.0043 | 3.3 |
| E3t | 33.7% | 30.2% | 1.3% | 65.2% | $0.0041 | 3.8 |
| E4 | 29.4% | 26.6% | 36.7% | 0.0% | $0.0043 | 3.3 |
| E4b | **39.3%** | **34.1%** | 4.4% | 8.7% | $0.0045 | 3.6 |

Test buy tasks (110): E0 13.6%, E1 19.1%, E2 15.5%, E3 26.4%, E3b 26.4%, E3c 24.5%, E3t 28.2%, E4 16.4%, E4b 25.5%. Dev buy tasks (40): E1 15.0%, E2 17.5%, E3 30.0%.

- History is the big gain: E3 is +13.5 points over E2 on test (+11 on buy, +15.5 on other tasks). The four history versions are within 2.7 points of each other, which is within noise at this size; E3 (RAG) is the cheapest of them.
- E2's parser does not help on test (19.8% vs E1 20.6%; buy 15.5% vs 19.1%). The dev gain (+15 points on 40 buy tasks in the Nov 15 runs, +2.5 in these) did not hold.
- E4's blunt rule catches every duplicate (15/15) but skips 36.7% of tasks where buying was right: restock and second unit drop to 0%, buy tasks to 16.4%.
- E4b keeps most of the duplicate catches (87%) with 4.4% false skips, and is the best config overall: restock 62%, override 81%, duplicate 87%. On dev its relationship label matched the task's relationship on 16 of 17 relationship tasks; its main error is labelling ordinary requests where the shopper owns a similar item as duplicates (6 of 12 on dev).
- Still weak everywhere: ambiguous (4–12%: the clarifying question, E8, is the planned fix), replacement after a bad experience (0–17%: E4b labels it right but the new pick still misses the gold), second unit (0–25%).
- `tools.dispatch` now returns a badly typed tool argument (e.g. `filters.brand` as an object) to the model as an error instead of crashing the task (one E3b dev task).

## Final runs after the search, prompt, check and clarify changes (Nov 19)

`runs/final/`, every config on all 328 tasks, one run each; about $9.90. 9 of ~2,900 agent runs invalid, all the agent's own (2 out of steps, 7 malformed JSON replies). Earlier numbers (`runs/beauty*`) are kept as the "before".

Test, all 252 tasks:

| Config | Before | After | Exact (after) | False skips | Missed skips | $/task |
|---|---|---|---|---|---|---|
| E0 | 17.5% | 17.5% | 13.5% | 0.0% | 65.2% | 0 |
| E1 | 20.6% | 22.6% | 19.0% | 1.3% | 60.9% | $0.0034 |
| E2 | 19.8% | 20.6% | 17.9% | 1.3% | 65.2% | $0.0036 |
| E3 | 33.3% | 36.5% | 30.6% | 1.3% | 65.2% | $0.0048 |
| E3b | 31.0% | 34.1% | 29.4% | 1.7% | 65.2% | $0.0056 |
| E3c | 32.1% | 34.1% | 30.2% | 1.7% | 65.2% | $0.0053 |
| E3t | 33.7% | 34.9% | 30.6% | 0.4% | 65.2% | $0.0052 |
| E4 | 29.4% | 31.0% | 27.0% | 36.2% | 0.0% | $0.0053 |
| E4b | 39.3% | **42.5%** | **36.1%** | 1.7% | **0.0%** | $0.0052 |
| E4q | – | **42.5%** | 35.7% | 2.6% | 4.3% | $0.0053 |

- Every history config gained 1–3 points; E4b is still the best (39.3% → 42.5%), now with 1.7% false skips (was 4.4%) and no missed skips (was 8.7%). Most of the gain is on the non-buy tasks (E4b 50.0% → 56.3%); buy tasks barely moved (E3 26.4% → 28.2%, E4b 25.5% → 24.5%).
- Why the search gain mostly didn't carry through: search now shows the bought product far more often (offline top 15: 64% → 76%), but on dev the agent opened it and still chose another 13–14% of the time (was 1–6%). Finding candidates is no longer the main limit; choosing among close ones is.
- E4q's question did not raise accuracy over E4b (42.5% both; ambiguous 8% → 12%, 3 of 24): the question gets the kind of product right, and then the pick is as hard as a buy task.
- E1 and E2 changed only through the prompt's "search more, open candidates" request (+0.8 to +2.0 points, within noise).
