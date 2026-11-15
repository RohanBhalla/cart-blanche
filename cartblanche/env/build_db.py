"""Build data/cartblanche.sqlite from Amazon Reviews 2023 by streaming, without saving raw files.

    python -m cartblanche.env.build_db                 # Beauty_and_Personal_Care, ~14 GB streamed
    python -m cartblanche.env.build_db --spot-check 20 # compare 20 stored products with the source

Three passes over Hugging Face files (each line is read once and either kept or dropped):
  1. 5-core index (user, product, rating, time; users and products with 5+ reviews):
     pick a random sample of users with 8+ reviews, and the set of 5-core products.
  2. Reviews: keep every review by a sampled user (their full history), and for each
     5-core product the K most helpful reviews (shown on its page).
  3. Metadata: keep products that are in the 5-core or in a sampled user's history.
     Long fields are trimmed; images, videos and IDs are dropped.

Every source can also be a local file path, which is how the tests run it.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sqlite3
import time
import urllib.request
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from cartblanche.env.db import DEFAULT_DB, INDEXES, SCHEMA, SqliteCatalog, connect

HF = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
DROP_DETAILS = {"UPC", "Package Dimensions", "Product Dimensions", "Item model number", "Manufacturer",
                "Is Discontinued By Manufacturer", "Date First Available", "Best Sellers Rank", "ASIN"}


def sources(category: str) -> dict[str, str]:
    return {
        "core": f"{HF}/benchmark/5core/rating_only/{category}.csv",
        "reviews": f"{HF}/raw/review_categories/{category}.jsonl",
        "meta": f"{HF}/raw/meta_categories/meta_{category}.jsonl",
    }


def lines(source: str, label: str = "") -> Iterator[bytes]:
    """Lines of a URL or a local file, with a progress line every million."""
    stream = urllib.request.urlopen(source) if source.startswith("http") else open(source, "rb")
    start = time.time()
    with stream:
        for i, line in enumerate(stream, 1):
            if label and i % 1_000_000 == 0:
                print(f"  {label}: {i / 1e6:.0f}M lines, {time.time() - start:.0f}s", flush=True)
            yield line


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# ---- pass 1 ----------------------------------------------------------------


def pick_users(core: Iterator[bytes], n_users: int, min_reviews: int, seed: int) -> tuple[dict[str, int], set[str]]:
    """Sample users with min_reviews+ 5-core reviews. Returns ({user: n_reviews}, 5-core product IDs)."""
    counts: Counter[str] = Counter()
    products: set[str] = set()
    next(core)  # header: user_id,parent_asin,rating,timestamp
    for line in core:
        user, product, *_ = line.decode().split(",")
        counts[user] += 1
        products.add(product)
    eligible = sorted(u for u, c in counts.items() if c >= min_reviews)
    chosen = random.Random(seed).sample(eligible, min(n_users, len(eligible)))
    print(f"  5-core: {len(counts):,} users, {len(eligible):,} with {min_reviews}+ reviews, "
          f"{len(products):,} products; sampled {len(chosen):,} users")
    return {u: counts[u] for u in chosen}, products


# ---- pass 2 ----------------------------------------------------------------


def collect_reviews(raw: Iterator[bytes], conn: sqlite3.Connection, users: set[str], core_products: set[str],
                    per_product: int) -> None:
    """Keep the sampled users' reviews and each 5-core product's top reviews by helpful votes."""
    top: dict[str, list[tuple[int, str, int, bool]]] = {}  # product -> min-heap of (helpful, time, rowid, is_history)
    for line in raw:
        r = json.loads(line)
        pid, uid = r["parent_asin"], r["user_id"]
        is_history = uid in users
        if not is_history and pid not in core_products:
            continue
        ts = iso(r["timestamp"])
        entry_key = (int(r.get("helpful_vote") or 0), ts)
        heap = top.setdefault(pid, []) if pid in core_products else None
        shown = heap is not None and (len(heap) < per_product or entry_key > heap[0][:2])
        if not shown and not is_history:
            continue
        cur = conn.execute(
            "INSERT INTO reviews (product_id, user_id, rating, title, text, timestamp, verified_purchase,"
            " helpful_vote, shown) VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, uid, r["rating"], cut(r.get("title") or "", 200), cut(r.get("text") or "", 2000), ts,
             int(bool(r.get("verified_purchase"))), entry_key[0], int(shown)),
        )
        if shown:
            if len(heap) == per_product:
                *_, evicted, evicted_is_history = heapq.heappop(heap)
                if evicted_is_history:  # still part of a user's history, just no longer on the page
                    conn.execute("UPDATE reviews SET shown = 0 WHERE rowid = ?", (evicted,))
                else:
                    conn.execute("DELETE FROM reviews WHERE rowid = ?", (evicted,))
            heapq.heappush(heap, (*entry_key, cur.lastrowid, is_history))
    # number each product's shown reviews 1..K, most helpful first
    rows = conn.execute(
        "SELECT rowid, product_id FROM reviews WHERE shown = 1 ORDER BY product_id, helpful_vote DESC, timestamp DESC"
    ).fetchall()
    numbering, last, n = [], None, 0
    for rowid, pid in rows:
        n = n + 1 if pid == last else 1
        last = pid
        numbering.append((n, rowid))
    conn.executemany("UPDATE reviews SET n = ? WHERE rowid = ?", numbering)


# ---- pass 3 ----------------------------------------------------------------


def product_row(m: dict) -> tuple:
    details = {k: cut(str(v), 200) for k, v in (m.get("details") or {}).items() if k not in DROP_DETAILS}
    brand = (m.get("details") or {}).get("Brand") or m.get("store")
    return (
        m["parent_asin"], cut(m.get("title") or "", 300), m.get("main_category"), json.dumps(m.get("categories") or []),
        brand, m.get("price") if isinstance(m.get("price"), (int, float)) else None, m.get("average_rating"),
        m.get("rating_number") or 0, json.dumps([cut(f, 300) for f in (m.get("features") or [])[:10]]),
        cut(" ".join(m.get("description") or []), 1000), json.dumps(details),
    )


def collect_products(meta: Iterator[bytes], conn: sqlite3.Connection, wanted: set[str]) -> None:
    for line in meta:
        m = json.loads(line)
        if m.get("parent_asin") in wanted and m.get("title"):
            conn.execute("INSERT OR REPLACE INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?)", product_row(m))


# ---- build -----------------------------------------------------------------


def build(src: dict[str, str], out: Path, n_users: int = 10_000, min_reviews: int = 8,
          per_product: int = 10, seed: int = 0) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    conn = connect(tmp)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")

    print("Pass 1/3: 5-core index")
    users, core_products = pick_users(lines(src["core"], "5-core"), n_users, min_reviews, seed)
    conn.executemany("INSERT INTO users VALUES (?, ?)", users.items())

    print("Pass 2/3: reviews")
    collect_reviews(lines(src["reviews"], "reviews"), conn, set(users), core_products, per_product)
    history_products = {row[0] for row in conn.execute(
        "SELECT DISTINCT product_id FROM reviews WHERE user_id IN (SELECT id FROM users)")}
    conn.commit()

    print("Pass 3/3: product metadata")
    collect_products(lines(src["meta"], "meta"), conn, core_products | history_products)
    conn.execute("DELETE FROM reviews WHERE shown = 1 AND product_id NOT IN (SELECT id FROM products)"
                 " AND user_id NOT IN (SELECT id FROM users)")
    conn.executescript(INDEXES)
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    tmp.replace(out)
    stats = SqliteCatalog(out).stats()
    print(json.dumps(stats, indent=2))
    return stats


def spot_check(db: Path, meta_source: str, n: int = 20, seed: int = 0) -> list[str]:
    """Re-read the source metadata and compare n random stored products field by field. Returns mismatches."""
    catalog = SqliteCatalog(db)
    ids = sorted(catalog.products)
    sample = set(random.Random(seed).sample(ids, min(n, len(ids))))
    problems, seen = [], 0
    for line in lines(meta_source, "meta"):
        m = json.loads(line)
        if m.get("parent_asin") not in sample:
            continue
        seen += 1
        p = catalog.products[m["parent_asin"]]
        expected = product_row(m)
        got = (p.id, p.title, p.main_category, json.dumps(p.categories), p.brand, p.price, p.average_rating,
               p.rating_number, json.dumps(p.features), p.description, json.dumps(p.details))
        problems += [f"{p.id}: {name} differs" for name, a, b in zip(
            ["id", "title", "main_category", "categories", "brand", "price", "average_rating", "rating_number",
             "features", "description", "details"], expected, got) if a != b]
        if seen == len(sample):
            break
    if seen < len(sample):
        problems.append(f"only {seen} of {len(sample)} sampled products found in the source")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--category", default="Beauty_and_Personal_Care")
    parser.add_argument("--out", type=Path, default=DEFAULT_DB)
    parser.add_argument("--users", type=int, default=10_000, help="users with 8+ reviews to sample")
    parser.add_argument("--reviews-per-product", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spot-check", type=int, metavar="N", help="only compare N stored products with the source")
    args = parser.parse_args()
    src = sources(args.category)
    if args.spot_check:
        problems = spot_check(args.out, src["meta"], args.spot_check)
        print("\n".join(problems) or f"All {args.spot_check} products match the source.")
        return
    build(src, args.out, args.users, per_product=args.reviews_per_product, seed=args.seed)


if __name__ == "__main__":
    main()
