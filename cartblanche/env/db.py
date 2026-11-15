"""The real store: a SQLite file built by env/build_db.py, read through the same
interface as the in-memory Catalog (catalog.products, catalog.reviews, catalog.users),
so TaskStore, the tools and the metrics work unchanged.

Tables
  products  one row per product (parent_asin), metadata trimmed
  reviews   every kept review: the sampled users' full histories, plus up to
            K reviews per product shown on its page (shown = 1, numbered n = 1..K
            by helpful votes, so their IDs are "<product>.review.<n>")
  users     the sampled users (8+ reviews)
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path

from cartblanche.schemas import HistoryItem, Product, Review, User, review_id

DEFAULT_DB = Path("data/cartblanche.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, main_category TEXT, categories TEXT, brand TEXT,
    price REAL, average_rating REAL, rating_number INTEGER, features TEXT, description TEXT, details TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    product_id TEXT NOT NULL, user_id TEXT NOT NULL, rating REAL, title TEXT, text TEXT, timestamp TEXT,
    verified_purchase INTEGER, helpful_vote INTEGER, shown INTEGER NOT NULL DEFAULT 0, n INTEGER
);
CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, n_reviews INTEGER);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS reviews_by_product ON reviews (product_id, shown, n);
CREATE INDEX IF NOT EXISTS reviews_by_user ON reviews (user_id, timestamp);
"""


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _product(row: sqlite3.Row) -> Product:
    return Product(
        id=row["id"], title=row["title"], main_category=row["main_category"] or "",
        categories=json.loads(row["categories"] or "[]"), brand=row["brand"], price=row["price"],
        average_rating=row["average_rating"], rating_number=row["rating_number"] or 0,
        features=json.loads(row["features"] or "[]"), description=row["description"] or "",
        details=json.loads(row["details"] or "{}"),
    )


class _Table(Mapping):
    """Read-only, cached dict view over one query. Missing keys raise KeyError, like a dict."""

    def __init__(self, conn: sqlite3.Connection, load, keys_sql: str):
        self._conn = conn
        self._get = lru_cache(maxsize=20_000)(load)
        self._keys_sql = keys_sql

    def __getitem__(self, key):
        value = self._get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return (row[0] for row in self._conn.execute(self._keys_sql))

    def __len__(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM ({self._keys_sql})").fetchone()[0]


class SqliteCatalog:
    def __init__(self, path: Path = DEFAULT_DB):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found; build it with: python -m cartblanche.env.build_db")
        self.conn = connect(path)
        self.products = _Table(self.conn, self._load_product, "SELECT id FROM products")
        self.reviews = _Table(self.conn, self._load_reviews, "SELECT DISTINCT product_id FROM reviews WHERE shown = 1")
        self.users = _Table(self.conn, self._load_user, "SELECT id FROM users")

    def _load_product(self, pid: str) -> Product | None:
        row = self.conn.execute("SELECT * FROM products WHERE id = ?", (pid,)).fetchone()
        return _product(row) if row else None

    def _load_reviews(self, pid: str) -> list[Review] | None:
        rows = self.conn.execute(
            "SELECT * FROM reviews WHERE product_id = ? AND shown = 1 ORDER BY n", (pid,)
        ).fetchall()
        if not rows:
            return None
        return [
            Review(id=review_id(pid, r["n"]), product_id=pid, user_id=r["user_id"], rating=r["rating"],
                   title=r["title"] or "", text=r["text"] or "", timestamp=r["timestamp"],
                   verified_purchase=bool(r["verified_purchase"]), helpful_vote=r["helpful_vote"] or 0)
            for r in rows
        ]

    def _load_user(self, uid: str) -> User | None:
        if not self.conn.execute("SELECT 1 FROM users WHERE id = ?", (uid,)).fetchone():
            return None
        rows = self.conn.execute(
            """SELECT r.*, p.title AS p_title, p.main_category, p.categories, p.brand, p.price
               FROM reviews r LEFT JOIN products p ON p.id = r.product_id
               WHERE r.user_id = ? ORDER BY r.timestamp""",
            (uid,),
        ).fetchall()
        history = [
            HistoryItem(product_id=r["product_id"], title=r["p_title"] or "", main_category=r["main_category"] or "",
                        categories=json.loads(r["categories"] or "[]"), brand=r["brand"], price=r["price"],
                        rating=r["rating"], review_title=r["title"] or "", review_text=r["text"] or "",
                        timestamp=r["timestamp"])
            for r in rows
        ]
        return User(id=uid, history=history)

    def stats(self) -> dict:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "products": q("SELECT COUNT(*) FROM products"),
            "products_with_price": q("SELECT COUNT(*) FROM products WHERE price IS NOT NULL"),
            "products_with_shown_reviews": q("SELECT COUNT(DISTINCT product_id) FROM reviews WHERE shown = 1"),
            "reviews": q("SELECT COUNT(*) FROM reviews"),
            "shown_reviews": q("SELECT COUNT(*) FROM reviews WHERE shown = 1"),
            "users": q("SELECT COUNT(*) FROM users"),
            "history_reviews": q("SELECT COUNT(*) FROM reviews WHERE user_id IN (SELECT id FROM users)"),
        }
