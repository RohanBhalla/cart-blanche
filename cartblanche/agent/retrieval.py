"""Rank texts by how close their meaning is to a query (used to pick relevant history).

Each history item's vector is computed once and saved on disk (cache/embeddings/), keyed by
the model name and the item's text, so later tasks and configs reuse it. Only the request is
embedded fresh. Callers pass only the shopper's visible history (their own purchases before
the task's as_of), so the ranking never sees other users or the held-out purchase.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = Path("cache/embeddings")


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # CPU on purpose: the Mac GPU backend crashes on some machines, and this model is small.
    return SentenceTransformer(MODEL_NAME, device="cpu")


def _cache_file(text: str, cache_dir: Path) -> Path:
    key = hashlib.sha256(f"{MODEL_NAME}\n{text}".encode()).hexdigest()
    return cache_dir / f"{key}.npy"


def _save(path: Path, vector: np.ndarray) -> None:
    """Write then rename, so a run working in parallel never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.save(f, vector)
    os.replace(tmp, path)


def embed_items(texts: list[str], cache_dir: Path | None = None) -> np.ndarray:
    """One vector per text, read from the disk cache when present; only new texts are encoded."""
    cache_dir = cache_dir or CACHE_DIR
    vectors: list[np.ndarray | None] = [None] * len(texts)
    missing = []
    for i, text in enumerate(texts):
        path = _cache_file(text, cache_dir)
        if path.exists():
            vectors[i] = np.load(path)
        else:
            missing.append(i)
    if missing:
        new = _model().encode([texts[i] for i in missing])
        for i, vector in zip(missing, new):
            vectors[i] = vector
            _save(_cache_file(texts[i], cache_dir), vector)
    return np.stack(vectors)


def embed_query(text: str) -> np.ndarray:
    """A request's vector. Not cached: each request is embedded once per task."""
    return _model().encode(text)


def rank_by_similarity(query: str, texts: list[str], k: int = 10, cache_dir: Path | None = None) -> list[int]:
    """Positions of the k texts closest in meaning to the query, best first."""
    if not texts:
        return []
    scores = util.cos_sim(embed_query(query), embed_items(texts, cache_dir))[0]
    order = scores.argsort(descending=True)
    return [int(i) for i in order[:k]]
