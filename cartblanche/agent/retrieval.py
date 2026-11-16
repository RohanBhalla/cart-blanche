"""Rank texts by how close their meaning is to a query (used to pick relevant history)."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # CPU on purpose: the Mac GPU backend crashes on some machines, and this model is small.
    return SentenceTransformer(MODEL_NAME, device="cpu")


def rank_by_similarity(query: str, texts: list[str], k: int = 10) -> list[int]:
    """Positions of the k texts closest in meaning to the query, best first."""
    if not texts:
        return []
    model = _model()
    scores = util.cos_sim(model.encode(query), model.encode(texts))[0]
    order = scores.argsort(descending=True)
    return [int(i) for i in order[:k]]