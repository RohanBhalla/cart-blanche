import numpy as np

import cartblanche.agent.retrieval as retrieval
from cartblanche.agent.retrieval import embed_items, rank_by_similarity

TEXTS = [
    "SPF 50 sunscreen spray for face",
    "CeraVe Daily Moisturizing Lotion for dry skin, fragrance free",
    "Lavender scented body wash",
]


def test_best_match_comes_first(tmp_path):
    assert rank_by_similarity("a fragrance free moisturizer for dry skin", TEXTS, cache_dir=tmp_path)[0] == 1


def test_k_limits_the_results(tmp_path):
    assert len(rank_by_similarity("moisturizer", TEXTS, k=2, cache_dir=tmp_path)) == 2


def test_empty_list_gives_empty_result(tmp_path):
    assert rank_by_similarity("anything", [], cache_dir=tmp_path) == []


def test_item_vectors_are_computed_once_and_reused(tmp_path, monkeypatch):
    encoded = []
    real = retrieval._model()

    class Counting:
        def encode(self, texts):
            encoded.extend([texts] if isinstance(texts, str) else texts)
            return real.encode(texts)

    monkeypatch.setattr(retrieval, "_model", lambda: Counting())
    first = embed_items(TEXTS[:2], tmp_path)
    assert encoded == TEXTS[:2] and len(list(tmp_path.glob("*.npy"))) == 2

    encoded.clear()
    second = embed_items(TEXTS, tmp_path)  # two cached, one new
    assert encoded == [TEXTS[2]]
    assert np.allclose(second[:2], first)


def test_cached_ranking_matches_uncached(tmp_path):
    query = "something for dry skin"
    fresh = rank_by_similarity(query, TEXTS, cache_dir=tmp_path / "a")  # computes and saves
    assert rank_by_similarity(query, TEXTS, cache_dir=tmp_path / "a") == fresh  # read back from disk
    assert rank_by_similarity(query, TEXTS, cache_dir=tmp_path / "b") == fresh  # a fresh cache agrees
