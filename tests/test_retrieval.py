from cartblanche.agent.retrieval import rank_by_similarity

TEXTS = [
    "SPF 50 sunscreen spray for face",
    "CeraVe Daily Moisturizing Lotion for dry skin, fragrance free",
    "Lavender scented body wash",
]


def test_best_match_comes_first():
    assert rank_by_similarity("a fragrance free moisturizer for dry skin", TEXTS)[0] == 1


def test_k_limits_the_results():
    assert len(rank_by_similarity("moisturizer", TEXTS, k=2)) == 2


def test_empty_list_gives_empty_result():
    assert rank_by_similarity("anything", []) == []