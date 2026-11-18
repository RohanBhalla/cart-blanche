from cartblanche.agent.history import describe, history_block, relevant_history
from cartblanche.schemas import HistoryItem


def make(title, rating=5, review="", brand="Acme", date="2023-01-05"):
    return HistoryItem(
        product_id=title, title=title, main_category="All Beauty",
        brand=brand, rating=rating, review_text=review, timestamp=date,
    )


ITEMS = [
    make("SPF 50 sunscreen spray", review="stings my eyes"),
    make("Rich moisturizing cream for dry skin", review="my skin feels soft all day"),
    make("Volumizing shampoo", review="fine hair loves it"),
]


def test_relevant_history_puts_the_closest_purchase_first():
    best = relevant_history("a moisturizer for dry skin", ITEMS, k=1)
    assert [i.title for i in best] == ["Rich moisturizing cream for dry skin"]


def test_relevant_history_returns_everything_when_k_is_large():
    assert len(relevant_history("anything", ITEMS, k=10)) == 3


def test_describe_shows_date_brand_rating_and_review():
    line = describe(make("Lip balm", rating=2, review="dried my lips", brand="Zed"))
    assert line == "2023-01-05 | Lip balm | Zed | rated 2 of 5 | review: dried my lips"


def test_history_block_numbers_the_items():
    block = history_block(ITEMS[:2])
    assert block.startswith("1. ") and "\n2. " in block


def test_history_block_says_so_when_there_is_no_history():
    assert history_block([]) == "The shopper has no past purchases on record."