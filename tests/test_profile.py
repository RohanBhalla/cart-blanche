from types import SimpleNamespace

import pytest

from cartblanche.agent.history import history_block, summarize_profile
from cartblanche.schemas import HistoryItem


def make(title, rating=5, review=""):
    return HistoryItem(
        product_id=title, title=title, main_category="All Beauty",
        brand="Acme", rating=rating, review_text=review, timestamp="2023-01-05",
    )


ITEMS = [
    make("Rich moisturizing cream", review="my dry skin feels soft"),
    make("SPF 50 sunscreen spray", rating=1, review="stings my eyes"),
]


class FakeLLM:
    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, messages, tools=None, tag="", **kwargs):
        self.calls.append((messages, tag))
        return SimpleNamespace(content=self.text)


def test_profile_is_written_from_the_history():
    llm = FakeLLM("  Dry skin. Likes fragrance free creams.  ")
    assert summarize_profile(ITEMS, llm, tag="t1") == "Dry skin. Likes fragrance free creams."
    messages, tag = llm.calls[0]
    assert tag == "t1/profile"
    assert messages[1]["content"] == history_block(ITEMS)


def test_profile_without_history_makes_no_model_call():
    llm = FakeLLM("never used")
    assert summarize_profile([], llm) == "The shopper has no past purchases on record."
    assert llm.calls == []


def test_empty_profile_reply_is_an_error():
    with pytest.raises(ValueError):
        summarize_profile(ITEMS, FakeLLM("   "))