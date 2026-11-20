from types import SimpleNamespace

from cartblanche.agent.history import describe, history_block, summarize_profile, tag_reasons
from cartblanche.schemas import HistoryItem


def make(title, rating=5, review=""):
    return HistoryItem(
        product_id=title, title=title, main_category="All Beauty",
        brand="Acme", rating=rating, review_text=review, timestamp="2023-01-05",
    )


ITEMS = [
    make("SPF 50 sunscreen spray", rating=1, review="stings my eyes and my face broke out"),
    make("Rose body lotion", review="smells amazing"),
]


class FakeLLM:
    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, messages, tools=None, tag="", **kwargs):
        self.calls.append((messages, tag))
        return SimpleNamespace(content=self.text)


def test_describe_shows_the_reason_when_given():
    line = describe(make("Lip balm", rating=2, review="dried my lips"), "quality")
    assert line == "2023-01-05 | Lip balm | Acme | rated 2 of 5 | reason: quality | review: dried my lips"


def test_history_block_puts_each_reason_on_its_own_line():
    block = history_block(ITEMS, ["reaction", "scent or shade"])
    first, second = block.splitlines()
    assert "reason: reaction" in first and "reason: scent or shade" in second


def test_one_call_labels_every_purchase():
    llm = FakeLLM('{"reasons": ["reaction", "scent or shade"]}')
    assert tag_reasons(ITEMS, llm, tag="t1") == ["reaction", "scent or shade"]
    assert len(llm.calls) == 1 and llm.calls[0][1] == "t1/tags"


def test_other_and_unknown_labels_are_dropped():
    llm = FakeLLM('{"reasons": ["Other", "price"]}')
    assert tag_reasons(ITEMS, llm) == [None, None]
    assert tag_reasons(ITEMS, FakeLLM('{"reasons": ["Reaction ", "other"]}')) == ["reaction", None]


def test_a_purchase_without_a_label_is_shown_without_a_reason():
    block = history_block(ITEMS, ["reaction", None])
    first, second = block.splitlines()
    assert "reason: reaction" in first and "reason" not in second


def test_no_history_means_no_model_call():
    llm = FakeLLM("never used")
    assert tag_reasons([], llm) == [] and llm.calls == []


def test_wrong_number_of_labels_drops_every_label():
    assert tag_reasons(ITEMS, FakeLLM('{"reasons": ["reaction"]}')) == [None, None]


def test_a_reply_without_json_drops_every_label():
    assert tag_reasons(ITEMS, FakeLLM("sorry, I cannot do that")) == [None, None]


def test_profile_writer_sees_the_reasons():
    llm = FakeLLM("Sensitive skin.")
    summarize_profile(ITEMS, llm, reasons=["reaction", "scent or shade"])
    assert "reason: reaction" in llm.calls[0][0][1]["content"]