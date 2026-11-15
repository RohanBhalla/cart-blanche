import json

from cartblanche.llm import LLM


class FakeCompletion:
    """Returns canned OpenAI-format responses and counts calls."""

    def __init__(self, tool_call: bool = False):
        self.calls = 0
        self.tool_call = tool_call

    def __call__(self, **request):
        self.calls += 1
        message = {"role": "assistant", "content": f"reply {self.calls}"}
        if self.tool_call:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_product", "arguments": '{"product_id": "p01"}'}}
                ],
            }
        return {"choices": [{"message": message}], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def make(tmp_path, fake, **kw):
    return LLM(model="fake/model", cache_dir=tmp_path / "cache", log_path=tmp_path / "calls.jsonl",
               completion_fn=fake, cost_fn=lambda raw: 0.001, **kw)


def test_cache_makes_reruns_free(tmp_path):
    fake = FakeCompletion()
    llm = make(tmp_path, fake)
    msgs = [{"role": "user", "content": "hi"}]
    first = llm.complete(msgs)
    second = llm.complete(msgs)
    assert fake.calls == 1
    assert not first.cached and second.cached
    assert second.content == first.content
    assert first.cost_usd == second.cost_usd == 0.001  # a cached reply keeps its original cost...
    assert llm.usage.spent_usd == 0.001  # ...but only the first call was paid for
    # a fresh wrapper over the same cache directory also hits
    assert make(tmp_path, fake).complete(msgs).cached and fake.calls == 1


def test_different_requests_miss_cache(tmp_path):
    fake = FakeCompletion()
    llm = make(tmp_path, fake)
    llm.complete([{"role": "user", "content": "a"}])
    llm.complete([{"role": "user", "content": "b"}])
    llm.complete([{"role": "user", "content": "a"}], tools=[{"type": "function", "function": {"name": "x"}}])
    assert fake.calls == 3


def test_usage_and_log(tmp_path):
    llm = make(tmp_path, FakeCompletion())
    msgs = [{"role": "user", "content": "hi"}]
    llm.complete(msgs, tag="t001/agent")
    llm.complete(msgs, tag="t001/agent")
    assert llm.usage.calls == 2 and llm.usage.cached_calls == 1
    assert llm.usage.prompt_tokens == 200 and llm.usage.completion_tokens == 40
    assert abs(llm.usage.cost_usd - 0.002) < 1e-9 and abs(llm.usage.spent_usd - 0.001) < 1e-9
    lines = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [line["cached"] for line in lines] == [False, True]
    assert lines[0]["tag"] == "t001/agent" and lines[0]["prompt_tokens"] == 100


def test_tool_calls_parsed(tmp_path):
    r = make(tmp_path, FakeCompletion(tool_call=True)).complete([{"role": "user", "content": "hi"}])
    assert r.content is None
    assert [(c.name, json.loads(c.arguments)) for c in r.tool_calls] == [("get_product", {"product_id": "p01"})]
    assert r.message["tool_calls"][0]["function"]["name"] == "get_product"
    assert r.tokens == 120


def test_no_cache_dir_always_calls(tmp_path):
    fake = FakeCompletion()
    llm = LLM(model="fake/model", cache_dir=None, completion_fn=fake, cost_fn=lambda raw: 0.0)
    msgs = [{"role": "user", "content": "hi"}]
    llm.complete(msgs)
    llm.complete(msgs)
    assert fake.calls == 2


def test_real_litellm_response_objects_parse(tmp_path):
    """Exercise the real LiteLLM path offline via its mock_response feature."""
    llm = LLM(model="gemini/gemini-3.5-flash-lite", cache_dir=tmp_path / "cache")
    r = llm.complete([{"role": "user", "content": "hi"}], mock_response="hello from mock")
    assert r.content == "hello from mock" and not r.cached
    again = llm.complete([{"role": "user", "content": "hi"}], mock_response="hello from mock")
    assert again.cached and again.content == "hello from mock"


class RateLimitError(Exception):  # same name LiteLLM uses; the wrapper matches by name
    pass


class Flaky:
    """Fails with the given errors first, then answers."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self, **request):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}


def test_rate_limits_are_retried_with_backoff(tmp_path):
    waits = []
    fake = Flaky(RateLimitError("429 per minute"), RateLimitError("429 per minute"))
    llm = make(tmp_path, fake, sleep=waits.append, retry_wait_s=10)
    assert llm.complete([{"role": "user", "content": "hi"}]).content == "ok"
    assert fake.calls == 3 and waits == [10, 20]


def test_retries_give_up_eventually(tmp_path):
    import pytest

    fake = Flaky(*[RateLimitError("429")] * 3)
    llm = make(tmp_path, fake, sleep=lambda s: None, max_retries=2)
    with pytest.raises(RateLimitError):
        llm.complete([{"role": "user", "content": "hi"}])
    assert fake.calls == 3


def test_daily_quota_stops_immediately(tmp_path):
    import pytest

    from cartblanche.llm import QuotaExhausted

    fake = Flaky(RateLimitError("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    llm = make(tmp_path, fake, sleep=lambda s: pytest.fail("should not wait"))
    with pytest.raises(QuotaExhausted):
        llm.complete([{"role": "user", "content": "hi"}])
    assert fake.calls == 1


def test_other_errors_are_not_retried(tmp_path):
    import pytest

    fake = Flaky(ValueError("bad request"))
    with pytest.raises(ValueError):
        make(tmp_path, fake, sleep=lambda s: pytest.fail("should not wait")).complete([{"role": "user", "content": "x"}])
    assert fake.calls == 1


def test_min_gap_between_real_calls_only(tmp_path):
    waits = []
    llm = make(tmp_path, FakeCompletion(), sleep=waits.append, min_gap_s=4)
    llm.complete([{"role": "user", "content": "a"}])
    llm.complete([{"role": "user", "content": "b"}])  # real call right after: waits
    llm.complete([{"role": "user", "content": "a"}])  # cached: no wait
    assert len(waits) == 1 and 3 < waits[0] <= 4
