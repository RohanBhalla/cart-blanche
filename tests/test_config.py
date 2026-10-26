from pathlib import Path

import pytest
from pydantic import ValidationError

from cartblanche.config import AgentKind, RunConfig, load_config

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


def test_all_experiment_configs_load():
    ids = {load_config(p).config_id for p in CONFIGS}
    assert ids == {"E0", "E1", "E2", "E3", "E3b", "E3c", "E4", "E4b", "E5", "E6", "E7", "E8"}


def test_each_step_adds_on_previous():
    c = {load_config(p).config_id: load_config(p) for p in CONFIGS}
    assert c["E0"].agent == AgentKind.RULES
    assert not c["E1"].intent_parser and c["E2"].intent_parser
    assert c["E7"].verifier and not c["E6"].verifier
    assert c["E8"].clarify and not c["E7"].clarify


def test_rules_agent_rejects_llm_components():
    with pytest.raises(ValidationError):
        RunConfig(config_id="bad", agent="rules", verifier=True)


def test_agent_and_judge_are_different_families(monkeypatch):
    monkeypatch.delenv("CARTBLANCHE_AGENT_MODEL", raising=False)
    monkeypatch.delenv("CARTBLANCHE_JUDGE_MODEL", raising=False)
    c = RunConfig(config_id="x")
    assert c.agent_model.split("/")[0] != c.judge_model.split("/")[0]


def test_model_names_overridable_from_env(monkeypatch):
    monkeypatch.setenv("CARTBLANCHE_JUDGE_MODEL", "openai/gpt-5.4-mini")
    assert RunConfig(config_id="x").judge_model == "openai/gpt-5.4-mini"
    assert RunConfig(config_id="x", judge_model="openai/other").judge_model == "openai/other"
