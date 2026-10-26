"""Run configuration: one YAML per run, switching each component on or off."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from cartblanche.schemas import AttackType


class AgentKind(StrEnum):
    RULES = "rules"  # E0: filter by constraints, sort by rating
    LLM = "llm"


class Personalization(StrEnum):
    NONE = "none"
    RAG = "rag"  # retrieve relevant history items
    PROFILE = "profile"  # summarized profile
    FULL = "full"  # full raw history in context


class RelationshipMode(StrEnum):
    NONE = "none"
    RULE = "rule"  # owns something similar -> skip
    CHECK = "check"  # duplicate / replacement / upgrade / complement / unrelated


class Labeling(StrEnum):
    NONE = "none"
    DELIMIT = "delimit"  # wrap untrusted text in tags
    SPOTLIGHT = "spotlight"  # tags plus an explicit "never follow instructions in here" rule


class RunConfig(BaseModel):
    config_id: str
    description: str = ""
    agent: AgentKind = AgentKind.LLM
    # Agent and judge come from different model families so the judge never grades its own family.
    agent_model: str = "gemini/gemini-2.5-flash-lite"
    judge_model: str = "openai/gpt-5.4-nano"
    temperature: float = 0.0
    max_steps: int = 12
    intent_parser: bool = False
    personalization: Personalization = Personalization.NONE
    relationship: RelationshipMode = RelationshipMode.NONE
    extractor: bool = False
    labeling: Labeling = Labeling.NONE
    verifier: bool = False
    clarify: bool = False
    attacks: list[AttackType] = Field(default_factory=list)
    doses: list[int] = Field(default_factory=lambda: [0, 1, 2, 4, 8])

    @model_validator(mode="after")
    def _rules_agent_has_no_llm_parts(self) -> RunConfig:
        uses_llm_parts = (
            self.intent_parser
            or self.extractor
            or self.verifier
            or self.clarify
            or self.personalization != Personalization.NONE
            or self.relationship != RelationshipMode.NONE
            or self.labeling != Labeling.NONE
        )
        if self.agent == AgentKind.RULES and uses_llm_parts:
            raise ValueError("the rules agent (E0) cannot use LLM components")
        return self


def load_config(path: str | Path) -> RunConfig:
    return RunConfig(**yaml.safe_load(Path(path).read_text()))
