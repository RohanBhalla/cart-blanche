"""Run configuration: one YAML per run, switching each component on or off."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from cartblanche.schemas import AttackType

# Agent and judge come from different model families so the judge never grades its own family.
DEFAULT_AGENT_MODEL = "gemini/gemini-2.5-flash-lite"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.4-nano"


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
    agent_model: str = Field(default_factory=lambda: os.getenv("CARTBLANCHE_AGENT_MODEL", DEFAULT_AGENT_MODEL))
    judge_model: str = Field(default_factory=lambda: os.getenv("CARTBLANCHE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
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
        if self.agent == AgentKind.RULES:
            on = [
                name
                for name, off in [
                    ("intent_parser", False),
                    ("personalization", Personalization.NONE),
                    ("relationship", RelationshipMode.NONE),
                    ("extractor", False),
                    ("labeling", Labeling.NONE),
                    ("verifier", False),
                    ("clarify", False),
                ]
                if getattr(self, name) != off
            ]
            if on:
                raise ValueError(f"rules agent cannot use LLM components: {on}")
        return self


def load_config(path: str | Path) -> RunConfig:
    load_dotenv()  # so CARTBLANCHE_*_MODEL overrides in .env apply
    return RunConfig(**yaml.safe_load(Path(path).read_text()))
