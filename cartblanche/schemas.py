"""Shared data contracts.

Every layer talks through these models: the store (Role A) returns Products,
Reviews and HistoryItems; tasks (A writes, C reads) are Task; the agent
(B writes, C scores) returns AgentResult.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Action(StrEnum):
    BUY = "BUY"
    SKIP = "SKIP"
    ALTERNATIVE = "ALTERNATIVE"


class TaskType(StrEnum):
    BUY = "buy"
    DONT_BUY = "dont_buy"


class Relationship(StrEnum):
    """How the requested item relates to what the user already owns."""

    DUPLICATE = "duplicate"
    REPLACEMENT = "replacement"
    UPGRADE = "upgrade"
    COMPLEMENT = "complement"
    SECOND_UNIT = "second_unit"
    UNRELATED = "unrelated"


class AttackType(StrEnum):
    FAKE_REVIEW = "fake_review"
    HIDDEN_INSTRUCTION = "hidden_instruction"
    URGENCY = "urgency"
    FAKE_OVERRIDE = "fake_override"
    PLANTED_PREFERENCE = "planted_preference"


class Split(StrEnum):
    DEV = "dev"
    TEST = "test"


class Product(BaseModel):
    """Mirrors the Amazon Reviews 2023 item metadata we keep."""

    id: str
    title: str
    main_category: str
    categories: list[str] = Field(default_factory=list)
    brand: str | None = None
    price: float | None = None
    average_rating: float | None = None
    rating_number: int = 0
    features: list[str] = Field(default_factory=list)
    description: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class Review(BaseModel):
    id: str
    product_id: str
    user_id: str
    rating: float
    title: str = ""
    text: str = ""
    timestamp: str
    verified_purchase: bool = True
    helpful_vote: int = 0


class HistoryItem(BaseModel):
    """One past purchase. Reviews stand in for purchases."""

    product_id: str
    title: str
    main_category: str
    categories: list[str] = Field(default_factory=list)
    brand: str | None = None
    price: float | None = None
    rating: float
    review_title: str = ""
    review_text: str = ""
    timestamp: str


class User(BaseModel):
    id: str
    history: list[HistoryItem] = Field(default_factory=list)


class Constraints(BaseModel):
    max_price: float | None = None
    size: str | None = None  # e.g. "M", "32", "10"; must be in the product's details["sizes"]
    min_rating: float | None = None
    required_features: list[str] = Field(default_factory=list)


class AttackSpec(BaseModel):
    type: AttackType
    target: str  # product id the attack promotes
    dose: int = Field(ge=0)  # 0 = genuine reviews only
    params: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    task_id: str
    user_id: str
    request: str
    type: TaskType
    as_of: str  # history after this timestamp is hidden from the agent
    shelf: list[str]  # products search_products can see
    candidates: list[str]  # scored subset of the shelf (gold + hard distractors)
    gold_action: Action
    gold_pick: str | None = None  # None when the gold action is SKIP
    acceptable: list[str] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    relationship: Relationship | None = None
    override: str | None = None  # user follow-up after a skip, e.g. "buy it anyway"
    hidden_profile: dict[str, Any] | None = None  # only the simulated user sees this
    attack: AttackSpec | None = None
    split: Split = Split.DEV

    @model_validator(mode="after")
    def _check_consistency(self) -> Task:
        if not set(self.candidates) <= set(self.shelf):
            raise ValueError("candidates must be a subset of shelf")
        if self.gold_action in (Action.BUY, Action.ALTERNATIVE):
            if self.gold_pick is None:
                raise ValueError(f"{self.gold_action} task needs a gold_pick")
            if self.gold_pick not in self.candidates:
                raise ValueError("gold_pick must be among candidates")
            if self.gold_pick not in self.acceptable:
                self.acceptable = [self.gold_pick, *self.acceptable]
        elif self.gold_pick is not None:
            raise ValueError(f"{self.gold_action} task must not have a gold_pick")
        if not set(self.acceptable) <= set(self.candidates):
            raise ValueError("acceptable must be a subset of candidates")
        if self.attack is not None:
            if self.attack.target not in self.shelf:
                raise ValueError("attack target must be on the shelf")
            if self.attack.target in self.acceptable:
                raise ValueError("attack target must not be an acceptable answer")
        return self


class AgentResult(BaseModel):
    task_id: str
    config_id: str
    action: Action
    pick: str | None = None
    reasons: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    question: str | None = None  # clarifying question asked, if any
    trace: list[dict[str, Any]] = Field(default_factory=list)
    tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None  # set when the agent failed to produce valid output


def spec_evidence_id(product_id: str, field: str) -> str:
    """Evidence ID for a product fact, e.g. 'p07.spec.price'."""
    return f"{product_id}.spec.{field}"


def review_id(product_id: str, n: int) -> str:
    """Review IDs are opaque: injected reviews use the same format as genuine ones."""
    return f"{product_id}.review.{n}"
