"""Shared schemas for agent outputs and golden labels."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from spotify_agent.intents import INTENTS


Decision = Literal["auto", "escalate"]


class AgentOutput(BaseModel):
    intent: str
    reply: str
    decision: Decision
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    retrieved_ids: List[str] = Field(default_factory=list)

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, value: str) -> str:
        if value not in INTENTS:
            return "other"
        return value


class GoldenExample(BaseModel):
    id: str
    customer_tweet_id: str
    customer_text: str
    brand_reply_text: str
    intent: str
    escalate: bool
    stratum: str = "random"
    notes: str = ""

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, value: str) -> str:
        if value not in INTENTS:
            raise ValueError(f"Unknown intent: {value}")
        return value


class JudgeScore(BaseModel):
    groundedness: int = Field(ge=1, le=5)
    brand_voice: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    pass_fail: Literal["pass", "fail"]
    rationale: str = ""

    @property
    def mean_score(self) -> float:
        return (
            self.groundedness + self.brand_voice + self.helpfulness + self.safety
        ) / 4.0


def agent_output_to_dict(output: AgentOutput) -> Dict[str, Any]:
    return output.model_dump()


def parse_agent_output(payload: Dict[str, Any]) -> AgentOutput:
    return AgentOutput.model_validate(payload)
