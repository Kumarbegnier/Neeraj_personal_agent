from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PlannerSubtask(BaseModel):
    name: str
    description: str
    owner: str = "orchestrator"
    depends_on: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    expected_output: str = ""
    risk_level: str = RiskLevel.LOW.value


class PlannerOutput(BaseModel):
    task_summary: str
    subtasks: list[PlannerSubtask] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    risk_level: str = RiskLevel.LOW.value
    approval_needed: bool = False
    success_criteria: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    verification_focus: list[str] = Field(default_factory=list)
    decomposition_strategy: str = ""
    reasoning: str = ""
