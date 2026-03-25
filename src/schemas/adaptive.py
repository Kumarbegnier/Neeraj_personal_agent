from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .context import (
    ApprovalState,
    ContextMemorySlice,
    ContextPacket,
    ProviderRouteHint,
    ProviderRoutingHints,
    ToolAvailabilitySnapshot,
    ToolCapability,
)


class ArchitectureMode(str, Enum):
    DIRECT_SINGLE_AGENT = "direct_single_agent"
    PLANNER_EXECUTOR = "planner_executor"
    MULTI_AGENT_RESEARCH = "multi_agent_research"
    BROWSER_HEAVY_VERIFIED = "browser_heavy_verified"
    COMMUNICATION_CRITIC = "communication_critic"


TaskIntensity = Literal["low", "moderate", "high"]
TaskRiskLevel = Literal["low", "medium", "high", "critical"]


class TaskCharacteristics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    complexity: TaskIntensity = "low"
    grounding_need: TaskIntensity = "low"
    tool_intensity: TaskIntensity = "low"
    risk_level: TaskRiskLevel = "low"
    parallelizability: TaskIntensity = "low"
    communication_intensity: TaskIntensity = "low"
    research_intensity: TaskIntensity = "low"
    browser_intensity: TaskIntensity = "low"


class ArchitectureReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_pattern: str
    summary: str
    decisive_factors: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    pattern_scores: dict[str, float] = Field(default_factory=dict)


class ArchitectureDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ArchitectureMode
    rationale: str
    reasoning: ArchitectureReasoning
    task_characteristics: TaskCharacteristics
    pattern_label: str = ""
    primary_agent: str = "general"
    supporting_agents: list[str] = Field(default_factory=list)
    requires_planning: bool = False
    requires_verifier: bool = True
    browser_heavy: bool = False
    critic_lane: bool = False
    parallel_fanout: int = 1
    loop_strategy: str = "react"
    stop_conditions: list[str] = Field(default_factory=list)


class LoopPhase(str, Enum):
    OBSERVE = "observe"
    SELECT_ARCHITECTURE = "select_architecture"
    CONTROL = "control"
    PLAN = "plan"
    ROUTE = "route"
    AGENT_DECIDE = "agent_decide"
    REASON = "reason"
    ACT = "act"
    VERIFY = "verify"
    REFLECT = "reflect"
    UPDATE_MEMORY = "update_memory"
    LOOP_CONTROL = "loop_control"
    COMPLETE = "complete"


class LoopState(BaseModel):
    iteration: int = 0
    max_iterations: int = 4
    retry_budget: int = 2
    retry_count: int = 0
    phase: LoopPhase = LoopPhase.OBSERVE
    architecture_mode: ArchitectureMode | None = None
    active_agent: str | None = None
    should_replan: bool = False
    ready_for_response: bool = False
    stop_conditions: list[str] = Field(
        default_factory=lambda: [
            "goal_achieved",
            "max_steps_reached",
            "retry_budget_exhausted",
            "stalled",
        ]
    )
    last_stop_trigger: str = "continue"
    last_stop_reason: str = ""
    last_verification_status: str = "pending"
    last_reflection_status: str = "pending"
    handoff_available: bool = False
    last_handoff_id: str | None = None
    compaction_applied: bool = False
    completed_phases: list[LoopPhase] = Field(default_factory=list)
    memory_checkpoints: list[str] = Field(default_factory=list)


ApprovalRiskState = ApprovalState
