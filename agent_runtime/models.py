from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from src.schemas.adaptive import ArchitectureDecision, LoopPhase, LoopState
from src.schemas.context import ContextPacket
from src.schemas.handoff import HandoffPacket
from src.schemas.observability import AutonomyMetrics, RuntimeTrace, StepTrace
from src.schemas.planner import PlannerOutput
from src.schemas.reflection import ReflectionOutput


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class Channel(str, Enum):
    voice = "voice"
    text = "text"
    api = "api"
    ui = "ui"


class AuthMode(str, Enum):
    bypass = "bypass"
    bearer = "bearer"
    api_key = "api_key"
    anonymous = "anonymous"


class PermissionMode(str, Enum):
    auto_approved = "auto_approved"
    confirm_required = "confirm_required"
    blocked = "blocked"


class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GatewayHeaders(BaseModel):
    authorization: str | None = None
    api_key: str | None = None
    client_id: str | None = None
    forwarded_for: str | None = None
    user_agent: str | None = None


class AuthContext(BaseModel):
    mode: AuthMode = AuthMode.anonymous
    is_authenticated: bool = False
    principal_id: str = "anonymous"
    role: str = "user"
    tenant_id: str | None = None
    reason: str = ""


class RateLimitStatus(BaseModel):
    allowed: bool = True
    limit: int = 60
    remaining: int = 59
    window_seconds: int = 60
    reason: str = ""


class GatewayResult(BaseModel):
    request_id: str = Field(default_factory=new_id)
    channel: Channel
    client_id: str = "unknown-client"
    accepted: bool = True
    normalized_message: str
    auth: AuthContext
    rate_limit: RateLimitStatus
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=new_id)
    memory_type: str
    content: str
    source: str
    score: float = 1.0
    salience: float = 0.5
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    last_accessed_at: datetime = Field(default_factory=utc_now)


class WorkingMemory(BaseModel):
    objective: str = ""
    distilled_context: str = ""
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    retrieved_facts: list[str] = Field(default_factory=list)
    plan_checkpoint: str = "context_loaded"


class ContextSignal(BaseModel):
    complexity: str = "moderate"
    risk_level: str = "low"
    requested_capabilities: list[str] = Field(default_factory=list)
    time_horizon: str = "session"
    collaboration_mode: str = "single-specialist"
    needs_memory_retrieval: bool = True


class MemorySnapshot(BaseModel):
    summary: str = "No prior conversation history."
    semantic: list[MemoryRecord] = Field(default_factory=list)
    episodic: list[ConversationTurn] = Field(default_factory=list)
    vector: list[MemoryRecord] = Field(default_factory=list)
    retrieved: list[MemoryRecord] = Field(default_factory=list)
    goal_stack: list[str] = Field(default_factory=list)
    open_loops: list[str] = Field(default_factory=list)
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)


class UserRequest(BaseModel):
    user_id: str = "anonymous"
    session_id: str = "default"
    channel: Channel = Channel.text
    message: str
    goals: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextSnapshot(BaseModel):
    user_id: str
    session_id: str
    channel: Channel
    latest_message: str
    gateway: GatewayResult
    preferences: dict[str, Any] = Field(default_factory=dict)
    history: list[ConversationTurn] = Field(default_factory=list)
    memory: MemorySnapshot = Field(default_factory=MemorySnapshot)
    active_goals: list[str] = Field(default_factory=list)
    system_goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    requested_capabilities: list[str] = Field(default_factory=list)
    signals: ContextSignal = Field(default_factory=ContextSignal)
    context_packet: ContextPacket | None = None
    handoff_packet: HandoffPacket | None = None
    execution_mode: str = "react"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ControlDecision(BaseModel):
    mode: str = "master-brain"
    intent: str
    control_notes: str
    preferred_agent: str | None = None
    urgency: str = "normal"
    complexity: str = "moderate"
    reasoning_mode: str = "deliberate"
    llm_role: str = "planner-and-synthesizer"
    coordination_pattern: str = "single-specialist"
    risk_level: str = "low"
    memory_strategy: str = "retrieve-and-ground"
    verification_mode: str = "standard"
    needs_tooling: bool = True
    architecture_mode: str | None = None


class ReActCycle(BaseModel):
    thought: str
    action: str
    observation: str


class PlanStep(BaseModel):
    name: str
    description: str
    owner: str = "orchestrator"
    depends_on: list[str] = Field(default_factory=list)
    status: str = "pending"
    step_type: str = "control"
    success_criteria: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    verification_focus: list[str] = Field(default_factory=list)
    requires_tools: list[str] = Field(default_factory=list)
    risk_level: str = "low"


class ExecutionPlan(PlannerOutput):
    objective: str
    react_cycles: list[ReActCycle] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    completion_state: str = "planned"


class TaskNode(BaseModel):
    node_id: str
    name: str
    description: str
    owner: str
    status: str = "pending"
    depends_on: list[str] = Field(default_factory=list)
    verification_required: bool = False


class TaskGraph(BaseModel):
    engine: str = "stateful-agent-loop"
    state: str = "planned"
    active_path: list[str] = Field(default_factory=list)
    nodes: list[TaskNode] = Field(default_factory=list)


class AgentRoute(BaseModel):
    agent_name: str
    rationale: str
    branch: str = "primary"
    confidence: float = 0.6


class SkillDescriptor(BaseModel):
    name: str
    description: str
    triggers: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class ToolAuditMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str | None = None
    tool_name: str = ""
    category: str = ""
    risk_level: str = "low"
    side_effect: str = "none"
    dry_run: bool = False
    retryable: bool = False
    input_schema: str = ""
    output_schema: str = ""
    contract_version: str = "tool-quality-v1"
    mcp_ready: bool = True
    recorded_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "pending"
    summary: str = ""
    postconditions_met: bool = False
    checks: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(default_factory=new_id)
    tool_name: str
    purpose: str = ""
    input_payload: dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "low"
    side_effect: str = "none"
    requires_confirmation: bool = False
    dry_run: bool = False
    retryable: bool | None = None
    verification_hint: str = ""
    expected_observation: str = ""
    priority: int = 5
    audit_metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str | None = None
    tool_name: str
    status: str
    normalized_input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    risk_level: str = "low"
    side_effect: str = "none"
    retryable: bool = False
    dry_run: bool = False
    input_schema: str = ""
    output_schema: str = ""
    blocked_reason: str | None = None
    verification: ToolVerificationResult = Field(default_factory=ToolVerificationResult)
    audit: ToolAuditMetadata = Field(default_factory=ToolAuditMetadata)


class AgentDecision(BaseModel):
    agent_name: str
    summary: str
    skill_names: list[str] = Field(default_factory=list)
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    reasoning: str = ""
    response_strategy: str = ""
    expected_deliverables: list[str] = Field(default_factory=list)
    claims_to_verify: list[str] = Field(default_factory=list)
    decision_notes: list[str] = Field(default_factory=list)


class ReasoningStep(BaseModel):
    objective: str
    thought: str
    reasoning_summary: str
    action_strategy: str
    candidate_tools: list[str] = Field(default_factory=list)
    selected_skills: list[str] = Field(default_factory=list)
    expected_observation: str = ""
    should_replan: bool = False
    replan_reason: str | None = None
    stop_signal: str = "continue"
    stop_reason: str | None = None


class ActionDecision(BaseModel):
    selected_tool_names: list[str] = Field(default_factory=list)
    deferred_tool_names: list[str] = Field(default_factory=list)
    rationale: str
    expected_outcome: str = ""
    requires_replan: bool = False
    replan_reason: str | None = None


ToolSelectionDecision = ActionDecision


class ExecutionResult(BaseModel):
    agent_name: str
    summary: str = ""
    tool_results: list[ToolResult] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    claims: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    confidence: float = 0.75
    goal_status: str = "in_progress"
    ready_for_response: bool = False
    requires_replan: bool = False
    next_focus: str = ""


class VerificationCheck(BaseModel):
    name: str
    status: str
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    severity: str = "info"


class VerificationResult(BaseModel):
    status: str
    summary: str
    checks: list[VerificationCheck] = Field(default_factory=list)
    verified_claims: list[str] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    confidence: float = 0.75
    retry_recommended: bool = False


VerificationReport = VerificationResult


class ReflectionReport(ReflectionOutput):
    status: str
    checks: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class StopDecision(BaseModel):
    should_stop: bool = False
    trigger: str = "continue"
    reason: str = ""
    ready_for_response: bool = False
    requires_replan: bool = False


class ModelExecutionRecord(BaseModel):
    task_type: str
    stage: str
    provider: str
    model: str
    status: str
    source: str = "fallback"
    latency_ms: int = 0
    used_fallback: bool = True
    reason: str = ""
    candidate_models: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelEvaluationRecord(BaseModel):
    task_type: str
    provider: str
    model: str
    score: float = 0.0
    notes: list[str] = Field(default_factory=list)
    compared_models: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured_output_validity: bool | None = None
    latency_ms: int | None = None
    task_success: bool | None = None
    response_completeness: float | None = None


class PermissionDecision(BaseModel):
    mode: PermissionMode
    requires_confirmation: bool = False
    reason: str


class SessionPermissionState(BaseModel):
    user_id: str
    session_id: str
    existing_session: bool = False
    history_turn_count: int = 0
    permission: PermissionDecision
    notes: list[str] = Field(default_factory=list)


class SafetyPolicyHit(BaseModel):
    policy: str
    severity: str
    reason: str
    affected_targets: list[str] = Field(default_factory=list)


class SafetyReport(BaseModel):
    status: str
    sandbox: str
    permission: PermissionDecision
    audit_log_saved: bool = False
    notes: list[str] = Field(default_factory=list)
    policy_hits: list[SafetyPolicyHit] = Field(default_factory=list)
    gated_actions: list[str] = Field(default_factory=list)
    risk_level: str = "low"


class TraceEvent(BaseModel):
    stage: str
    detail: str
    payload: dict[str, Any] = Field(default_factory=dict)


class StructuredResponse(BaseModel):
    response: str
    highlights: list[str] = Field(default_factory=list)
    approval_note: str = ""


class ObservationRecord(BaseModel):
    observation_id: str = Field(default_factory=new_id)
    step_index: int = 0
    source: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class StateTransition(BaseModel):
    transition_id: str = Field(default_factory=new_id)
    step_index: int
    formula: str = "S_{t+1} = F(S_t, O_t)"
    prior_status: str
    next_status: str
    observations: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    selected_route: str = "general"
    retry_recommended: bool = False
    replan_required: bool = False
    ready_for_response: bool = False
    adaptive_constraints: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    route_bias: str | None = None
    architecture_mode: str | None = None
    loop_phase: str = "update_memory"
    termination_signal: str = "continue"
    created_at: datetime = Field(default_factory=utc_now)


class ReActStepTrace(BaseModel):
    step_index: int
    observe_summary: str = ""
    context_summary: str = ""
    architecture_mode: str | None = None
    architecture_summary: str = ""
    reasoning_summary: str = ""
    action_summary: str = ""
    verification_summary: str = ""
    reflection_summary: str = ""
    selected_route: str = "general"
    selected_tools: list[str] = Field(default_factory=list)
    observed_evidence: list[str] = Field(default_factory=list)
    memory_checkpoint: str = ""
    memory_checkpoints: list[str] = Field(default_factory=list)
    loop_phases: list[LoopPhase] = Field(default_factory=list)
    replan_triggered: bool = False
    stop_trigger: str = "continue"
    stop_reason: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class AgentState(BaseModel):
    state_id: str = Field(default_factory=new_id)
    request: UserRequest
    headers: GatewayHeaders = Field(default_factory=GatewayHeaders)
    status: str = "initialized"
    step_index: int = 0
    max_steps: int = 4
    goal_status: str = "in_progress"
    termination_reason: str = ""
    response_ready: bool = False
    gateway: GatewayResult | None = None
    session: SessionPermissionState | None = None
    memory: MemorySnapshot = Field(default_factory=MemorySnapshot)
    context: ContextSnapshot | None = None
    context_packet: ContextPacket | None = None
    handoff_packet: HandoffPacket | None = None
    architecture: ArchitectureDecision | None = None
    control: ControlDecision | None = None
    loop_state: LoopState = Field(default_factory=LoopState)
    plan: ExecutionPlan | None = None
    task_graph: TaskGraph | None = None
    route: AgentRoute | None = None
    skills: list[SkillDescriptor] = Field(default_factory=list)
    decision: AgentDecision | None = None
    reasoning: ReasoningStep | None = None
    tool_selection: ActionDecision | None = None
    pending_tool_requests: list[ToolRequest] = Field(default_factory=list)
    last_tool_results: list[ToolResult] = Field(default_factory=list)
    tool_history: list[ToolResult] = Field(default_factory=list)
    execution: ExecutionResult | None = None
    verification: VerificationResult | None = None
    reflection: ReflectionReport | None = None
    stop_decision: StopDecision | None = None
    safety: SafetyReport | None = None
    model_runs: list[ModelExecutionRecord] = Field(default_factory=list)
    model_evaluations: list[ModelEvaluationRecord] = Field(default_factory=list)
    observations: list[ObservationRecord] = Field(default_factory=list)
    reasoning_notes: list[str] = Field(default_factory=list)
    autonomy_metrics: AutonomyMetrics = Field(default_factory=AutonomyMetrics)
    step_traces: list[StepTrace] = Field(default_factory=list)
    runtime_trace: RuntimeTrace | None = None
    react_trace: list[ReActStepTrace] = Field(default_factory=list)
    state_transitions: list[StateTransition] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    adaptive_constraints: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    route_bias: str | None = None
    retry_count: int = 0
    replan_count: int = 0
    needs_replan: bool = False
    needs_retry: bool = False
    final_response: str = ""


class InteractionResponse(BaseModel):
    request_id: str
    response: str
    confirmation: str
    assigned_agent: str
    gateway: GatewayResult
    session: SessionPermissionState
    control: ControlDecision
    context_packet: ContextPacket | None = None
    handoff_packet: HandoffPacket | None = None
    architecture: ArchitectureDecision | None = None
    loop_state: LoopState = Field(default_factory=LoopState)
    plan: ExecutionPlan
    task_graph: TaskGraph
    skills: list[SkillDescriptor] = Field(default_factory=list)
    verification: VerificationResult
    reflection: ReflectionReport
    safety: SafetyReport
    memory: MemorySnapshot
    autonomy_metrics: AutonomyMetrics = Field(default_factory=AutonomyMetrics)
    runtime_trace: RuntimeTrace | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    react_trace: list[ReActStepTrace] = Field(default_factory=list)
    step_traces: list[StepTrace] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    state_transitions: list[StateTransition] = Field(default_factory=list)
    model_runs: list[ModelExecutionRecord] = Field(default_factory=list)
    model_evaluations: list[ModelEvaluationRecord] = Field(default_factory=list)
    state_id: str
    loop_count: int = 0
    termination_reason: str = ""


class SessionState(BaseModel):
    user_id: str
    session_id: str
    preferences: dict[str, Any] = Field(default_factory=dict)
    history: list[ConversationTurn] = Field(default_factory=list)
    memory: MemorySnapshot = Field(default_factory=MemorySnapshot)
