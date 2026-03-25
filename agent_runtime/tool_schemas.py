from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.browser import (
    BrowserGoal,
    BrowserStateSnapshot,
    BrowserVerificationResult,
    BrowserVerificationStatus,
)

from src.tools.base import ToolInputModel, ToolOutputModel


class ToolListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = ""
    url: str = ""
    snippet: str = ""


class EmptyToolInput(ToolInputModel):
    pass


class SessionHistoryOutput(ToolOutputModel):
    turn_count: int = 0


class MemoryEntriesOutput(ToolOutputModel):
    count: int = 0
    entries: list[str] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)


class WorkingMemoryOutput(ToolOutputModel):
    objective: str = ""
    distilled_context: str = ""
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    retrieved_facts: list[str] = Field(default_factory=list)
    checkpoint: str = ""


class GoalStackOutput(ToolOutputModel):
    active_goals: list[str] = Field(default_factory=list)
    goal_stack: list[str] = Field(default_factory=list)
    system_goals: list[str] = Field(default_factory=list)


class CapabilityMapOutput(ToolOutputModel):
    interfaces: list[str] = Field(default_factory=list)
    specialized_agents: list[str] = Field(default_factory=list)
    tool_categories: list[str] = Field(default_factory=list)
    requested_capabilities: list[str] = Field(default_factory=list)
    execution_layer: list[str] = Field(default_factory=list)


class ExecutionCatalogOutput(ToolOutputModel):
    connectors: dict[str, str] = Field(default_factory=dict)
    governance: dict[str, str] = Field(default_factory=dict)


class SkillManifestOutput(ToolOutputModel):
    skills: list[str] = Field(default_factory=list)


class PlanAnalyzerInput(ToolInputModel):
    objective: str = ""
    step_count: int = 0
    success_criteria: list[str] = Field(default_factory=list)
    verification_focus: list[str] = Field(default_factory=list)


class PlanAnalyzerOutput(ToolOutputModel):
    objective: str = ""
    step_count: int = 0
    success_criteria: list[str] = Field(default_factory=list)
    verification_focus: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class VerificationHarnessInput(ToolInputModel):
    checks: list[str] = Field(default_factory=list)
    mode: str = "standard"


class VerificationHarnessOutput(ToolOutputModel):
    check_count: int = 0
    checks: list[str] = Field(default_factory=list)
    mode: str = "standard"
    risk_level: str = "low"


class RiskMonitorInput(ToolInputModel):
    risk_level: str = ""
    action: str = "inspect"
    requires_confirmation: bool = False


class RiskMonitorOutput(ToolOutputModel):
    risk_level: str = "low"
    requested_action: str = "inspect"
    requires_confirmation: bool = False
    constraints: list[str] = Field(default_factory=list)


class ConnectorActionInput(ToolInputModel):
    action: str = "inspect"
    target: str = ""
    operation: str = ""
    database: str = ""
    repository_action: str = ""
    calendar_action: str = ""
    document_action: str = ""
    browser: str = ""


class ConnectorAvailabilityOutput(ToolOutputModel):
    connector: str = ""
    target: str = ""
    requested_action: str = ""
    operation: str = ""
    database: str = ""
    repository_action: str = ""
    calendar_action: str = ""
    document_action: str = ""
    browser: str = ""


class EmailDraftInput(ToolInputModel):
    to: list[str] = Field(default_factory=list)
    subject: str = ""
    body: str = ""


class EmailDraftOutput(ToolOutputModel):
    draft_id: str = ""
    to: list[str] = Field(default_factory=list)
    subject: str = ""
    body_preview: str = ""
    approval_required: bool = True


class SearchWebInput(ToolInputModel):
    query: str = ""


class SearchWebOutput(ToolOutputModel):
    query: str = ""
    results: list[ToolListItem] = Field(default_factory=list)
    source: str = ""


class BrowserSearchInput(ToolInputModel):
    query: str = ""
    session: str = ""


class BrowserSearchOutput(ToolOutputModel):
    query: str = ""
    browser_session: str = ""
    results: list[ToolListItem] = Field(default_factory=list)


class BrowserGoalVerificationInput(ToolInputModel):
    goal: BrowserGoal
    snapshot: BrowserStateSnapshot


class BrowserGoalVerificationOutput(ToolOutputModel):
    status: BrowserVerificationStatus = "in_progress"
    goal_reached: bool = False
    requires_confirmation: bool = False
    dangerous_action_detected: bool = False
    stop_before_submit_triggered: bool = False
    matched_indicators: list[str] = Field(default_factory=list)
    missing_indicators: list[str] = Field(default_factory=list)
    verification: BrowserVerificationResult = Field(default_factory=BrowserVerificationResult)
    snapshot: BrowserStateSnapshot = Field(default_factory=BrowserStateSnapshot)


class SaveMemoryInput(ToolInputModel):
    content: str = ""
    tags: list[str] = Field(default_factory=list)
    source: str = "save_memory_tool"
    salience: float = 0.8


class SaveMemoryOutput(ToolOutputModel):
    saved: bool = False
    memory_id: str = ""
    content: str = ""
    tags: list[str] = Field(default_factory=list)


class LoadRecentMemoryInput(ToolInputModel):
    limit: int = 4


class LoadRecentMemoryOutput(ToolOutputModel):
    history: list[str] = Field(default_factory=list)
    semantic: list[str] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)


class SummarizeFileInput(ToolInputModel):
    path: str = Field(min_length=1)


class SummarizeFileOutput(ToolOutputModel):
    path: str = ""
    line_count: int = 0


class GenerateCodeInput(ToolInputModel):
    language: str = "python"
    objective: str = ""


class GenerateCodeOutput(ToolOutputModel):
    language: str = "python"
    objective: str = ""
    generated_code: str = ""


class OpenPageInput(ToolInputModel):
    url: str = "https://example.com"
    browser: str = "playwright"


class OpenPageOutput(ToolOutputModel):
    url: str = "https://example.com"
    browser: str = "playwright"


class ExtractPageTextInput(ToolInputModel):
    html: str = ""
    text: str = ""


class ExtractPageTextOutput(ToolOutputModel):
    text: str = ""
    length: int = 0


class CreateTaskRecordInput(ToolInputModel):
    title: str = ""
    status: str = "planned"


class CreateTaskRecordOutput(ToolOutputModel):
    task_id: str = ""
    title: str = ""
    task_status: str = "planned"
    stored: bool = False
