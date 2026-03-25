from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


BrowserVerificationStatus = Literal[
    "goal_reached",
    "in_progress",
    "requires_confirmation",
    "blocked",
    "dry_run",
]


class ScreenshotCheckpointMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = ""
    path: str = ""
    captured_at: str = ""
    source: str = ""
    note: str = ""

    @field_validator("path", mode="before")
    @classmethod
    def normalize_path(cls, value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return str(Path(text))


class BrowserGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""
    target_url: str = ""
    target_title: str = ""
    success_indicators: list[str] = Field(default_factory=list)
    required_text: list[str] = Field(default_factory=list)
    avoided_text: list[str] = Field(default_factory=list)
    dom_hints: list[str] = Field(default_factory=list)
    action_kind: str = "inspect"
    action_target: str = ""
    stop_before_submit: bool = True
    allow_submit: bool = False
    approval_required_for_dangerous_actions: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrowserStateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_name: str = ""
    page_url: str = ""
    page_title: str = ""
    page_text_snapshot: str = ""
    dom_text_summary: str = ""
    extracted_text_blocks: list[str] = Field(default_factory=list)
    screenshot_checkpoint: ScreenshotCheckpointMetadata = Field(default_factory=ScreenshotCheckpointMetadata)
    action_kind: str = "inspect"
    action_target: str = ""
    approval_granted: bool = False
    dangerous_action_candidates: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrowserVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: BrowserVerificationStatus = "in_progress"
    summary: str = ""
    goal_reached: bool = False
    requires_confirmation: bool = False
    should_stop: bool = False
    dangerous_action_detected: bool = False
    dangerous_action_reasons: list[str] = Field(default_factory=list)
    stop_before_submit_triggered: bool = False
    matched_indicators: list[str] = Field(default_factory=list)
    missing_indicators: list[str] = Field(default_factory=list)
    verification_notes: list[str] = Field(default_factory=list)
    recommended_next_action: str = ""
    page_text_snapshot: str = ""
    dom_text_summary: str = ""
    screenshot_checkpoint: ScreenshotCheckpointMetadata = Field(default_factory=ScreenshotCheckpointMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)
