from __future__ import annotations

from pydantic import BaseModel, Field


class ReflectionOutput(BaseModel):
    summary: str
    issues: list[str] = Field(default_factory=list)
    repairs: list[str] = Field(default_factory=list)
    lessons: list[str] = Field(default_factory=list)
    retry_recommended: bool = False
    retry_reason: str | None = None
    route_bias: str | None = None
    blocked_tools: list[str] = Field(default_factory=list)
    confidence: float = 0.75
