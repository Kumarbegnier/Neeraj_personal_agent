from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from src.runtime.models import ContextSnapshot, ToolRequest, ToolResult
from src.schemas.catalog import ToolDescriptor


class ToolExecutionContext(BaseModel):
    user_id: str
    session_id: str
    channel: str
    active_goals: list[str] = Field(default_factory=list)

    @classmethod
    def from_context(cls, context: ContextSnapshot) -> "ToolExecutionContext":
        return cls(
            user_id=context.user_id,
            session_id=context.session_id,
            channel=context.channel.value,
            active_goals=context.active_goals,
        )

class BaseTool(Protocol):
    descriptor: ToolDescriptor

    def run(self, context: ContextSnapshot, request: ToolRequest) -> ToolResult:
        ...
