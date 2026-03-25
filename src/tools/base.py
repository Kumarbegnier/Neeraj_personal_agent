from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.runtime.models import (
    ContextSnapshot,
    ToolAuditMetadata,
    ToolRequest,
    ToolResult,
    ToolVerificationResult,
)
from src.schemas.catalog import ToolDescriptor


class ToolExecutionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


class ToolInputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ready"
    summary: str = ""
    dry_run: bool = False
    preview_action: str = ""


ToolHandler = Callable[[ContextSnapshot, dict[str, Any]], dict[str, Any]]
ToolVerifier = Callable[[ContextSnapshot, BaseModel, BaseModel], ToolVerificationResult]
DryRunBuilder = Callable[[ToolRequest, BaseModel], BaseModel]


@dataclass(frozen=True)
class ToolContract:
    descriptor: ToolDescriptor
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler
    verifier: ToolVerifier
    retryable: bool = True
    dry_run_builder: DryRunBuilder | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.descriptor.name,
            "description": self.descriptor.description,
            "category": self.descriptor.category,
            "risk_level": self.descriptor.risk_level,
            "side_effect": self.descriptor.side_effect,
            "retryable": self.retryable,
            "supports_dry_run": self.descriptor.supports_dry_run,
            "mcp_ready": self.descriptor.mcp_ready,
            "contract_version": self.descriptor.contract_version,
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
        }

    def build_dry_run_output(self, request: ToolRequest, validated_input: BaseModel) -> BaseModel:
        if self.dry_run_builder is not None:
            return self.dry_run_builder(request, validated_input)
        return self.output_model.model_validate(
            {
                "status": "dry_run",
                "summary": request.purpose or f"Dry run preview for '{self.descriptor.name}'.",
                "dry_run": True,
                "preview_action": request.purpose or f"Preview {self.descriptor.name}",
            }
        )

    def build_audit_metadata(
        self,
        *,
        request: ToolRequest,
        risk_level: str,
        side_effect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolAuditMetadata:
        return ToolAuditMetadata(
            call_id=request.call_id,
            tool_name=self.descriptor.name,
            category=self.descriptor.category,
            risk_level=risk_level,
            side_effect=side_effect,
            dry_run=request.dry_run,
            retryable=self.retryable if request.retryable is None else request.retryable,
            input_schema=self.input_model.__name__,
            output_schema=self.output_model.__name__,
            contract_version=self.descriptor.contract_version,
            mcp_ready=self.descriptor.mcp_ready,
            metadata=metadata or {},
        )


class BaseTool(Protocol):
    contract: ToolContract

    def run(self, context: ContextSnapshot, request: ToolRequest) -> ToolResult:
        ...
