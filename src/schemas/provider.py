from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .routing import ModelProvider, ModelTaskType


class ProviderMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class StructuredOutputSchema(BaseModel):
    name: str
    json_schema: dict[str, Any]
    strict: bool = True


class ProviderRequest(BaseModel):
    task_type: ModelTaskType
    model: str
    messages: list[ProviderMessage]
    temperature: float = 0.1
    max_tokens: int = 2048
    structured_output: StructuredOutputSchema | None = None
    web_grounded: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    provider: ModelProvider
    model: str
    content: str
    latency_ms: int = 0
    finish_reason: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(BaseModel):
    provider: ModelProvider
    configured: bool
    default_model: str
    supports_structured_output: bool = True
    supports_tool_calling: bool = False
    supports_web_grounding: bool = False
