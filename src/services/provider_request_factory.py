from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.schemas.provider import ProviderMessage, ProviderRequest, StructuredOutputSchema
from src.schemas.routing import ModelTaskType


def build_structured_provider_request(
    *,
    task_type: ModelTaskType,
    model: str,
    system_prompt: str,
    user_prompt: str,
    output_type: type[BaseModel],
    metadata: dict[str, Any] | None = None,
    web_grounded: bool = False,
) -> ProviderRequest:
    return ProviderRequest(
        task_type=task_type,
        model=model,
        messages=[
            ProviderMessage(role="system", content=system_prompt),
            ProviderMessage(role="user", content=user_prompt),
        ],
        structured_output=StructuredOutputSchema(
            name=output_type.__name__,
            json_schema=output_type.model_json_schema(),
        ),
        web_grounded=web_grounded,
        metadata=metadata or {},
    )
