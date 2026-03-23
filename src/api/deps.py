from __future__ import annotations

from functools import lru_cache

from fastapi import Header

from src.schemas.chat import GatewayHeaders
from src.services.orchestration_service import OrchestrationService


@lru_cache(maxsize=1)
def get_orchestration_service() -> OrchestrationService:
    return OrchestrationService()


def get_gateway_headers(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_client_id: str | None = Header(default=None),
    x_forwarded_for: str | None = Header(default=None),
    user_agent: str | None = Header(default=None),
) -> GatewayHeaders:
    return GatewayHeaders(
        authorization=authorization,
        api_key=x_api_key,
        client_id=x_client_id,
        forwarded_for=x_forwarded_for,
        user_agent=user_agent,
    )
