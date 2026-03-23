from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from src.schemas.catalog import AgentCatalog, AuditLogResponse, ToolCatalog
from src.schemas.platform import ChatResponse, ExecuteResponse, HealthResponse, PlanResponse
from src.runtime.models import InteractionResponse, SessionState
from src.runtime.workflow import StageDescriptor
from src.schemas.platform import ArchitectureResponse


class ApiClientError(RuntimeError):
    """Raised when the frontend cannot complete a backend request."""


@dataclass(frozen=True)
class RequestEnvelope:
    user_id: str
    session_id: str
    message: str
    selected_model: str
    approval_granted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        metadata = {
            "frontend": "streamlit",
            "selected_model": self.selected_model,
            "approval_granted": self.approval_granted,
            **self.metadata,
        }
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "channel": "ui",
            "message": self.message,
            "preferences": {
                "selected_model": self.selected_model,
            },
            "metadata": metadata,
        }


class ApiClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def health(self) -> HealthResponse:
        payload = self._request("GET", "/health")
        return HealthResponse.model_validate(payload)

    def architecture(self) -> list[StageDescriptor]:
        payload = self._request("GET", "/architecture")
        return ArchitectureResponse.model_validate(payload).stages

    def agents(self) -> AgentCatalog:
        payload = self._request("GET", "/agents")
        return AgentCatalog.model_validate(payload)

    def tools(self) -> ToolCatalog:
        payload = self._request("GET", "/tools")
        return ToolCatalog.model_validate(payload)

    def audit_logs(self, limit: int = 100) -> AuditLogResponse:
        payload = self._request("GET", f"/audit/logs?limit={limit}")
        return AuditLogResponse.model_validate(payload)

    def session_state(self, user_id: str, session_id: str) -> SessionState:
        payload = self._request("GET", f"/sessions/{user_id}/{session_id}")
        return SessionState.model_validate(payload)

    def plan(self, request: RequestEnvelope) -> PlanResponse:
        payload = self._request("POST", "/plan", json=request.as_payload())
        return PlanResponse.model_validate(payload)

    def chat(self, request: RequestEnvelope) -> InteractionResponse:
        payload = self._request("POST", "/chat", json=request.as_payload())
        return ChatResponse.model_validate(payload).interaction

    def execute(self, request: RequestEnvelope) -> InteractionResponse:
        payload = self._request("POST", "/execute", json=request.as_payload())
        return ExecuteResponse.model_validate(payload).interaction

    def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout_seconds) as client:
                response = client.request(method=method, url=path, json=json)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise ApiClientError(
                f"Backend request failed with status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiClientError(
                f"Unable to reach the FastAPI backend at {self._base_url}. {exc}"
            ) from exc
