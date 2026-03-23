from __future__ import annotations

from typing import Any

from src.agents import get_agent_catalog
from src.memory import EpisodicStore, MemoryManager, SemanticStore
from src.runtime import ARCHITECTURE_STAGES, Orchestrator, build_default_orchestrator
from src.runtime.models import (
    AgentState,
    GatewayHeaders,
    InteractionResponse,
    SessionState,
    UserRequest,
)
from src.schemas.catalog import AgentCatalog, AuditLogResponse, ToolCatalog
from src.core.config import get_settings
from src.safety.audit import AuditService
from src.services.runtime_lifecycle import RuntimeLifecycleService
from src.services.llm_service import LLMService
from src.tools import get_tool_catalog


class OrchestrationService:
    def __init__(
        self,
        *,
        orchestrator: Orchestrator | None = None,
        llm_service: LLMService | None = None,
        episodic_store: EpisodicStore | None = None,
        semantic_store: SemanticStore | None = None,
        audit_service: AuditService | None = None,
        lifecycle_service: RuntimeLifecycleService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.llm_service = llm_service or LLMService()
        self.episodic_store = episodic_store or EpisodicStore()
        self.semantic_store = semantic_store or SemanticStore()
        self.audit_service = audit_service or AuditService()
        self._orchestrator = orchestrator or build_default_orchestrator()
        self.memory_manager = MemoryManager(
            self._orchestrator.memory_system,
            self.episodic_store,
            self.semantic_store,
            self.settings,
        )
        self.lifecycle = lifecycle_service or RuntimeLifecycleService(
            memory_manager=self.memory_manager,
            audit_service=self.audit_service,
        )

    def get_architecture(self):
        return ARCHITECTURE_STAGES

    def list_agents(self) -> AgentCatalog:
        return get_agent_catalog()

    def list_tools(self) -> ToolCatalog:
        return get_tool_catalog()

    def get_audit_logs(self, limit: int = 100) -> AuditLogResponse:
        return self.lifecycle.recent_audit_events(limit=limit)

    def health(self) -> dict[str, Any]:
        return {
            "name": self.settings.app_name,
            "status": "running",
            "environment": self.settings.environment,
            "llm": self.llm_service.health(),
            "memory": self.episodic_store.health(),
            "semantic_memory": self.semantic_store.health(),
            "audit": self.audit_service.health(),
        }

    def get_session_state(self, user_id: str, session_id: str) -> SessionState:
        return self.lifecycle.session_state(user_id=user_id, session_id=session_id)

    def handle_interaction(
        self,
        request: UserRequest,
        headers: GatewayHeaders,
    ) -> InteractionResponse:
        self.lifecycle.prepare_request(request)
        response = self._orchestrator.handle(request=request, headers=headers)
        self.lifecycle.finalize_interaction(request, response)
        return response

    def execute_interaction(
        self,
        request: UserRequest,
        headers: GatewayHeaders,
    ) -> InteractionResponse:
        return self.handle_interaction(request=request, headers=headers)

    def plan_interaction(
        self,
        request: UserRequest,
        headers: GatewayHeaders,
    ) -> AgentState:
        self.lifecycle.prepare_request(request)
        state = self._orchestrator.preview_plan(request=request, headers=headers)
        self.lifecycle.record_plan_preview(request, state)
        return state
