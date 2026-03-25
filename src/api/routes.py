from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI

from src.api.deps import get_gateway_headers, get_orchestration_service
from src.core.config import get_settings
from src.core.logging import configure_logging
from src.runtime.models import RuntimeTrace
from src.schemas.catalog import AgentCatalog, AuditLogResponse, ToolCatalog
from src.schemas.chat import GatewayHeaders, InteractionResponse, SessionState, UserRequest
from src.schemas.platform import (
    ArchitectureResponse,
    ChatRequest,
    ChatResponse,
    ExecuteRequest,
    ExecuteResponse,
    HealthResponse,
    PlanRequest,
    PlanResponse,
)
from src.schemas.routing import TaskFamilyRoutingWinner
from src.services.orchestration_service import OrchestrationService

router = APIRouter()


@router.get("/")
def read_root():
    settings = get_settings()
    return {
        "message": f"Welcome to {settings.app_name}!",
        "available_routes": [
            "/health",
            "/chat",
            "/plan",
            "/execute",
            "/architecture",
            "/agents",
            "/tools",
            "/audit/logs",
            "/observability/runtime-traces",
            "/routing/evaluation-winners",
            "/sessions/{user_id}/{session_id}",
        ],
    }


@router.get("/health", response_model=HealthResponse)
def get_health(
    service: OrchestrationService = Depends(get_orchestration_service),
):
    return service.health()


@router.get("/status", response_model=HealthResponse)
def get_status(
    service: OrchestrationService = Depends(get_orchestration_service),
):
    return service.health()


@router.get("/architecture", response_model=ArchitectureResponse)
def get_architecture(
    service: OrchestrationService = Depends(get_orchestration_service),
):
    return ArchitectureResponse(stages=service.get_architecture())


@router.get("/agents", response_model=AgentCatalog)
def get_agents(
    service: OrchestrationService = Depends(get_orchestration_service),
):
    return service.list_agents()


@router.get("/tools", response_model=ToolCatalog)
def get_tools(
    service: OrchestrationService = Depends(get_orchestration_service),
):
    return service.list_tools()


@router.get("/audit/logs", response_model=AuditLogResponse)
def get_audit_logs(
    limit: int = 100,
    service: OrchestrationService = Depends(get_orchestration_service),
):
    return service.get_audit_logs(limit=limit)


@router.get("/observability/runtime-traces", response_model=list[RuntimeTrace])
def get_runtime_traces(
    limit: int = 25,
    service: OrchestrationService = Depends(get_orchestration_service),
):
    return service.get_runtime_traces(limit=limit)


@router.get("/routing/evaluation-winners", response_model=list[TaskFamilyRoutingWinner])
def get_routing_evaluation_winners(
    limit: int = 12,
    service: OrchestrationService = Depends(get_orchestration_service),
):
    return service.get_evaluation_winners(limit=limit)


@router.get("/sessions/{user_id}/{session_id}", response_model=SessionState)
def get_session_state(
    user_id: str,
    session_id: str,
    service: OrchestrationService = Depends(get_orchestration_service),
):
    return service.get_session_state(user_id=user_id, session_id=session_id)


@router.post("/chat", response_model=ChatResponse)
def create_chat(
    request: ChatRequest,
    headers: GatewayHeaders = Depends(get_gateway_headers),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    interaction = service.handle_interaction(request=request, headers=headers)
    return ChatResponse(interaction=interaction)


@router.post("/plan", response_model=PlanResponse)
def create_plan(
    request: PlanRequest,
    headers: GatewayHeaders = Depends(get_gateway_headers),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    state = service.plan_interaction(request=request, headers=headers)
    return PlanResponse(
        request_id=state.gateway.request_id if state.gateway else "preview",
        state_id=state.state_id,
        assigned_agent=state.route.agent_name if state.route else "general",
        control=state.control,
        plan=state.plan,
        task_graph=state.task_graph,
        skills=state.skills,
        memory=state.memory,
        permission=state.session.permission,
        trace=state.trace,
    )


@router.post("/execute", response_model=ExecuteResponse)
def execute_plan(
    request: ExecuteRequest,
    headers: GatewayHeaders = Depends(get_gateway_headers),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    interaction = service.execute_interaction(request=request, headers=request.headers or headers)
    return ExecuteResponse(interaction=interaction)


@router.post("/interactions", response_model=InteractionResponse)
def create_interaction(
    request: UserRequest,
    headers: GatewayHeaders = Depends(get_gateway_headers),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    return service.handle_interaction(request=request, headers=headers)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name)
    app.include_router(router)
    return app
