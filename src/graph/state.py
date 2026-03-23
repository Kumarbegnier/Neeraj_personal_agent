from __future__ import annotations

from pydantic import BaseModel

from src.runtime.models import (
    ContextSnapshot,
    ControlDecision,
    ExecutionPlan,
    GatewayResult,
    InteractionResponse,
    SessionPermissionState,
    TaskGraph,
    UserRequest,
)


class GraphState(BaseModel):
    request: UserRequest
    gateway: GatewayResult | None = None
    session: SessionPermissionState | None = None
    context: ContextSnapshot | None = None
    control: ControlDecision | None = None
    plan: ExecutionPlan | None = None
    task_graph: TaskGraph | None = None
    response: InteractionResponse | None = None
