from __future__ import annotations

from dotenv import load_dotenv

from .agents import build_agent_registry
from .context_builder import ContextBuilder
from .control import OrchestratorBrain
from .execution import ExecutionEngine
from .gateway import InterfaceGateway
from .memory import MemorySystem, dump_model
from .models import (
    ControlDecision,
    ExecutionPlan,
    AgentState,
    GatewayHeaders,
    InteractionResponse,
    PermissionDecision,
    PermissionMode,
    StateTransition,
    TaskGraph,
    TraceEvent,
    UserRequest,
)
from .planner import Planner
from .reflection import ReflectionEngine
from .response_helpers import build_confirmation, reviewed_safety, skipped_reflection, skipped_verification
from .responder import ResponseComposer
from .router import AgentRouter
from .router_executor import RouterExecutor
from .safety import SafetyPermissions
from .session_permissions import SessionPermissionManager
from .skills import SkillLibrary
from .task_graph import TaskGraphEngine
from .tool_registry import ToolRegistry
from .verification import VerificationEngine


class Orchestrator:
    def __init__(
        self,
        memory_system: MemorySystem,
        api_gateway: InterfaceGateway,
        session_permission_manager: SessionPermissionManager,
        context_builder: ContextBuilder,
        orchestrator_brain: OrchestratorBrain,
        planner: Planner,
        task_graph_engine: TaskGraphEngine,
        router_executor: RouterExecutor,
        execution_engine: ExecutionEngine,
        verification_engine: VerificationEngine,
        reflection_engine: ReflectionEngine,
        safety_permissions: SafetyPermissions,
        response_composer: ResponseComposer,
    ) -> None:
        self.memory_system = memory_system
        self.api_gateway = api_gateway
        self.session_permission_manager = session_permission_manager
        self.context_builder = context_builder
        self.orchestrator_brain = orchestrator_brain
        self.planner = planner
        self.task_graph_engine = task_graph_engine
        self.router_executor = router_executor
        self.execution_engine = execution_engine
        self.verification_engine = verification_engine
        self.reflection_engine = reflection_engine
        self.safety_permissions = safety_permissions
        self.response_composer = response_composer

    def handle(
        self,
        request: UserRequest,
        headers: GatewayHeaders | None = None,
    ) -> InteractionResponse:
        state = AgentState(
            request=request,
            headers=headers or GatewayHeaders(),
            max_steps=self._max_steps_for(request),
        )
        self._append_trace(
            state,
            "User / UI",
            f"Received a {request.channel.value} request for session '{request.session_id}'.",
            {"user_id": request.user_id},
        )

        state.gateway = self.api_gateway.process(request, headers=state.headers)
        self._append_trace(
            state,
            "API Gateway",
            "Validated the request through auth, normalization, and rate policies.",
            {
                "accepted": state.gateway.accepted,
                "client_id": state.gateway.client_id,
                "auth_mode": state.gateway.auth.mode.value,
                "rate_remaining": state.gateway.rate_limit.remaining,
            },
        )

        state.session = self.session_permission_manager.prepare(request, state.gateway, self.memory_system)
        self._append_trace(
            state,
            "Session + Permissions",
            state.session.permission.reason,
            {
                "existing_session": state.session.existing_session,
                "history_turn_count": state.session.history_turn_count,
                "permission_mode": state.session.permission.mode.value,
                "requires_confirmation": state.session.permission.requires_confirmation,
            },
        )

        if not state.gateway.accepted or state.session.permission.mode == PermissionMode.blocked:
            return self._blocked_response(state)

        while state.step_index < state.max_steps:
            state.step_index += 1
            prior_signature = self._state_signature(state)
            self._append_trace(
                state,
                "Iteration Start",
                f"Starting loop iteration {state.step_index}.",
                {
                    "step_index": state.step_index,
                    "retry_count": state.retry_count,
                    "replan_count": state.replan_count,
                    "needs_replan": state.needs_replan,
                },
            )
            self._observe(state)
            self._control(state)
            self._plan(state)
            self._route(state)
            self._agent_decide(state)
            self._execute(state)
            self._verify(state)
            self._reflect(state)
            self._update_state(state, prior_signature)
            self._append_trace(
                state,
                "Iteration End",
                f"Completed loop iteration {state.step_index}.",
                {
                    "step_index": state.step_index,
                    "verification_status": state.verification.status if state.verification else "unknown",
                    "ready_for_response": state.execution.ready_for_response if state.execution else False,
                    "route_bias": state.route_bias,
                    "blocked_tools": state.blocked_tools,
                },
            )

            should_retry = (
                (state.verification.retry_recommended if state.verification else False)
                or state.needs_retry
                or (state.execution.requires_replan if state.execution else False)
            )
            if should_retry:
                state.retry_count += 1
                state.needs_replan = True
                state.goal_status = "retrying"
                self._append_trace(
                    state,
                    "Loop Control",
                    "Verification or reflection requested another closed-loop iteration.",
                    {
                        "retry_count": state.retry_count,
                        "replan_count": state.replan_count,
                        "route_bias": state.route_bias,
                        "blocked_tools": state.blocked_tools,
                    },
                )
                continue

            if state.execution and state.execution.ready_for_response:
                state.response_ready = True
                state.goal_status = "achieved"
                state.termination_reason = "goal_achieved"
                break

        if not state.termination_reason:
            state.termination_reason = "max_steps_reached"
            state.goal_status = "max_steps_reached"
        state.response_ready = True

        if state.plan is not None:
            state.plan = self.planner.complete(state.plan)
        if state.task_graph is not None:
            state.task_graph = self.task_graph_engine.finalize(state.task_graph)

        state.safety = self.safety_permissions.review(state)
        state.safety.permission = self._merge_permissions(state.session.permission, state.safety.permission)
        state.final_response = self.response_composer.synthesize_from_state(state)

        self.memory_system.save_turn(
            request,
            state.final_response,
            metadata={
                "assigned_agent": state.route.agent_name if state.route else "general",
                "actions": state.execution.actions if state.execution else [],
                "approval_mode": state.safety.permission.mode.value,
                "verification_summary": state.verification.summary if state.verification else "",
                "reflection_lessons": state.reflection.lessons if state.reflection else [],
                "unresolved": state.execution.unresolved if state.execution else [],
                "route_bias": state.route_bias,
            },
            active_goals=state.context.active_goals if state.context else [],
        )
        state.memory = self.memory_system.build_snapshot(
            request.user_id,
            request.session_id,
            active_goals=state.context.active_goals if state.context else [],
            query=request.message,
            constraints=(state.context.constraints if state.context else []) + state.adaptive_constraints,
        )
        self._append_trace(
            state,
            "Persistent Memory Update",
            "Updated episodic, semantic, vector, and working memory after the loop converged.",
            {
                "history_length": self.memory_system.history_length(request.user_id, request.session_id),
                "goal_stack": state.memory.goal_stack,
                "retrieved_count": len(state.memory.retrieved),
            },
        )

        confirmation = build_confirmation(
            state.route.agent_name if state.route else "general",
            state.execution.actions if state.execution else [],
            state.safety.permission,
        )
        self._append_trace(
            state,
            "Response / Approval / Action Result",
            confirmation,
            {
                "assigned_agent": state.route.agent_name if state.route else "general",
                "approval_mode": state.safety.permission.mode.value,
                "requires_confirmation": state.safety.permission.requires_confirmation,
                "safety_risk_level": state.safety.risk_level,
                "termination_reason": state.termination_reason,
            },
        )

        response = InteractionResponse(
            request_id=state.gateway.request_id,
            response=state.final_response,
            confirmation=confirmation,
            assigned_agent=state.route.agent_name if state.route else "general",
            gateway=state.gateway,
            session=state.session,
            control=state.control,
            plan=state.plan,
            task_graph=state.task_graph or TaskGraph(state="completed", active_path=[], nodes=[]),
            skills=state.skills,
            verification=state.verification or skipped_verification(),
            reflection=state.reflection or skipped_reflection(),
            safety=state.safety or reviewed_safety(),
            memory=state.memory,
            trace=state.trace,
            tool_results=state.tool_history,
            state_transitions=state.state_transitions,
            state_id=state.state_id,
            loop_count=state.step_index,
            termination_reason=state.termination_reason,
        )

        self.memory_system.log_interaction(
            {
                "request": dump_model(request),
                "state": dump_model(state),
                "response": dump_model(response),
                "trace": [dump_model(event) for event in state.trace],
            }
        )
        return response

    def session_state(self, user_id: str, session_id: str):
        return self.memory_system.get_session_state(user_id, session_id)

    def preview_plan(
        self,
        request: UserRequest,
        headers: GatewayHeaders | None = None,
    ) -> AgentState:
        state = AgentState(
            request=request,
            headers=headers or GatewayHeaders(),
            max_steps=self._max_steps_for(request),
        )
        self._append_trace(
            state,
            "User / UI",
            f"Received a {request.channel.value} planning request for session '{request.session_id}'.",
            {"user_id": request.user_id},
        )
        state.gateway = self.api_gateway.process(request, headers=state.headers)
        self._append_trace(
            state,
            "API Gateway",
            "Validated the planning request through auth, normalization, and rate policies.",
            {
                "accepted": state.gateway.accepted,
                "client_id": state.gateway.client_id,
                "auth_mode": state.gateway.auth.mode.value,
            },
        )
        state.session = self.session_permission_manager.prepare(request, state.gateway, self.memory_system)
        self._append_trace(
            state,
            "Session + Permissions",
            state.session.permission.reason,
            {
                "permission_mode": state.session.permission.mode.value,
                "requires_confirmation": state.session.permission.requires_confirmation,
            },
        )
        if not state.gateway.accepted or state.session.permission.mode == PermissionMode.blocked:
            state.status = "blocked"
            state.memory = self.memory_system.build_snapshot(
                request.user_id,
                request.session_id,
                query=request.message,
            )
            state.control = ControlDecision(
                intent="rejected",
                control_notes="Planning stopped because preflight rejected the request.",
                preferred_agent="general",
                urgency="normal",
                complexity="low",
                reasoning_mode="none",
                llm_role="not_invoked",
                coordination_pattern="blocked",
                risk_level="high",
                memory_strategy="skip",
                verification_mode="skipped",
                needs_tooling=False,
            )
            state.plan = ExecutionPlan(
                objective=request.message,
                reasoning="Preflight checks blocked plan preview.",
                react_cycles=[],
                steps=[],
                assumptions=[],
                constraints=["Execution is blocked pending permission review."],
                success_criteria=[],
                failure_modes=[],
                verification_focus=[],
                decomposition_strategy="blocked",
                completion_state="blocked",
            )
            state.task_graph = TaskGraph(state="blocked", active_path=["session_permissions"], nodes=[])
            return state

        state.step_index = 1
        self._observe(state)
        self._control(state)
        self._plan(state)
        self._route(state)
        return state

    def _observe(self, state: AgentState) -> None:
        state.status = "observe"
        self.context_builder.observe(state, self.memory_system)
        self._append_trace(
            state,
            "Observe",
            f"Step {state.step_index}: refreshed context from memory, goals, constraints, and prior observations.",
            {
                "retrieved_count": len(state.memory.retrieved),
                "active_goals": state.context.active_goals if state.context else [],
                "constraints": state.context.constraints if state.context else [],
                "adaptive_constraints": state.adaptive_constraints,
            },
        )

    def _control(self, state: AgentState) -> None:
        state.status = "control"
        state.control = self.orchestrator_brain.decide(state)
        reasoning_note = (
            f"Step {state.step_index}: controlling over {len(state.memory.retrieved)} retrieved memories, "
            f"{len([obs for obs in state.observations if obs.step_index <= state.step_index])} observations, "
            f"retry_count={state.retry_count}, replan_count={state.replan_count}, route_bias={state.route_bias or 'none'}."
        )
        state.reasoning_notes.append(reasoning_note)
        self._append_trace(
            state,
            "Control",
            f"Step {state.step_index}: recomputed control policy from the live AgentState.",
            {
                "intent": state.control.intent,
                "preferred_agent": state.control.preferred_agent,
                "complexity": state.control.complexity,
                "risk_level": state.control.risk_level,
                "retry_count": state.retry_count,
                "reasoning_note": reasoning_note,
            },
        )

    def _plan(self, state: AgentState) -> None:
        state.status = "plan"
        replanning = state.needs_replan and state.plan is not None
        state.plan = self.planner.create_plan(state)
        if replanning:
            state.replan_count += 1
        state.needs_replan = False
        if state.control is None:
            raise ValueError("Control must exist before task-graph planning.")
        state.task_graph = self.task_graph_engine.build(state.plan, state.control)
        self._append_trace(
            state,
            "Plan",
            (
                f"Step {state.step_index}: regenerated the execution plan."
                if replanning
                else f"Step {state.step_index}: generated the execution plan."
            ),
            {
                "replanning": replanning,
                "replan_count": state.replan_count,
                "constraints": state.plan.constraints,
                "verification_focus": state.plan.verification_focus,
            },
        )

    def _route(self, state: AgentState) -> None:
        state.status = "route"
        state.route = self.router_executor.route(state)
        state.skills = self.router_executor.skills_for(state, state.route.agent_name)
        state.decision = None
        state.pending_tool_requests = []
        if state.task_graph is not None:
            state.task_graph = self.task_graph_engine.mark_route(state.task_graph, state.route.agent_name)
        self._append_trace(
            state,
            "Route",
            f"Step {state.step_index}: selected the next specialist branch from the current AgentState.",
            {
                "route": state.route.agent_name,
                "route_rationale": state.route.rationale,
                "skills": [skill.name for skill in state.skills],
                "route_bias": state.route_bias,
            },
        )

    def _agent_decide(self, state: AgentState) -> None:
        state.status = "agent_decide"
        if state.route is None:
            raise ValueError("Route must exist before agent decision.")
        state.decision, _agent = self.router_executor.decide(state, state.route.agent_name, state.skills)
        state.pending_tool_requests = state.decision.tool_requests
        if state.task_graph is not None:
            state.task_graph = self.task_graph_engine.mark_decision(
                state.task_graph,
                state.route.agent_name,
                tool_count=len(state.pending_tool_requests),
            )
        self._append_trace(
            state,
            "Agent Decide",
            f"Step {state.step_index}: the routed specialist converted state into a structured action decision.",
            {
                "agent_name": state.decision.agent_name,
                "skills": state.decision.skill_names,
                "tools_requested": [tool.tool_name for tool in state.pending_tool_requests],
                "claims_to_verify": state.decision.claims_to_verify,
                "blocked_tools": state.blocked_tools,
            },
        )

    def _execute(self, state: AgentState) -> None:
        state.status = "execute"
        if state.route is None:
            raise ValueError("Route must exist before execution.")
        if state.decision is None:
            raise ValueError("Agent decision must exist before execution.")
        agent = self.router_executor.agent_for(state.route.agent_name)
        state.execution = self.execution_engine.execute(state, agent)
        state.goal_status = state.execution.goal_status
        state.last_tool_results = state.execution.tool_results
        state.tool_history.extend(state.last_tool_results)
        self._record_observations(state)

        for tool_result in state.last_tool_results:
            self._append_trace(
                state,
                "Execute",
                f"Connector '{tool_result.tool_name}' completed with status '{tool_result.status}'.",
                {
                    "output": tool_result.output,
                    "evidence": tool_result.evidence,
                    "risk_level": tool_result.risk_level,
                },
            )

    def _verify(self, state: AgentState) -> None:
        state.status = "verify"
        state.verification = self.verification_engine.verify(state)
        state.needs_retry = state.verification.retry_recommended
        self._append_trace(
            state,
            "Verify",
            state.verification.summary,
            {
                "status": state.verification.status,
                "verified_claims": state.verification.verified_claims,
                "unverified_claims": state.verification.unverified_claims,
                "gaps": state.verification.gaps,
            },
        )

    def _reflect(self, state: AgentState) -> None:
        state.status = "reflect"
        state.reflection = self.reflection_engine.review(state)
        self._append_trace(
            state,
            "Reflect",
            state.reflection.summary,
            {
                "status": state.reflection.status,
                "issues": state.reflection.issues,
                "repairs": state.reflection.repairs,
                "route_bias": state.route_bias,
                "blocked_tools": state.blocked_tools,
            },
        )

    def _update_state(self, state: AgentState, prior_signature: dict[str, object]) -> None:
        state.status = "update_state"
        state.memory = self.memory_system.checkpoint_iteration(state, checkpoint="update_state")
        if state.context is not None:
            state.context.memory = state.memory
        transition = StateTransition(
            step_index=state.step_index,
            prior_status=str(prior_signature.get("status", "initialized")),
            next_status=state.status,
            observations=[
                observation.summary
                for observation in state.observations
                if observation.step_index == state.step_index
            ][:4],
            reasoning_summary=state.reasoning_notes[-1] if state.reasoning_notes else "",
            selected_route=state.route.agent_name if state.route else "general",
            retry_recommended=state.verification.retry_recommended if state.verification else False,
            replan_required=state.needs_replan or state.needs_retry,
            ready_for_response=state.execution.ready_for_response if state.execution else False,
            adaptive_constraints=list(state.adaptive_constraints),
            blocked_tools=list(state.blocked_tools),
            route_bias=state.route_bias,
            termination_signal=(
                "goal_achieved"
                if state.execution and state.execution.ready_for_response and not (state.verification and state.verification.retry_recommended)
                else "retry"
                if state.verification and state.verification.retry_recommended
                else "continue"
            ),
        )
        state.state_transitions.append(transition)
        self._append_trace(
            state,
            "Update State",
            f"Step {state.step_index}: applied S_(t+1) = F(S_t, O_t) and checkpointed the next live state.",
            {
                "formula": transition.formula,
                "selected_route": transition.selected_route,
                "retry_recommended": transition.retry_recommended,
                "ready_for_response": transition.ready_for_response,
                "retrieved_count": len(state.memory.retrieved),
                "open_loops": state.memory.open_loops,
                "working_checkpoint": state.memory.working_memory.plan_checkpoint,
            },
        )

    def _state_signature(self, state: AgentState) -> dict[str, object]:
        return {
            "status": state.status,
            "route_bias": state.route_bias,
            "blocked_tools": list(state.blocked_tools),
            "retry_count": state.retry_count,
            "replan_count": state.replan_count,
        }

    def _record_observations(self, state: AgentState) -> None:
        from .models import ObservationRecord

        for tool_result in state.last_tool_results:
            state.observations.append(
                ObservationRecord(
                    step_index=state.step_index,
                    source=tool_result.tool_name,
                    summary=tool_result.evidence[0] if tool_result.evidence else f"{tool_result.tool_name}:{tool_result.status}",
                    payload=tool_result.output,
                    evidence=tool_result.evidence,
                )
            )

        if state.execution:
            state.observations.append(
                ObservationRecord(
                    step_index=state.step_index,
                    source=f"{state.execution.agent_name}_summary",
                    summary=state.execution.summary,
                    payload=state.execution.artifacts,
                    evidence=state.execution.observations,
                )
            )

    def _blocked_response(self, state: AgentState) -> InteractionResponse:
        state.status = "blocked"
        state.termination_reason = "preflight_blocked"
        state.goal_status = "blocked"
        state.response_ready = True
        state.final_response = "The request could not continue because preflight gateway or permission checks rejected it."
        state.control = ControlDecision(
            intent="rejected",
            control_notes="Execution stopped before the agent loop could start.",
            preferred_agent="general",
            urgency="normal",
            complexity="low",
            reasoning_mode="none",
            llm_role="not_invoked",
            coordination_pattern="blocked",
            risk_level="high",
            memory_strategy="skip",
            verification_mode="skipped",
            needs_tooling=False,
        )
        state.plan = ExecutionPlan(
            objective=state.request.message.strip(),
            reasoning="Preflight checks blocked the loop before planning could begin.",
            react_cycles=[],
            steps=[],
            assumptions=["The request cannot proceed until gateway or permission issues are resolved."],
            constraints=["Do not execute actions while the request is blocked."],
            success_criteria=["Surface the block reason clearly to the caller."],
            failure_modes=["The runtime proceeds despite preflight rejection."],
            verification_focus=[],
            decomposition_strategy="blocked",
            completion_state="blocked",
        )
        state.task_graph = TaskGraph(
            state="blocked",
            active_path=["api_gateway", "session_permissions"],
            nodes=[],
        )
        verification = skipped_verification(
            "Verification skipped because execution never progressed past preflight checks."
        )
        reflection = skipped_reflection(
            "Reflection skipped because execution never progressed past preflight checks."
        )
        safety = reviewed_safety(
            state.session.permission,
            status="blocked",
            notes=["The request was blocked before execution."],
            risk_level="high",
        )
        state.memory = self.memory_system.build_snapshot(state.request.user_id, state.request.session_id, query=state.request.message)
        self._append_trace(
            state,
            "Response / Approval / Action Result",
            "Request blocked before execution.",
            {
                "assigned_agent": "general",
                "approval_mode": state.session.permission.mode.value,
            },
        )

        response = InteractionResponse(
            request_id=state.gateway.request_id,
            response=state.final_response,
            confirmation="Request blocked before execution.",
            assigned_agent="general",
            gateway=state.gateway,
            session=state.session,
            control=state.control,
            plan=state.plan,
            task_graph=state.task_graph,
            skills=[],
            verification=verification,
            reflection=reflection,
            safety=safety,
            memory=state.memory,
            trace=state.trace,
            tool_results=[],
            state_transitions=[],
            state_id=state.state_id,
            loop_count=0,
            termination_reason=state.termination_reason,
        )

        self.memory_system.log_interaction(
            {
                "request": dump_model(state.request),
                "state": dump_model(state),
                "response": dump_model(response),
                "trace": [dump_model(event) for event in state.trace],
            }
        )
        return response

    def _merge_permissions(
        self,
        preflight: PermissionDecision,
        final_review: PermissionDecision,
    ) -> PermissionDecision:
        if PermissionMode.blocked in {preflight.mode, final_review.mode}:
            return PermissionDecision(
                mode=PermissionMode.blocked,
                requires_confirmation=False,
                reason="Execution was blocked by permission policy.",
            )

        if preflight.requires_confirmation or final_review.requires_confirmation:
            return PermissionDecision(
                mode=PermissionMode.confirm_required,
                requires_confirmation=True,
                reason=f"{preflight.reason} {final_review.reason}".strip(),
            )

        return PermissionDecision(
            mode=PermissionMode.auto_approved,
            requires_confirmation=False,
            reason=f"{preflight.reason} {final_review.reason}".strip(),
        )

    def _append_trace(
        self,
        state: AgentState,
        stage: str,
        detail: str,
        payload: dict,
    ) -> None:
        state.trace.append(TraceEvent(stage=stage, detail=detail, payload=payload))

    def _max_steps_for(self, request: UserRequest) -> int:
        requested = request.metadata.get("max_steps")
        if isinstance(requested, int) and 1 <= requested <= 8:
            return requested
        return 4


def build_default_orchestrator() -> Orchestrator:
    load_dotenv()
    memory_system = MemorySystem()
    skill_library = SkillLibrary()
    api_gateway = InterfaceGateway()
    session_permission_manager = SessionPermissionManager()
    context_builder = ContextBuilder()
    orchestrator_brain = OrchestratorBrain()
    planner = Planner()
    task_graph_engine = TaskGraphEngine()
    agents = build_agent_registry()
    tool_registry = ToolRegistry(memory_system, skill_library)
    execution_engine = ExecutionEngine(tool_registry)
    router_executor = RouterExecutor(
        agent_router=AgentRouter(),
        skill_library=skill_library,
        agents=agents,
    )
    verification_engine = VerificationEngine()
    reflection_engine = ReflectionEngine()
    safety_permissions = SafetyPermissions()
    response_composer = ResponseComposer()

    return Orchestrator(
        memory_system=memory_system,
        api_gateway=api_gateway,
        session_permission_manager=session_permission_manager,
        context_builder=context_builder,
        orchestrator_brain=orchestrator_brain,
        planner=planner,
        task_graph_engine=task_graph_engine,
        router_executor=router_executor,
        execution_engine=execution_engine,
        verification_engine=verification_engine,
        reflection_engine=reflection_engine,
        safety_permissions=safety_permissions,
        response_composer=response_composer,
    )
