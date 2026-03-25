from __future__ import annotations

from dotenv import load_dotenv

from src.schemas.adaptive import LoopPhase
from src.schemas.context import ProviderRouteHint, ProviderRoutingHints
from src.schemas.routing import ModelTaskType
from src.services.llm_service import LLMService
from src.services.planner_service import PlannerService
from src.services.reflection_service import ReflectionService

from .agents import build_agent_registry
from .architecture_selector import ArchitectureSelector
from .claim_verifier import ClaimVerifier
from .context_builder import ContextBuilder
from .control import OrchestratorBrain
from .execution import ExecutionEngine
from .gateway import InterfaceGateway
from .handoff_engine import HandoffEngine
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
from .observability import RuntimeObservabilityEngine
from .planner import Planner
from .reasoning import ReasoningEngine
from .reflection import ReflectionEngine
from .response_helpers import build_confirmation, reviewed_safety, skipped_reflection, skipped_verification
from .responder import ResponseComposer
from .router import AgentRouter
from .router_executor import RouterExecutor
from .safety import SafetyPermissions
from .session_permissions import SessionPermissionManager
from .skills import SkillLibrary
from .stopping import StoppingEngine
from .task_graph import TaskGraphEngine
from .tool_registry import ToolRegistry
from .tool_selection import ToolSelectionEngine
from .runtime_utils import selected_model_override
from .verification import VerificationEngine


class Orchestrator:
    def __init__(
        self,
        memory_system: MemorySystem,
        api_gateway: InterfaceGateway,
        session_permission_manager: SessionPermissionManager,
        context_builder: ContextBuilder,
        architecture_selector: ArchitectureSelector,
        orchestrator_brain: OrchestratorBrain,
        planner: Planner,
        task_graph_engine: TaskGraphEngine,
        router_executor: RouterExecutor,
        reasoning_engine: ReasoningEngine,
        tool_selection_engine: ToolSelectionEngine,
        execution_engine: ExecutionEngine,
        verification_engine: VerificationEngine,
        reflection_engine: ReflectionEngine,
        handoff_engine: HandoffEngine,
        runtime_observability: RuntimeObservabilityEngine,
        stopping_engine: StoppingEngine,
        safety_permissions: SafetyPermissions,
        response_composer: ResponseComposer,
        llm_service: LLMService | None = None,
    ) -> None:
        self.memory_system = memory_system
        self.api_gateway = api_gateway
        self.session_permission_manager = session_permission_manager
        self.context_builder = context_builder
        self.architecture_selector = architecture_selector
        self.orchestrator_brain = orchestrator_brain
        self.planner = planner
        self.task_graph_engine = task_graph_engine
        self.router_executor = router_executor
        self.reasoning_engine = reasoning_engine
        self.tool_selection_engine = tool_selection_engine
        self.execution_engine = execution_engine
        self.verification_engine = verification_engine
        self.reflection_engine = reflection_engine
        self.handoff_engine = handoff_engine
        self.runtime_observability = runtime_observability
        self.stopping_engine = stopping_engine
        self.safety_permissions = safety_permissions
        self.response_composer = response_composer
        self._llm_service = llm_service

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
        self._bootstrap_loop_state(state)
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
            state.reasoning = None
            state.tool_selection = None
            state.execution = None
            state.verification = None
            state.reflection = None
            state.stop_decision = None
            state.last_tool_results = []
            prior_signature = self._state_signature(state)
            react_trace = self._new_react_trace(state)
            self._start_iteration(state, react_trace)
            self._append_trace(
                state,
                "Iteration Start",
                f"Starting loop iteration {state.step_index}.",
                {
                    "step_index": state.step_index,
                    "retry_count": state.retry_count,
                    "replan_count": state.replan_count,
                    "needs_replan": state.needs_replan,
                    "retry_budget": state.loop_state.retry_budget,
                },
            )
            self._observe(state, react_trace)
            self._select_architecture(state, react_trace)
            self._control(state, react_trace)
            self._ensure_plan(state, react_trace)
            self._route(state, react_trace)
            react_trace.selected_route = state.route.agent_name if state.route else "general"
            self._agent_decide(state, react_trace)
            self._reason(state, react_trace)
            self._act(state, react_trace)
            self._verify(state, react_trace)
            self._reflect(state, react_trace)
            self._update_state(state, prior_signature, react_trace)
            self._stop(state, react_trace)
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
                    "stop_trigger": state.stop_decision.trigger if state.stop_decision else "continue",
                },
            )
            self.runtime_observability.record_iteration(state)

            if state.stop_decision and state.stop_decision.should_stop:
                state.response_ready = True
                state.goal_status = "achieved" if state.stop_decision.trigger == "goal_achieved" else state.stop_decision.trigger
                state.termination_reason = state.stop_decision.trigger
                break

            if state.stop_decision and state.stop_decision.requires_replan:
                state.retry_count += 1
                state.needs_replan = True
                state.goal_status = "retrying"
                self._append_trace(
                    state,
                    "Loop Control",
                    "The ReAct loop requested another iteration with updated reasoning or plan state.",
                    {
                        "retry_count": state.retry_count,
                        "replan_count": state.replan_count,
                        "route_bias": state.route_bias,
                        "blocked_tools": state.blocked_tools,
                        "retry_budget": state.loop_state.retry_budget,
                        "reason": state.stop_decision.reason if state.stop_decision else "",
                    },
                )
                continue

        if not state.termination_reason:
            state.termination_reason = "max_steps_reached"
            state.goal_status = "max_steps_reached"
        state.response_ready = True
        self._finalize_loop_state(state)

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
                "handoff_summary": (
                    state.handoff_packet.summary.completed_work_summary if state.handoff_packet else ""
                ),
                "handoff_open_questions": (
                    [question.question for question in state.handoff_packet.open_questions[:3]]
                    if state.handoff_packet
                    else []
                ),
                "claim_verification_summary": (
                    state.claim_verification.summary if state.claim_verification else ""
                ),
                "unsupported_claims": (
                    state.claim_verification.unsupported_claims[:3]
                    if state.claim_verification
                    else []
                ),
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
        self.runtime_observability.finalize_trace(state)

        response = InteractionResponse(
            request_id=state.gateway.request_id,
            response=state.final_response,
            confirmation=confirmation,
            assigned_agent=state.route.agent_name if state.route else "general",
            gateway=state.gateway,
            session=state.session,
            control=state.control,
            context_packet=state.context_packet,
            handoff_packet=state.handoff_packet,
            architecture=state.architecture,
            loop_state=state.loop_state,
            plan=state.plan,
            task_graph=state.task_graph or TaskGraph(state="completed", active_path=[], nodes=[]),
            skills=state.skills,
            verification=state.verification or skipped_verification(),
            claim_verification=state.claim_verification,
            reflection=state.reflection or skipped_reflection(),
            safety=state.safety or reviewed_safety(),
            memory=state.memory,
            autonomy_metrics=state.autonomy_metrics,
            runtime_trace=state.runtime_trace,
            trace=state.trace,
            react_trace=state.react_trace,
            step_traces=state.step_traces,
            tool_results=state.tool_history,
            state_transitions=state.state_transitions,
            model_runs=state.model_runs,
            model_evaluations=state.model_evaluations,
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
        self._bootstrap_loop_state(state)
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
                task_summary="Planning preview stopped because preflight checks blocked the request.",
                subtasks=[],
                required_tools=[],
                risk_level="high",
                approval_needed=True,
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
        preview_trace = self._new_react_trace(state)
        self._start_iteration(state, preview_trace)
        self._observe(state, preview_trace)
        self._select_architecture(state, preview_trace)
        self._control(state, preview_trace)
        self._plan(state, preview_trace)
        self._route(state, preview_trace)
        return state

    def _observe(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.OBSERVE)
        self.context_builder.observe(
            state,
            self.memory_system,
            provider_routing_hints=self._provider_routing_hints(state),
        )
        react_trace.observe_summary = (
            f"Loaded {len(state.memory.retrieved)} retrieved memories and "
            f"{len(state.context.active_goals) if state.context else 0} active goal(s)."
        )
        react_trace.context_summary = state.context_packet.context_summary if state.context_packet else ""
        self._append_trace(
            state,
            "Observe",
            f"Step {state.step_index}: refreshed context from memory, goals, constraints, and prior observations.",
            {
                "retrieved_count": len(state.memory.retrieved),
                "active_goals": state.context.active_goals if state.context else [],
                "constraints": state.context.constraints if state.context else [],
                "execution_mode": state.context.execution_mode if state.context else "react",
                "adaptive_constraints": state.adaptive_constraints,
            },
        )
        self._complete_phase(state, LoopPhase.OBSERVE)

    def _select_architecture(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.SELECT_ARCHITECTURE)
        state.architecture = self.architecture_selector.select(state)
        state.loop_state.architecture_mode = state.architecture.mode
        react_trace.architecture_mode = state.architecture.mode.value
        react_trace.architecture_summary = state.architecture.reasoning.summary
        if state.context is not None:
            state.context.execution_mode = state.architecture.loop_strategy
            state.context.metadata["architecture_mode"] = state.architecture.mode.value
        if state.context_packet is not None:
            state.context_packet = state.context_packet.model_copy(
                update={
                    "current_execution_mode": state.architecture.loop_strategy,
                    "metadata": {
                        **state.context_packet.metadata,
                        "architecture_mode": state.architecture.mode.value,
                    },
                }
            )
            if state.context is not None:
                state.context.context_packet = state.context_packet
        self._append_trace(
            state,
            "Select Architecture",
            state.architecture.reasoning.summary,
            {
                "mode": state.architecture.mode.value,
                "pattern_label": state.architecture.pattern_label,
                "primary_agent": state.architecture.primary_agent,
                "supporting_agents": state.architecture.supporting_agents,
                "requires_planning": state.architecture.requires_planning,
                "requires_verifier": state.architecture.requires_verifier,
                "critic_lane": state.architecture.critic_lane,
                "parallel_fanout": state.architecture.parallel_fanout,
                "task_characteristics": state.architecture.task_characteristics.model_dump(),
                "pattern_scores": state.architecture.reasoning.pattern_scores,
            },
        )
        self._complete_phase(state, LoopPhase.SELECT_ARCHITECTURE)

    def _control(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.CONTROL)
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
        self._complete_phase(state, LoopPhase.CONTROL)

    def _plan(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.PLAN)
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
        self._complete_phase(state, LoopPhase.PLAN)

    def _ensure_plan(self, state: AgentState, react_trace) -> None:
        if state.plan is None or state.needs_replan:
            self._plan(state, react_trace)
            return
        self._enter_phase(state, react_trace, LoopPhase.PLAN)
        self._append_trace(
            state,
            "Plan",
            f"Step {state.step_index}: reused the current execution plan because no replanning trigger was active.",
            {
                "replanning": False,
                "replan_count": state.replan_count,
                "constraints": state.plan.constraints if state.plan else [],
            },
        )
        self._complete_phase(state, LoopPhase.PLAN)

    def _route(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.ROUTE)
        state.route = self.router_executor.route(state)
        state.skills = self.router_executor.skills_for(state, state.route.agent_name)
        state.decision = None
        state.pending_tool_requests = []
        state.loop_state.active_agent = state.route.agent_name
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
        self._complete_phase(state, LoopPhase.ROUTE)

    def _agent_decide(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.AGENT_DECIDE)
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
        self._complete_phase(state, LoopPhase.AGENT_DECIDE)

    def _reason(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.REASON)
        if state.decision is None:
            raise ValueError("Agent decision must exist before reasoning.")
        state.reasoning = self.reasoning_engine.reason(state)
        react_trace.reasoning_summary = state.reasoning.reasoning_summary
        state.reasoning_notes.append(state.reasoning.reasoning_summary)
        if state.reasoning.should_replan and state.reasoning.replan_reason:
            state.needs_replan = True
            if state.reasoning.replan_reason not in state.adaptive_constraints:
                state.adaptive_constraints.append(state.reasoning.replan_reason)
        self._append_trace(
            state,
            "Reason",
            state.reasoning.reasoning_summary,
            {
                "thought": state.reasoning.thought,
                "action_strategy": state.reasoning.action_strategy,
                "candidate_tools": state.reasoning.candidate_tools,
                "should_replan": state.reasoning.should_replan,
                "replan_reason": state.reasoning.replan_reason,
            },
        )
        self._complete_phase(state, LoopPhase.REASON)

    def _act(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.ACT)
        if state.route is None:
            raise ValueError("Route must exist before execution.")
        if state.decision is None:
            raise ValueError("Agent decision must exist before execution.")
        state.tool_selection, selected_requests = self.tool_selection_engine.select(state)
        state.pending_tool_requests = selected_requests
        react_trace.selected_tools = [request.tool_name for request in selected_requests]
        if state.task_graph is not None:
            state.task_graph = self.task_graph_engine.mark_reasoning(state.task_graph, react_trace.selected_tools)
        self._append_trace(
            state,
            "Tool Selection",
            state.tool_selection.rationale,
            {
                "selected_tools": state.tool_selection.selected_tool_names,
                "deferred_tools": state.tool_selection.deferred_tool_names,
                "expected_outcome": state.tool_selection.expected_outcome,
                "requires_replan": state.tool_selection.requires_replan,
            },
        )
        if state.tool_selection.requires_replan or state.reasoning.should_replan:
            state.needs_replan = True
            state.needs_retry = True
            if state.tool_selection.replan_reason and state.tool_selection.replan_reason not in state.adaptive_constraints:
                state.adaptive_constraints.append(state.tool_selection.replan_reason)
            react_trace.action_summary = "Action deferred because reasoning or tool selection requested replanning."
            self._complete_phase(state, LoopPhase.ACT)
            return

        agent = self.router_executor.agent_for(state.route.agent_name)
        state.execution = self.execution_engine.execute(state, agent, selected_requests)
        state.goal_status = state.execution.goal_status
        state.last_tool_results = state.execution.tool_results
        state.tool_history.extend(state.last_tool_results)
        self._record_observations(state)
        react_trace.action_summary = state.execution.summary
        react_trace.observed_evidence = state.execution.observations[:6]

        for tool_result in state.last_tool_results:
            self._append_trace(
                state,
                "Act",
                f"Connector '{tool_result.tool_name}' completed with status '{tool_result.status}'.",
                {
                    "output": tool_result.output,
                    "evidence": tool_result.evidence,
                    "risk_level": tool_result.risk_level,
                },
            )
        self._complete_phase(state, LoopPhase.ACT)

    def _verify(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.VERIFY)
        if state.execution is None:
            state.verification = skipped_verification(
                "Verification skipped because the iteration replanned before acting."
            )
            react_trace.verification_summary = state.verification.summary
            state.loop_state.last_verification_status = state.verification.status
            self._append_trace(
                state,
                "Verify",
                state.verification.summary,
                {"status": state.verification.status},
            )
            self._complete_phase(state, LoopPhase.VERIFY)
            return
        state.verification = self.verification_engine.verify(state)
        state.needs_retry = state.verification.retry_recommended
        react_trace.verification_summary = state.verification.summary
        state.loop_state.last_verification_status = state.verification.status
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
        self._complete_phase(state, LoopPhase.VERIFY)

    def _reflect(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.REFLECT)
        if state.execution is None:
            state.reflection = skipped_reflection(
                "Reflection on execution skipped because reasoning requested replanning before acting."
            )
            react_trace.reflection_summary = state.reflection.summary
            state.loop_state.last_reflection_status = state.reflection.status
            self._append_trace(
                state,
                "Reflect",
                state.reflection.summary,
                {"status": state.reflection.status},
            )
            self._complete_phase(state, LoopPhase.REFLECT)
            return
        state.reflection = self.reflection_engine.review(state)
        react_trace.reflection_summary = state.reflection.summary
        state.loop_state.last_reflection_status = state.reflection.status
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
        self._complete_phase(state, LoopPhase.REFLECT)

    def _update_state(self, state: AgentState, prior_signature: dict[str, object], react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.UPDATE_MEMORY)
        checkpoint_name = f"react_iteration_{state.step_index}"
        state.memory = self.memory_system.checkpoint_iteration(state, checkpoint=checkpoint_name)
        if state.context is not None:
            state.context.memory = state.memory
        react_trace.memory_checkpoint = state.memory.working_memory.plan_checkpoint
        react_trace.memory_checkpoints.append(state.memory.working_memory.plan_checkpoint)
        react_trace.replan_triggered = state.needs_replan or state.needs_retry
        if checkpoint_name not in state.loop_state.memory_checkpoints:
            state.loop_state.memory_checkpoints.append(checkpoint_name)
        state.loop_state.should_replan = state.needs_replan or state.needs_retry
        state.loop_state.ready_for_response = state.execution.ready_for_response if state.execution else False
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
            architecture_mode=state.architecture.mode.value if state.architecture else None,
            loop_phase=LoopPhase.UPDATE_MEMORY.value,
            termination_signal=(
                "goal_achieved"
                if state.execution and state.execution.ready_for_response and not (state.verification and state.verification.retry_recommended)
                else "retry"
                if state.needs_replan or state.needs_retry
                else "retry"
                if state.verification and state.verification.retry_recommended
                else "continue"
            ),
        )
        state.state_transitions.append(transition)
        state.react_trace.append(react_trace)
        handoff_compaction = self._refresh_handoff_state(state)
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
                "handoff_id": state.handoff_packet.handoff_id if state.handoff_packet else None,
                "compaction_applied": bool(handoff_compaction),
                "compaction_trimmed": handoff_compaction,
            },
        )
        self._complete_phase(state, LoopPhase.UPDATE_MEMORY)

    def _refresh_handoff_state(self, state: AgentState) -> dict[str, int]:
        state.handoff_packet = self.handoff_engine.build_handoff(state)
        state.loop_state.handoff_available = True
        state.loop_state.last_handoff_id = state.handoff_packet.handoff_id
        state.loop_state.compaction_applied = False

        if state.context is not None:
            state.context.handoff_packet = state.handoff_packet
            state.context.metadata["handoff_packet"] = state.handoff_packet.model_dump(mode="json")
        if state.context_packet is not None:
            state.context_packet = state.context_packet.model_copy(
                update={
                    "metadata": {
                        **state.context_packet.metadata,
                        "handoff_packet": state.handoff_packet.model_dump(mode="json"),
                    }
                }
            )
            if state.context is not None:
                state.context.context_packet = state.context_packet

        if not self.handoff_engine.should_compact(state):
            return {}

        trimmed = self.handoff_engine.compact_state(state)
        state.loop_state.compaction_applied = bool(trimmed)
        return trimmed

    def _stop(self, state: AgentState, react_trace) -> None:
        self._enter_phase(state, react_trace, LoopPhase.LOOP_CONTROL)
        state.stop_decision = self.stopping_engine.decide(state)
        react_trace.stop_trigger = state.stop_decision.trigger
        react_trace.stop_reason = state.stop_decision.reason
        state.loop_state.last_stop_trigger = state.stop_decision.trigger
        state.loop_state.last_stop_reason = state.stop_decision.reason
        state.loop_state.ready_for_response = state.stop_decision.ready_for_response
        state.loop_state.should_replan = state.stop_decision.requires_replan
        self._append_trace(
            state,
            "Loop Control",
            state.stop_decision.reason,
            {
                "trigger": state.stop_decision.trigger,
                "should_stop": state.stop_decision.should_stop,
                "ready_for_response": state.stop_decision.ready_for_response,
                "requires_replan": state.stop_decision.requires_replan,
            },
        )
        self._complete_phase(state, LoopPhase.LOOP_CONTROL)

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
        state.loop_state.last_stop_trigger = "preflight_blocked"
        state.loop_state.last_stop_reason = "Execution was blocked before the ReAct loop could start."
        self._finalize_loop_state(state)
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
            task_summary="The request cannot proceed because preflight checks blocked execution.",
            subtasks=[],
            required_tools=[],
            risk_level="high",
            approval_needed=True,
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
        self.runtime_observability.record_preflight_block(state)
        self.runtime_observability.finalize_trace(state)

        response = InteractionResponse(
            request_id=state.gateway.request_id,
            response=state.final_response,
            confirmation="Request blocked before execution.",
            assigned_agent="general",
            gateway=state.gateway,
            session=state.session,
            control=state.control,
            context_packet=state.context_packet,
            handoff_packet=state.handoff_packet,
            architecture=state.architecture,
            loop_state=state.loop_state,
            plan=state.plan,
            task_graph=state.task_graph,
            skills=[],
            verification=verification,
            claim_verification=state.claim_verification,
            reflection=reflection,
            safety=safety,
            memory=state.memory,
            autonomy_metrics=state.autonomy_metrics,
            runtime_trace=state.runtime_trace,
            trace=state.trace,
            react_trace=state.react_trace,
            step_traces=state.step_traces,
            tool_results=[],
            state_transitions=[],
            model_runs=state.model_runs,
            model_evaluations=state.model_evaluations,
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

    def _new_react_trace(self, state: AgentState):
        from .models import ReActStepTrace

        return ReActStepTrace(step_index=state.step_index, selected_route=state.route.agent_name if state.route else "general")

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

    def _retry_budget_for(self, request: UserRequest) -> int:
        requested = request.metadata.get("retry_budget")
        max_retry_budget = max(0, self._max_steps_for(request) - 1)
        if isinstance(requested, int):
            return max(0, min(requested, max_retry_budget))
        return min(2, max_retry_budget)

    def _stop_conditions_for(self, request: UserRequest) -> list[str]:
        requested = request.metadata.get("stop_conditions")
        if isinstance(requested, list):
            normalized = [str(item).strip() for item in requested if str(item).strip()]
            if normalized:
                return normalized
        return [
            "goal_achieved",
            "max_steps_reached",
            "retry_budget_exhausted",
            "stalled",
        ]

    def _provider_routing_hints(self, state: AgentState) -> ProviderRoutingHints:
        if self._llm_service is None:
            return ProviderRoutingHints()

        selected_model = selected_model_override(state)
        task_types = (
            ModelTaskType.ORCHESTRATION,
            ModelTaskType.PLANNING,
            ModelTaskType.REASONING,
            ModelTaskType.COMMUNICATION,
            ModelTaskType.RESEARCH,
            ModelTaskType.WEB_GROUNDING,
        )
        hints: list[ProviderRouteHint] = []
        for task_type in task_types:
            decision = self._llm_service.model_router.route(task_type, selected_model=selected_model)
            hints.append(
                ProviderRouteHint(
                    task_type=task_type.value,
                    provider=decision.provider.value,
                    model=decision.model,
                    reason=decision.reason,
                    candidate_models=list(decision.candidate_models),
                )
            )
        return ProviderRoutingHints(hints=hints)

    def _bootstrap_loop_state(self, state: AgentState) -> None:
        state.loop_state.max_iterations = state.max_steps
        state.loop_state.retry_budget = self._retry_budget_for(state.request)
        state.loop_state.retry_count = state.retry_count
        state.loop_state.stop_conditions = self._stop_conditions_for(state.request)
        state.loop_state.phase = LoopPhase.OBSERVE
        state.loop_state.completed_phases = []
        state.loop_state.memory_checkpoints = []
        state.loop_state.should_replan = state.needs_replan
        state.loop_state.ready_for_response = state.response_ready
        state.loop_state.last_stop_trigger = "continue"
        state.loop_state.last_stop_reason = ""
        state.loop_state.last_verification_status = "pending"
        state.loop_state.last_reflection_status = "pending"
        state.loop_state.handoff_available = False
        state.loop_state.last_handoff_id = None
        state.loop_state.compaction_applied = False

    def _start_iteration(self, state: AgentState, react_trace) -> None:
        state.loop_state.iteration = state.step_index
        state.loop_state.max_iterations = state.max_steps
        state.loop_state.retry_count = state.retry_count
        state.loop_state.completed_phases = []
        state.loop_state.should_replan = state.needs_replan
        state.loop_state.ready_for_response = False
        state.loop_state.architecture_mode = state.architecture.mode if state.architecture else None
        state.loop_state.active_agent = state.route.agent_name if state.route else None
        state.loop_state.handoff_available = state.handoff_packet is not None
        state.loop_state.last_handoff_id = state.handoff_packet.handoff_id if state.handoff_packet else None
        state.loop_state.compaction_applied = False
        react_trace.architecture_mode = state.architecture.mode.value if state.architecture else None

    def _enter_phase(self, state: AgentState, react_trace, phase: LoopPhase) -> None:
        state.status = phase.value
        state.loop_state.phase = phase
        state.loop_state.iteration = state.step_index
        state.loop_state.max_iterations = state.max_steps
        state.loop_state.retry_count = state.retry_count
        state.loop_state.architecture_mode = state.architecture.mode if state.architecture else None
        if state.route is not None:
            state.loop_state.active_agent = state.route.agent_name
        if phase not in react_trace.loop_phases:
            react_trace.loop_phases.append(phase)
        react_trace.architecture_mode = state.architecture.mode.value if state.architecture else None

    def _complete_phase(self, state: AgentState, phase: LoopPhase) -> None:
        if phase not in state.loop_state.completed_phases:
            state.loop_state.completed_phases.append(phase)

    def _finalize_loop_state(self, state: AgentState) -> None:
        state.loop_state.phase = LoopPhase.COMPLETE
        state.loop_state.iteration = state.step_index
        state.loop_state.max_iterations = state.max_steps
        state.loop_state.retry_count = state.retry_count
        state.loop_state.should_replan = state.needs_replan or state.needs_retry
        state.loop_state.ready_for_response = state.response_ready
        state.loop_state.architecture_mode = state.architecture.mode if state.architecture else None
        state.loop_state.active_agent = state.route.agent_name if state.route else None
        state.loop_state.handoff_available = state.handoff_packet is not None
        state.loop_state.last_handoff_id = state.handoff_packet.handoff_id if state.handoff_packet else None
        self._complete_phase(state, LoopPhase.COMPLETE)


def build_default_orchestrator(llm_service: LLMService | None = None) -> Orchestrator:
    load_dotenv()
    resolved_llm_service = llm_service or LLMService()
    memory_system = MemorySystem()
    skill_library = SkillLibrary()
    api_gateway = InterfaceGateway()
    session_permission_manager = SessionPermissionManager()
    context_builder = ContextBuilder()
    architecture_selector = ArchitectureSelector()
    orchestrator_brain = OrchestratorBrain(resolved_llm_service)
    reflection_service = ReflectionService(resolved_llm_service)
    planner_service = PlannerService(resolved_llm_service, reflection_service)
    planner = Planner(resolved_llm_service, planner_service)
    reasoning_engine = ReasoningEngine(resolved_llm_service, reflection_service)
    tool_selection_engine = ToolSelectionEngine(resolved_llm_service)
    task_graph_engine = TaskGraphEngine()
    agents = build_agent_registry(resolved_llm_service)
    tool_registry = ToolRegistry(memory_system, skill_library)
    execution_engine = ExecutionEngine(tool_registry, resolved_llm_service)
    router_executor = RouterExecutor(
        agent_router=AgentRouter(),
        skill_library=skill_library,
        agents=agents,
    )
    claim_verifier = ClaimVerifier()
    verification_engine = VerificationEngine(resolved_llm_service, claim_verifier=claim_verifier)
    reflection_engine = ReflectionEngine(resolved_llm_service)
    handoff_engine = HandoffEngine()
    runtime_observability = RuntimeObservabilityEngine()
    stopping_engine = StoppingEngine()
    safety_permissions = SafetyPermissions()
    response_composer = ResponseComposer(
        resolved_llm_service,
        reflection_service,
        claim_verifier=claim_verifier,
    )

    return Orchestrator(
        memory_system=memory_system,
        api_gateway=api_gateway,
        session_permission_manager=session_permission_manager,
        context_builder=context_builder,
        architecture_selector=architecture_selector,
        orchestrator_brain=orchestrator_brain,
        planner=planner,
        task_graph_engine=task_graph_engine,
        router_executor=router_executor,
        reasoning_engine=reasoning_engine,
        tool_selection_engine=tool_selection_engine,
        execution_engine=execution_engine,
        verification_engine=verification_engine,
        reflection_engine=reflection_engine,
        handoff_engine=handoff_engine,
        runtime_observability=runtime_observability,
        stopping_engine=stopping_engine,
        safety_permissions=safety_permissions,
        response_composer=response_composer,
        llm_service=resolved_llm_service,
    )
