from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.schemas.catalog import ToolDescriptor
from src.schemas.context import (
    ApprovalState,
    ContextEpisodicMemory,
    ContextMemorySlice,
    ContextPacket,
    ContextSemanticMemoryResult,
    NormalizedUserRequest,
    ProviderRoutingHints,
    ToolAvailabilitySnapshot,
    ToolCapability,
)
from src.tools.catalog import get_tool_descriptors

from .memory import MemorySystem
from .models import (
    AgentState,
    ContextSignal,
    ContextSnapshot,
    GatewayResult,
    MemorySnapshot,
    SessionPermissionState,
    UserRequest,
)
from .runtime_utils import dedupe_preserve_order, recent_observation_summaries


ToolDescriptorLoader = Callable[[], list[ToolDescriptor]]


class ContextEngine:
    def __init__(self, tool_descriptor_loader: ToolDescriptorLoader | None = None) -> None:
        self._tool_descriptor_loader = tool_descriptor_loader or get_tool_descriptors

    def build_context(
        self,
        *,
        request: UserRequest,
        gateway: GatewayResult,
        memory_snapshot: MemorySnapshot,
        active_goals: Sequence[str],
        requested_capabilities: Sequence[str],
        constraints: Sequence[str],
        preferences: Mapping[str, Any] | None = None,
        available_tools: Sequence[ToolDescriptor] | None = None,
        approval_state: ApprovalState | None = None,
        provider_routing_hints: ProviderRoutingHints | None = None,
        current_execution_mode: str = "react",
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextPacket:
        memory_slice = self._build_memory_slice(
            memory_snapshot,
            retrieval_query=memory_snapshot.working_memory.objective or gateway.normalized_message,
        )
        tool_availability = self._build_tool_availability(
            available_tools if available_tools is not None else self._tool_descriptor_loader()
        )
        resolved_preferences = dict(preferences) if preferences is not None else dict(request.preferences)
        packet = ContextPacket(
            normalized_user_request=NormalizedUserRequest(
                raw_message=request.message,
                normalized_message=gateway.normalized_message,
                channel=request.channel.value,
                goals=list(request.goals),
                preferences=resolved_preferences,
                metadata=dict(request.metadata),
            ),
            memory=memory_slice,
            active_goals=list(active_goals),
            requested_capabilities=list(requested_capabilities),
            constraints=list(constraints),
            tool_availability=tool_availability,
            approval_state=approval_state or ApprovalState(),
            provider_routing_hints=provider_routing_hints or ProviderRoutingHints(),
            current_execution_mode=current_execution_mode,
            metadata=dict(metadata or {}),
        )
        return packet.model_copy(
            update={
                "context_summary": self._context_summary(
                    normalized_message=packet.normalized_user_request.normalized_message,
                    active_goals=packet.active_goals,
                    requested_capabilities=packet.requested_capabilities,
                    semantic_memory=packet.memory.semantic_memory_results,
                    constraints=packet.constraints,
                    execution_mode=packet.current_execution_mode,
                )
            }
        )

    def observe(
        self,
        state: AgentState,
        memory_system: MemorySystem,
        *,
        provider_routing_hints: ProviderRoutingHints | None = None,
        execution_mode: str | None = None,
    ) -> AgentState:
        gateway = state.gateway
        if gateway is None:
            raise ValueError("Gateway state must be available before context observation.")

        merged_preferences = memory_system.merge_preferences(
            state.request.user_id,
            state.request.session_id,
            state.request.preferences,
        )
        active_goals = self._derive_goals(state)
        constraints = self._derive_constraints(state, merged_preferences)
        requested_capabilities = self._derive_capabilities(state)
        signals = self._derive_signals(state, active_goals, requested_capabilities)

        memory_system.update_goals(state.request.user_id, state.request.session_id, active_goals)
        memory_snapshot = memory_system.build_snapshot(
            state.request.user_id,
            state.request.session_id,
            active_goals=active_goals,
            query=self._memory_query(state),
            constraints=constraints,
        )
        self._refresh_working_memory(
            state=state,
            gateway=gateway,
            memory_snapshot=memory_snapshot,
            constraints=constraints,
        )

        approval_state = self._build_approval_state(
            session=state.session,
            risk_level=signals.risk_level,
            blocked_tools=state.blocked_tools,
        )
        current_execution_mode = execution_mode or self._execution_mode(state)
        packet = self.build_context(
            request=state.request,
            gateway=gateway,
            memory_snapshot=memory_snapshot,
            active_goals=active_goals,
            requested_capabilities=requested_capabilities,
            constraints=constraints,
            preferences=merged_preferences,
            approval_state=approval_state,
            provider_routing_hints=provider_routing_hints,
            current_execution_mode=current_execution_mode,
            metadata=self._packet_metadata(state),
        )
        context = ContextSnapshot(
            user_id=state.request.user_id,
            session_id=state.request.session_id,
            channel=state.request.channel,
            latest_message=gateway.normalized_message,
            gateway=gateway,
            preferences=merged_preferences,
            history=memory_system.get_history(state.request.user_id, state.request.session_id)[-6:],
            memory=memory_snapshot,
            active_goals=active_goals,
            system_goals=[
                "stay_safe",
                "be_traceable",
                "verify_before_claiming",
                "treat_llm_as_component_not_system",
            ],
            constraints=constraints,
            requested_capabilities=requested_capabilities,
            signals=signals,
            context_packet=packet,
            handoff_packet=state.handoff_packet,
            execution_mode=current_execution_mode,
            metadata={
                **dict(state.request.metadata),
                **self._packet_metadata(state),
                "approval_granted": approval_state.approval_granted,
            },
        )

        state.memory = memory_snapshot
        state.context = context
        state.context_packet = packet
        return state

    def build(
        self,
        request: UserRequest,
        gateway: GatewayResult,
        memory_system: MemorySystem,
        *,
        session: SessionPermissionState | None = None,
        provider_routing_hints: ProviderRoutingHints | None = None,
        execution_mode: str | None = None,
    ) -> ContextSnapshot:
        state = AgentState(request=request, gateway=gateway, session=session)
        observed_state = self.observe(
            state,
            memory_system,
            provider_routing_hints=provider_routing_hints,
            execution_mode=execution_mode,
        )
        if observed_state.context is None:
            raise ValueError("Context observation did not produce a context snapshot.")
        return observed_state.context

    def _build_memory_slice(
        self,
        memory_snapshot: MemorySnapshot,
        *,
        retrieval_query: str,
    ) -> ContextMemorySlice:
        episodic_memory = [
            ContextEpisodicMemory(
                role=turn.role,
                content=turn.content,
                timestamp=turn.timestamp,
                metadata=dict(turn.metadata),
            )
            for turn in memory_snapshot.episodic[-4:]
        ]
        semantic_results = [
            ContextSemanticMemoryResult(
                memory_id=record.memory_id,
                memory_type=record.memory_type,
                content=record.content,
                source=record.source,
                score=record.score,
                salience=record.salience,
                tags=list(record.tags),
                attributes=dict(record.attributes),
            )
            for record in memory_snapshot.retrieved[:6]
        ]
        working_memory = memory_snapshot.working_memory
        return ContextMemorySlice(
            summary=memory_snapshot.summary,
            retrieval_query=retrieval_query,
            recent_episodic_memory=episodic_memory,
            semantic_memory_results=semantic_results,
            working_memory_summary=working_memory.distilled_context,
            open_loops=list(memory_snapshot.open_loops),
            goal_stack=list(memory_snapshot.goal_stack),
            retrieved_facts=list(working_memory.retrieved_facts),
        )

    def _build_tool_availability(
        self,
        descriptors: Sequence[ToolDescriptor],
    ) -> ToolAvailabilitySnapshot:
        tools = [
            ToolCapability(
                name=descriptor.name,
                category=descriptor.category,
                description=descriptor.description,
                risk_level=descriptor.risk_level,
                side_effect=descriptor.side_effect,
                structured_output=descriptor.structured_output,
                retryable=descriptor.retryable,
                supports_dry_run=descriptor.supports_dry_run,
                mcp_ready=descriptor.mcp_ready,
                contract_version=descriptor.contract_version,
            )
            for descriptor in descriptors
        ]
        categories = dedupe_preserve_order(tool.category for tool in tools)
        risky_tools = [tool.name for tool in tools if tool.risk_level in {"medium", "high", "critical"}]
        approval_gated_tools = [
            tool.name
            for tool in tools
            if tool.side_effect in {"draft", "send", "delete", "deploy", "purchase", "shutdown"}
            or tool.risk_level in {"high", "critical"}
        ]
        return ToolAvailabilitySnapshot(
            available_tools=tools,
            categories=categories,
            risky_tools=risky_tools,
            approval_gated_tools=approval_gated_tools,
            total_tools=len(tools),
        )

    def _build_approval_state(
        self,
        *,
        session: SessionPermissionState | None,
        risk_level: str,
        blocked_tools: Sequence[str],
    ) -> ApprovalState:
        if session is None:
            return ApprovalState(
                permission_mode="auto_approved",
                requires_confirmation=False,
                risk_level=risk_level,
                approval_granted=True,
                blocked_tools=list(blocked_tools),
                rationale="No session permission state was provided, so the context defaulted to auto-approved.",
            )

        permission = session.permission
        return ApprovalState(
            permission_mode=permission.mode.value,
            requires_confirmation=permission.requires_confirmation,
            risk_level=risk_level,
            approval_granted=permission.mode.value == "auto_approved" and not permission.requires_confirmation,
            blocked_tools=list(blocked_tools),
            gated_actions=["confirmation_required"] if permission.requires_confirmation else [],
            rationale=permission.reason,
        )

    def _context_summary(
        self,
        *,
        normalized_message: str,
        active_goals: Sequence[str],
        requested_capabilities: Sequence[str],
        semantic_memory: Sequence[ContextSemanticMemoryResult],
        constraints: Sequence[str],
        execution_mode: str,
    ) -> str:
        goal_summary = ", ".join(active_goals[:3]) or "respond_helpfully"
        capability_summary = ", ".join(requested_capabilities[:3]) or "reasoning"
        memory_summary = "; ".join(result.content for result in semantic_memory[:2]) or "No high-signal retrieved memory."
        constraint_summary = "; ".join(constraints[:2]) or "No additional adaptive constraints."
        return (
            f"Request: {normalized_message[:120]} | Goals: {goal_summary} | "
            f"Capabilities: {capability_summary} | Mode: {execution_mode} | "
            f"Memory: {memory_summary} | Constraints: {constraint_summary}"
        )[:700]

    def _packet_metadata(self, state: AgentState) -> dict[str, Any]:
        metadata = {
            "step_index": state.step_index,
            "retry_count": state.retry_count,
            "replan_count": state.replan_count,
            "blocked_tools": list(state.blocked_tools),
            "route_bias": state.route_bias,
        }
        if state.handoff_packet is not None:
            metadata.update(
                {
                    "handoff_packet": state.handoff_packet.model_dump(mode="json"),
                    "handoff_summary": state.handoff_packet.summary.completed_work_summary,
                    "handoff_next_actions": list(state.handoff_packet.next_actions[:3]),
                }
            )
        return metadata

    def _refresh_working_memory(
        self,
        *,
        state: AgentState,
        gateway: GatewayResult,
        memory_snapshot: MemorySnapshot,
        constraints: Sequence[str],
    ) -> None:
        working_memory = memory_snapshot.working_memory
        working_memory.objective = gateway.normalized_message
        working_memory.constraints = list(constraints)
        working_memory.plan_checkpoint = "observe"
        recent_observations = recent_observation_summaries(state, limit=3)
        if recent_observations:
            working_memory.distilled_context = (
                f"{memory_snapshot.summary} | Recent loop observations: {'; '.join(recent_observations)}"
            )[:500]
            for summary in recent_observations:
                if summary not in working_memory.retrieved_facts:
                    working_memory.retrieved_facts.append(summary)
        if state.handoff_packet is not None:
            working_memory.distilled_context = state.handoff_packet.reusable_context[:500]
            for question in state.handoff_packet.open_questions[:3]:
                if question.question not in working_memory.open_questions:
                    working_memory.open_questions.append(question.question)
            for item in state.handoff_packet.memory_snapshot.retrieved_facts[:3]:
                if item not in working_memory.retrieved_facts:
                    working_memory.retrieved_facts.append(item)
        if state.reflection and state.reflection.lessons:
            for lesson in state.reflection.lessons[:2]:
                if lesson not in working_memory.assumptions:
                    working_memory.assumptions.append(lesson)

    def _execution_mode(self, state: AgentState) -> str:
        if state.architecture is not None:
            return state.architecture.loop_strategy
        if state.context is not None and state.context.execution_mode:
            return state.context.execution_mode
        if state.context is not None and state.context.signals.collaboration_mode == "modular-orchestration":
            return "modular-orchestration"
        return "react"

    def _derive_goals(self, state: AgentState) -> list[str]:
        lowered = state.request.message.lower()
        goals = list(state.request.goals)
        retrieved_text = " ".join(record.content.lower() for record in state.memory.retrieved)

        keyword_map = {
            "design_system": ("architecture", "orchestrator", "workflow", "agent", "pipeline", "system"),
            "ship_code": ("code", "coding", "debug", "fix", "implement", "backend", "frontend", "api", "build"),
            "communicate": ("email", "message", "reply", "draft", "send"),
            "research": ("research", "find", "search", "lookup", "compare", "summarize"),
            "browse_web": ("browser", "website", "web", "page"),
            "manage_tasks": ("schedule", "calendar", "task", "remind", "meeting"),
            "inspect_files": ("file", "document", "pdf", "report", "doc"),
            "maintain_memory": ("remember", "history", "memory", "context"),
            "verify_outputs": ("verify", "verification", "check", "test", "prove"),
            "enforce_safety": ("safe", "safety", "guardrail", "approval", "gating"),
        }

        for goal, keywords in keyword_map.items():
            if any(keyword in lowered for keyword in keywords) and goal not in goals:
                goals.append(goal)

        if "verification summary" in retrieved_text and "verify_outputs" not in goals:
            goals.append("verify_outputs")
        if state.needs_retry and "verify_outputs" not in goals:
            goals.append("verify_outputs")
        if not goals:
            goals.append("respond_helpfully")
        return goals

    def _derive_constraints(
        self,
        state: AgentState,
        preferences: Mapping[str, Any],
    ) -> list[str]:
        lowered = state.request.message.lower()
        constraints: list[str] = list(state.adaptive_constraints)

        if "architecture" in lowered or "system" in lowered:
            constraints.append("Favor modular composition over monolithic control flow.")
        if "safety" in lowered or "approval" in lowered:
            constraints.append("Keep potentially external side effects behind explicit safety gates.")
        if "memory" in lowered:
            constraints.append("Ground responses in retrieved memory rather than recent turns alone.")
        if preferences:
            constraints.append("Respect persisted user preferences during planning and response generation.")
        if state.request.channel.value == "voice":
            constraints.append("Prefer concise spoken-ready outputs.")
        if state.retry_count > 0:
            constraints.append("Change strategy instead of repeating the previous failed attempt.")
        if state.blocked_tools:
            constraints.append(f"Avoid blocked tools: {', '.join(state.blocked_tools)}.")
        if state.session is not None and state.session.permission.requires_confirmation:
            constraints.append("Do not perform approval-gated side effects before confirmation is granted.")

        return dedupe_preserve_order(constraints)

    def _derive_capabilities(self, state: AgentState) -> list[str]:
        lowered = state.request.message.lower()
        retrieved_text = " ".join(record.content.lower() for record in state.memory.retrieved)
        capability_map = {
            "planning": ("plan", "planner", "roadmap", "decompose"),
            "coding": ("code", "build", "implement", "debug", "architecture"),
            "memory": ("memory", "context", "remember"),
            "verification": ("verify", "test", "check", "prove"),
            "reflection": ("reflect", "reflection", "repair"),
            "safety": ("safe", "safety", "approve", "gating", "permission"),
            "research": ("research", "search", "compare"),
            "browser": ("browser", "website", "page", "web"),
            "documents": ("file", "document", "pdf", "report"),
            "communication": ("email", "message", "reply", "draft"),
            "coordination": ("task", "schedule", "meeting", "calendar"),
        }

        selected = [
            capability
            for capability, keywords in capability_map.items()
            if any(keyword in lowered for keyword in keywords)
        ]
        if "verification summary" in retrieved_text and "verification" not in selected:
            selected.append("verification")
        if any("reflection lesson" in record.content.lower() for record in state.memory.retrieved) and "memory" not in selected:
            selected.append("memory")
        return selected or ["reasoning"]

    def _memory_query(self, state: AgentState) -> str:
        query_parts = [state.request.message]
        query_parts.extend(
            observation.summary
            for observation in state.observations
            if observation.step_index in {state.step_index, max(0, state.step_index - 1)}
        )
        if state.execution and state.execution.next_focus:
            query_parts.append(state.execution.next_focus)
        if state.handoff_packet is not None:
            query_parts.append(state.handoff_packet.summary.completed_work_summary)
            query_parts.extend(question.question for question in state.handoff_packet.open_questions[:2])
            query_parts.extend(state.handoff_packet.next_actions[:2])
        query_parts.extend(state.adaptive_constraints[:2])
        if state.route_bias:
            query_parts.append(f"preferred route {state.route_bias}")
        return " ".join(part for part in query_parts if part).strip()

    def _derive_signals(
        self,
        state: AgentState,
        active_goals: Sequence[str],
        requested_capabilities: Sequence[str],
    ) -> ContextSignal:
        lowered = state.request.message.lower()
        complexity = "moderate"
        if len(requested_capabilities) >= 4 or len(active_goals) >= 4 or state.replan_count > 0:
            complexity = "high"
        elif len(state.request.message.split()) <= 12 and len(requested_capabilities) <= 2:
            complexity = "low"

        risk_level = self._fallback_risk_level(state)
        if state.retry_count > 0 and risk_level == "low":
            risk_level = "medium"
        if state.session is not None and state.session.permission.requires_confirmation and risk_level == "low":
            risk_level = "medium"

        time_horizon = "session"
        if any(term in lowered for term in ("roadmap", "long-term", "future")):
            time_horizon = "multi-turn"
        elif any(term in lowered for term in ("urgent", "asap", "immediately")):
            time_horizon = "immediate"

        collaboration_mode = "single-specialist"
        if len(requested_capabilities) >= 3 or state.route_bias is not None:
            collaboration_mode = "modular-orchestration"

        return ContextSignal(
            complexity=complexity,
            risk_level=risk_level,
            requested_capabilities=list(requested_capabilities),
            time_horizon=time_horizon,
            collaboration_mode=collaboration_mode,
            needs_memory_retrieval=bool(requested_capabilities or active_goals),
        )

    def _fallback_risk_level(self, state: AgentState) -> str:
        lowered = state.request.message.lower()
        if any(term in lowered for term in ("delete", "deploy", "purchase", "send", "shutdown")):
            return "high"
        if any(term in lowered for term in ("write", "change", "modify", "build")):
            return "medium"
        return "low"
