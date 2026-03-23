from __future__ import annotations

from .memory import MemorySystem
from .models import AgentState, ContextSignal, ContextSnapshot
from .runtime_utils import dedupe_preserve_order, recent_observation_summaries


class ContextEngineeringHub:
    def observe(self, state: AgentState, memory_system: MemorySystem) -> AgentState:
        request = state.request
        gateway = state.gateway
        if gateway is None:
            raise ValueError("Gateway state must be available before context observation.")

        merged_preferences = memory_system.merge_preferences(
            request.user_id, request.session_id, request.preferences
        )
        active_goals = self._derive_goals(state)
        constraints = self._derive_constraints(state, merged_preferences)
        requested_capabilities = self._derive_capabilities(state)
        signals = self._derive_signals(state, active_goals, requested_capabilities)

        memory_system.update_goals(request.user_id, request.session_id, active_goals)
        memory_snapshot = memory_system.build_snapshot(
            request.user_id,
            request.session_id,
            active_goals=active_goals,
            query=self._memory_query(state),
            constraints=constraints,
        )

        memory_snapshot.working_memory.objective = gateway.normalized_message
        memory_snapshot.working_memory.constraints = constraints
        memory_snapshot.working_memory.plan_checkpoint = "observe"
        recent_observations = recent_observation_summaries(state, limit=3)
        if recent_observations:
            memory_snapshot.working_memory.distilled_context = (
                f"{memory_snapshot.summary} | Recent loop observations: {'; '.join(recent_observations)}"
            )[:500]
            for summary in recent_observations:
                if summary not in memory_snapshot.working_memory.retrieved_facts:
                    memory_snapshot.working_memory.retrieved_facts.append(summary)
        if state.reflection and state.reflection.lessons:
            for lesson in state.reflection.lessons[:2]:
                if lesson not in memory_snapshot.working_memory.assumptions:
                    memory_snapshot.working_memory.assumptions.append(lesson)

        context = ContextSnapshot(
            user_id=request.user_id,
            session_id=request.session_id,
            channel=request.channel,
            latest_message=gateway.normalized_message,
            gateway=gateway,
            preferences=merged_preferences,
            history=memory_system.get_history(request.user_id, request.session_id)[-6:],
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
            metadata={
                **request.metadata,
                "step_index": state.step_index,
                "retry_count": state.retry_count,
                "replan_count": state.replan_count,
                "blocked_tools": state.blocked_tools,
                "route_bias": state.route_bias,
            },
        )

        state.memory = memory_snapshot
        state.context = context
        state.memory = memory_system.checkpoint_iteration(state, checkpoint="observe")
        state.context.memory = state.memory
        return state

    def build(
        self,
        request,
        gateway,
        memory_system: MemorySystem,
    ) -> ContextSnapshot:
        state = AgentState(request=request, gateway=gateway)
        return self.observe(state, memory_system).context or ContextSnapshot(
            user_id=request.user_id,
            session_id=request.session_id,
            channel=request.channel,
            latest_message=request.message,
            gateway=gateway,
        )

    def _derive_goals(self, state: AgentState) -> list[str]:
        request = state.request
        lowered = request.message.lower()
        goals = list(request.goals)
        retrieved_text = " ".join(record.content.lower() for record in state.memory.retrieved)

        keyword_map = {
            "design_system": ("architecture", "orchestrator", "workflow", "agent", "pipeline", "system"),
            "ship_code": ("code", "coding", "debug", "fix", "implement", "backend", "frontend", "api", "build"),
            "communicate": ("email", "message", "reply", "draft", "send"),
            "research": ("research", "find", "search", "lookup", "compare", "summarize"),
            "browse_web": ("browser", "website", "web", "page"),
            "manage_tasks": ("schedule", "calendar", "task", "remind", "meeting"),
            "inspect_files": ("file", "document", "pdf", "report", "doc"),
            "maintain_memory": ("remember", "history", "memory", "preference", "context"),
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

    def _derive_constraints(self, state: AgentState, preferences: dict[str, object]) -> list[str]:
        request = state.request
        lowered = request.message.lower()
        constraints: list[str] = list(state.adaptive_constraints)

        if "architecture" in lowered or "system" in lowered:
            constraints.append("Favor modular composition over monolithic control flow.")
        if "safety" in lowered or "approval" in lowered:
            constraints.append("Keep potentially external side effects behind explicit safety gates.")
        if "memory" in lowered:
            constraints.append("Ground responses in retrieved memory rather than recent turns alone.")
        if preferences:
            constraints.append("Respect persisted user preferences during planning and response generation.")
        if request.channel.value == "voice":
            constraints.append("Prefer concise spoken-ready outputs.")
        if state.retry_count > 0:
            constraints.append("Change strategy instead of repeating the previous failed attempt.")
        if state.blocked_tools:
            constraints.append(f"Avoid blocked tools: {', '.join(state.blocked_tools)}.")

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
        query_parts.extend(state.adaptive_constraints[:2])
        if state.route_bias:
            query_parts.append(f"preferred route {state.route_bias}")
        return " ".join(part for part in query_parts if part).strip()

    def _derive_signals(
        self,
        state: AgentState,
        active_goals: list[str],
        requested_capabilities: list[str],
    ) -> ContextSignal:
        lowered = state.request.message.lower()
        complexity = "moderate"
        if len(requested_capabilities) >= 4 or len(active_goals) >= 4 or state.replan_count > 0:
            complexity = "high"
        elif len(state.request.message.split()) <= 12 and len(requested_capabilities) <= 2:
            complexity = "low"

        risk_level = "low"
        if any(term in lowered for term in ("delete", "deploy", "purchase", "send", "shutdown")):
            risk_level = "high"
        elif any(term in lowered for term in ("write", "change", "modify", "build")):
            risk_level = "medium"
        if state.retry_count > 0 and risk_level == "low":
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
            requested_capabilities=requested_capabilities,
            time_horizon=time_horizon,
            collaboration_mode=collaboration_mode,
            needs_memory_retrieval=bool(requested_capabilities or active_goals),
        )
