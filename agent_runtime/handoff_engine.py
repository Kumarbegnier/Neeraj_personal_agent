from __future__ import annotations

from typing import Any

from src.schemas.context import (
    ApprovalState,
    ContextEpisodicMemory,
    ContextMemorySlice,
    ContextSemanticMemoryResult,
)
from src.schemas.handoff import HandoffPacket, HandoffSummary, OpenQuestion

from .models import AgentState, MemorySnapshot
from .runtime_utils import dedupe_preserve_order


class HandoffEngine:
    """Compacts live execution state into a reusable handoff packet."""

    def build_handoff(self, state: AgentState) -> HandoffPacket:
        approval_state = self._approval_state(state)
        memory_snapshot = self._memory_snapshot(state.memory, state.request.message)
        open_questions = self._open_questions(state)
        next_actions = self._next_actions(state, open_questions)
        summary = HandoffSummary(
            objective=state.plan.objective if state.plan is not None else state.request.message,
            completed_work_summary=self._completed_work_summary(state),
            completed_steps=self._completed_steps(state),
            current_status=self._current_status(state),
            verification_summary=state.verification.summary if state.verification else "",
            reflection_summary=state.reflection.summary if state.reflection else "",
        )
        reusable_context = self._reusable_context(
            summary=summary,
            open_questions=open_questions,
            next_actions=next_actions,
            blocked_tools=self._blocked_tools(state),
        )

        return HandoffPacket(
            summary=summary,
            open_questions=open_questions,
            next_actions=next_actions,
            blocked_tools=self._blocked_tools(state),
            approval_state=approval_state,
            memory_snapshot=memory_snapshot,
            source_agent=state.route.agent_name if state.route else "general",
            architecture_mode=state.architecture.mode.value if state.architecture else "",
            loop_iteration=state.step_index,
            reusable_context=reusable_context,
            metadata={
                "retry_count": state.retry_count,
                "replan_count": state.replan_count,
                "termination_reason": state.termination_reason,
            },
        )

    def should_compact(self, state: AgentState) -> bool:
        if bool(state.request.metadata.get("disable_handoff_compaction")):
            return False
        if bool(state.request.metadata.get("force_handoff_compaction")):
            return True
        if state.step_index < 2:
            return False

        multi_turn = bool(state.context and state.context.signals.time_horizon == "multi-turn")
        high_complexity = bool(state.context and state.context.signals.complexity == "high")
        return any(
            (
                state.max_steps >= 3,
                state.retry_count > 0,
                state.replan_count > 0,
                state.needs_replan,
                state.needs_retry,
                len(state.observations) > 8,
                len(state.tool_history) > 10,
                high_complexity,
                multi_turn,
            )
        )

    def compact_state(self, state: AgentState) -> dict[str, int]:
        packet = state.handoff_packet or self.build_handoff(state)
        state.handoff_packet = packet

        trim_targets = {
            "observations": (len(state.observations), 6),
            "tool_history": (len(state.tool_history), 12),
            "reasoning_notes": (len(state.reasoning_notes), 6),
            "trace": (len(state.trace), 24),
            "state_transitions": (len(state.state_transitions), 6),
            "react_trace": (len(state.react_trace), 3),
            "model_runs": (len(state.model_runs), 10),
            "model_evaluations": (len(state.model_evaluations), 10),
        }

        state.observations = state.observations[-6:]
        state.tool_history = state.tool_history[-12:]
        state.reasoning_notes = state.reasoning_notes[-6:]
        state.trace = state.trace[-24:]
        state.state_transitions = state.state_transitions[-6:]
        state.react_trace = state.react_trace[-3:]
        state.model_runs = state.model_runs[-10:]
        state.model_evaluations = state.model_evaluations[-10:]

        working_memory = state.memory.working_memory
        working_memory.distilled_context = packet.reusable_context[:500]
        working_memory.open_questions = [question.question for question in packet.open_questions[:4]]
        working_memory.retrieved_facts = dedupe_preserve_order(
            [*packet.summary.completed_steps, *packet.memory_snapshot.retrieved_facts]
        )[:6]
        state.memory.open_loops = (
            [question.question for question in packet.open_questions if question.blocking][:4]
            or state.memory.open_loops[:4]
        )

        if state.context is not None:
            state.context.memory = state.memory
            state.context.handoff_packet = packet
            state.context.metadata = {
                **state.context.metadata,
                "handoff_packet": packet.model_dump(mode="json"),
                "compacted_state": True,
            }
        if state.context_packet is not None:
            state.context_packet = state.context_packet.model_copy(
                update={
                    "metadata": {
                        **state.context_packet.metadata,
                        "handoff_packet": packet.model_dump(mode="json"),
                        "compacted_state": True,
                    }
                }
            )
            if state.context is not None:
                state.context.context_packet = state.context_packet

        return {
            name: max(0, before - limit)
            for name, (before, limit) in trim_targets.items()
            if before > limit
        }

    def _approval_state(self, state: AgentState) -> ApprovalState:
        if state.context_packet is not None:
            return state.context_packet.approval_state.model_copy(
                update={"blocked_tools": self._blocked_tools(state)}
            )

        if state.session is None:
            return ApprovalState(approval_granted=True)

        permission = state.session.permission
        return ApprovalState(
            permission_mode=permission.mode.value,
            requires_confirmation=permission.requires_confirmation,
            risk_level=state.context.signals.risk_level if state.context else "low",
            approval_granted=permission.mode.value == "auto_approved" and not permission.requires_confirmation,
            blocked_tools=self._blocked_tools(state),
            gated_actions=["confirmation_required"] if permission.requires_confirmation else [],
            rationale=permission.reason,
        )

    def _memory_snapshot(self, memory: MemorySnapshot, objective: str) -> ContextMemorySlice:
        return ContextMemorySlice(
            summary=memory.summary,
            retrieval_query=memory.working_memory.objective or objective,
            recent_episodic_memory=[
                ContextEpisodicMemory(
                    role=turn.role,
                    content=turn.content,
                    timestamp=turn.timestamp,
                    metadata=dict(turn.metadata),
                )
                for turn in memory.episodic[-4:]
            ],
            semantic_memory_results=[
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
                for record in memory.retrieved[:6]
            ],
            working_memory_summary=memory.working_memory.distilled_context,
            open_loops=list(memory.open_loops),
            goal_stack=list(memory.goal_stack),
            retrieved_facts=list(memory.working_memory.retrieved_facts),
        )

    def _completed_work_summary(self, state: AgentState) -> str:
        parts = [
            f"Completed {state.step_index} iteration(s) for the current objective.",
            state.execution.summary if state.execution else "",
            f"Verification: {state.verification.summary}" if state.verification else "",
            f"Reflection: {state.reflection.summary}" if state.reflection else "",
        ]
        return " ".join(part for part in parts if part)[:700]

    def _completed_steps(self, state: AgentState) -> list[str]:
        steps = [
            f"Selected route: {state.route.agent_name}" if state.route else "",
            *(state.execution.actions if state.execution else []),
            state.execution.summary if state.execution else "",
            state.verification.summary if state.verification else "",
            state.reflection.summary if state.reflection else "",
        ]
        return dedupe_preserve_order(step for step in steps if step)[:6]

    def _current_status(self, state: AgentState) -> str:
        if state.execution is not None and state.execution.goal_status:
            return state.execution.goal_status
        if state.goal_status:
            return state.goal_status
        return state.status

    def _open_questions(self, state: AgentState) -> list[OpenQuestion]:
        owner = state.route.agent_name if state.route else "general"
        questions: list[OpenQuestion] = []
        seen: set[str] = set()

        candidates: list[tuple[str, str, bool]] = [
            (question, "working_memory", False)
            for question in state.memory.working_memory.open_questions
        ]
        candidates.extend((gap, "verification", True) for gap in (state.verification.gaps if state.verification else []))
        if state.execution is not None and state.execution.next_focus:
            candidates.append((state.execution.next_focus, "execution", True))
        if state.stop_decision is not None and state.stop_decision.requires_replan and state.stop_decision.reason:
            candidates.append((state.stop_decision.reason, "loop_control", True))

        for question, source, blocking in candidates:
            normalized = question.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            questions.append(
                OpenQuestion(
                    question=normalized,
                    source=source,
                    blocking=blocking,
                    suggested_owner=owner,
                    related_tools=list(state.blocked_tools)[:3],
                )
            )
            if len(questions) >= 6:
                break
        return questions

    def _next_actions(self, state: AgentState, open_questions: list[OpenQuestion]) -> list[str]:
        actions = list(state.reflection.next_actions if state.reflection else [])
        if state.execution is not None and state.execution.next_focus:
            actions.append(f"Focus next on: {state.execution.next_focus}")
        if state.reasoning is not None and state.reasoning.should_replan:
            actions.append("Replan before the next act phase.")
        if state.verification is not None and state.verification.retry_recommended:
            actions.append("Address verification gaps before making stronger claims.")
        if not actions and open_questions:
            actions.append(f"Resolve: {open_questions[0].question}")
        return dedupe_preserve_order(action for action in actions if action)[:6]

    def _blocked_tools(self, state: AgentState) -> list[str]:
        blocked = list(state.blocked_tools)
        blocked.extend(
            result.tool_name
            for result in state.last_tool_results
            if result.status in {"gated", "error", "unavailable", "verification_failed"}
        )
        return dedupe_preserve_order(blocked)[:8]

    def _reusable_context(
        self,
        *,
        summary: HandoffSummary,
        open_questions: list[OpenQuestion],
        next_actions: list[str],
        blocked_tools: list[str],
    ) -> str:
        question_summary = "; ".join(question.question for question in open_questions[:3]) or "No open questions."
        action_summary = "; ".join(next_actions[:3]) or "No next actions yet."
        blocked_summary = ", ".join(blocked_tools[:4]) or "none"
        return (
            f"Objective: {summary.objective[:160]} | Completed: {summary.completed_work_summary[:260]} | "
            f"Open questions: {question_summary[:220]} | Next actions: {action_summary[:220]} | "
            f"Blocked tools: {blocked_summary}"
        )[:900]
