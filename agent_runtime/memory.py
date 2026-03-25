from __future__ import annotations
from threading import RLock
from typing import Any, Iterable

from .models import (
    AgentState,
    ConversationTurn,
    MemoryRecord,
    MemorySnapshot,
    SessionState,
    UserRequest,
    WorkingMemory,
    utc_now,
)
from .runtime_utils import dedupe_preserve_order, tokenize_words


def dump_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

class MemorySystem:
    def __init__(self) -> None:
        self._history: dict[str, list[ConversationTurn]] = {}
        self._preferences: dict[str, dict[str, Any]] = {}
        self._semantic: dict[str, list[MemoryRecord]] = {}
        self._vector: dict[str, list[MemoryRecord]] = {}
        self._goals: dict[str, list[str]] = {}
        self._logs: list[dict[str, Any]] = []
        self._lock = RLock()

    def _key(self, user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    def get_history(self, user_id: str, session_id: str) -> list[ConversationTurn]:
        key = self._key(user_id, session_id)
        with self._lock:
            return list(self._history.get(key, []))

    def get_preferences(self, user_id: str, session_id: str) -> dict[str, Any]:
        key = self._key(user_id, session_id)
        with self._lock:
            return dict(self._preferences.get(key, {}))

    def merge_preferences(
        self, user_id: str, session_id: str, incoming: dict[str, Any]
    ) -> dict[str, Any]:
        key = self._key(user_id, session_id)
        with self._lock:
            existing_preferences = self._preferences.get(key, {})
            merged = dict(existing_preferences)
            changed_preferences = {
                pref_key: pref_value
                for pref_key, pref_value in incoming.items()
                if existing_preferences.get(pref_key) != pref_value
            }
            merged.update(incoming)
            self._preferences[key] = merged

            semantic = self._semantic.setdefault(key, [])
            for pref_key, pref_value in changed_preferences.items():
                semantic.append(
                    MemoryRecord(
                        memory_type="semantic",
                        content=f"User preference {pref_key}={pref_value}",
                        source="preference_update",
                        salience=0.8,
                        tags=["preference", pref_key],
                        attributes={"key": pref_key, "value": pref_value},
                    )
                )

            return dict(merged)

    def update_goals(self, user_id: str, session_id: str, goals: list[str]) -> list[str]:
        key = self._key(user_id, session_id)
        with self._lock:
            existing = self._goals.setdefault(key, [])
            for goal in goals:
                if goal not in existing:
                    existing.append(goal)
            return list(existing)

    def ingest_history(
        self,
        user_id: str,
        session_id: str,
        turns: list[ConversationTurn],
    ) -> None:
        key = self._key(user_id, session_id)
        with self._lock:
            history = self._history.setdefault(key, [])
            existing = {(turn.role, turn.content, turn.timestamp.isoformat()) for turn in history}
            for turn in turns:
                signature = (turn.role, turn.content, turn.timestamp.isoformat())
                if signature not in existing:
                    history.append(turn)
                    existing.add(signature)

    def ingest_semantic_records(
        self,
        user_id: str,
        session_id: str,
        records: list[MemoryRecord],
    ) -> None:
        key = self._key(user_id, session_id)
        with self._lock:
            semantic = self._semantic.setdefault(key, [])
            existing = {(record.memory_id, record.content) for record in semantic}
            for record in records:
                signature = (record.memory_id, record.content)
                if signature not in existing:
                    semantic.append(record)
                    existing.add(signature)

    def append_memory_record(
        self,
        user_id: str,
        session_id: str,
        record: MemoryRecord,
    ) -> None:
        key = self._key(user_id, session_id)
        with self._lock:
            semantic = self._semantic.setdefault(key, [])
            semantic.append(record)

    def summarize_history(self, user_id: str, session_id: str, limit: int = 4) -> str:
        recent_turns = self.get_history(user_id, session_id)[-limit:]
        if not recent_turns:
            return "No prior conversation history."

        summary_parts = []
        for turn in recent_turns:
            snippet = turn.content.strip().replace("\n", " ")
            summary_parts.append(f"{turn.role}: {snippet[:80]}")
        return " | ".join(summary_parts)

    def retrieve_records(
        self,
        user_id: str,
        session_id: str,
        query: str,
        memory_types: Iterable[str] | None = None,
        limit: int = 6,
    ) -> list[MemoryRecord]:
        key = self._key(user_id, session_id)
        query_tokens = tokenize_words(query)
        target_types = set(memory_types or ("semantic", "vector"))

        with self._lock:
            candidates: list[MemoryRecord] = []
            if "semantic" in target_types:
                candidates.extend(self._semantic.get(key, []))
            if "vector" in target_types:
                candidates.extend(self._vector.get(key, []))

            if not candidates:
                return []

            ranked = sorted(
                (self._rank_record(record, query_tokens) for record in candidates),
                key=lambda item: item.score,
                reverse=True,
            )

            selected: list[MemoryRecord] = []
            for record in ranked:
                if len(selected) >= limit:
                    break
                if record.score <= 0:
                    continue
                record.last_accessed_at = utc_now()
                selected.append(record)

            return [record.model_copy(deep=True) for record in selected]

    def build_snapshot(
        self,
        user_id: str,
        session_id: str,
        active_goals: list[str] | None = None,
        query: str | None = None,
        constraints: list[str] | None = None,
    ) -> MemorySnapshot:
        key = self._key(user_id, session_id)
        with self._lock:
            goal_stack = list(self._goals.get(key, []))
            if active_goals:
                for goal in active_goals:
                    if goal not in goal_stack:
                        goal_stack.append(goal)

            recent_history = list(self._history.get(key, []))[-6:]
            semantic = list(self._semantic.get(key, []))
            vector = list(self._vector.get(key, []))

        retrieved = self.retrieve_records(
            user_id=user_id,
            session_id=session_id,
            query=query or "recent session state",
            limit=6,
        )

        working_memory = self._build_working_memory(
            objective=query or "",
            history_summary=self.summarize_history(user_id, session_id),
            retrieved=retrieved,
            goal_stack=goal_stack,
            constraints=constraints or [],
        )

        open_loops = goal_stack[-3:]
        return MemorySnapshot(
            summary=self.summarize_history(user_id, session_id),
            semantic=semantic[-8:],
            episodic=recent_history,
            vector=vector[-8:],
            retrieved=retrieved,
            goal_stack=goal_stack,
            open_loops=open_loops,
            working_memory=working_memory,
        )

    def checkpoint_iteration(
        self,
        state: AgentState,
        checkpoint: str,
    ) -> MemorySnapshot:
        base = state.memory.model_copy(deep=True)
        latest_observations = [
            observation.summary
            for observation in state.observations
            if observation.step_index == state.step_index
        ][-4:]

        transient_records = [
            MemoryRecord(
                memory_type="working",
                content=f"Step {state.step_index} observation: {summary}",
                source="iteration_checkpoint",
                salience=0.95,
                tags=["working", "iteration", f"step_{state.step_index}"],
            )
            for summary in latest_observations
        ]

        merged_retrieved = transient_records + list(base.retrieved)
        deduped_retrieved: list[MemoryRecord] = []
        seen_content: set[str] = set()
        for record in merged_retrieved:
            if record.content in seen_content:
                continue
            deduped_retrieved.append(record)
            seen_content.add(record.content)
            if len(deduped_retrieved) >= 6:
                break

        base.retrieved = deduped_retrieved
        working_memory = base.working_memory
        working_memory.plan_checkpoint = checkpoint
        if state.execution and state.execution.summary:
            working_memory.distilled_context = (
                f"{base.summary} | Step {state.step_index}: {state.execution.summary}"
            )[:500]
        if state.execution and state.execution.next_focus:
            question = f"Next focus: {state.execution.next_focus}"
            if question not in working_memory.open_questions:
                working_memory.open_questions.insert(0, question)
        for constraint in state.adaptive_constraints:
            if constraint not in working_memory.constraints:
                working_memory.constraints.append(constraint)
        for lesson in (state.reflection.lessons if state.reflection else [])[:2]:
            if lesson not in working_memory.assumptions:
                working_memory.assumptions.append(lesson)
        for summary in latest_observations:
            if summary not in working_memory.retrieved_facts:
                working_memory.retrieved_facts.append(summary)
        base.open_loops = dedupe_preserve_order(
            [*base.open_loops, *(state.execution.unresolved if state.execution else [])]
        )[:4]
        return base

    def save_turn(
        self,
        request: UserRequest,
        response_text: str,
        metadata: dict[str, Any],
        active_goals: list[str],
    ) -> None:
        key = self._key(request.user_id, request.session_id)
        with self._lock:
            history = self._history.setdefault(key, [])
            history.append(
                ConversationTurn(
                    role="user",
                    content=request.message,
                    metadata={
                        "channel": request.channel.value,
                        "request_metadata": request.metadata,
                        "goals": active_goals,
                    },
                )
            )
            history.append(
                ConversationTurn(
                    role="assistant",
                    content=response_text,
                    metadata=metadata,
                )
            )

            semantic = self._semantic.setdefault(key, [])
            semantic.extend(
                [
                    MemoryRecord(
                        memory_type="semantic",
                        content=f"Last assigned agent: {metadata.get('assigned_agent', 'general')}",
                        source="orchestration_metadata",
                        salience=0.7,
                        tags=["agent", metadata.get("assigned_agent", "general")],
                    ),
                    MemoryRecord(
                        memory_type="semantic",
                        content=f"Latest objective: {request.message[:200]}",
                        source="objective_capture",
                        salience=0.9,
                        tags=["objective", *active_goals],
                    ),
                ]
            )

            verification_summary = metadata.get("verification_summary")
            if verification_summary:
                semantic.append(
                    MemoryRecord(
                        memory_type="semantic",
                        content=f"Verification summary: {verification_summary}",
                        source="verification",
                        salience=0.8,
                        tags=["verification"],
                    )
                )

            reflection_lessons = metadata.get("reflection_lessons", [])
            for lesson in reflection_lessons[:2]:
                semantic.append(
                    MemoryRecord(
                        memory_type="semantic",
                        content=f"Reflection lesson: {lesson}",
                        source="reflection",
                        salience=0.75,
                        tags=["reflection"],
                    )
                )

            handoff_summary = metadata.get("handoff_summary")
            if handoff_summary:
                semantic.append(
                    MemoryRecord(
                        memory_type="semantic",
                        content=f"Handoff summary: {handoff_summary}",
                        source="handoff",
                        salience=0.72,
                        tags=["handoff", "continuity"],
                    )
                )

            vector = self._vector.setdefault(key, [])
            vector.extend(
                [
                    MemoryRecord(
                        memory_type="vector",
                        content=request.message[:240],
                        source="user_turn",
                        salience=0.85,
                        tags=active_goals,
                    ),
                    MemoryRecord(
                        memory_type="vector",
                        content=response_text[:240],
                        source="assistant_turn",
                        salience=0.8,
                        tags=[metadata.get("assigned_agent", "general")],
                    ),
                ]
            )

            goals = self._goals.setdefault(key, [])
            for goal in active_goals:
                if goal not in goals:
                    goals.append(goal)

    def log_interaction(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._logs.append(payload)

    def get_session_state(self, user_id: str, session_id: str) -> SessionState:
        return SessionState(
            user_id=user_id,
            session_id=session_id,
            preferences=self.get_preferences(user_id, session_id),
            history=self.get_history(user_id, session_id),
            memory=self.build_snapshot(user_id, session_id),
        )

    def history_length(self, user_id: str, session_id: str) -> int:
        return len(self.get_history(user_id, session_id))

    def _rank_record(self, record: MemoryRecord, query_tokens: set[str]) -> MemoryRecord:
        if not query_tokens:
            return record.model_copy(update={"score": record.salience})

        content_tokens = tokenize_words(record.content)
        tag_tokens = {token.lower() for token in record.tags}
        overlap = len(query_tokens & content_tokens)
        tag_overlap = len(query_tokens & tag_tokens)
        score = round((overlap * 1.3) + (tag_overlap * 0.8) + record.salience, 3)
        return record.model_copy(update={"score": score})

    def _build_working_memory(
        self,
        objective: str,
        history_summary: str,
        retrieved: list[MemoryRecord],
        goal_stack: list[str],
        constraints: list[str],
    ) -> WorkingMemory:
        retrieved_facts = [record.content for record in retrieved[:4]]
        assumptions = [
            "Treat the model as a reasoning component rather than the entire system.",
        ]
        if goal_stack:
            assumptions.append(f"Active goals remain anchored on: {', '.join(goal_stack[:3])}.")
        if not retrieved_facts:
            assumptions.append("Little relevant long-term memory was found, so recent context dominates.")

        open_questions = []
        if not retrieved_facts:
            open_questions.append("Should the system store a stronger architectural summary after this turn?")
        if objective and "build" in objective.lower():
            open_questions.append("Which modules need the deepest restructuring to satisfy the objective?")

        return WorkingMemory(
            objective=objective,
            distilled_context=history_summary,
            assumptions=assumptions,
            constraints=constraints,
            open_questions=open_questions,
            retrieved_facts=retrieved_facts,
            plan_checkpoint="context_grounded",
        )

MemoryStore = MemorySystem
