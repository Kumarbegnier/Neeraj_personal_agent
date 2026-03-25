from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.schemas.adaptive import (
    ArchitectureDecision,
    ArchitectureMode,
    ArchitectureReasoning,
    TaskCharacteristics,
)

from .models import AgentState

_INTENSITY_SCORE = {"low": 0.0, "moderate": 1.0, "high": 2.0}
_RISK_SCORE = {"low": 0.0, "medium": 1.0, "high": 2.0, "critical": 3.0}


@dataclass(frozen=True)
class ExecutionPattern:
    mode: ArchitectureMode
    label: str
    loop_strategy: str
    default_primary_agent: str
    default_supporting_agents: tuple[str, ...] = ()
    requires_planning: bool = False
    requires_verifier: bool = True
    browser_heavy: bool = False
    critic_lane: bool = False
    parallel_fanout: int = 1


class ArchitectureSelector:
    def __init__(self, patterns: Sequence[ExecutionPattern] | None = None) -> None:
        self._patterns = tuple(patterns or self._default_patterns())
        if not self._patterns:
            raise ValueError("ArchitectureSelector requires at least one execution pattern.")

    def select(self, state: AgentState) -> ArchitectureDecision:
        context = state.context
        packet = state.context_packet or (context.context_packet if context else None)
        if context is None or packet is None:
            raise ValueError("Context packet must exist before architecture selection.")

        characteristics = self._task_characteristics(state)
        scorecard = {
            pattern.mode: self._score_pattern(pattern, characteristics, state)
            for pattern in self._patterns
        }
        selected_pattern = self._patterns[0]
        selected_score = scorecard[selected_pattern.mode]
        for pattern in self._patterns[1:]:
            pattern_score = scorecard[pattern.mode]
            if pattern_score > selected_score:
                selected_pattern = pattern
                selected_score = pattern_score

        primary_agent = self._primary_agent_for(selected_pattern, state)
        supporting_agents = self._supporting_agents_for(
            selected_pattern,
            capabilities=packet.requested_capabilities,
            characteristics=characteristics,
        )
        requires_verifier = selected_pattern.requires_verifier or selected_pattern.browser_heavy
        reasoning = self._reasoning_for(
            selected_pattern,
            characteristics=characteristics,
            scorecard=scorecard,
        )

        return ArchitectureDecision(
            mode=selected_pattern.mode,
            rationale=reasoning.summary,
            reasoning=reasoning,
            task_characteristics=characteristics,
            pattern_label=selected_pattern.label,
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            requires_planning=selected_pattern.requires_planning,
            requires_verifier=requires_verifier,
            browser_heavy=selected_pattern.browser_heavy,
            critic_lane=selected_pattern.critic_lane,
            parallel_fanout=self._parallel_fanout_for(selected_pattern, characteristics),
            loop_strategy=selected_pattern.loop_strategy,
            stop_conditions=self._stop_conditions_for(
                selected_pattern,
                requires_verifier=requires_verifier,
            ),
        )

    def _task_characteristics(self, state: AgentState) -> TaskCharacteristics:
        context = state.context
        packet = state.context_packet
        if context is None or packet is None:
            raise ValueError("Context packet must exist before deriving task characteristics.")

        lowered = self._surface(state)
        capabilities = set(packet.requested_capabilities)
        goals = set(packet.active_goals)

        complexity_points = {"low": 1, "moderate": 2, "high": 3}.get(context.signals.complexity, 2)
        complexity_points += 1 if len(packet.active_goals) >= 3 else 0
        complexity_points += 1 if len(packet.requested_capabilities) >= 4 else 0
        complexity_points += 1 if state.retry_count > 0 or state.replan_count > 0 or state.needs_replan else 0
        complexity_points += 1 if len(lowered.split()) >= 28 else 0

        grounding_points = 0
        grounding_points += sum(
            capability in capabilities
            for capability in ("research", "browser", "documents", "verification")
        )
        grounding_points += sum(
            keyword in lowered
            for keyword in (
                "latest",
                "current",
                "verify",
                "verification",
                "source",
                "sources",
                "evidence",
                "grounded",
                "compare",
                "cite",
            )
        )
        grounding_points += 1 if packet.approval_state.risk_level in {"high", "critical"} else 0

        tool_points = sum(
            capability in capabilities
            for capability in (
                "coding",
                "browser",
                "documents",
                "research",
                "verification",
                "memory",
                "coordination",
            )
        )
        tool_points += 1 if packet.tool_availability.total_tools >= 8 else 0
        tool_points += 1 if state.retry_count > 0 else 0

        parallel_points = 0
        parallel_points += 1 if len(packet.active_goals) >= 3 else 0
        parallel_points += 1 if len(packet.requested_capabilities) >= 3 else 0
        parallel_points += 1 if {"research", "documents"} & capabilities else 0
        parallel_points += 1 if any(
            keyword in lowered for keyword in ("compare", "multiple", "across", "several", "parallel")
        ) else 0

        communication_points = 0
        communication_points += 2 if "communication" in capabilities or "communicate" in goals else 0
        communication_points += sum(
            keyword in lowered
            for keyword in (
                "email",
                "message",
                "reply",
                "draft",
                "tone",
                "customer",
                "client",
                "send",
                "critic",
                "critique",
                "review",
            )
        )

        research_points = 0
        research_points += 2 if "research" in capabilities or "research" in goals else 0
        research_points += 1 if "documents" in capabilities else 0
        research_points += sum(
            keyword in lowered
            for keyword in (
                "research",
                "search",
                "compare",
                "evidence",
                "source",
                "summary",
                "findings",
                "synthesize",
            )
        )

        browser_points = 0
        browser_points += 2 if "browser" in capabilities or "browse_web" in goals else 0
        browser_points += sum(
            keyword in lowered
            for keyword in (
                "browser",
                "web",
                "website",
                "page",
                "navigate",
                "latest",
                "current",
                "live",
            )
        )

        return TaskCharacteristics(
            complexity=self._intensity_from_points(complexity_points),
            grounding_need=self._intensity_from_points(grounding_points),
            tool_intensity=self._intensity_from_points(tool_points),
            risk_level=packet.approval_state.risk_level,
            parallelizability=self._intensity_from_points(parallel_points),
            communication_intensity=self._intensity_from_points(communication_points),
            research_intensity=self._intensity_from_points(research_points),
            browser_intensity=self._intensity_from_points(browser_points),
        )

    def _score_pattern(
        self,
        pattern: ExecutionPattern,
        characteristics: TaskCharacteristics,
        state: AgentState,
    ) -> float:
        packet = state.context_packet
        if packet is None:
            raise ValueError("Context packet must exist before scoring architecture patterns.")

        capabilities = set(packet.requested_capabilities)
        lowered = self._surface(state)

        complexity = _INTENSITY_SCORE[characteristics.complexity]
        grounding = _INTENSITY_SCORE[characteristics.grounding_need]
        tool_intensity = _INTENSITY_SCORE[characteristics.tool_intensity]
        risk = _RISK_SCORE[characteristics.risk_level]
        parallel = _INTENSITY_SCORE[characteristics.parallelizability]
        communication = _INTENSITY_SCORE[characteristics.communication_intensity]
        research = _INTENSITY_SCORE[characteristics.research_intensity]
        browser = _INTENSITY_SCORE[characteristics.browser_intensity]

        score = 0.0
        if pattern.mode == ArchitectureMode.DIRECT_SINGLE_AGENT:
            score += max(0.0, 3.4 - (complexity * 1.1))
            score += max(0.0, 2.0 - (grounding * 0.8))
            score += max(0.0, 2.0 - (tool_intensity * 0.9))
            score += max(0.0, 1.5 - (risk * 0.5))
            score += max(0.0, 1.2 - (parallel * 0.6))
            score += 0.4 if state.retry_count == 0 and state.replan_count == 0 else -0.8
            score -= 1.4 if browser >= 2.0 else 0.0
            score -= 0.8 if research >= 2.0 else 0.0
            score -= 0.9 if communication >= 2.0 and risk >= 1.0 else 0.0
        elif pattern.mode == ArchitectureMode.PLANNER_EXECUTOR:
            score += 1.2 + (complexity * 1.3) + (tool_intensity * 1.4)
            score += 1.2 if "coding" in capabilities or "planning" in capabilities else 0.0
            score += 0.7 if state.retry_count > 0 or state.needs_replan or state.replan_count > 0 else 0.0
            score += 0.3 if grounding >= 1.0 and browser < 2.0 else 0.0
            score -= 1.6 if browser >= 2.0 else 0.0
            score -= 0.4 if communication >= 2.0 else 0.0
        elif pattern.mode == ArchitectureMode.MULTI_AGENT_RESEARCH:
            score += 1.0 + (research * 1.9) + (grounding * 1.5) + (parallel * 1.6)
            score += 0.8 if "documents" in capabilities else 0.0
            score += 0.5 if "browser" in capabilities else 0.0
            score += 0.5 if complexity >= 1.0 else 0.0
            score -= 1.0 if communication >= 2.0 else 0.0
            score -= 0.7 if browser >= 2.0 and "research" not in capabilities else 0.0
        elif pattern.mode == ArchitectureMode.BROWSER_HEAVY_VERIFIED:
            score += 1.4 + (browser * 2.5) + (grounding * 1.8)
            score += 1.0 if "browser" in capabilities else 0.0
            score += 0.7 if risk >= 1.0 else 0.0
            score += 0.5 if research >= 1.0 else 0.0
            score -= 0.9 if communication >= 2.0 else 0.0
        elif pattern.mode == ArchitectureMode.COMMUNICATION_CRITIC:
            score += 1.0 + (communication * 2.4) + (risk * 1.0)
            score += 0.9 if packet.approval_state.requires_confirmation else 0.0
            score += 0.6 if grounding >= 1.0 else 0.0
            score += 0.6 if any(
                keyword in lowered for keyword in ("tone", "customer", "stakeholder", "critique", "review")
            ) else 0.0
            score -= 1.2 if browser >= 2.0 else 0.0
            score -= 0.6 if research >= 2.0 else 0.0
        return round(score, 3)

    def _primary_agent_for(self, pattern: ExecutionPattern, state: AgentState) -> str:
        packet = state.context_packet
        if packet is None:
            raise ValueError("Context packet must exist before selecting a primary agent.")

        if pattern.mode == ArchitectureMode.DIRECT_SINGLE_AGENT:
            return self._best_specialist(packet.requested_capabilities, state.request.message)
        if pattern.mode == ArchitectureMode.PLANNER_EXECUTOR and "coordination" in packet.requested_capabilities:
            return "task"
        if pattern.mode == ArchitectureMode.PLANNER_EXECUTOR and "coding" not in packet.requested_capabilities:
            return self._best_specialist(packet.requested_capabilities, state.request.message)
        return pattern.default_primary_agent

    def _supporting_agents_for(
        self,
        pattern: ExecutionPattern,
        *,
        capabilities: Sequence[str],
        characteristics: TaskCharacteristics,
    ) -> list[str]:
        selected = list(pattern.default_supporting_agents)
        capability_set = set(capabilities)

        if pattern.mode == ArchitectureMode.PLANNER_EXECUTOR:
            if "verification" in capability_set or characteristics.grounding_need == "high":
                selected.append("research")
            selected.append("general")
        elif pattern.mode == ArchitectureMode.MULTI_AGENT_RESEARCH:
            if "browser" in capability_set or characteristics.browser_intensity == "high":
                selected.append("web")
            if "documents" in capability_set:
                selected.append("file")
            selected.append("general")
        elif pattern.mode == ArchitectureMode.BROWSER_HEAVY_VERIFIED:
            selected.extend(["research", "general"])
            if "documents" in capability_set:
                selected.append("file")
        elif pattern.mode == ArchitectureMode.COMMUNICATION_CRITIC:
            selected.append("general")
            if characteristics.grounding_need != "low":
                selected.append("research")

        return self._dedupe(selected)

    def _parallel_fanout_for(
        self,
        pattern: ExecutionPattern,
        characteristics: TaskCharacteristics,
    ) -> int:
        if pattern.mode == ArchitectureMode.MULTI_AGENT_RESEARCH:
            return 3 if characteristics.parallelizability == "high" else 2
        if pattern.mode in {
            ArchitectureMode.BROWSER_HEAVY_VERIFIED,
            ArchitectureMode.COMMUNICATION_CRITIC,
            ArchitectureMode.PLANNER_EXECUTOR,
        }:
            return max(pattern.parallel_fanout, 2)
        return pattern.parallel_fanout

    def _stop_conditions_for(
        self,
        pattern: ExecutionPattern,
        *,
        requires_verifier: bool,
    ) -> list[str]:
        conditions = ["goal_achieved", "max_steps_reached", "stalled"]
        if pattern.requires_planning:
            conditions.append("replan_requested")
        if requires_verifier:
            conditions.append("verification_passed")
        if pattern.browser_heavy:
            conditions.append("browser_evidence_verified")
        if pattern.critic_lane:
            conditions.append("critic_approved")
        return conditions

    def _reasoning_for(
        self,
        pattern: ExecutionPattern,
        *,
        characteristics: TaskCharacteristics,
        scorecard: dict[ArchitectureMode, float],
    ) -> ArchitectureReasoning:
        labels_by_mode = {candidate.mode: candidate.label for candidate in self._patterns}
        ordered_scores = sorted(
            (
                (labels_by_mode[candidate_mode], score)
                for candidate_mode, score in scorecard.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        runner_up = ordered_scores[1][0] if len(ordered_scores) > 1 else "none"
        summary = (
            f"Selected '{pattern.label}' because complexity={characteristics.complexity}, "
            f"grounding={characteristics.grounding_need}, tool_intensity={characteristics.tool_intensity}, "
            f"risk={characteristics.risk_level}, and parallelizability={characteristics.parallelizability} "
            f"fit this execution pattern better than '{runner_up}'."
        )
        decisive_factors = [
            f"Task complexity was classified as {characteristics.complexity}.",
            f"Grounding need was classified as {characteristics.grounding_need}.",
            f"Tool intensity was classified as {characteristics.tool_intensity}.",
            f"Risk level was classified as {characteristics.risk_level}.",
            f"Parallelizability was classified as {characteristics.parallelizability}.",
        ]
        if pattern.mode == ArchitectureMode.MULTI_AGENT_RESEARCH:
            decisive_factors.append(
                f"Research intensity was {characteristics.research_intensity}, which favors evidence fan-out."
            )
        if pattern.mode == ArchitectureMode.BROWSER_HEAVY_VERIFIED:
            decisive_factors.append(
                f"Browser intensity was {characteristics.browser_intensity}, so the verifier-backed web path is preferred."
            )
        if pattern.mode == ArchitectureMode.COMMUNICATION_CRITIC:
            decisive_factors.append(
                f"Communication intensity was {characteristics.communication_intensity}, which justifies a critic lane."
            )

        tradeoffs = {
            ArchitectureMode.DIRECT_SINGLE_AGENT: [
                "Keeps orchestration overhead low but offers less decomposition and parallelism.",
            ],
            ArchitectureMode.PLANNER_EXECUTOR: [
                "Adds planning overhead in exchange for cleaner decomposition and retry handling.",
            ],
            ArchitectureMode.MULTI_AGENT_RESEARCH: [
                "Introduces coordination cost but improves evidence gathering across independent lanes.",
            ],
            ArchitectureMode.BROWSER_HEAVY_VERIFIED: [
                "Runs slower because browser work and verification both stay in the critical path.",
            ],
            ArchitectureMode.COMMUNICATION_CRITIC: [
                "Adds a critique pass, which slows delivery slightly but reduces tone and safety regressions.",
            ],
        }[pattern.mode]

        return ArchitectureReasoning(
            selected_pattern=pattern.label,
            summary=summary,
            decisive_factors=decisive_factors,
            tradeoffs=tradeoffs,
            pattern_scores={label: round(score, 2) for label, score in ordered_scores},
        )

    def _best_specialist(self, capabilities: Sequence[str], message: str) -> str:
        lowered = message.lower()
        capability_set = set(capabilities)
        if "browser" in capability_set:
            return "web"
        if "communication" in capability_set:
            return "communication"
        if "documents" in capability_set:
            return "file"
        if "coordination" in capability_set:
            return "task"
        if "research" in capability_set and "coding" not in capability_set:
            return "research"
        if "coding" in capability_set or any(
            keyword in lowered for keyword in ("build", "implement", "debug", "fix", "architecture")
        ):
            return "coding"
        return "general"

    def _surface(self, state: AgentState) -> str:
        packet = state.context_packet
        context = state.context
        memory = state.memory
        parts = [
            state.request.message,
            " ".join(packet.active_goals) if packet else "",
            " ".join(packet.constraints) if packet else "",
            " ".join(packet.requested_capabilities) if packet else "",
            " ".join(observation.summary for observation in state.observations[-4:]),
            " ".join(record.content for record in memory.retrieved[:4]),
            context.memory.working_memory.distilled_context if context else "",
        ]
        return " ".join(part.strip().lower() for part in parts if part).strip()

    def _intensity_from_points(self, points: int) -> str:
        if points <= 1:
            return "low"
        if points <= 3:
            return "moderate"
        return "high"

    def _dedupe(self, values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _default_patterns(self) -> tuple[ExecutionPattern, ...]:
        return (
            ExecutionPattern(
                mode=ArchitectureMode.DIRECT_SINGLE_AGENT,
                label="direct single-agent/tool path",
                loop_strategy="react_direct",
                default_primary_agent="general",
                requires_verifier=False,
            ),
            ExecutionPattern(
                mode=ArchitectureMode.PLANNER_EXECUTOR,
                label="planner + executor path",
                loop_strategy="react_planner_executor",
                default_primary_agent="coding",
                requires_planning=True,
            ),
            ExecutionPattern(
                mode=ArchitectureMode.MULTI_AGENT_RESEARCH,
                label="multi-agent research path",
                loop_strategy="react_multi_agent_research",
                default_primary_agent="research",
                requires_planning=True,
                parallel_fanout=2,
            ),
            ExecutionPattern(
                mode=ArchitectureMode.BROWSER_HEAVY_VERIFIED,
                label="browser-heavy path with verifier",
                loop_strategy="react_browser_verified",
                default_primary_agent="web",
                requires_planning=True,
                requires_verifier=True,
                browser_heavy=True,
                parallel_fanout=2,
            ),
            ExecutionPattern(
                mode=ArchitectureMode.COMMUNICATION_CRITIC,
                label="communication-heavy path with critic lane",
                loop_strategy="react_communication_critic",
                default_primary_agent="communication",
                requires_planning=True,
                requires_verifier=True,
                critic_lane=True,
                parallel_fanout=2,
            ),
        )
