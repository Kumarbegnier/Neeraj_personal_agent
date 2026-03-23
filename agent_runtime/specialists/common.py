from __future__ import annotations

from ..models import (
    AgentDecision,
    AgentState,
    ExecutionResult,
    SkillDescriptor,
    ToolRequest,
    ToolResult,
)


def skill_names(skills: list[SkillDescriptor]) -> list[str]:
    return [skill.name for skill in skills]


def tool_request(
    tool_name: str,
    purpose: str,
    *,
    payload: dict | None = None,
    priority: int = 5,
    risk_level: str = "low",
    side_effect: str = "none",
    requires_confirmation: bool = False,
    verification_hint: str = "",
    expected_observation: str = "",
) -> ToolRequest:
    return ToolRequest(
        tool_name=tool_name,
        purpose=purpose,
        input_payload=payload or {},
        priority=priority,
        risk_level=risk_level,
        side_effect=side_effect,
        requires_confirmation=requires_confirmation,
        verification_hint=verification_hint,
        expected_observation=expected_observation,
    )


def status_counts(results: list[ToolResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def unresolved_results(results: list[ToolResult]) -> list[str]:
    return [
        f"{result.tool_name}:{result.status}"
        for result in results
        if result.status in {"error", "gated", "unavailable"}
    ]


def confidence_from_results(results: list[ToolResult], base: float = 0.7) -> float:
    if not results:
        return base
    successes = sum(1 for result in results if result.status == "success")
    return round(min(0.98, 0.4 + (successes / len(results)) * 0.55), 2)


def memory_text(state: AgentState) -> str:
    return " ".join(record.content.lower() for record in state.memory.retrieved)


def filter_blocked_requests(state: AgentState, requests: list[ToolRequest]) -> list[ToolRequest]:
    blocked = {tool_name.lower() for tool_name in state.blocked_tools}
    filtered = [request for request in requests if request.tool_name.lower() not in blocked]
    deduped: list[ToolRequest] = []
    seen: set[str] = set()
    for request in sorted(filtered, key=lambda item: item.priority):
        if request.tool_name not in seen:
            deduped.append(request)
            seen.add(request.tool_name)
    return deduped


def observations_from_results(results: list[ToolResult]) -> list[str]:
    observations = []
    for result in results:
        if result.evidence:
            observations.append(f"{result.tool_name}:{result.evidence[0]}")
        else:
            observations.append(f"{result.tool_name}:{result.status}")
    return observations[:6]


def ready_for_response(results: list[ToolResult], unresolved: list[str]) -> bool:
    if not results:
        return False
    return not unresolved and any(result.status == "success" for result in results)


def build_execution_result(
    *,
    agent_name: str,
    decision: AgentDecision,
    tool_results: list[ToolResult],
    summary: str,
    actions: list[str],
    artifacts: dict[str, object],
    unresolved_focus: str,
    ready_focus: str = "Ready for final synthesis.",
    base_confidence: float = 0.7,
) -> ExecutionResult:
    unresolved = unresolved_results(tool_results)
    ready = ready_for_response(tool_results, unresolved)
    return ExecutionResult(
        agent_name=agent_name,
        summary=summary,
        tool_results=tool_results,
        actions=actions,
        artifacts=artifacts,
        claims=decision.claims_to_verify,
        observations=observations_from_results(tool_results),
        unresolved=unresolved,
        confidence=confidence_from_results(tool_results, base=base_confidence),
        goal_status="ready" if ready else "in_progress",
        ready_for_response=ready,
        requires_replan=bool(unresolved),
        next_focus=unresolved_focus if unresolved else ready_focus,
    )
