from __future__ import annotations

from ..models import AgentDecision, AgentState, ExecutionResult, SkillDescriptor, ToolResult
from .base import BaseAgent
from .common import (
    build_execution_result,
    filter_blocked_requests,
    memory_text,
    skill_names,
    status_counts,
    tool_request,
)


class CodingAgent(BaseAgent):
    name = "coding"

    def decide(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
        retrieved_memory = memory_text(state)
        requests = [
            tool_request("working_memory", "Load distilled constraints and retrieved facts.", priority=1),
            tool_request(
                "plan_analyzer",
                "Inspect current plan structure and completion criteria.",
                payload={
                    "objective": state.plan.objective if state.plan else state.request.message,
                    "step_count": len(state.plan.steps) if state.plan else 0,
                    "success_criteria": state.plan.success_criteria if state.plan else [],
                    "verification_focus": state.plan.verification_focus if state.plan else [],
                },
                priority=2,
            ),
            tool_request("semantic_memory", "Retrieve relevant architectural memory.", priority=3),
            tool_request("capability_map", "Inspect available orchestration surfaces.", priority=4),
            tool_request(
                "verification_harness",
                "Prepare checks for architecture claims.",
                payload={"checks": state.plan.verification_focus if state.plan else [], "mode": "strict"},
                priority=5,
            ),
            tool_request(
                "generate_code",
                "Generate a starter implementation sketch from the current objective.",
                payload={"objective": state.request.message, "language": state.request.metadata.get("language", "python")},
                priority=6,
            ),
        ]
        if "repository" in retrieved_memory or "github" in retrieved_memory or state.retry_count == 0:
            requests.append(
                tool_request(
                    "github_adapter",
                    "Prepare repository inspection operations.",
                    payload={"repository_action": "inspect"},
                    priority=7,
                )
            )

        return AgentDecision(
            agent_name=self.name,
            summary="Prepared a coding action set whose tool mix is influenced by working memory, reflection, and retrieved architecture context.",
            skill_names=skill_names(skills),
            tool_requests=filter_blocked_requests(state, requests),
            reasoning="Treat the model as planner while using memory and verification outputs to adapt the coding tool mix.",
            response_strategy="Compose a final answer from verified coding state only.",
            expected_deliverables=["Architecture-aware execution", "Verified implementation state"],
            claims_to_verify=[
                "Planning, memory, tools, verification, reflection, and safety are causally connected.",
                "Retrieved memory influences routing and tool choice.",
            ],
            decision_notes=[f"Retry count is {state.retry_count}.", f"Route bias is {state.route_bias or 'none'}."],
        )

    def assess(
        self,
        state: AgentState,
        decision: AgentDecision,
        tool_results: list[ToolResult],
    ) -> ExecutionResult:
        return build_execution_result(
            agent_name=self.name,
            decision=decision,
            tool_results=tool_results,
            summary="Coding state refreshed with plan analysis, architectural memory, and execution-surface observations.",
            actions=[
                "Loaded working memory",
                "Analyzed the current plan",
                "Prepared verification checks",
                "Inspected available execution connectors",
            ],
            artifacts={
                "plan_step_count": len(state.plan.steps) if state.plan else 0,
                "requested_capabilities": state.context.requested_capabilities if state.context else [],
                "tool_status_counts": status_counts(tool_results),
            },
            unresolved_focus="Change strategy before retrying unsupported architecture claims.",
            base_confidence=0.78,
        )
