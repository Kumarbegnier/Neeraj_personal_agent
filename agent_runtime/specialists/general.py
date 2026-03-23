from __future__ import annotations

from src.services.modeling.types import ModelTaskType

from ..models import AgentDecision, AgentState, ExecutionResult, SkillDescriptor, ToolResult
from .base import BaseAgent
from .common import build_execution_result, filter_blocked_requests, skill_names, status_counts, tool_request


class GeneralAgent(BaseAgent):
    name = "general"
    decision_task_type = ModelTaskType.REASONING

    def build_decision(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
        requests = [
            tool_request("capability_map", "Inspect the available system surface.", priority=1),
            tool_request("working_memory", "Load the current objective and constraints.", priority=2),
            tool_request("load_recent_memory", "Load durable recent memory.", payload={"limit": 4}, priority=3),
            tool_request(
                "verification_harness",
                "Prepare lightweight response checks.",
                payload={"checks": state.plan.verification_focus[:3] if state.plan else [], "mode": "standard"},
                priority=4,
            ),
        ]
        if state.retry_count > 0 and "semantic_memory" not in state.blocked_tools:
            requests.append(tool_request("semantic_memory", "Retrieve broader context before retrying.", priority=2))
        if state.retry_count == 0 and "save_memory" not in state.blocked_tools:
            requests.append(
                tool_request(
                    "save_memory",
                    "Persist a concise semantic memory for future steps.",
                    payload={"content": state.request.message},
                    priority=5,
                )
            )

        return AgentDecision(
            agent_name=self.name,
            summary="Prepared a general action set using the shared stateful loop.",
            skill_names=skill_names(skills),
            tool_requests=filter_blocked_requests(state, requests),
            reasoning="Keep simple requests inside the same stateful loop so verification and reflection still bind.",
            response_strategy="Provide a concise final synthesis from verified state only.",
            expected_deliverables=["Traceable default observations"],
            claims_to_verify=["The default path preserves memory, verification, and safety behavior."],
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
            summary="General execution state updated on the shared loop.",
            actions=["Used the shared stateful orchestration rail"],
            artifacts={"tool_status_counts": status_counts(tool_results)},
            unresolved_focus="Retry with a more specific specialist if general evidence is weak.",
        )
