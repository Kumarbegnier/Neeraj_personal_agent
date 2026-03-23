from __future__ import annotations

from src.services.modeling.types import ModelTaskType

from ..models import AgentDecision, AgentState, ExecutionResult, SkillDescriptor, ToolResult
from .base import BaseAgent
from .common import build_execution_result, filter_blocked_requests, skill_names, status_counts, tool_request


class ResearchAgent(BaseAgent):
    name = "research"
    decision_task_type = ModelTaskType.RESEARCH

    def build_decision(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
        requests = [
            tool_request("working_memory", "Load retrieved facts and open questions.", priority=1),
            tool_request("vector_memory", "Retrieve prior evidence fragments.", priority=2),
            tool_request(
                "search_web",
                "Collect web search results relevant to the current query.",
                payload={"query": state.request.message},
                priority=3,
            ),
            tool_request(
                "load_recent_memory",
                "Load durable recent memory before synthesizing research findings.",
                payload={"limit": 4},
                priority=4,
            ),
            tool_request(
                "verification_harness",
                "Prepare evidence checks.",
                payload={"checks": state.plan.verification_focus if state.plan else [], "mode": "standard"},
                priority=5,
            ),
        ]
        return AgentDecision(
            agent_name=self.name,
            summary="Prepared a research action set combining retrieval, browsing, and verification.",
            skill_names=skill_names(skills),
            tool_requests=filter_blocked_requests(state, requests),
            reasoning="Use retrieved memory and evidence tools to gather observations before final synthesis.",
            response_strategy="Synthesize only from verified evidence in the final stage.",
            expected_deliverables=["Evidence collection lane", "Grounded synthesis state"],
            claims_to_verify=[
                "Research outputs are grounded in retrieved evidence.",
                "Verification is binding before final synthesis.",
            ],
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
            summary="Research state updated with retrieval-backed evidence observations.",
            actions=["Loaded retrieved evidence", "Prepared browsing and document review", "Attached verification checks"],
            artifacts={"tool_status_counts": status_counts(tool_results)},
            unresolved_focus="Broaden evidence collection if verification remains weak.",
        )
