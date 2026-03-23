from __future__ import annotations

from src.services.modeling.types import ModelTaskType

from ..models import AgentDecision, AgentState, ExecutionResult, SkillDescriptor, ToolResult
from .base import BaseAgent
from .common import build_execution_result, filter_blocked_requests, skill_names, status_counts, tool_request


class FileAgent(BaseAgent):
    name = "file"
    decision_task_type = ModelTaskType.REASONING

    def build_decision(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
        requests = [
            tool_request(
                "summarize_file",
                "Prepare file inspection.",
                payload={"path": state.request.metadata.get("path", "")},
                priority=1,
            ),
            tool_request("vector_memory", "Retrieve similar document context.", priority=2),
            tool_request("working_memory", "Load objective and constraints for file analysis.", priority=3),
            tool_request("session_history", "Load prior document context.", priority=4),
            tool_request(
                "verification_harness",
                "Prepare analysis checks.",
                payload={"checks": state.plan.verification_focus if state.plan else [], "mode": "standard"},
                priority=5,
            ),
        ]
        return AgentDecision(
            agent_name=self.name,
            summary="Prepared a document-centric action set with retrieval and verification support.",
            skill_names=skill_names(skills),
            tool_requests=filter_blocked_requests(state, requests),
            reasoning="Document work should connect retrieval, inspection, and verification before final synthesis.",
            response_strategy="Summarize document state only after the loop converges.",
            expected_deliverables=["Document analysis observations"],
            claims_to_verify=["File analysis is grounded in retrieval and explicit checks."],
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
            summary="File-analysis state updated with document and retrieval observations.",
            actions=["Prepared document inspection", "Loaded retrieval context", "Attached verification checks"],
            artifacts={"tool_status_counts": status_counts(tool_results)},
            unresolved_focus="Change evidence sources if file-analysis verification remains weak.",
        )
