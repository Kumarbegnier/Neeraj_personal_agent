from __future__ import annotations

from ..models import AgentDecision, AgentState, ExecutionResult, SkillDescriptor, ToolResult
from .base import BaseAgent
from .common import build_execution_result, filter_blocked_requests, skill_names, status_counts, tool_request


class TaskAgent(BaseAgent):
    name = "task"

    def decide(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
        requests = [
            tool_request("session_history", "Load prior task context.", priority=1),
            tool_request("working_memory", "Load active constraints and goals.", priority=2),
            tool_request(
                "calendar_adapter",
                "Prepare schedule operations.",
                payload={"calendar_action": "review_schedule"},
                priority=3,
            ),
            tool_request(
                "create_task_record",
                "Create a structured task record for the current workflow.",
                payload={"title": state.request.message[:80], "status": "planned"},
                priority=4,
            ),
            tool_request("skill_manifest", "Confirm reusable task workflows.", priority=5),
        ]
        if self._memory_driven_verification(state):
            requests.append(
                tool_request(
                    "verification_harness",
                    "Prepare task-operation verification checks.",
                    payload={"checks": state.plan.verification_focus if state.plan else [], "mode": "standard"},
                    priority=3,
                )
            )
        return AgentDecision(
            agent_name=self.name,
            summary="Prepared a task-coordination action set with operational context and risk review.",
            skill_names=skill_names(skills),
            tool_requests=filter_blocked_requests(state, requests),
            reasoning="Operational work should remain reviewable and approval-aware across loop iterations.",
            response_strategy="Summarize task state after verification and safety settle.",
            expected_deliverables=["Schedule-aware operational observations"],
            claims_to_verify=["Task execution remains confirmation-first."],
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
            summary="Task state updated with schedule and operational observations.",
            actions=["Loaded task context", "Prepared schedule review", "Checked operational risk"],
            artifacts={"tool_status_counts": status_counts(tool_results)},
            unresolved_focus="Adjust approval posture if task tools remain gated.",
        )
