from __future__ import annotations

from src.services.modeling.types import ModelTaskType

from ..models import AgentDecision, AgentState, ExecutionResult, SkillDescriptor, ToolResult
from .base import BaseAgent
from .common import (
    build_execution_result,
    filter_blocked_requests,
    skill_names,
    status_counts,
    tool_request,
)


class CommunicationAgent(BaseAgent):
    name = "communication"
    decision_task_type = ModelTaskType.COMMUNICATION

    def build_decision(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
        lowered = state.request.message.lower()
        live_send = any(word in lowered for word in ("send", "deliver", "email now", "message now"))
        dispatch_action = "send_message" if live_send else "draft_message"
        requests = [
            tool_request("session_history", "Load recent messaging context.", priority=1),
            tool_request("working_memory", "Surface constraints and message intent.", priority=2),
            tool_request(
                "risk_monitor",
                "Assess whether outbound actions need approval.",
                payload={
                    "action": dispatch_action,
                    "risk_level": state.context.signals.risk_level if state.context else "low",
                },
                priority=3,
            ),
            tool_request("skill_manifest", "Confirm reusable communication workflows.", priority=4),
            tool_request(
                "send_email_draft",
                "Prepare the outbound communication path.",
                payload={
                    "to": state.request.metadata.get("to", []),
                    "subject": state.request.metadata.get("subject", f"Draft about {state.request.message[:40]}"),
                    "body": state.request.metadata.get("body", state.request.message),
                    "action": dispatch_action,
                },
                priority=5,
                side_effect="send" if live_send else "draft",
                requires_confirmation=live_send,
                verification_hint=(
                    "Ensure live outbound actions stay gated until approval arrives."
                    if live_send
                    else "Ensure this remains a draft flow, not a live send."
                ),
            ),
        ]
        if self._memory_driven_verification(state):
            requests.append(
                tool_request(
                    "verification_harness",
                    "Prepare checks for outbound communication claims.",
                    payload={"checks": state.plan.verification_focus if state.plan else [], "mode": "strict"},
                    priority=2,
                )
            )

        return AgentDecision(
            agent_name=self.name,
            summary="Prepared a communication action set grounded in context, risk state, and verification readiness.",
            skill_names=skill_names(skills),
            tool_requests=filter_blocked_requests(state, requests),
            reasoning="Use recent history, working memory, and risk checks before draft-oriented outbound work.",
            response_strategy="Summarize communication progress after the loop converges.",
            expected_deliverables=["Context-aware communication flow", "Approval-aware outbound lane"],
            claims_to_verify=[
                (
                    "Live outbound communication remains confirmation-first."
                    if live_send
                    else "The communication flow remains confirmation-first."
                ),
                "Risk state is evaluated before outbound action.",
            ],
            decision_notes=[f"Blocked tools filtered: {', '.join(state.blocked_tools) or 'none'}."],
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
            summary="Communication workbench updated with context, risk posture, and draft-path status.",
            actions=["Loaded conversation context", "Assessed message risk", "Prepared draft-only dispatch"],
            artifacts={"tool_status_counts": status_counts(tool_results)},
            unresolved_focus="Tighten approval messaging if outbound tools remain gated.",
        )
