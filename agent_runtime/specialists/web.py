from __future__ import annotations

from ..models import AgentDecision, AgentState, ExecutionResult, SkillDescriptor, ToolResult
from .base import BaseAgent
from .common import build_execution_result, filter_blocked_requests, skill_names, status_counts, tool_request


class WebAgent(BaseAgent):
    name = "web"

    def decide(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
        requests = [
            tool_request("working_memory", "Load the current web objective and constraints.", priority=1),
            tool_request(
                "browser_search",
                "Prepare browser-first evidence collection.",
                payload={"query": state.request.message, "session": state.request.session_id},
                priority=2,
            ),
            tool_request(
                "open_page",
                "Open a page for downstream extraction.",
                payload={"url": state.request.metadata.get("url", "https://example.com")},
                priority=3,
            ),
            tool_request(
                "extract_page_text",
                "Extract textual content from the opened page or provided HTML.",
                payload={"text": state.request.metadata.get("page_text", state.request.message)},
                priority=4,
            ),
            tool_request("capability_map", "Confirm web execution capabilities.", priority=5),
        ]
        if self._memory_driven_verification(state):
            requests.append(
                tool_request(
                    "verification_harness",
                    "Prepare web-task verification checks.",
                    payload={"checks": state.plan.verification_focus if state.plan else [], "mode": "strict"},
                    priority=3,
                )
            )

        return AgentDecision(
            agent_name=self.name,
            summary="Prepared a browser-centered action set tied to the shared state loop.",
            skill_names=skill_names(skills),
            tool_requests=filter_blocked_requests(state, requests),
            reasoning="Keep browser work inside the same memory, verification, and safety loop as every other action.",
            response_strategy="Describe browser progress only after the loop converges.",
            expected_deliverables=["Browser execution observations"],
            claims_to_verify=["Web automation remains inside the shared orchestrated loop."],
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
            summary="Web state updated with browser and API observations.",
            actions=["Prepared browser interaction", "Bound web work to the shared loop"],
            artifacts={"tool_status_counts": status_counts(tool_results)},
            unresolved_focus="Change execution surface if browser tools stay gated.",
        )
