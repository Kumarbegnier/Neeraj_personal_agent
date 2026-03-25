from __future__ import annotations

from typing import Any

from src.services.modeling.types import ModelTaskType

from ..models import AgentDecision, AgentState, ExecutionResult, SkillDescriptor, ToolResult
from .base import BaseAgent
from .common import (
    confidence_from_results,
    filter_blocked_requests,
    observations_from_results,
    skill_names,
    status_counts,
    tool_request,
    unresolved_results,
)


class WebAgent(BaseAgent):
    name = "web"
    decision_task_type = ModelTaskType.WEB_GROUNDING

    def build_decision(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
        browser_metadata = self._browser_metadata(state)
        risky_browser_action = self._risky_browser_action(browser_metadata)
        requests = [
            tool_request("working_memory", "Load the current web objective and constraints.", priority=1),
            tool_request(
                "browser_search",
                "Prepare browser-first evidence collection.",
                payload={"query": state.request.message, "session": state.request.session_id},
                priority=2,
                audit_metadata=self._browser_audit_metadata(state, "browser_search"),
            ),
            tool_request(
                "open_page",
                "Open a page for downstream extraction.",
                payload={"url": state.request.metadata.get("url", "https://example.com")},
                priority=3,
                audit_metadata=self._browser_audit_metadata(state, "open_page"),
            ),
            tool_request(
                "extract_page_text",
                "Extract textual content from the opened page or provided HTML.",
                payload={"text": state.request.metadata.get("page_text", state.request.message)},
                priority=4,
                audit_metadata=self._browser_audit_metadata(state, "extract_page_text"),
            ),
            tool_request("capability_map", "Confirm web execution capabilities.", priority=5),
        ]
        if risky_browser_action is not None:
            requests.append(
                tool_request(
                    "browser_adapter",
                    "Prepare the requested browser interaction without executing risky side effects.",
                    payload={
                        "action": risky_browser_action["action_kind"],
                        "target": risky_browser_action["action_target"],
                        "browser": browser_metadata.get("browser", "playwright"),
                    },
                    priority=4,
                    risk_level="high",
                    side_effect=risky_browser_action["side_effect"],
                    requires_confirmation=True,
                    verification_hint="Keep dangerous browser actions approval-gated.",
                    expected_observation="Approval gate or readiness signal for the intended browser action.",
                    audit_metadata={
                        "browser_goal": self._step_goal(state, "extract_page_text"),
                        "action_kind": risky_browser_action["action_kind"],
                        "action_target": risky_browser_action["action_target"],
                    },
                )
            )
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
            summary="Prepared a browser-centered action set with post-step goal verification and submit-aware safety guards.",
            skill_names=skill_names(skills),
            tool_requests=filter_blocked_requests(state, requests),
            reasoning=(
                "Keep browser work inside the same memory, verification, and safety loop as every other action, "
                "and verify each major browser step against an explicit goal."
            ),
            response_strategy="Describe browser progress only after the loop converges.",
            expected_deliverables=["Browser execution observations", "Browser goal verification checkpoints"],
            claims_to_verify=[
                "Web automation remains inside the shared orchestrated loop.",
                "Major browser steps are verified before the loop proceeds.",
            ],
        )

    def assess(
        self,
        state: AgentState,
        decision: AgentDecision,
        tool_results: list[ToolResult],
    ) -> ExecutionResult:
        browser_verifications = [
            result for result in tool_results if result.tool_name == "verify_browser_goal"
        ]
        confirmation_required = any(
            bool(result.output.get("requires_confirmation")) for result in browser_verifications
        )
        submit_guard_triggered = any(
            bool(result.output.get("stop_before_submit_triggered")) for result in browser_verifications
        )
        all_browser_goals_reached = bool(browser_verifications) and all(
            bool(result.output.get("goal_reached")) for result in browser_verifications
        )
        unresolved = unresolved_results(tool_results)
        if browser_verifications and not all_browser_goals_reached and "browser_goal_not_reached" not in unresolved:
            unresolved.append("browser_goal_not_reached")
        if confirmation_required and "browser_action_requires_confirmation" not in unresolved:
            unresolved.append("browser_action_requires_confirmation")
        if submit_guard_triggered and "browser_submit_guard_triggered" not in unresolved:
            unresolved.append("browser_submit_guard_triggered")

        summary = "Verified browser progress after each major step."
        next_focus = "Ready for final synthesis."
        goal_status = "ready"
        if submit_guard_triggered:
            summary = "Browser progress paused because the stop-before-submit guard triggered."
            next_focus = "Keep the browser flow paused until the user confirms the risky action."
            goal_status = "blocked"
        elif confirmation_required:
            summary = "Browser progress found a dangerous action that now requires confirmation."
            next_focus = "Request confirmation before continuing the browser action."
            goal_status = "blocked"
        elif browser_verifications and not all_browser_goals_reached:
            summary = "Browser work produced observations, but at least one step did not reach its goal yet."
            next_focus = "Retry or change the browser strategy before continuing."
            goal_status = "in_progress"

        ready = all_browser_goals_reached and not unresolved and not confirmation_required and not submit_guard_triggered

        return ExecutionResult(
            agent_name=self.name,
            summary=summary,
            tool_results=tool_results,
            actions=[
                "Prepared browser interaction",
                "Verified browser goals after major steps",
                "Bound web work to the shared loop",
            ],
            artifacts={
                "tool_status_counts": status_counts(tool_results),
                "browser_goal_statuses": [
                    {
                        "step_name": result.output.get("snapshot", {}).get("step_name", ""),
                        "status": result.output.get("status", ""),
                        "goal_reached": result.output.get("goal_reached", False),
                        "requires_confirmation": result.output.get("requires_confirmation", False),
                    }
                    for result in browser_verifications
                ],
                "screenshot_checkpoints": [
                    result.output.get("verification", {}).get("screenshot_checkpoint", {})
                    for result in browser_verifications
                    if any(
                        result.output.get("verification", {})
                        .get("screenshot_checkpoint", {})
                        .get(field, "")
                        for field in ("label", "path", "captured_at", "source", "note")
                    )
                ],
            },
            claims=decision.claims_to_verify,
            observations=observations_from_results(tool_results),
            unresolved=unresolved,
            confidence=confidence_from_results(tool_results, base=0.74),
            goal_status=goal_status,
            ready_for_response=ready,
            requires_replan=bool(unresolved),
            next_focus=next_focus,
        )

    def _browser_audit_metadata(self, state: AgentState, step_name: str) -> dict[str, Any]:
        browser_metadata = self._browser_metadata(state)
        action_context = self._action_context_for_step(browser_metadata, step_name)
        screenshot_checkpoint = self._screenshot_checkpoint(browser_metadata, step_name)
        return {
            "browser_goal": self._step_goal(state, step_name),
            "page_url": str(browser_metadata.get("url", "https://example.com")),
            "page_title": str(browser_metadata.get("page_title", "")),
            "dom_text_summary": str(browser_metadata.get("dom_text_summary", "")),
            "screenshot_checkpoint": screenshot_checkpoint,
            "action_kind": action_context["action_kind"],
            "action_target": action_context["action_target"],
            "dangerous_action_candidates": action_context["dangerous_action_candidates"],
        }

    def _browser_metadata(self, state: AgentState) -> dict[str, Any]:
        return dict(state.request.metadata)

    def _step_goal(self, state: AgentState, step_name: str) -> dict[str, Any]:
        browser_metadata = self._browser_metadata(state)
        url = str(browser_metadata.get("url", "https://example.com"))
        success_indicators = self._list_value(browser_metadata.get("success_indicators"))
        required_text = self._list_value(browser_metadata.get("required_text"))
        dom_hints = self._list_value(browser_metadata.get("dom_hints"))
        if step_name == "browser_search":
            return {
                "description": "Collect browser search results for the current browsing objective.",
                "success_indicators": [state.request.message],
                "action_kind": "search",
                "action_target": state.request.message,
                "metadata": {"step_name": step_name},
            }
        if step_name == "open_page":
            return {
                "description": "Open the target page for browser inspection.",
                "target_url": url,
                "target_title": str(browser_metadata.get("page_title", "")),
                "success_indicators": success_indicators,
                "action_kind": "open",
                "action_target": url,
                "metadata": {"step_name": step_name},
            }
        return {
            "description": "Extract grounded text from the current page before any risky browser action.",
            "target_url": url,
            "required_text": required_text,
            "success_indicators": success_indicators,
            "dom_hints": dom_hints,
            "action_kind": str(browser_metadata.get("browser_action", "extract")),
            "action_target": str(browser_metadata.get("action_target", url)),
            "stop_before_submit": bool(browser_metadata.get("stop_before_submit", True)),
            "allow_submit": bool(browser_metadata.get("allow_submit", False)),
            "metadata": {"step_name": step_name},
        }

    def _action_context_for_step(
        self,
        browser_metadata: dict[str, Any],
        step_name: str,
    ) -> dict[str, Any]:
        url = str(browser_metadata.get("url", "https://example.com"))
        if step_name == "browser_search":
            return {
                "action_kind": "search",
                "action_target": str(browser_metadata.get("search_target", "")),
                "dangerous_action_candidates": [],
            }
        if step_name == "open_page":
            return {
                "action_kind": "open",
                "action_target": url,
                "dangerous_action_candidates": [],
            }
        explicit_action = str(browser_metadata.get("browser_action", "extract"))
        action_target = str(browser_metadata.get("action_target", url))
        dangerous_candidates = [explicit_action, action_target] if explicit_action != "extract" else []
        return {
            "action_kind": explicit_action,
            "action_target": action_target,
            "dangerous_action_candidates": dangerous_candidates,
        }

    def _risky_browser_action(self, browser_metadata: dict[str, Any]) -> dict[str, str] | None:
        action_kind = str(browser_metadata.get("browser_action", "")).strip().lower()
        if not action_kind:
            return None
        if not any(term in action_kind for term in ("submit", "send", "confirm", "purchase", "pay", "delete")):
            return None
        side_effect = "send"
        if any(term in action_kind for term in ("purchase", "pay", "checkout")):
            side_effect = "purchase"
        elif "delete" in action_kind:
            side_effect = "delete"
        return {
            "action_kind": action_kind,
            "action_target": str(browser_metadata.get("action_target", "")).strip(),
            "side_effect": side_effect,
        }

    def _screenshot_checkpoint(
        self,
        browser_metadata: dict[str, Any],
        step_name: str,
    ) -> dict[str, str]:
        checkpoint = browser_metadata.get("screenshot_checkpoint")
        if isinstance(checkpoint, dict):
            return {
                "label": str(checkpoint.get("label", step_name)),
                "path": str(checkpoint.get("path", "")),
                "captured_at": str(checkpoint.get("captured_at", "")),
                "source": str(checkpoint.get("source", "browser_agent")),
                "note": str(checkpoint.get("note", "")),
            }
        return {
            "label": step_name,
            "path": str(browser_metadata.get("screenshot_path", "")),
            "captured_at": str(browser_metadata.get("screenshot_captured_at", "")),
            "source": "browser_agent",
            "note": "",
        }

    def _list_value(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if value is None:
            return []
        text = str(value).strip()
        return [text] if text else []
