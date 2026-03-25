from __future__ import annotations

import unittest
from pathlib import Path

from agent_runtime.browser_goal_verifier import BrowserGoalVerifier
from agent_runtime.memory import MemorySystem
from agent_runtime.models import (
    AuthContext,
    Channel,
    ContextSignal,
    ContextSnapshot,
    GatewayResult,
    MemorySnapshot,
    RateLimitStatus,
    ToolRequest,
    UserRequest,
    AgentState,
)
from agent_runtime.skills import SkillLibrary
from agent_runtime.specialists.web import WebAgent
from agent_runtime.tools import ToolLayer
from src.schemas.browser import BrowserGoal, BrowserStateSnapshot


def build_context(*, approval_granted: bool = True) -> ContextSnapshot:
    gateway = GatewayResult(
        channel=Channel.text,
        normalized_message="browser goal verification test",
        auth=AuthContext(),
        rate_limit=RateLimitStatus(),
    )
    return ContextSnapshot(
        user_id="user-1",
        session_id="session-1",
        channel=Channel.text,
        latest_message="browser goal verification test",
        gateway=gateway,
        memory=MemorySnapshot(),
        active_goals=["browse_web"],
        system_goals=["stay_safe"],
        constraints=[],
        requested_capabilities=["browser", "verification"],
        signals=ContextSignal(complexity="moderate", risk_level="medium"),
        metadata={"approval_granted": approval_granted},
    )


class BrowserGoalVerifierTests(unittest.TestCase):
    def test_verifier_marks_goal_reached_from_text_dom_and_screenshot_metadata(self) -> None:
        verifier = BrowserGoalVerifier()
        goal = BrowserGoal(
            description="Reach the checkout review page.",
            target_url="https://example.com/checkout",
            required_text=["Order Summary"],
            dom_hints=["Proceed to Checkout"],
        )
        snapshot = BrowserStateSnapshot(
            step_name="extract_page_text",
            page_url="https://example.com/checkout",
            page_title="Checkout Review",
            page_text_snapshot="Order Summary is visible with the current basket items.",
            dom_text_summary="Proceed to Checkout button is visible.",
            screenshot_checkpoint={
                "label": "checkout-review",
                "path": r"C:\tmp\checkout-review.png",
                "captured_at": "2026-03-25T10:00:00Z",
                "source": "playwright",
            },
        )

        result = verifier.verify(goal, snapshot)

        self.assertTrue(result.goal_reached)
        self.assertEqual(result.status, "goal_reached")
        self.assertEqual(result.screenshot_checkpoint.path, str(Path(r"C:\tmp\checkout-review.png")))

    def test_verifier_detects_dangerous_submit_and_triggers_stop_guard(self) -> None:
        verifier = BrowserGoalVerifier()
        goal = BrowserGoal(
            description="Review the form before any submit.",
            action_kind="submit_form",
            action_target="Place order",
            stop_before_submit=True,
            allow_submit=False,
        )
        snapshot = BrowserStateSnapshot(
            step_name="extract_page_text",
            page_url="https://example.com/checkout",
            page_text_snapshot="Place order and confirm purchase controls are visible.",
            action_kind="submit_form",
            action_target="Place order",
            approval_granted=False,
        )

        result = verifier.verify(goal, snapshot)

        self.assertFalse(result.goal_reached)
        self.assertTrue(result.requires_confirmation)
        self.assertTrue(result.stop_before_submit_triggered)
        self.assertEqual(result.status, "blocked")

    def test_tool_layer_auto_verifies_major_browser_steps(self) -> None:
        layer = ToolLayer(MemorySystem(), SkillLibrary())
        context = build_context(approval_granted=True)

        requests = [
            ToolRequest(
                tool_name="browser_search",
                input_payload={"query": "Find docs", "session": "session-1"},
                priority=1,
                audit_metadata={
                    "browser_goal": {
                        "description": "Collect browser search results for the current browsing objective.",
                        "success_indicators": ["Find docs"],
                        "action_kind": "search",
                        "action_target": "Find docs",
                    }
                },
            ),
            ToolRequest(
                tool_name="open_page",
                input_payload={"url": "https://example.com/docs"},
                priority=2,
                audit_metadata={
                    "browser_goal": {
                        "description": "Open the target page for browser inspection.",
                        "target_url": "https://example.com/docs",
                        "action_kind": "open",
                        "action_target": "https://example.com/docs",
                    }
                },
            ),
            ToolRequest(
                tool_name="extract_page_text",
                input_payload={"text": "Implementation details and API references"},
                priority=3,
                audit_metadata={
                    "browser_goal": {
                        "description": "Extract grounded text from the current page.",
                        "required_text": ["Implementation details"],
                        "action_kind": "extract",
                        "action_target": "https://example.com/docs",
                    }
                },
            ),
        ]

        results = layer.run_many(requests, context)

        self.assertEqual(
            [result.tool_name for result in results],
            [
                "browser_search",
                "verify_browser_goal",
                "open_page",
                "verify_browser_goal",
                "extract_page_text",
                "verify_browser_goal",
            ],
        )
        verification_results = [result for result in results if result.tool_name == "verify_browser_goal"]
        self.assertEqual(len(verification_results), 3)
        self.assertTrue(all(result.status == "success" for result in verification_results))
        self.assertEqual(
            verification_results[0].output["snapshot"]["step_name"],
            "browser_search",
        )

    def test_web_agent_gates_risky_browser_actions_until_approval(self) -> None:
        state = AgentState(
            request=UserRequest(
                message="Review the checkout page and then submit the order.",
                metadata={
                    "url": "https://example.com/checkout",
                    "browser_action": "submit_form",
                    "action_target": "Place order",
                    "page_text": "Order Summary and Place order controls are visible.",
                },
            )
        )
        decision = WebAgent().build_decision(state, [])
        adapter_request = next(
            request for request in decision.tool_requests if request.tool_name == "browser_adapter"
        )
        extract_request = next(
            request for request in decision.tool_requests if request.tool_name == "extract_page_text"
        )

        self.assertTrue(adapter_request.requires_confirmation)
        self.assertEqual(adapter_request.side_effect, "send")
        self.assertEqual(extract_request.audit_metadata["action_kind"], "submit_form")

        gated_result = ToolLayer(MemorySystem(), SkillLibrary()).run(
            adapter_request,
            build_context(approval_granted=False),
        )

        self.assertEqual(gated_result.status, "gated")
        self.assertEqual(gated_result.blocked_reason, "confirmation_required")


if __name__ == "__main__":
    unittest.main()
