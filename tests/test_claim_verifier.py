from __future__ import annotations

import unittest

from agent_runtime.claim_verifier import ClaimVerifier
from agent_runtime.models import (
    AgentDecision,
    AgentRoute,
    AgentState,
    ExecutionPlan,
    ExecutionResult,
    MemoryRecord,
    ToolResult,
    ToolVerificationResult,
    UserRequest,
)
from agent_runtime.responder import ResponseComposer
from agent_runtime.verification import VerificationEngine


def build_plan(objective: str) -> ExecutionPlan:
    return ExecutionPlan(
        objective=objective,
        task_summary="Verify claims against collected evidence.",
        subtasks=[],
        required_tools=["search_web", "load_recent_memory"],
        risk_level="medium",
        approval_needed=False,
        reasoning="Use retrieved evidence before final synthesis.",
        react_cycles=[],
        steps=[],
        assumptions=[],
        constraints=[],
        success_criteria=["Ground major claims in evidence."],
        failure_modes=[],
        verification_focus=["claim grounding"],
        decomposition_strategy="react",
    )


def build_research_state() -> AgentState:
    objective = "Research whether irreversible actions require approval."
    state = AgentState(
        request=UserRequest(message=objective),
    )
    state.plan = build_plan(objective)
    state.route = AgentRoute(agent_name="research", rationale="Research mode requires evidence collection.")
    state.decision = AgentDecision(
        agent_name="research",
        summary="Collect evidence and verify the claims.",
        claims_to_verify=[
            "Approvals are required for irreversible actions.",
            "The system already deployed code to production.",
        ],
    )
    state.execution = ExecutionResult(
        agent_name="research",
        summary="The policy proves the system already deployed code to production.",
        observations=[
            "A policy page states that approvals are required before irreversible actions.",
            "Retrieved notes repeat the approval requirement for high-impact operations.",
        ],
        claims=list(state.decision.claims_to_verify),
        ready_for_response=True,
        goal_status="ready",
    )
    state.last_tool_results = [
        ToolResult(
            tool_name="search_web",
            status="success",
            output={
                "summary": "Search results say approvals are required before irreversible actions.",
                "results": [
                    {
                        "title": "Policy Guide",
                        "snippet": "Approvals are required for irreversible actions before execution.",
                    }
                ],
            },
            evidence=[
                "Policy Guide: approvals are required for irreversible actions before execution.",
            ],
            verification=ToolVerificationResult(
                status="passed",
                summary="Tool postconditions passed.",
                postconditions_met=True,
            ),
        )
    ]
    state.memory.retrieved = [
        MemoryRecord(
            memory_type="semantic",
            content="Compliance notes: irreversible actions require approval before execution.",
            source="semantic_memory",
            tags=["policy", "approval"],
        )
    ]
    return state


class ClaimVerifierTests(unittest.TestCase):
    def test_skips_low_risk_tasks_by_default(self) -> None:
        state = AgentState(request=UserRequest(message="Summarize this helper function."))

        report = ClaimVerifier().verify(
            state,
            candidate_response="The helper normalizes whitespace and returns a string.",
            source="response",
        )

        self.assertFalse(report.enabled)
        self.assertEqual(report.status, "skipped")
        self.assertEqual(report.disabled_reason, "low_risk_task")

    def test_links_candidate_claims_to_evidence(self) -> None:
        state = build_research_state()

        report = ClaimVerifier().verify(
            state,
            candidate_response=(
                "Approvals are required for irreversible actions. "
                "The system already deployed code to production."
            ),
            source="final_response",
        )

        self.assertTrue(report.enabled)
        self.assertIn("Approvals are required for irreversible actions", report.supported_claims)
        self.assertIn("The system already deployed code to production", report.unsupported_claims)
        supported = next(
            claim for claim in report.claims if claim.claim_text == "Approvals are required for irreversible actions"
        )
        self.assertTrue(supported.evidence_links)
        self.assertIn(supported.evidence_links[0].source_type, {"tool_evidence", "memory", "observation"})

    def test_verification_engine_attaches_claim_report(self) -> None:
        state = build_research_state()

        report = VerificationEngine(claim_verifier=ClaimVerifier()).verify(state)

        self.assertIsNotNone(report.claim_verification)
        self.assertTrue(report.claim_verification.enabled)
        self.assertIn("The system already deployed code to production.", report.unverified_claims)
        self.assertEqual(state.claim_verification, report.claim_verification)

    def test_response_composer_adds_caution_when_claims_remain_unsupported(self) -> None:
        state = build_research_state()
        state.verification = VerificationEngine(claim_verifier=ClaimVerifier()).verify(state)

        response = ResponseComposer(claim_verifier=ClaimVerifier()).synthesize_from_state(state)

        self.assertIn("Caution:", response)
        self.assertIsNotNone(state.claim_verification)
        self.assertEqual(state.claim_verification.status, "needs_attention")
        self.assertTrue(state.claim_verification.unsupported_claims)


if __name__ == "__main__":
    unittest.main()
