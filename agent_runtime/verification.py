from __future__ import annotations

from .models import AgentState, VerificationCheck, VerificationReport
from .runtime_utils import tokenize_words


class VerificationEngine:
    def verify(self, state: AgentState) -> VerificationReport:
        if state.plan is None or state.decision is None or state.execution is None:
            raise ValueError("Plan, decision, and execution must exist before verification.")

        checks: list[VerificationCheck] = []
        gaps: list[str] = []
        verified_claims: list[str] = []
        unverified_claims: list[str] = []

        tool_failures = [result for result in state.last_tool_results if result.status not in {"success"}]
        checks.append(
            VerificationCheck(
                name="tool_execution_health",
                status="passed" if not tool_failures else "needs_attention",
                rationale="Tool calls should succeed or be explicitly handled.",
                evidence=[f"{result.tool_name}:{result.status}" for result in state.last_tool_results],
                severity="medium" if tool_failures else "info",
            )
        )
        if tool_failures:
            gaps.append("One or more tool calls were gated, unavailable, or errored.")

        fresh_observations = state.execution.observations or [obs.summary for obs in state.observations if obs.step_index == state.step_index]
        checks.append(
            VerificationCheck(
                name="fresh_observations_present",
                status="passed" if fresh_observations else "needs_attention",
                rationale="Each loop iteration should add fresh observations to state.",
                evidence=fresh_observations[:4] or ["No fresh observations were recorded."],
                severity="high" if not fresh_observations else "info",
            )
        )
        if not fresh_observations:
            gaps.append("The last action did not add fresh observations to the AgentState.")

        checks.append(
            VerificationCheck(
                name="success_criteria_defined",
                status="passed" if state.plan.success_criteria else "needs_attention",
                rationale="Plans should expose explicit completion criteria.",
                evidence=state.plan.success_criteria[:4] or ["No success criteria were provided."],
                severity="medium",
            )
        )
        if not state.plan.success_criteria:
            gaps.append("The plan lacks explicit success criteria.")

        evidence_text = " ".join(
            [
                state.execution.summary,
                *state.execution.actions,
                *state.execution.observations,
                *(evidence for result in state.last_tool_results for evidence in result.evidence),
                *(record.content for record in state.memory.retrieved[:3]),
            ]
        )
        evidence_tokens = tokenize_words(evidence_text)

        claims = state.execution.claims or state.decision.claims_to_verify
        for claim in claims:
            claim_tokens = tokenize_words(claim)
            overlap = len(claim_tokens & evidence_tokens)
            if overlap >= max(2, len(claim_tokens) // 4):
                verified_claims.append(claim)
            else:
                unverified_claims.append(claim)

        checks.append(
            VerificationCheck(
                name="claim_grounding",
                status="passed" if not unverified_claims else "needs_attention",
                rationale="Claims should be grounded in observations, tool evidence, or retrieved memory.",
                evidence=(verified_claims or ["No claims were verified."])[:4],
                severity="high" if unverified_claims else "info",
            )
        )
        if unverified_claims:
            gaps.append("Some claims were not sufficiently grounded in evidence.")

        if state.retry_count > 0:
            checks.append(
                VerificationCheck(
                    name="strategy_changed_after_retry",
                    status=(
                        "passed"
                        if state.replan_count > 0 or state.route_bias is not None or state.blocked_tools
                        else "needs_attention"
                    ),
                    rationale="A retry should change the control state rather than repeat the same attempt.",
                    evidence=[
                        f"replan_count={state.replan_count}",
                        f"route_bias={state.route_bias}",
                        f"blocked_tools={state.blocked_tools}",
                    ],
                    severity="high",
                )
            )
            if not (state.replan_count > 0 or state.route_bias is not None or state.blocked_tools):
                gaps.append("The runtime retried without changing strategy state.")

        passed = sum(1 for check in checks if check.status == "passed")
        confidence = round(passed / len(checks), 2) if checks else 0.0
        status = "passed" if not gaps else "needs_attention"
        summary = (
            "Verification found the current state sufficiently grounded."
            if not gaps
            else "Verification found grounding or strategy gaps that must change the next loop step."
        )

        return VerificationReport(
            status=status,
            summary=summary,
            checks=checks,
            verified_claims=verified_claims,
            unverified_claims=unverified_claims,
            gaps=gaps,
            confidence=confidence,
            retry_recommended=bool(gaps),
        )
