from __future__ import annotations

import json

from src.services.llm_service import LLMService
from src.services.modeling.types import ModelTaskType

from .claim_verifier import ClaimVerifier
from .models import AgentState, VerificationCheck, VerificationReport
from .runtime_utils import record_model_result, selected_model_override, tokenize_words


class VerificationEngine:
    def __init__(
        self,
        llm_service: LLMService | None = None,
        claim_verifier: ClaimVerifier | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._claim_verifier = claim_verifier or ClaimVerifier()

    def verify(self, state: AgentState) -> VerificationReport:
        fallback = self._fallback_report(state)
        if self._llm_service is None:
            return fallback

        model_result = self._llm_service.generate_structured(
            task_type=ModelTaskType.REASONING,
            stage="verification",
            output_type=VerificationReport,
            system_prompt=(
                "You are the verification and reasoning model for a hybrid multi-model agent system. "
                "Return a structured verification report that is evidence-aware and strict about grounding."
            ),
            user_prompt="\n".join(
                [
                    f"Objective: {state.request.message}",
                    f"Architecture mode: {state.architecture.mode.value if state.architecture else ''}",
                    f"Latest execution summary: {state.execution.summary if state.execution else ''}",
                    f"Tool statuses: {json.dumps([result.status for result in state.last_tool_results], ensure_ascii=True)}",
                    f"Claims to verify: {json.dumps(state.execution.claims if state.execution else [], ensure_ascii=True)}",
                    f"Fallback verification: {fallback.model_dump_json()}",
                ]
            ),
            fallback_output=fallback,
            selected_model=selected_model_override(state),
            metadata={"step_index": state.step_index},
        )
        record_model_result(state, model_result)
        report = VerificationReport.model_validate(model_result.output.model_dump())
        if report.claim_verification is None and fallback.claim_verification is not None:
            report = report.model_copy(update={"claim_verification": fallback.claim_verification})
        state.claim_verification = report.claim_verification
        return report

    def _fallback_report(self, state: AgentState) -> VerificationReport:
        if state.plan is None or state.decision is None or state.execution is None:
            raise ValueError("Plan, decision, and execution must exist before verification.")

        checks: list[VerificationCheck] = []
        gaps: list[str] = []
        verified_claims: list[str] = []
        weakly_supported_claims: list[str] = []
        unverified_claims: list[str] = []

        tool_failures = [result for result in state.last_tool_results if result.status not in {"success"}]
        contract_failures = [
            result
            for result in state.last_tool_results
            if result.verification.status not in {"passed", "skipped"}
        ]
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

        checks.append(
            VerificationCheck(
                name="tool_contract_postconditions",
                status="passed" if not contract_failures else "needs_attention",
                rationale="Each tool result should satisfy its typed contract and postcondition verifier.",
                evidence=[
                    f"{result.tool_name}:{result.verification.status}"
                    for result in state.last_tool_results
                ]
                or ["No tool contract verification results were recorded."],
                severity="high" if contract_failures else "info",
            )
        )
        if contract_failures:
            gaps.append("One or more tool outputs failed contract validation or postcondition checks.")

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

        claim_report = self._claim_verifier.verify(state, source="verification")
        state.claim_verification = claim_report
        if claim_report.enabled:
            verified_claims = list(claim_report.supported_claims)
            weakly_supported_claims = list(claim_report.weakly_supported_claims)
            unverified_claims = list(claim_report.unsupported_claims)
            checks.append(
                VerificationCheck(
                    name="claim_grounding",
                    status="passed" if claim_report.status == "passed" else "needs_attention",
                    rationale="Claims should be grounded in observations, tool evidence, or retrieved memory.",
                    evidence=[
                        f"{claim.claim_text} -> "
                        f"{', '.join(link.source_name or link.source_type for link in claim.evidence_links[:2])}"
                        for claim in claim_report.claims[:4]
                    ]
                    or ["No major claims were extracted for explicit verification."],
                    severity="high" if weakly_supported_claims or unverified_claims else "info",
                )
            )
            if weakly_supported_claims:
                gaps.append("Some claims are only weakly supported and should be tightened before final output.")
            if unverified_claims:
                gaps.append("Some claims were not sufficiently grounded in evidence.")
        else:
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

        if state.architecture and state.architecture.browser_heavy:
            browser_evidence = [
                evidence
                for result in state.last_tool_results
                if result.tool_name in {
                    "browser_adapter",
                    "browser_search",
                    "open_page",
                    "extract_page_text",
                    "verify_browser_goal",
                }
                for evidence in result.evidence
            ]
            checks.append(
                VerificationCheck(
                    name="browser_evidence_present",
                    status="passed" if browser_evidence else "needs_attention",
                    rationale="Browser-heavy paths should produce browser-grounded evidence.",
                    evidence=browser_evidence[:4] or ["No browser-grounded evidence was recorded."],
                    severity="high",
                )
            )
            if not browser_evidence:
                gaps.append("The browser-heavy path did not produce browser-grounded evidence.")

        browser_goal_results = [
            result for result in state.last_tool_results if result.tool_name == "verify_browser_goal"
        ]
        if browser_goal_results:
            unsafe_browser_results = [
                result
                for result in browser_goal_results
                if not bool(result.output.get("goal_reached"))
                or bool(result.output.get("requires_confirmation"))
                or bool(result.output.get("stop_before_submit_triggered"))
            ]
            checks.append(
                VerificationCheck(
                    name="browser_goal_verified",
                    status="passed" if not unsafe_browser_results else "needs_attention",
                    rationale="Major browser steps should verify that the intended goal state was reached safely.",
                    evidence=[
                        (
                            f"{result.output.get('snapshot', {}).get('step_name', 'browser')}:"
                            f"{result.output.get('status', 'unknown')}"
                        )
                        for result in browser_goal_results
                    ],
                    severity="high" if unsafe_browser_results else "info",
                )
            )
            if unsafe_browser_results:
                gaps.append("Browser goal verification did not confirm safe goal completion for every major step.")
            if any(bool(result.output.get("requires_confirmation")) for result in browser_goal_results):
                gaps.append("A browser action requires explicit approval before execution can continue.")
            if any(bool(result.output.get("stop_before_submit_triggered")) for result in browser_goal_results):
                gaps.append("The stop-before-submit guard triggered and paused the browser flow.")

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
            weakly_supported_claims=weakly_supported_claims,
            unverified_claims=unverified_claims,
            gaps=gaps,
            confidence=confidence,
            retry_recommended=bool(gaps),
            claim_verification=claim_report,
        )
