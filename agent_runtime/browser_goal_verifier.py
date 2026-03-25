from __future__ import annotations

from src.schemas.browser import BrowserGoal, BrowserStateSnapshot, BrowserVerificationResult


class BrowserGoalVerifier:
    _dangerous_terms = (
        "submit",
        "send",
        "confirm",
        "approve",
        "purchase",
        "pay",
        "checkout",
        "delete",
        "remove",
        "transfer",
        "publish",
        "book",
    )
    _submit_terms = (
        "submit",
        "send",
        "confirm",
        "checkout",
        "purchase",
        "pay",
        "place order",
        "book",
    )

    def verify(
        self,
        goal: BrowserGoal,
        snapshot: BrowserStateSnapshot,
    ) -> BrowserVerificationResult:
        normalized_snapshot = snapshot.model_copy(
            update={"screenshot_checkpoint": snapshot.screenshot_checkpoint}
        )
        combined_text = self._combined_text(normalized_snapshot)
        matched_indicators, missing_indicators = self._match_indicators(goal, normalized_snapshot, combined_text)
        dangerous_reasons = self._dangerous_action_reasons(goal, normalized_snapshot, combined_text)
        stop_before_submit = self._stop_before_submit(goal, normalized_snapshot, dangerous_reasons)
        requires_confirmation = (
            bool(dangerous_reasons)
            and goal.approval_required_for_dangerous_actions
            and not normalized_snapshot.approval_granted
        )
        avoided_present = [item for item in goal.avoided_text if self._contains(combined_text, item)]
        if avoided_present:
            missing_indicators.extend(f"avoid:{item}" for item in avoided_present)

        explicit_targets = bool(
            goal.target_url
            or goal.target_title
            or goal.success_indicators
            or goal.required_text
            or goal.dom_hints
        )
        observed_state = bool(
            normalized_snapshot.page_url
            or normalized_snapshot.page_title
            or normalized_snapshot.page_text_snapshot
            or normalized_snapshot.dom_text_summary
            or normalized_snapshot.extracted_text_blocks
        )
        goal_reached = (
            (not explicit_targets and observed_state) or not missing_indicators
        ) and not requires_confirmation and not stop_before_submit

        verification_notes = []
        if matched_indicators:
            verification_notes.append("Observed browser evidence matched the current step goal.")
        if missing_indicators:
            verification_notes.append("Some expected browser indicators were still missing.")
        if dangerous_reasons:
            verification_notes.append("A potentially dangerous browser action was detected.")
        if stop_before_submit:
            verification_notes.append("The stop-before-submit guard prevented continuation.")

        status = "goal_reached" if goal_reached else "in_progress"
        recommended_next_action = "continue_browsing"
        if requires_confirmation:
            status = "requires_confirmation"
            recommended_next_action = "request_user_approval"
        if stop_before_submit:
            status = "blocked"
            recommended_next_action = "pause_before_submit"
        if goal_reached:
            recommended_next_action = "proceed_to_next_browser_step"

        summary = self._summary_for(
            goal_reached=goal_reached,
            requires_confirmation=requires_confirmation,
            stop_before_submit=stop_before_submit,
            snapshot=normalized_snapshot,
            goal=goal,
        )

        return BrowserVerificationResult(
            status=status,
            summary=summary,
            goal_reached=goal_reached,
            requires_confirmation=requires_confirmation,
            should_stop=goal_reached or requires_confirmation or stop_before_submit,
            dangerous_action_detected=bool(dangerous_reasons),
            dangerous_action_reasons=dangerous_reasons,
            stop_before_submit_triggered=stop_before_submit,
            matched_indicators=matched_indicators,
            missing_indicators=missing_indicators,
            verification_notes=verification_notes,
            recommended_next_action=recommended_next_action,
            page_text_snapshot=normalized_snapshot.page_text_snapshot,
            dom_text_summary=normalized_snapshot.dom_text_summary,
            screenshot_checkpoint=normalized_snapshot.screenshot_checkpoint,
            metadata={
                "step_name": normalized_snapshot.step_name,
                "action_kind": normalized_snapshot.action_kind,
                "action_target": normalized_snapshot.action_target,
                "goal_description": goal.description,
            },
        )

    def _match_indicators(
        self,
        goal: BrowserGoal,
        snapshot: BrowserStateSnapshot,
        combined_text: str,
    ) -> tuple[list[str], list[str]]:
        matched: list[str] = []
        missing: list[str] = []

        if goal.target_url:
            if self._contains(snapshot.page_url.lower(), goal.target_url):
                matched.append(f"url:{goal.target_url}")
            else:
                missing.append(f"url:{goal.target_url}")
        if goal.target_title:
            if self._contains(snapshot.page_title.lower(), goal.target_title):
                matched.append(f"title:{goal.target_title}")
            else:
                missing.append(f"title:{goal.target_title}")

        for indicator in self._dedupe(
            [*goal.success_indicators, *goal.required_text, *goal.dom_hints]
        ):
            label = indicator.strip()
            if not label:
                continue
            if self._contains(combined_text, label):
                matched.append(label)
            else:
                missing.append(label)
        return matched, missing

    def _dangerous_action_reasons(
        self,
        goal: BrowserGoal,
        snapshot: BrowserStateSnapshot,
        _: str,
    ) -> list[str]:
        action_surface = " ".join(
            [
                goal.action_kind,
                goal.action_target,
                snapshot.action_kind,
                snapshot.action_target,
            ]
        ).lower()
        reasons = [
            f"action_contains:{term}"
            for term in self._dangerous_terms
            if term in action_surface
        ]
        candidate_surface = " ".join(snapshot.dangerous_action_candidates).lower()
        reasons.extend(
            f"candidate_contains:{term}"
            for term in self._dangerous_terms
            if term in candidate_surface
        )
        return self._dedupe(reasons)

    def _stop_before_submit(
        self,
        goal: BrowserGoal,
        snapshot: BrowserStateSnapshot,
        dangerous_reasons: list[str],
    ) -> bool:
        if not goal.stop_before_submit or goal.allow_submit:
            return False
        action_surface = " ".join(
            [
                goal.action_kind,
                goal.action_target,
                snapshot.action_kind,
                snapshot.action_target,
            ]
        ).lower()
        return bool(dangerous_reasons) and any(term in action_surface for term in self._submit_terms)

    def _summary_for(
        self,
        *,
        goal_reached: bool,
        requires_confirmation: bool,
        stop_before_submit: bool,
        snapshot: BrowserStateSnapshot,
        goal: BrowserGoal,
    ) -> str:
        if stop_before_submit:
            return (
                f"Stopped before executing '{snapshot.action_kind or goal.action_kind}' because the "
                "submit guard is active."
            )
        if requires_confirmation:
            return "A dangerous browser action was detected and now requires explicit approval."
        if goal_reached:
            return f"Verified browser goal for step '{snapshot.step_name or 'browser'}'."
        return f"Browser goal for step '{snapshot.step_name or 'browser'}' is not yet satisfied."

    def _combined_text(self, snapshot: BrowserStateSnapshot) -> str:
        parts = [
            snapshot.page_url,
            snapshot.page_title,
            snapshot.page_text_snapshot,
            snapshot.dom_text_summary,
            *snapshot.extracted_text_blocks,
        ]
        return " ".join(part for part in parts if part).lower()

    def _contains(self, haystack: str, needle: str) -> bool:
        normalized = needle.strip().lower()
        return bool(normalized) and normalized in haystack

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped
