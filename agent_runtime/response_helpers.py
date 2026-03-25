from __future__ import annotations

from .models import (
    PermissionDecision,
    PermissionMode,
    ReflectionReport,
    SafetyReport,
    VerificationReport,
)


def build_confirmation(
    agent_name: str,
    actions: list[str],
    permission: PermissionDecision,
) -> str:
    action_count = len(actions)
    approval_clause = (
        "Approval required before side effects."
        if permission.requires_confirmation
        else "No additional approval gate triggered."
    )
    return (
        f"Completed via the '{agent_name}' agent with {action_count} action"
        f"{'' if action_count == 1 else 's'}. {approval_clause}"
    )


def skipped_verification(summary: str = "Verification did not run.") -> VerificationReport:
    return VerificationReport(
        status="skipped",
        summary=summary,
        checks=[],
        verified_claims=[],
        weakly_supported_claims=[],
        unverified_claims=[],
        gaps=[],
        confidence=0.0,
        retry_recommended=False,
        claim_verification=None,
    )


def skipped_reflection(summary: str = "Reflection did not run.") -> ReflectionReport:
    return ReflectionReport(
        status="skipped",
        summary=summary,
        checks=[],
        issues=[],
        repairs=[],
        lessons=[],
        next_actions=[],
        confidence=0.0,
    )


def reviewed_safety(
    permission: PermissionDecision | None = None,
    *,
    status: str = "reviewed",
    notes: list[str] | None = None,
    policy_hits: list | None = None,
    gated_actions: list[str] | None = None,
    risk_level: str = "low",
) -> SafetyReport:
    resolved_permission = permission or PermissionDecision(
        mode=PermissionMode.auto_approved,
        requires_confirmation=False,
        reason="No safety review was required.",
    )
    return SafetyReport(
        status=status,
        sandbox="workspace-write",
        permission=resolved_permission,
        audit_log_saved=True,
        notes=notes or [],
        policy_hits=policy_hits or [],
        gated_actions=gated_actions or [],
        risk_level=risk_level,
    )
