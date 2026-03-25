from __future__ import annotations

from .models import AgentState, PermissionDecision, PermissionMode, SafetyPolicyHit, SafetyReport


class SafetyPermissions:
    def review(self, state: AgentState) -> SafetyReport:
        lowered = state.request.message.lower()
        approval_granted = bool(state.request.metadata.get("approval_granted"))
        policy_hits: list[SafetyPolicyHit] = []
        gated_actions: list[str] = []

        if any(term in lowered for term in ("delete", "destroy", "wipe", "shutdown")):
            policy_hits.append(
                SafetyPolicyHit(
                    policy="destructive_action",
                    severity="high",
                    reason="The request contains destructive-operation language.",
                    affected_targets=["request_message"],
                )
            )

        if any(term in lowered for term in ("purchase", "pay", "wire", "send")):
            policy_hits.append(
                SafetyPolicyHit(
                    policy="external_side_effect",
                    severity="high",
                    reason="The request references real-world side effects or outbound execution.",
                    affected_targets=["request_message"],
                )
            )

        if state.decision:
            for tool in state.decision.tool_requests:
                if tool.requires_confirmation or tool.side_effect in {"send", "delete", "deploy", "purchase", "shutdown"}:
                    gated_actions.append(tool.tool_name)
                    policy_hits.append(
                        SafetyPolicyHit(
                            policy="tool_approval_gate",
                            severity="high" if tool.side_effect != "none" else "medium",
                            reason=f"Tool '{tool.tool_name}' may cause side effect '{tool.side_effect}'.",
                            affected_targets=[tool.tool_name],
                        )
                    )
                elif tool.risk_level == "medium":
                    policy_hits.append(
                        SafetyPolicyHit(
                            policy="medium_risk_tool",
                            severity="medium",
                            reason=f"Tool '{tool.tool_name}' is classified as medium risk.",
                            affected_targets=[tool.tool_name],
                        )
                    )

        if (
            state.verification
            and (state.verification.weakly_supported_claims or state.verification.unverified_claims)
            and state.execution
            and state.execution.actions
        ):
            policy_hits.append(
                SafetyPolicyHit(
                    policy="unverified_execution_claims",
                    severity="medium",
                    reason="Some execution claims remain weakly supported or insufficiently verified.",
                    affected_targets=state.execution.actions[:3],
                )
            )

        risk_level = "low"
        if any(hit.severity == "high" for hit in policy_hits):
            risk_level = "high"
        elif policy_hits:
            risk_level = "medium"

        requires_confirmation = (bool(gated_actions) or risk_level == "high") and not approval_granted
        mode = PermissionMode.confirm_required if requires_confirmation else PermissionMode.auto_approved
        if state.session and state.session.permission.mode == PermissionMode.blocked:
            mode = PermissionMode.blocked
            requires_confirmation = False

        permission = PermissionDecision(
            mode=mode,
            requires_confirmation=requires_confirmation,
            reason=(
                "Safety review recorded elevated risk, but the caller already granted approval."
                if approval_granted and (gated_actions or risk_level == "high")
                else
                "Safety review found side effects or elevated risk that require confirmation."
                if mode == PermissionMode.confirm_required
                else "Safety review found no action that requires additional confirmation."
                if mode == PermissionMode.auto_approved
                else "Safety review blocked execution."
            ),
        )

        notes = [
            f"Reviewed {len(state.execution.actions) if state.execution else 0} action(s) and {len(state.decision.tool_requests) if state.decision else 0} tool request(s).",
            "Sandbox assumption: workspace-write local development environment.",
        ]
        if approval_granted:
            notes.append("Caller approval metadata was present for this request.")

        return SafetyReport(
            status="reviewed" if mode != PermissionMode.blocked else "blocked",
            sandbox="workspace-write",
            permission=permission,
            audit_log_saved=True,
            notes=notes,
            policy_hits=policy_hits,
            gated_actions=gated_actions,
            risk_level=risk_level,
        )
