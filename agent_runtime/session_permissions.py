from __future__ import annotations

from .memory import MemorySystem
from .models import GatewayResult, PermissionDecision, PermissionMode, SessionPermissionState, UserRequest


class SessionPermissionManager:
    def prepare(
        self,
        request: UserRequest,
        gateway: GatewayResult,
        memory_system: MemorySystem,
    ) -> SessionPermissionState:
        history_count = memory_system.history_length(request.user_id, request.session_id)
        existing_session = history_count > 0
        lowered = request.message.lower()
        approval_granted = bool(request.metadata.get("approval_granted"))

        risky_terms = ("delete", "purchase", "pay", "send", "deploy", "shutdown", "destroy", "wipe")
        contains_risky_language = any(term in lowered for term in risky_terms)
        requires_confirmation = contains_risky_language and not approval_granted
        mode = PermissionMode.confirm_required if requires_confirmation else PermissionMode.auto_approved
        reason = (
            "Preflight permission check flagged potentially external, destructive, or irreversible language."
            if requires_confirmation
            else "Caller supplied explicit approval for a potentially sensitive request."
            if approval_granted and contains_risky_language
            else "Session initialized without any preflight permission blockers."
        )

        if not gateway.accepted:
            mode = PermissionMode.blocked
            requires_confirmation = False
            reason = "Gateway rejection prevented session execution."

        notes = [
            f"Gateway auth mode: {gateway.auth.mode.value}.",
            f"Existing session: {'yes' if existing_session else 'no'}.",
            f"History turn count before execution: {history_count}.",
        ]

        return SessionPermissionState(
            user_id=request.user_id,
            session_id=request.session_id,
            existing_session=existing_session,
            history_turn_count=history_count,
            permission=PermissionDecision(
                mode=mode,
                requires_confirmation=requires_confirmation,
                reason=reason,
            ),
            notes=notes,
        )
