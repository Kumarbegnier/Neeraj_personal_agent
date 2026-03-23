from __future__ import annotations

from src.runtime.models import PermissionDecision

from src.core.permissions import permission_requires_approval


class ApprovalService:
    def needs_approval(self, permission: PermissionDecision) -> bool:
        return permission_requires_approval(permission)
