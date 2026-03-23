from __future__ import annotations

from src.runtime.models import PermissionDecision, PermissionMode


def permission_requires_approval(permission: PermissionDecision) -> bool:
    return (
        permission.mode == PermissionMode.confirm_required
        or permission.requires_confirmation
    )
