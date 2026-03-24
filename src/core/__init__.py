__all__ = ["Settings", "get_settings", "permission_requires_approval"]


def __getattr__(name: str):
    if name in {"Settings", "get_settings"}:
        from .config import Settings, get_settings

        return {
            "Settings": Settings,
            "get_settings": get_settings,
        }[name]
    if name == "permission_requires_approval":
        from .permissions import permission_requires_approval

        return permission_requires_approval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
