__all__ = ["FrontendConfig", "bootstrap_page", "get_frontend_config"]


def __getattr__(name: str):
    if name == "bootstrap_page":
        from .bootstrap import bootstrap_page

        return bootstrap_page
    if name in {"FrontendConfig", "get_frontend_config"}:
        from .config import FrontendConfig, get_frontend_config

        return {
            "FrontendConfig": FrontendConfig,
            "get_frontend_config": get_frontend_config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
