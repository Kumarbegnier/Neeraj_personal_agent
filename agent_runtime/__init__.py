__all__ = ["ARCHITECTURE_STAGES", "Orchestrator", "build_default_orchestrator"]


def __getattr__(name: str):
    if name == "ARCHITECTURE_STAGES":
        from .architecture import ARCHITECTURE_STAGES

        return ARCHITECTURE_STAGES
    if name in {"Orchestrator", "build_default_orchestrator"}:
        from .orchestrator import Orchestrator, build_default_orchestrator

        return {
            "Orchestrator": Orchestrator,
            "build_default_orchestrator": build_default_orchestrator,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
