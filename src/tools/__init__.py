from .base import ToolDescriptor, ToolExecutionContext
from .catalog import get_tool_catalog, get_tool_descriptors
from src.schemas.catalog import ToolCatalog

__all__ = [
    "ToolDescriptor",
    "ToolExecutionContext",
    "ToolCatalog",
    "ToolRegistry",
    "get_tool_catalog",
    "get_tool_descriptors",
]


def __getattr__(name: str):
    if name == "ToolRegistry":
        from .registry import ToolRegistry

        return ToolRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
