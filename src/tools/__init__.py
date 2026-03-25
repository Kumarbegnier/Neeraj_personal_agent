from .base import ToolContract, ToolDescriptor, ToolExecutionContext, ToolInputModel, ToolOutputModel
from .catalog import get_tool_catalog, get_tool_descriptors
from src.schemas.catalog import ToolCatalog

__all__ = [
    "ToolDescriptor",
    "ToolExecutionContext",
    "ToolCatalog",
    "ToolContract",
    "ToolRegistry",
    "get_tool_catalog",
    "get_tool_descriptors",
    "ToolInputModel",
    "ToolOutputModel",
]


def __getattr__(name: str):
    if name == "ToolRegistry":
        from .registry import ToolRegistry

        return ToolRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
