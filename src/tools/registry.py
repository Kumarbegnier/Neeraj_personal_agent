from __future__ import annotations

from src.runtime.models import ContextSnapshot, ToolRequest, ToolResult
from agent_runtime.tool_registry import ToolRegistry as RuntimeToolRegistry
from src.schemas.catalog import ToolCatalog
from src.tools.catalog import get_tool_catalog


class ToolRegistry:
    def __init__(self, runtime_registry: RuntimeToolRegistry) -> None:
        self._runtime_registry = runtime_registry

    def catalog(self) -> ToolCatalog:
        return get_tool_catalog()

    def execute(self, request: ToolRequest, context: ContextSnapshot) -> ToolResult:
        return self._runtime_registry.run(request, context)
