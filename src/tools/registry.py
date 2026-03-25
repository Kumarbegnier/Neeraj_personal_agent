from __future__ import annotations

from src.runtime.models import ContextSnapshot, ToolRequest, ToolResult
from agent_runtime.tool_registry import ToolRegistry as RuntimeToolRegistry
from src.schemas.catalog import ToolCatalog


class ToolRegistry:
    def __init__(self, runtime_registry: RuntimeToolRegistry) -> None:
        self._runtime_registry = runtime_registry

    def catalog(self) -> ToolCatalog:
        return ToolCatalog(tools=self._runtime_registry.catalog())

    def contracts(self) -> list[dict[str, object]]:
        return self._runtime_registry.contracts()

    def execute(self, request: ToolRequest, context: ContextSnapshot) -> ToolResult:
        return self._runtime_registry.run(request, context)
