from __future__ import annotations

from src.graph.nodes.context_builder import ContextBuilderNode
from src.graph.nodes.executor import ExecutorNode
from src.graph.nodes.memory_update import MemoryUpdateNode
from src.graph.nodes.planner import PlannerNode
from src.graph.nodes.reflection import ReflectionNode
from src.graph.nodes.router import RouterNode


class GraphBuilder:
    def build(self) -> dict[str, object]:
        return {
            "context_builder": ContextBuilderNode,
            "planner": PlannerNode,
            "router": RouterNode,
            "executor": ExecutorNode,
            "reflection": ReflectionNode,
            "memory_update": MemoryUpdateNode,
        }
