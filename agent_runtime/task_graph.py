from __future__ import annotations

from .models import ControlDecision, ExecutionPlan, TaskGraph, TaskNode


class TaskGraphEngine:
    def build(self, plan: ExecutionPlan, control: ControlDecision) -> TaskGraph:
        nodes = [
            TaskNode(
                node_id=step.name,
                name=step.name.replace("_", " ").title(),
                description=step.description,
                owner=step.owner,
                status=step.status,
                depends_on=step.depends_on,
                verification_required=step.step_type in {"verification", "reflection"},
            )
            for step in plan.steps
        ]

        active_path = [node.node_id for node in nodes if node.status in {"completed", "in_progress"}]
        if not active_path:
            active_path = [nodes[0].node_id] if nodes else []

        return TaskGraph(
            state="planned",
            active_path=active_path,
            nodes=nodes,
        )

    def mark_route(self, graph: TaskGraph, agent_name: str) -> TaskGraph:
        for node in graph.nodes:
            if node.node_id == "route":
                node.status = "completed"
            elif node.node_id == "agent_decide":
                node.status = "in_progress"
                node.description = f"Prepare the {agent_name} specialist decision from the current routed state."

        graph.state = "active"
        graph.active_path = ["route", agent_name, "agent_decide"]
        return graph

    def mark_decision(
        self,
        graph: TaskGraph,
        agent_name: str,
        tool_count: int = 0,
    ) -> TaskGraph:
        for node in graph.nodes:
            if node.node_id == "agent_decide":
                node.status = "completed"
            elif node.node_id == "execute":
                node.status = "in_progress"
                node.description = (
                    f"Execute the {agent_name} branch while collecting tool evidence"
                    f" from {tool_count} requested tool call(s)."
                )

        graph.state = "active"
        graph.active_path = ["route", agent_name, "agent_decide", "execute"]
        return graph

    def activate(self, graph: TaskGraph, agent_name: str) -> TaskGraph:
        return self.mark_decision(graph, agent_name)

    def finalize(self, graph: TaskGraph) -> TaskGraph:
        for node in graph.nodes:
            node.status = "completed"
        graph.state = "completed"
        graph.active_path = ["execute", "verify", "reflect", "update_state", "finalize_response"]
        return graph
