from __future__ import annotations

import unittest

from agent_runtime.context_engine import ContextEngine
from agent_runtime.memory import MemorySystem
from agent_runtime.models import (
    AgentState,
    AuthContext,
    Channel,
    ConversationTurn,
    GatewayResult,
    MemoryRecord,
    MemorySnapshot,
    PermissionDecision,
    PermissionMode,
    RateLimitStatus,
    SessionPermissionState,
    UserRequest,
    WorkingMemory,
)
from src.schemas.catalog import ToolDescriptor
from src.schemas.context import ApprovalState, ProviderRouteHint, ProviderRoutingHints


def build_gateway(normalized_message: str) -> GatewayResult:
    return GatewayResult(
        channel=Channel.text,
        normalized_message=normalized_message,
        auth=AuthContext(),
        rate_limit=RateLimitStatus(),
    )


class ContextEngineTests(unittest.TestCase):
    def test_build_context_creates_research_grade_packet(self) -> None:
        request = UserRequest(
            message="Implement a research-grade context engine.",
            goals=["ship_code"],
            preferences={"tone": "concise"},
        )
        gateway = build_gateway("implement a research-grade context engine")
        memory_snapshot = MemorySnapshot(
            summary="user: implement a research-grade context engine",
            episodic=[
                ConversationTurn(role="user", content="Please build a modular runtime."),
                ConversationTurn(role="assistant", content="I will keep providers isolated."),
            ],
            retrieved=[
                MemoryRecord(
                    memory_type="semantic",
                    content="Previous architectural guidance emphasized isolated services.",
                    source="semantic_store",
                    score=2.3,
                    salience=0.9,
                    tags=["architecture", "modularity"],
                )
            ],
            goal_stack=["ship_code", "verify_outputs"],
            open_loops=["Confirm the context packet shape."],
            working_memory=WorkingMemory(
                objective="implement a research-grade context engine",
                distilled_context="The new context layer must remain modular and provider-agnostic.",
                retrieved_facts=["Isolated services are preferred."],
                plan_checkpoint="context_grounded",
            ),
        )
        approval_state = ApprovalState(
            permission_mode="confirm_required",
            requires_confirmation=True,
            risk_level="medium",
            approval_granted=False,
            blocked_tools=["send_email_draft"],
            gated_actions=["confirmation_required"],
            rationale="Outbound actions require confirmation.",
        )
        provider_hints = ProviderRoutingHints(
            hints=[
                ProviderRouteHint(
                    task_type="planning",
                    provider="deepseek",
                    model="planner-x",
                    reason="Planning prefers the default planning route.",
                    candidate_models=["planner-x", "planner-y"],
                )
            ]
        )
        engine = ContextEngine(
            tool_descriptor_loader=lambda: [
                ToolDescriptor(name="search_web", category="research", description="Search the web."),
                ToolDescriptor(
                    name="send_email_draft",
                    category="communication",
                    description="Draft an outbound email.",
                    risk_level="medium",
                    side_effect="draft",
                ),
            ]
        )

        packet = engine.build_context(
            request=request,
            gateway=gateway,
            memory_snapshot=memory_snapshot,
            active_goals=["ship_code", "verify_outputs"],
            requested_capabilities=["coding", "verification"],
            constraints=["Keep the context layer isolated from providers."],
            approval_state=approval_state,
            provider_routing_hints=provider_hints,
            current_execution_mode="react_planner_executor",
        )

        self.assertEqual(
            packet.normalized_user_request.normalized_message,
            "implement a research-grade context engine",
        )
        self.assertEqual(packet.memory.retrieval_query, "implement a research-grade context engine")
        self.assertEqual(packet.memory.semantic_memory_results[0].content, memory_snapshot.retrieved[0].content)
        self.assertEqual(packet.tool_availability.total_tools, 2)
        self.assertIn("send_email_draft", packet.tool_availability.approval_gated_tools)
        self.assertTrue(packet.approval_state.requires_confirmation)
        self.assertEqual(packet.provider_routing_hints.hints[0].provider, "deepseek")
        self.assertEqual(packet.current_execution_mode, "react_planner_executor")
        self.assertIn("Mode: react_planner_executor", packet.context_summary)

    def test_observe_updates_state_with_context_packet(self) -> None:
        memory_system = MemorySystem()
        request = UserRequest(
            user_id="user-1",
            session_id="session-1",
            message="Implement the context layer and keep approval gates safe.",
        )
        gateway = build_gateway("implement the context layer and keep approval gates safe")
        session = SessionPermissionState(
            user_id=request.user_id,
            session_id=request.session_id,
            permission=PermissionDecision(
                mode=PermissionMode.confirm_required,
                requires_confirmation=True,
                reason="Approval-gated actions need confirmation.",
            ),
        )
        memory_system.ingest_history(
            request.user_id,
            request.session_id,
            [
                ConversationTurn(role="user", content="We need a cleaner context service."),
                ConversationTurn(role="assistant", content="I will isolate it from providers."),
            ],
        )
        memory_system.ingest_semantic_records(
            request.user_id,
            request.session_id,
            [
                MemoryRecord(
                    memory_type="semantic",
                    content="Reflection lesson: keep routing outside the context builder.",
                    source="reflection",
                    salience=0.8,
                    tags=["reflection", "routing"],
                )
            ],
        )
        state = AgentState(request=request, gateway=gateway, session=session, step_index=1)
        state.blocked_tools = ["send_email_draft"]
        provider_hints = ProviderRoutingHints(
            hints=[
                ProviderRouteHint(
                    task_type="reasoning",
                    provider="openai",
                    model="gpt-runtime",
                    reason="Reasoning follows the configured default route.",
                )
            ]
        )
        engine = ContextEngine(
            tool_descriptor_loader=lambda: [
                ToolDescriptor(name="working_memory", category="memory", description="Inspect working memory."),
                ToolDescriptor(
                    name="send_email_draft",
                    category="communication",
                    description="Draft an outbound email.",
                    risk_level="medium",
                    side_effect="draft",
                ),
            ]
        )

        observed = engine.observe(
            state,
            memory_system,
            provider_routing_hints=provider_hints,
            execution_mode="react_context_first",
        )

        self.assertIs(observed, state)
        self.assertIsNotNone(state.context)
        self.assertIsNotNone(state.context_packet)
        self.assertEqual(state.context.execution_mode, "react_context_first")
        self.assertEqual(state.context_packet.approval_state.permission_mode, "confirm_required")
        self.assertFalse(state.context.metadata["approval_granted"])
        self.assertIn("coding", state.context_packet.requested_capabilities)
        self.assertIn("safety", state.context_packet.requested_capabilities)
        self.assertEqual(state.context_packet.provider_routing_hints.hints[0].task_type, "reasoning")
        self.assertEqual(state.context_packet.memory.recent_episodic_memory[0].role, "user")
        self.assertIn("send_email_draft", state.context_packet.tool_availability.approval_gated_tools)


if __name__ == "__main__":
    unittest.main()
