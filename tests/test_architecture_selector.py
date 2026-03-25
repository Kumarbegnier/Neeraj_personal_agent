from __future__ import annotations

import unittest

from agent_runtime.architecture_selector import ArchitectureSelector
from agent_runtime.context_engine import ContextEngine
from agent_runtime.models import (
    AgentState,
    AuthContext,
    Channel,
    ContextSignal,
    ContextSnapshot,
    GatewayResult,
    MemoryRecord,
    MemorySnapshot,
    RateLimitStatus,
    UserRequest,
    WorkingMemory,
)
from src.schemas.adaptive import ArchitectureMode
from src.schemas.catalog import ToolDescriptor
from src.schemas.context import ApprovalState


def build_gateway(normalized_message: str) -> GatewayResult:
    return GatewayResult(
        channel=Channel.text,
        normalized_message=normalized_message,
        auth=AuthContext(),
        rate_limit=RateLimitStatus(),
    )


def build_state(
    message: str,
    *,
    capabilities: list[str],
    goals: list[str],
    complexity: str,
    risk_level: str,
    requires_confirmation: bool = False,
    retrieved: list[MemoryRecord] | None = None,
) -> AgentState:
    request = UserRequest(message=message, goals=goals)
    gateway = build_gateway(message.lower())
    memory_snapshot = MemorySnapshot(
        summary=f"user: {message}",
        retrieved=retrieved or [],
        goal_stack=list(goals),
        working_memory=WorkingMemory(
            objective=message.lower(),
            distilled_context="Architecture selection should stay modular and typed.",
            retrieved_facts=["The selector should remain provider-agnostic."],
            plan_checkpoint="context_loaded",
        ),
    )
    packet = ContextEngine(
        tool_descriptor_loader=lambda: [
            ToolDescriptor(name="search_web", category="research", description="Search the web."),
            ToolDescriptor(name="open_browser", category="browser", description="Browse a page."),
            ToolDescriptor(name="read_document", category="documents", description="Inspect a document."),
            ToolDescriptor(name="draft_message", category="communication", description="Draft a message."),
            ToolDescriptor(name="generate_code", category="coding", description="Generate code."),
            ToolDescriptor(name="run_tests", category="verification", description="Run verification."),
        ]
    ).build_context(
        request=request,
        gateway=gateway,
        memory_snapshot=memory_snapshot,
        active_goals=goals,
        requested_capabilities=capabilities,
        constraints=["Keep providers and routes outside the selector."],
        approval_state=ApprovalState(
            permission_mode="confirm_required" if requires_confirmation else "auto_approved",
            requires_confirmation=requires_confirmation,
            risk_level=risk_level,
            approval_granted=not requires_confirmation,
            rationale="High-stakes outbound actions require confirmation." if requires_confirmation else "",
        ),
        current_execution_mode="react",
    )

    state = AgentState(request=request, gateway=gateway, step_index=1)
    state.memory = memory_snapshot
    state.context_packet = packet
    state.context = ContextSnapshot(
        user_id=request.user_id,
        session_id=request.session_id,
        channel=request.channel,
        latest_message=gateway.normalized_message,
        gateway=gateway,
        memory=memory_snapshot,
        active_goals=list(goals),
        system_goals=["stay_safe", "be_traceable"],
        constraints=list(packet.constraints),
        requested_capabilities=list(capabilities),
        signals=ContextSignal(
            complexity=complexity,
            risk_level=risk_level,
            requested_capabilities=list(capabilities),
            collaboration_mode="modular-orchestration" if len(capabilities) >= 3 else "single-specialist",
            needs_memory_retrieval=True,
        ),
        context_packet=packet,
        execution_mode="react",
    )
    return state


class ArchitectureSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = ArchitectureSelector()

    def test_selects_direct_single_agent_path_for_small_low_risk_tasks(self) -> None:
        state = build_state(
            "Fix a small parsing bug in execution.py.",
            capabilities=["coding"],
            goals=["ship_code"],
            complexity="low",
            risk_level="low",
        )

        decision = self.selector.select(state)

        self.assertEqual(decision.mode, ArchitectureMode.DIRECT_SINGLE_AGENT)
        self.assertEqual(decision.primary_agent, "coding")
        self.assertFalse(decision.requires_planning)
        self.assertEqual(decision.task_characteristics.complexity, "low")
        self.assertTrue(decision.reasoning.pattern_scores)

    def test_selects_planner_executor_for_complex_tool_heavy_builds(self) -> None:
        state = build_state(
            "Implement a coordinated refactor across planning, execution, memory, and verification layers.",
            capabilities=["coding", "planning", "memory", "verification"],
            goals=["ship_code", "verify_outputs", "maintain_memory"],
            complexity="high",
            risk_level="medium",
        )

        decision = self.selector.select(state)

        self.assertEqual(decision.mode, ArchitectureMode.PLANNER_EXECUTOR)
        self.assertTrue(decision.requires_planning)
        self.assertEqual(decision.loop_strategy, "react_planner_executor")
        self.assertIn("general", decision.supporting_agents)

    def test_selects_multi_agent_research_for_parallel_evidence_tasks(self) -> None:
        state = build_state(
            "Research and compare multiple approaches across notes, documents, and prior findings.",
            capabilities=["research", "documents", "verification"],
            goals=["research", "verify_outputs", "inspect_files"],
            complexity="high",
            risk_level="medium",
            retrieved=[
                MemoryRecord(
                    memory_type="semantic",
                    content="Previous work favored evidence-driven architecture decisions.",
                    source="semantic_store",
                    tags=["research", "evidence"],
                )
            ],
        )

        decision = self.selector.select(state)

        self.assertEqual(decision.mode, ArchitectureMode.MULTI_AGENT_RESEARCH)
        self.assertGreaterEqual(decision.parallel_fanout, 2)
        self.assertEqual(decision.primary_agent, "research")
        self.assertIn("general", decision.supporting_agents)

    def test_selects_browser_heavy_verified_for_live_web_grounding(self) -> None:
        state = build_state(
            "Browse current product websites and verify the latest pricing with sources.",
            capabilities=["browser", "research", "verification"],
            goals=["browse_web", "research", "verify_outputs"],
            complexity="moderate",
            risk_level="medium",
        )

        decision = self.selector.select(state)

        self.assertEqual(decision.mode, ArchitectureMode.BROWSER_HEAVY_VERIFIED)
        self.assertTrue(decision.browser_heavy)
        self.assertTrue(decision.requires_verifier)
        self.assertEqual(decision.task_characteristics.browser_intensity, "high")

    def test_selects_communication_critic_for_high_stakes_messaging(self) -> None:
        state = build_state(
            "Draft a customer escalation email, critique the tone, and prepare it for sending.",
            capabilities=["communication", "verification"],
            goals=["communicate", "verify_outputs"],
            complexity="moderate",
            risk_level="high",
            requires_confirmation=True,
        )

        decision = self.selector.select(state)

        self.assertEqual(decision.mode, ArchitectureMode.COMMUNICATION_CRITIC)
        self.assertTrue(decision.critic_lane)
        self.assertEqual(decision.primary_agent, "communication")
        self.assertIn("general", decision.supporting_agents)
        self.assertEqual(decision.task_characteristics.communication_intensity, "high")


if __name__ == "__main__":
    unittest.main()
