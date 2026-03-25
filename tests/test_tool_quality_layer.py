from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_runtime.memory import MemorySystem
from agent_runtime.models import (
    AuthContext,
    Channel,
    ContextSignal,
    ContextSnapshot,
    GatewayResult,
    MemorySnapshot,
    RateLimitStatus,
    ToolRequest,
)
from agent_runtime.tools import ToolLayer
from agent_runtime.skills import SkillLibrary


def build_context(*, approval_granted: bool = True) -> ContextSnapshot:
    gateway = GatewayResult(
        channel=Channel.text,
        normalized_message="test tool contract",
        auth=AuthContext(),
        rate_limit=RateLimitStatus(),
    )
    return ContextSnapshot(
        user_id="user-1",
        session_id="session-1",
        channel=Channel.text,
        latest_message="test tool contract",
        gateway=gateway,
        memory=MemorySnapshot(),
        active_goals=["ship_code"],
        system_goals=["stay_safe"],
        constraints=[],
        requested_capabilities=["coding"],
        signals=ContextSignal(complexity="low", risk_level="low"),
        metadata={"approval_granted": approval_granted},
    )


class ToolQualityLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory_system = MemorySystem()
        self.layer = ToolLayer(self.memory_system, SkillLibrary())
        self.context = build_context()

    def test_invalid_input_is_rejected_by_strict_schema(self) -> None:
        result = self.layer.run(
            ToolRequest(tool_name="summarize_file", input_payload={}),
            self.context,
        )

        self.assertEqual(result.status, "invalid_input")
        self.assertEqual(result.input_schema, "SummarizeFileInput")
        self.assertFalse(result.verification.postconditions_met)
        self.assertEqual(result.audit.tool_name, "summarize_file")

    def test_dry_run_prevents_side_effects_and_records_audit_metadata(self) -> None:
        before = self.memory_system.build_snapshot("user-1", "session-1").semantic

        result = self.layer.run(
            ToolRequest(
                tool_name="save_memory",
                input_payload={"content": "persist me", "tags": ["dry-run"]},
                dry_run=True,
            ),
            self.context,
        )

        after = self.memory_system.build_snapshot("user-1", "session-1").semantic
        self.assertEqual(result.status, "dry_run")
        self.assertTrue(result.dry_run)
        self.assertEqual(len(before), len(after))
        self.assertEqual(result.verification.status, "skipped")
        self.assertTrue(result.audit.dry_run)
        self.assertEqual(result.audit.input_schema, "SaveMemoryInput")

    def test_successful_execution_includes_postcondition_verification(self) -> None:
        result = self.layer.run(
            ToolRequest(
                tool_name="generate_code",
                input_payload={"language": "python", "objective": "build a helper"},
            ),
            self.context,
        )

        self.assertEqual(result.status, "success")
        self.assertTrue(result.verification.postconditions_met)
        self.assertEqual(result.output_schema, "GenerateCodeOutput")
        self.assertTrue(result.retryable)
        self.assertTrue(result.output["generated_code"])

    def test_registry_exposes_mcp_ready_contract_descriptions(self) -> None:
        contracts = self.layer.contracts()
        search_contract = next(contract for contract in contracts if contract["name"] == "search_web")

        self.assertTrue(search_contract["supports_dry_run"])
        self.assertTrue(search_contract["mcp_ready"])
        self.assertIn("properties", search_contract["input_schema"])
        self.assertIn("properties", search_contract["output_schema"])

    def test_successful_file_summary_uses_typed_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "sample.txt"
            target.write_text("one\ntwo\nthree\n", encoding="utf-8")

            result = self.layer.run(
                ToolRequest(
                    tool_name="summarize_file",
                    input_payload={"path": str(target)},
                ),
                self.context,
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.output["status"], "summarized")
        self.assertEqual(result.verification.status, "passed")
        self.assertEqual(result.audit.output_schema, "SummarizeFileOutput")


if __name__ == "__main__":
    unittest.main()
