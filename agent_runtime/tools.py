from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from src.agents import get_agent_descriptors
from src.schemas.browser import BrowserGoal, BrowserStateSnapshot
from src.schemas.catalog import ToolDescriptor
from src.tools.base import ToolContract, ToolVerifier
from src.tools.catalog import get_tool_descriptor_map

from .browser_goal_verifier import BrowserGoalVerifier
from .memory import MemorySystem
from .models import (
    ContextSnapshot,
    MemoryRecord,
    ToolAuditMetadata,
    ToolRequest,
    ToolResult,
    ToolVerificationResult,
)
from .skills import SkillLibrary
from .tool_schemas import (
    BrowserGoalVerificationInput,
    BrowserGoalVerificationOutput,
    BrowserSearchInput,
    BrowserSearchOutput,
    CapabilityMapOutput,
    ConnectorActionInput,
    ConnectorAvailabilityOutput,
    CreateTaskRecordInput,
    CreateTaskRecordOutput,
    EmailDraftInput,
    EmailDraftOutput,
    EmptyToolInput,
    ExecutionCatalogOutput,
    ExtractPageTextInput,
    ExtractPageTextOutput,
    GenerateCodeInput,
    GenerateCodeOutput,
    GoalStackOutput,
    LoadRecentMemoryInput,
    LoadRecentMemoryOutput,
    MemoryEntriesOutput,
    OpenPageInput,
    OpenPageOutput,
    PlanAnalyzerInput,
    PlanAnalyzerOutput,
    RiskMonitorInput,
    RiskMonitorOutput,
    SaveMemoryInput,
    SaveMemoryOutput,
    SearchWebInput,
    SearchWebOutput,
    SessionHistoryOutput,
    SkillManifestOutput,
    SummarizeFileInput,
    SummarizeFileOutput,
    VerificationHarnessInput,
    VerificationHarnessOutput,
    WorkingMemoryOutput,
)


MAX_SUMMARY_FILE_BYTES = 1_000_000
BROWSER_MAJOR_STEP_NAMES = frozenset({"browser_search", "open_page", "extract_page_text"})


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, set, tuple)):
        return bool(value)
    return True


def _postcondition_verifier(
    *,
    required_fields: tuple[str, ...] = (),
    accepted_statuses: tuple[str, ...] = (),
) -> ToolVerifier:
    def verify(
        _: ContextSnapshot,
        __: BaseModel,
        output: BaseModel,
    ) -> ToolVerificationResult:
        if getattr(output, "dry_run", False):
            return ToolVerificationResult(
                status="skipped",
                summary="Dry run skipped postcondition verification.",
                postconditions_met=True,
                checks=["Dry run prevented side effects while preserving the contract shape."],
            )

        failures: list[str] = []
        checks: list[str] = []
        output_status = str(getattr(output, "status", "") or "")
        if accepted_statuses:
            checks.append(f"Expected output status in {list(accepted_statuses)}.")
            if output_status not in accepted_statuses:
                failures.append(
                    f"Output status '{output_status or 'missing'}' is outside the accepted set."
                )
        for field_name in required_fields:
            value = getattr(output, field_name, None)
            checks.append(f"Required field '{field_name}' must be populated.")
            if not _has_value(value):
                failures.append(f"Field '{field_name}' did not satisfy the postcondition.")

        if failures:
            return ToolVerificationResult(
                status="failed",
                summary="Tool output did not satisfy its postconditions.",
                postconditions_met=False,
                checks=checks,
                failures=failures,
                details={"status": output_status},
            )

        summary = (
            "Tool output satisfied schema validation and postcondition checks."
            if checks
            else "Tool output satisfied schema validation."
        )
        return ToolVerificationResult(
            status="passed",
            summary=summary,
            postconditions_met=True,
            checks=checks,
            details={"status": output_status},
        )

    return verify


class ToolLayer:
    def __init__(self, memory_system: MemorySystem, skill_library: SkillLibrary) -> None:
        self._memory_system = memory_system
        self._skill_library = skill_library
        self._catalog = get_tool_descriptor_map()
        self._browser_goal_verifier = BrowserGoalVerifier()
        self._tools = self._build_registry()

    def _contract(
        self,
        tool_name: str,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        handler,
        *,
        required_fields: tuple[str, ...] = (),
        accepted_statuses: tuple[str, ...] = (),
        verifier: ToolVerifier | None = None,
    ) -> ToolContract:
        descriptor = self._catalog[tool_name].model_copy(deep=True)
        return ToolContract(
            descriptor=descriptor,
            input_model=input_model,
            output_model=output_model,
            handler=handler,
            verifier=verifier
            or _postcondition_verifier(
                required_fields=required_fields,
                accepted_statuses=accepted_statuses,
            ),
            retryable=descriptor.retryable,
        )

    def _build_registry(self) -> dict[str, ToolContract]:
        contracts = {
            "session_history": self._contract(
                "session_history",
                EmptyToolInput,
                SessionHistoryOutput,
                self._session_history,
                required_fields=("turn_count", "summary"),
            ),
            "semantic_memory": self._contract(
                "semantic_memory",
                EmptyToolInput,
                MemoryEntriesOutput,
                self._semantic_memory,
                required_fields=("count",),
            ),
            "vector_memory": self._contract(
                "vector_memory",
                EmptyToolInput,
                MemoryEntriesOutput,
                self._vector_memory,
                required_fields=("count",),
            ),
            "working_memory": self._contract(
                "working_memory",
                EmptyToolInput,
                WorkingMemoryOutput,
                self._working_memory,
                required_fields=("objective", "checkpoint"),
            ),
            "goal_stack": self._contract(
                "goal_stack",
                EmptyToolInput,
                GoalStackOutput,
                self._goal_stack,
            ),
        }
        contracts.update(
            {
                "capability_map": self._contract(
                    "capability_map",
                    EmptyToolInput,
                    CapabilityMapOutput,
                    self._capability_map,
                    required_fields=("interfaces", "specialized_agents", "tool_categories"),
                ),
                "execution_catalog": self._contract(
                    "execution_catalog",
                    EmptyToolInput,
                    ExecutionCatalogOutput,
                    self._execution_catalog,
                    required_fields=("connectors", "governance"),
                ),
                "skill_manifest": self._contract(
                    "skill_manifest",
                    EmptyToolInput,
                    SkillManifestOutput,
                    self._skill_manifest,
                    required_fields=("skills",),
                ),
                "plan_analyzer": self._contract(
                    "plan_analyzer",
                    PlanAnalyzerInput,
                    PlanAnalyzerOutput,
                    self._plan_analyzer,
                    required_fields=("objective",),
                ),
                "verification_harness": self._contract(
                    "verification_harness",
                    VerificationHarnessInput,
                    VerificationHarnessOutput,
                    self._verification_harness,
                    required_fields=("check_count", "mode"),
                ),
                "risk_monitor": self._contract(
                    "risk_monitor",
                    RiskMonitorInput,
                    RiskMonitorOutput,
                    self._risk_monitor,
                    required_fields=("risk_level", "requested_action"),
                ),
            }
        )
        contracts.update(self._connector_contracts())
        contracts.update(self._artifact_contracts())
        missing_handlers = set(self._catalog) - set(contracts)
        if missing_handlers:
            raise ValueError(f"Missing tool handlers for: {sorted(missing_handlers)}")
        return contracts

    def _connector_contracts(self) -> dict[str, ToolContract]:
        return {
            "api_dispatcher": self._contract(
                "api_dispatcher",
                ConnectorActionInput,
                ConnectorAvailabilityOutput,
                self._api_dispatcher,
                required_fields=("connector", "requested_action"),
                accepted_statuses=("available",),
            ),
            "browser_adapter": self._contract(
                "browser_adapter",
                ConnectorActionInput,
                ConnectorAvailabilityOutput,
                self._browser_adapter,
                required_fields=("connector", "target"),
                accepted_statuses=("available",),
            ),
            "os_adapter": self._contract(
                "os_adapter",
                ConnectorActionInput,
                ConnectorAvailabilityOutput,
                self._os_adapter,
                required_fields=("connector", "operation"),
                accepted_statuses=("available",),
            ),
            "database_adapter": self._contract(
                "database_adapter",
                ConnectorActionInput,
                ConnectorAvailabilityOutput,
                self._database_adapter,
                required_fields=("connector", "database"),
                accepted_statuses=("available",),
            ),
            "github_adapter": self._contract(
                "github_adapter",
                ConnectorActionInput,
                ConnectorAvailabilityOutput,
                self._github_adapter,
                required_fields=("connector", "repository_action"),
                accepted_statuses=("available",),
            ),
            "calendar_adapter": self._contract(
                "calendar_adapter",
                ConnectorActionInput,
                ConnectorAvailabilityOutput,
                self._calendar_adapter,
                required_fields=("connector", "calendar_action"),
                accepted_statuses=("available",),
            ),
            "document_adapter": self._contract(
                "document_adapter",
                ConnectorActionInput,
                ConnectorAvailabilityOutput,
                self._document_adapter,
                required_fields=("connector", "document_action"),
                accepted_statuses=("available",),
            ),
        }

    def _artifact_contracts(self) -> dict[str, ToolContract]:
        return {
            "send_email_draft": self._contract(
                "send_email_draft",
                EmailDraftInput,
                EmailDraftOutput,
                self._send_email_draft,
                required_fields=("draft_id", "subject", "body_preview", "approval_required"),
                accepted_statuses=("draft_saved",),
            ),
            "search_web": self._contract(
                "search_web",
                SearchWebInput,
                SearchWebOutput,
                self._search_web,
                required_fields=("query", "results", "source"),
            ),
            "browser_search": self._contract(
                "browser_search",
                BrowserSearchInput,
                BrowserSearchOutput,
                self._browser_search,
                required_fields=("query", "browser_session", "results"),
                accepted_statuses=("browser_search_prepared",),
            ),
            "verify_browser_goal": self._contract(
                "verify_browser_goal",
                BrowserGoalVerificationInput,
                BrowserGoalVerificationOutput,
                self._verify_browser_goal,
                required_fields=("verification", "snapshot"),
                accepted_statuses=("goal_reached", "in_progress", "requires_confirmation", "blocked", "dry_run"),
                verifier=self._browser_goal_postcondition_verifier(),
            ),
            "save_memory": self._contract(
                "save_memory",
                SaveMemoryInput,
                SaveMemoryOutput,
                self._save_memory,
                required_fields=("saved", "memory_id", "content"),
            ),
            "load_recent_memory": self._contract(
                "load_recent_memory",
                LoadRecentMemoryInput,
                LoadRecentMemoryOutput,
                self._load_recent_memory,
            ),
            "summarize_file": self._contract(
                "summarize_file",
                SummarizeFileInput,
                SummarizeFileOutput,
                self._summarize_file,
                required_fields=("path", "summary"),
                accepted_statuses=("summarized",),
            ),
            "generate_code": self._contract(
                "generate_code",
                GenerateCodeInput,
                GenerateCodeOutput,
                self._generate_code,
                required_fields=("language", "generated_code"),
            ),
            "open_page": self._contract(
                "open_page",
                OpenPageInput,
                OpenPageOutput,
                self._open_page,
                required_fields=("url", "browser"),
                accepted_statuses=("opened_placeholder",),
            ),
            "extract_page_text": self._contract(
                "extract_page_text",
                ExtractPageTextInput,
                ExtractPageTextOutput,
                self._extract_page_text,
                required_fields=("text", "length"),
                accepted_statuses=("extracted",),
            ),
            "create_task_record": self._contract(
                "create_task_record",
                CreateTaskRecordInput,
                CreateTaskRecordOutput,
                self._create_task_record,
                required_fields=("task_id", "title", "stored", "task_status"),
            ),
        }

    def catalog(self) -> list[ToolDescriptor]:
        return [
            contract.descriptor.model_copy(deep=True)
            for contract in sorted(self._tools.values(), key=lambda item: item.descriptor.name)
        ]

    def contracts(self) -> list[dict[str, Any]]:
        return [
            contract.describe()
            for contract in sorted(self._tools.values(), key=lambda item: item.descriptor.name)
        ]

    def _descriptor_for(self, tool_name: str) -> ToolDescriptor | None:
        descriptor = self._catalog.get(tool_name)
        return descriptor.model_copy(deep=True) if descriptor is not None else None

    def run(self, request: ToolRequest, context: ContextSnapshot) -> ToolResult:
        contract = self._tools.get(request.tool_name)
        descriptor = self._descriptor_for(request.tool_name)
        if contract is None or descriptor is None:
            return ToolResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                status="unavailable",
                output={"message": f"Tool '{request.tool_name}' is not registered."},
                evidence=[f"Tool '{request.tool_name}' is absent from the registry."],
                risk_level=request.risk_level,
                retryable=False,
                verification=ToolVerificationResult(
                    status="failed",
                    summary="The requested tool is not registered.",
                    postconditions_met=False,
                    failures=[f"Tool '{request.tool_name}' is missing from the registry."],
                ),
                audit=ToolAuditMetadata(
                    call_id=request.call_id,
                    tool_name=request.tool_name,
                    risk_level=request.risk_level,
                    metadata=dict(request.audit_metadata),
                ),
            )

        risk_level = (
            descriptor.risk_level
            if descriptor is not None and request.risk_level == "low"
            else request.risk_level
        )
        side_effect = (
            descriptor.side_effect
            if descriptor is not None and request.side_effect == "none"
            else request.side_effect
        )
        retryable = descriptor.retryable if request.retryable is None else request.retryable

        try:
            validated_input = contract.input_model.model_validate(request.input_payload)
        except ValidationError as exc:
            return self._invalid_input_result(contract, request, risk_level, side_effect, exc)

        approval_granted = bool(context.metadata.get("approval_granted"))
        requires_confirmation = request.requires_confirmation or self._should_gate(risk_level, side_effect)
        normalized_input = validated_input.model_dump(mode="json")

        if not approval_granted and requires_confirmation and not request.dry_run:
            return self._gated_result(
                contract=contract,
                request=request,
                normalized_input=normalized_input,
                risk_level=risk_level,
                side_effect=side_effect,
                retryable=retryable,
            )

        if request.dry_run:
            return self._dry_run_result(
                contract=contract,
                request=request,
                validated_input=validated_input,
                normalized_input=normalized_input,
                risk_level=risk_level,
                side_effect=side_effect,
                retryable=retryable,
            )

        try:
            raw_output = contract.handler(context, validated_input.model_dump(mode="python"))
            validated_output = contract.output_model.model_validate(raw_output)
            if not getattr(validated_output, "summary", ""):
                validated_output = validated_output.model_copy(
                    update={
                        "summary": (
                            f"Tool '{request.tool_name}' completed with status "
                            f"'{getattr(validated_output, 'status', 'ready')}'."
                        )
                    }
                )
            verification = contract.verifier(context, validated_input, validated_output)
            result_status = "success" if verification.postconditions_met else "verification_failed"
            return self._success_result(
                contract=contract,
                request=request,
                normalized_input=normalized_input,
                output_payload=validated_output.model_dump(mode="json"),
                verification=verification,
                risk_level=risk_level,
                side_effect=side_effect,
                retryable=retryable,
                status=result_status,
            )
        except ValidationError as exc:
            return self._invalid_output_result(
                contract=contract,
                request=request,
                normalized_input=normalized_input,
                risk_level=risk_level,
                side_effect=side_effect,
                retryable=retryable,
                error=exc,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            return self._error_result(
                contract=contract,
                request=request,
                normalized_input=normalized_input,
                risk_level=risk_level,
                side_effect=side_effect,
                retryable=retryable,
                error=exc,
            )

    def run_many(self, requests: list[ToolRequest], context: ContextSnapshot) -> list[ToolResult]:
        ordered = sorted(requests, key=lambda request: request.priority)
        results: list[ToolResult] = []
        for request in ordered:
            result = self.run(request, context)
            results.append(result)
            follow_up_request = self._browser_verification_request(context, request, result)
            if follow_up_request is not None:
                results.append(self.run(follow_up_request, context))
        return results

    def _browser_verification_request(
        self,
        context: ContextSnapshot,
        request: ToolRequest,
        result: ToolResult,
    ) -> ToolRequest | None:
        if request.tool_name not in BROWSER_MAJOR_STEP_NAMES or result.status != "success":
            return None
        goal_payload = (
            request.audit_metadata.get("browser_goal")
            or request.input_payload.get("browser_goal")
            or context.metadata.get("browser_goal")
        )
        if not goal_payload:
            return None

        goal = BrowserGoal.model_validate(goal_payload)
        snapshot = self._browser_snapshot_for_step(context, request, result)
        return ToolRequest(
            tool_name="verify_browser_goal",
            purpose=f"Verify whether the browser goal was reached after '{request.tool_name}'.",
            input_payload={
                "goal": goal.model_dump(mode="json"),
                "snapshot": snapshot.model_dump(mode="json"),
            },
            priority=request.priority,
            verification_hint="Confirm the browser goal was reached before the loop continues.",
            expected_observation="Structured browser-goal verification for the last major browser step.",
            audit_metadata={
                "triggered_by": request.tool_name,
                "goal_description": goal.description,
                "phase": "post_browser_step",
            },
        )

    def _browser_snapshot_for_step(
        self,
        context: ContextSnapshot,
        request: ToolRequest,
        result: ToolResult,
    ) -> BrowserStateSnapshot:
        output = result.output
        search_text = self._stringify_browser_results(output.get("results"))
        page_text = str(output.get("text") or request.input_payload.get("text") or search_text or "")
        screenshot_payload = (
            request.audit_metadata.get("screenshot_checkpoint")
            or request.input_payload.get("screenshot_checkpoint")
            or output.get("screenshot_checkpoint")
            or {}
        )
        return BrowserStateSnapshot(
            step_name=request.tool_name,
            page_url=str(
                output.get("url")
                or request.input_payload.get("url")
                or request.audit_metadata.get("page_url")
                or ""
            ),
            page_title=str(output.get("title") or request.audit_metadata.get("page_title") or ""),
            page_text_snapshot=page_text[:1000],
            dom_text_summary=str(
                output.get("summary") or request.audit_metadata.get("dom_text_summary") or ""
            ),
            extracted_text_blocks=[
                text
                for text in (
                    search_text,
                    str(output.get("query") or ""),
                    str(output.get("summary") or ""),
                )
                if text
            ][:4],
            screenshot_checkpoint=screenshot_payload,
            action_kind=str(
                request.audit_metadata.get("action_kind")
                or request.input_payload.get("action_kind")
                or "inspect"
            ),
            action_target=str(
                request.audit_metadata.get("action_target")
                or request.input_payload.get("action_target")
                or output.get("url")
                or ""
            ),
            approval_granted=bool(context.metadata.get("approval_granted")),
            dangerous_action_candidates=[
                str(candidate)
                for candidate in request.audit_metadata.get("dangerous_action_candidates", [])
            ],
            metadata={
                "triggered_by": request.tool_name,
                "tool_status": result.status,
                "browser": output.get("browser", request.input_payload.get("browser", "")),
            },
        )

    def _stringify_browser_results(self, results: Any) -> str:
        if not isinstance(results, list):
            return ""
        parts: list[str] = []
        for item in results[:5]:
            if isinstance(item, dict):
                for key in ("title", "url", "snippet"):
                    value = item.get(key)
                    if value:
                        parts.append(str(value))
            else:
                parts.append(str(item))
        return " ".join(parts)

    def _browser_goal_postcondition_verifier(self) -> ToolVerifier:
        def verify(
            _: ContextSnapshot,
            __: BaseModel,
            output: BaseModel,
        ) -> ToolVerificationResult:
            if getattr(output, "dry_run", False):
                return ToolVerificationResult(
                    status="skipped",
                    summary="Dry run skipped browser-goal verification.",
                    postconditions_met=True,
                    checks=["Browser verification ran in preview mode only."],
                )

            checks = [
                "The browser step should reach its declared goal.",
                "Dangerous browser actions should require confirmation.",
                "Submit-like browser actions should stop before execution unless explicitly allowed.",
            ]
            failures: list[str] = []
            if not bool(getattr(output, "goal_reached", False)):
                failures.append("The browser goal was not reached for the current step.")
            if bool(getattr(output, "requires_confirmation", False)):
                failures.append("A dangerous browser action requires explicit approval.")
            if bool(getattr(output, "stop_before_submit_triggered", False)):
                failures.append("The stop-before-submit guard prevented continuation.")

            if failures:
                return ToolVerificationResult(
                    status="failed",
                    summary=str(getattr(output, "summary", "") or "Browser goal verification did not pass."),
                    postconditions_met=False,
                    checks=checks,
                    failures=failures,
                    details={
                        "status": getattr(output, "status", ""),
                        "matched_indicators": list(getattr(output, "matched_indicators", [])),
                        "missing_indicators": list(getattr(output, "missing_indicators", [])),
                    },
                )

            return ToolVerificationResult(
                status="passed",
                summary=str(getattr(output, "summary", "") or "Browser goal verification passed."),
                postconditions_met=True,
                checks=checks,
                details={
                    "status": getattr(output, "status", ""),
                    "matched_indicators": list(getattr(output, "matched_indicators", [])),
                },
            )

        return verify

    def _invalid_input_result(
        self,
        contract: ToolContract,
        request: ToolRequest,
        risk_level: str,
        side_effect: str,
        error: ValidationError,
    ) -> ToolResult:
        failures = [err["msg"] for err in error.errors()]
        return ToolResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            status="invalid_input",
            output={"message": "Input validation failed.", "errors": error.errors()},
            evidence=[f"Input validation failed for '{request.tool_name}'."],
            risk_level=risk_level,
            side_effect=side_effect,
            retryable=False,
            input_schema=contract.input_model.__name__,
            output_schema=contract.output_model.__name__,
            verification=ToolVerificationResult(
                status="failed",
                summary="Input validation failed.",
                postconditions_met=False,
                failures=failures,
                details={"errors": error.errors()},
            ),
            audit=contract.build_audit_metadata(
                request=request,
                risk_level=risk_level,
                side_effect=side_effect,
                metadata={
                    **request.audit_metadata,
                    "phase": "input_validation",
                    "errors": error.errors(),
                },
            ),
        )

    def _gated_result(
        self,
        *,
        contract: ToolContract,
        request: ToolRequest,
        normalized_input: dict[str, Any],
        risk_level: str,
        side_effect: str,
        retryable: bool,
    ) -> ToolResult:
        return ToolResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            status="gated",
            normalized_input=normalized_input,
            output={
                "message": "Tool call was held behind an approval gate.",
                "side_effect": side_effect,
            },
            evidence=[f"Approval gate triggered for side effect '{side_effect}'."],
            risk_level=risk_level,
            side_effect=side_effect,
            retryable=retryable,
            input_schema=contract.input_model.__name__,
            output_schema=contract.output_model.__name__,
            blocked_reason="confirmation_required",
            verification=ToolVerificationResult(
                status="failed",
                summary="Approval policy gated the tool call.",
                postconditions_met=False,
                failures=["confirmation_required"],
            ),
            audit=contract.build_audit_metadata(
                request=request,
                risk_level=risk_level,
                side_effect=side_effect,
                metadata={**request.audit_metadata, "phase": "approval_gate"},
            ),
        )

    def _dry_run_result(
        self,
        *,
        contract: ToolContract,
        request: ToolRequest,
        validated_input: BaseModel,
        normalized_input: dict[str, Any],
        risk_level: str,
        side_effect: str,
        retryable: bool,
    ) -> ToolResult:
        dry_run_output = contract.build_dry_run_output(request, validated_input)
        return ToolResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            status="dry_run",
            normalized_input=normalized_input,
            output=dry_run_output.model_dump(mode="json"),
            evidence=[getattr(dry_run_output, "preview_action", "") or f"Dry run for {request.tool_name}."],
            risk_level=risk_level,
            side_effect=side_effect,
            retryable=retryable,
            dry_run=True,
            input_schema=contract.input_model.__name__,
            output_schema=contract.output_model.__name__,
            verification=ToolVerificationResult(
                status="skipped",
                summary="Dry run completed without mutating external state.",
                postconditions_met=True,
                checks=["Side effects were skipped by request."],
            ),
            audit=contract.build_audit_metadata(
                request=request,
                risk_level=risk_level,
                side_effect=side_effect,
                metadata={**request.audit_metadata, "phase": "dry_run"},
            ),
        )

    def _success_result(
        self,
        *,
        contract: ToolContract,
        request: ToolRequest,
        normalized_input: dict[str, Any],
        output_payload: dict[str, Any],
        verification: ToolVerificationResult,
        risk_level: str,
        side_effect: str,
        retryable: bool,
        status: str,
    ) -> ToolResult:
        return ToolResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            status=status,
            normalized_input=normalized_input,
            output=output_payload,
            evidence=self._derive_evidence(output_payload),
            risk_level=risk_level,
            side_effect=side_effect,
            retryable=retryable,
            input_schema=contract.input_model.__name__,
            output_schema=contract.output_model.__name__,
            verification=verification,
            audit=contract.build_audit_metadata(
                request=request,
                risk_level=risk_level,
                side_effect=side_effect,
                metadata={**request.audit_metadata, "phase": "execution"},
            ),
        )

    def _invalid_output_result(
        self,
        *,
        contract: ToolContract,
        request: ToolRequest,
        normalized_input: dict[str, Any],
        risk_level: str,
        side_effect: str,
        retryable: bool,
        error: ValidationError,
    ) -> ToolResult:
        failures = [err["msg"] for err in error.errors()]
        return ToolResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            status="invalid_output",
            normalized_input=normalized_input,
            output={"message": "Output validation failed.", "errors": error.errors()},
            evidence=[f"Output validation failed for '{request.tool_name}'."],
            risk_level=risk_level,
            side_effect=side_effect,
            retryable=retryable,
            input_schema=contract.input_model.__name__,
            output_schema=contract.output_model.__name__,
            verification=ToolVerificationResult(
                status="failed",
                summary="Output validation failed.",
                postconditions_met=False,
                failures=failures,
                details={"errors": error.errors()},
            ),
            audit=contract.build_audit_metadata(
                request=request,
                risk_level=risk_level,
                side_effect=side_effect,
                metadata={
                    **request.audit_metadata,
                    "phase": "output_validation",
                    "errors": error.errors(),
                },
            ),
        )

    def _error_result(
        self,
        *,
        contract: ToolContract,
        request: ToolRequest,
        normalized_input: dict[str, Any],
        risk_level: str,
        side_effect: str,
        retryable: bool,
        error: Exception,
    ) -> ToolResult:
        return ToolResult(
            call_id=request.call_id,
            tool_name=request.tool_name,
            status="error",
            normalized_input=normalized_input,
            output={"message": str(error)},
            evidence=[f"{request.tool_name} raised an exception: {error}"],
            risk_level=risk_level,
            side_effect=side_effect,
            retryable=retryable,
            input_schema=contract.input_model.__name__,
            output_schema=contract.output_model.__name__,
            verification=ToolVerificationResult(
                status="failed",
                summary="Tool execution raised an exception.",
                postconditions_met=False,
                failures=[str(error)],
            ),
            audit=contract.build_audit_metadata(
                request=request,
                risk_level=risk_level,
                side_effect=side_effect,
                metadata={**request.audit_metadata, "phase": "exception"},
            ),
        )

    def _should_gate(self, risk_level: str, side_effect: str) -> bool:
        if side_effect in {"delete", "deploy", "send", "purchase", "shutdown"}:
            return True
        return risk_level == "high" and side_effect != "none"

    def _derive_evidence(self, output: dict[str, Any]) -> list[str]:
        evidence = []
        for key, value in output.items():
            if isinstance(value, list):
                snippet = ", ".join(str(item) for item in value[:3])
                evidence.append(f"{key}: {snippet}")
            elif isinstance(value, dict):
                evidence.append(f"{key}: {', '.join(value.keys())}")
            else:
                evidence.append(f"{key}: {value}")
        return evidence[:6]

    def _session_history(self, context: ContextSnapshot, _: dict[str, Any]) -> dict[str, Any]:
        history = self._memory_system.get_history(context.user_id, context.session_id)
        return {
            "turn_count": len(history),
            "summary": self._memory_system.summarize_history(context.user_id, context.session_id),
        }

    def _semantic_memory(self, context: ContextSnapshot, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "count": len(context.memory.semantic),
            "entries": [entry.content for entry in context.memory.semantic[-4:]],
            "retrieved": [entry.content for entry in context.memory.retrieved[:4]],
        }

    def _vector_memory(self, context: ContextSnapshot, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "count": len(context.memory.vector),
            "entries": [entry.content for entry in context.memory.vector[-4:]],
            "retrieved": [entry.content for entry in context.memory.retrieved[:4]],
        }

    def _working_memory(self, context: ContextSnapshot, _: dict[str, Any]) -> dict[str, Any]:
        wm = context.memory.working_memory
        return {
            "objective": wm.objective,
            "distilled_context": wm.distilled_context,
            "assumptions": wm.assumptions,
            "constraints": wm.constraints,
            "open_questions": wm.open_questions,
            "retrieved_facts": wm.retrieved_facts,
            "checkpoint": wm.plan_checkpoint,
        }

    def _goal_stack(self, context: ContextSnapshot, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "active_goals": context.active_goals,
            "goal_stack": context.memory.goal_stack,
            "system_goals": context.system_goals,
        }

    def _capability_map(self, context: ContextSnapshot, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "interfaces": ["voice", "text", "api", "ui"],
            "specialized_agents": [descriptor.key for descriptor in get_agent_descriptors()],
            "tool_categories": sorted({descriptor.category for descriptor in self._catalog.values()}),
            "requested_capabilities": context.requested_capabilities,
            "execution_layer": ["memory retrieval", "tool adapters", "verification checks", "safety gates"],
        }

    def _execution_catalog(self, _: ContextSnapshot, __: dict[str, Any]) -> dict[str, Any]:
        return {
            "connectors": {name: "available" for name in sorted(self._catalog)},
            "governance": {
                "approval_gates": "enabled",
                "evidence_capture": "enabled",
            },
        }

    def _skill_manifest(self, _: ContextSnapshot, __: dict[str, Any]) -> dict[str, Any]:
        return {"skills": [skill.name for skill in self._skill_library.catalog()]}

    def _plan_analyzer(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        objective = str(payload.get("objective") or context.latest_message)
        return {
            "objective": objective,
            "step_count": int(payload.get("step_count", 0)),
            "success_criteria": list(payload.get("success_criteria", [])),
            "verification_focus": list(payload.get("verification_focus", [])),
            "constraints": list(context.constraints),
            "summary": f"Plan analysis prepared for '{objective[:80]}'.",
        }

    def _verification_harness(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        checks = payload.get("checks") or context.memory.open_loops or context.system_goals[:3]
        return {
            "check_count": len(checks),
            "checks": checks,
            "mode": payload.get("mode", "standard"),
            "risk_level": context.signals.risk_level,
            "summary": "Verification checks were prepared for the current step.",
        }

    def _risk_monitor(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        resolved_risk = payload.get("risk_level", context.signals.risk_level) or context.signals.risk_level
        return {
            "risk_level": resolved_risk,
            "requested_action": payload.get("action", "inspect"),
            "requires_confirmation": payload.get("requires_confirmation", False),
            "constraints": context.constraints,
            "summary": f"Risk posture is currently classified as '{resolved_risk}'.",
        }

    def _api_dispatcher(self, _: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "connector": "api_dispatcher",
            "status": "available",
            "requested_action": payload.get("action", "inspect"),
            "summary": "API integration surface is reachable.",
        }

    def _browser_adapter(self, _: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "connector": "browser_adapter",
            "status": "available",
            "target": payload.get("target", "browser-session"),
            "summary": "Browser automation surface is reachable.",
        }

    def _os_adapter(self, _: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "connector": "os_adapter",
            "status": "available",
            "operation": payload.get("operation", "workspace-inspection"),
            "summary": "Operating system inspection surface is reachable.",
        }

    def _database_adapter(self, _: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "connector": "database_adapter",
            "status": "available",
            "database": payload.get("database", "configured-store"),
            "summary": "Database access surface is reachable.",
        }

    def _github_adapter(self, _: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "connector": "github_adapter",
            "status": "available",
            "repository_action": payload.get("repository_action", "inspect"),
            "summary": "Repository integration surface is reachable.",
        }

    def _calendar_adapter(self, _: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "connector": "calendar_adapter",
            "status": "available",
            "calendar_action": payload.get("calendar_action", "review"),
            "summary": "Calendar coordination surface is reachable.",
        }

    def _document_adapter(self, _: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "connector": "document_adapter",
            "status": "available",
            "document_action": payload.get("document_action", "summarize"),
            "summary": "Document ingestion surface is reachable.",
        }

    def _send_email_draft(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        subject = payload.get("subject", f"Draft regarding: {context.latest_message[:50]}")
        body = payload.get(
            "body",
            f"Hello,\n\nThis draft was prepared for the session objective: {context.latest_message}\n",
        )
        return {
            "draft_id": f"draft-{context.session_id}-{len(subject)}",
            "to": payload.get("to", []),
            "subject": subject,
            "body_preview": body[:200],
            "approval_required": True,
            "status": "draft_saved",
            "summary": "A draft email artifact was prepared without sending it.",
        }

    def _search_web(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload.get("query", context.latest_message)
        return {
            "query": query,
            "results": [
                {
                    "title": f"Research lead for {query[:40]}",
                    "url": "https://example.com/research-lead",
                    "snippet": f"Synthesized starter result for query '{query[:60]}'.",
                },
                {
                    "title": "System design notes",
                    "url": "https://example.com/system-design",
                    "snippet": "Structured evidence placeholder for web search integration.",
                },
            ],
            "source": "stubbed_web_search",
            "summary": f"Prepared structured research leads for '{query[:80]}'.",
        }

    def _browser_search(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload.get("query", context.latest_message)
        return {
            "query": query,
            "browser_session": payload.get("session", context.session_id),
            "results": [
                {
                    "title": f"Browser search result for {query[:40]}",
                    "url": "https://example.com/browser-result",
                },
            ],
            "status": "browser_search_prepared",
            "summary": f"Prepared a browser-first search flow for '{query[:80]}'.",
        }

    def _verify_browser_goal(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        goal = BrowserGoal.model_validate(payload.get("goal", {}))
        snapshot_payload = dict(payload.get("snapshot", {}))
        snapshot_payload.setdefault("approval_granted", bool(context.metadata.get("approval_granted")))
        snapshot = BrowserStateSnapshot.model_validate(snapshot_payload)
        verification = self._browser_goal_verifier.verify(goal, snapshot)
        return {
            "status": verification.status,
            "summary": verification.summary,
            "goal_reached": verification.goal_reached,
            "requires_confirmation": verification.requires_confirmation,
            "dangerous_action_detected": verification.dangerous_action_detected,
            "stop_before_submit_triggered": verification.stop_before_submit_triggered,
            "matched_indicators": verification.matched_indicators,
            "missing_indicators": verification.missing_indicators,
            "verification": verification.model_dump(mode="json"),
            "snapshot": snapshot.model_dump(mode="json"),
        }

    def _save_memory(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        content = payload.get("content", context.latest_message)
        tags = payload.get("tags", ["manual_memory"])
        record = MemoryRecord(
            memory_type="semantic",
            content=content,
            source=payload.get("source", "save_memory_tool"),
            salience=float(payload.get("salience", 0.8)),
            tags=[str(tag) for tag in tags],
            attributes={"tool": "save_memory"},
        )
        self._memory_system.append_memory_record(context.user_id, context.session_id, record)
        return {
            "saved": True,
            "memory_id": record.memory_id,
            "content": record.content,
            "tags": record.tags,
            "summary": "Persisted a semantic memory record.",
        }

    def _load_recent_memory(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        limit = int(payload.get("limit", 4))
        snapshot = self._memory_system.build_snapshot(
            context.user_id,
            context.session_id,
            active_goals=context.active_goals,
            query=context.latest_message,
            constraints=context.constraints,
        )
        return {
            "history": [turn.content for turn in snapshot.episodic[-limit:]],
            "semantic": [record.content for record in snapshot.semantic[-limit:]],
            "retrieved": [record.content for record in snapshot.retrieved[:limit]],
            "summary": "Loaded recent episodic and semantic memory slices.",
        }

    def _summarize_file(self, _: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(payload["path"]))
        if not path.exists() or not path.is_file():
            return {
                "status": "not_found",
                "path": str(path),
                "summary": f"File '{path}' does not exist.",
                "line_count": 0,
            }
        if path.stat().st_size > MAX_SUMMARY_FILE_BYTES:
            return {
                "status": "too_large",
                "path": str(path),
                "summary": (
                    f"File '{path}' is larger than the safe summary limit of "
                    f"{MAX_SUMMARY_FILE_BYTES} bytes."
                ),
                "line_count": 0,
            }
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        summary = " ".join(lines[:5])[:400]
        return {
            "status": "summarized",
            "path": str(path),
            "summary": summary or "The file was empty.",
            "line_count": len(lines),
        }

    def _generate_code(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        language = payload.get("language", "python")
        objective = payload.get("objective", context.latest_message)
        if language.lower() == "python":
            generated = (
                'def solve_task() -> str:\n'
                f'    """Starter implementation for: {objective[:60]}."""\n'
                '    return "TODO: complete implementation"\n'
            )
        else:
            generated = f"// Starter code for: {objective[:60]}"
        return {
            "language": language,
            "objective": objective,
            "generated_code": generated,
            "summary": f"Generated a starter {language} implementation scaffold.",
        }

    def _open_page(self, _: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        url = payload.get("url", "https://example.com")
        return {
            "url": url,
            "status": "opened_placeholder",
            "browser": payload.get("browser", "playwright"),
            "summary": f"Prepared a browser open-page action for '{url}'.",
        }

    def _extract_page_text(self, _: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        html = payload.get("html") or payload.get("text") or ""
        condensed = " ".join(str(html).split())
        return {
            "text": condensed[:500],
            "length": len(condensed),
            "status": "extracted",
            "summary": "Extracted text from the supplied page payload.",
        }

    def _create_task_record(self, context: ContextSnapshot, payload: dict[str, Any]) -> dict[str, Any]:
        title = payload.get("title", context.latest_message[:80])
        task_status = payload.get("status", "planned")
        record = MemoryRecord(
            memory_type="semantic",
            content=f"Task record: {title} [{task_status}]",
            source="create_task_record",
            salience=0.7,
            tags=["task", task_status],
            attributes={"title": title, "status": task_status},
        )
        self._memory_system.append_memory_record(context.user_id, context.session_id, record)
        return {
            "task_id": record.memory_id,
            "title": title,
            "task_status": task_status,
            "stored": True,
            "summary": "Stored a structured task record in shared memory.",
        }
