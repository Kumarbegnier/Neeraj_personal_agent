from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.agents import get_agent_descriptors
from src.schemas.catalog import ToolDescriptor
from src.tools.catalog import get_tool_catalog, get_tool_descriptor_map

from .memory import MemorySystem
from .models import ContextSnapshot, MemoryRecord, ToolRequest, ToolResult
from .skills import SkillLibrary


ToolHandler = Callable[[ContextSnapshot, dict], dict]
MAX_SUMMARY_FILE_BYTES = 1_000_000


class ToolLayer:
    def __init__(self, memory_system: MemorySystem, skill_library: SkillLibrary) -> None:
        self._memory_system = memory_system
        self._skill_library = skill_library
        self._catalog = get_tool_descriptor_map()
        self._tools = self._build_registry()

    def _build_registry(self) -> dict[str, ToolHandler]:
        handlers = {
            "session_history": self._session_history,
            "semantic_memory": self._semantic_memory,
            "vector_memory": self._vector_memory,
            "working_memory": self._working_memory,
            "goal_stack": self._goal_stack,
            "capability_map": self._capability_map,
            "execution_catalog": self._execution_catalog,
            "skill_manifest": self._skill_manifest,
            "plan_analyzer": self._plan_analyzer,
            "verification_harness": self._verification_harness,
            "risk_monitor": self._risk_monitor,
            "api_dispatcher": self._api_dispatcher,
            "browser_adapter": self._browser_adapter,
            "os_adapter": self._os_adapter,
            "database_adapter": self._database_adapter,
            "github_adapter": self._github_adapter,
            "calendar_adapter": self._calendar_adapter,
            "document_adapter": self._document_adapter,
            "send_email_draft": self._send_email_draft,
            "search_web": self._search_web,
            "browser_search": self._browser_search,
            "save_memory": self._save_memory,
            "load_recent_memory": self._load_recent_memory,
            "summarize_file": self._summarize_file,
            "generate_code": self._generate_code,
            "open_page": self._open_page,
            "extract_page_text": self._extract_page_text,
            "create_task_record": self._create_task_record,
        }
        missing_handlers = set(self._catalog) - set(handlers)
        if missing_handlers:
            raise ValueError(f"Missing tool handlers for: {sorted(missing_handlers)}")
        return handlers

    def catalog(self) -> list[ToolDescriptor]:
        return get_tool_catalog().tools

    def _descriptor_for(self, tool_name: str) -> ToolDescriptor | None:
        descriptor = self._catalog.get(tool_name)
        return descriptor.model_copy(deep=True) if descriptor is not None else None

    def run(self, request: ToolRequest, context: ContextSnapshot) -> ToolResult:
        handler = self._tools.get(request.tool_name)
        descriptor = self._descriptor_for(request.tool_name)
        approval_granted = bool(context.metadata.get("approval_granted"))
        if handler is None:
            return ToolResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                status="unavailable",
                output={"message": f"Tool '{request.tool_name}' is not registered."},
                evidence=[f"Tool '{request.tool_name}' is absent from the registry."],
                risk_level=descriptor.risk_level if descriptor is not None else request.risk_level,
            )

        risk_level = descriptor.risk_level if descriptor is not None and request.risk_level == "low" else request.risk_level
        side_effect = descriptor.side_effect if descriptor is not None and request.side_effect == "none" else request.side_effect
        requires_confirmation = request.requires_confirmation or self._should_gate(risk_level, side_effect)
        if not approval_granted and requires_confirmation:
            return ToolResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                status="gated",
                output={
                    "message": "Tool call was held behind an approval gate.",
                    "side_effect": side_effect,
                },
                evidence=[f"Approval gate triggered for side effect '{side_effect}'."],
                risk_level=risk_level,
                blocked_reason="confirmation_required",
            )

        try:
            output = handler(context, request.input_payload)
            evidence = self._derive_evidence(output)
            return ToolResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                status="success",
                output=output,
                evidence=evidence,
                risk_level=risk_level,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            return ToolResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                status="error",
                output={"message": str(exc)},
                evidence=[f"{request.tool_name} raised an exception: {exc}"],
                risk_level=risk_level,
            )

    def run_many(self, requests: list[ToolRequest], context: ContextSnapshot) -> list[ToolResult]:
        ordered = sorted(requests, key=lambda request: request.priority)
        return [self.run(request, context) for request in ordered]

    def _should_gate(self, risk_level: str, side_effect: str) -> bool:
        if side_effect in {"delete", "deploy", "send", "purchase", "shutdown"}:
            return True
        return risk_level == "high" and side_effect != "none"

    def _derive_evidence(self, output: dict) -> list[str]:
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

    def _session_history(self, context: ContextSnapshot, _: dict) -> dict:
        history = self._memory_system.get_history(context.user_id, context.session_id)
        return {
            "turn_count": len(history),
            "summary": self._memory_system.summarize_history(context.user_id, context.session_id),
        }

    def _semantic_memory(self, context: ContextSnapshot, _: dict) -> dict:
        return {
            "count": len(context.memory.semantic),
            "entries": [entry.content for entry in context.memory.semantic[-4:]],
            "retrieved": [entry.content for entry in context.memory.retrieved[:4]],
        }

    def _vector_memory(self, context: ContextSnapshot, _: dict) -> dict:
        return {
            "count": len(context.memory.vector),
            "entries": [entry.content for entry in context.memory.vector[-4:]],
            "retrieved": [entry.content for entry in context.memory.retrieved[:4]],
        }

    def _working_memory(self, context: ContextSnapshot, _: dict) -> dict:
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

    def _goal_stack(self, context: ContextSnapshot, _: dict) -> dict:
        return {
            "active_goals": context.active_goals,
            "goal_stack": context.memory.goal_stack,
            "system_goals": context.system_goals,
        }

    def _capability_map(self, context: ContextSnapshot, _: dict) -> dict:
        return {
            "interfaces": ["voice", "text", "api", "ui"],
            "specialized_agents": [descriptor.key for descriptor in get_agent_descriptors()],
            "tool_categories": sorted({descriptor.category for descriptor in self._catalog.values()}),
            "requested_capabilities": context.requested_capabilities,
            "execution_layer": ["memory retrieval", "tool adapters", "verification checks", "safety gates"],
        }

    def _execution_catalog(self, _: ContextSnapshot, __: dict) -> dict:
        return {
            "connectors": {name: "available" for name in sorted(self._catalog)},
            "governance": {
                "approval_gates": "enabled",
                "evidence_capture": "enabled",
            },
        }

    def _skill_manifest(self, _: ContextSnapshot, __: dict) -> dict:
        return {
            "skills": [skill.name for skill in self._skill_library.catalog()],
        }

    def _plan_analyzer(self, context: ContextSnapshot, payload: dict) -> dict:
        return {
            "objective": payload.get("objective", context.latest_message),
            "step_count": payload.get("step_count", 0),
            "success_criteria": payload.get("success_criteria", []),
            "verification_focus": payload.get("verification_focus", []),
            "constraints": context.constraints,
        }

    def _verification_harness(self, context: ContextSnapshot, payload: dict) -> dict:
        checks = payload.get("checks") or context.memory.open_loops or context.system_goals[:3]
        return {
            "check_count": len(checks),
            "checks": checks,
            "mode": payload.get("mode", "standard"),
            "risk_level": context.signals.risk_level,
        }

    def _risk_monitor(self, context: ContextSnapshot, payload: dict) -> dict:
        return {
            "risk_level": payload.get("risk_level", context.signals.risk_level),
            "requested_action": payload.get("action", "inspect"),
            "requires_confirmation": payload.get("requires_confirmation", False),
            "constraints": context.constraints,
        }

    def _api_dispatcher(self, _: ContextSnapshot, payload: dict) -> dict:
        return {
            "connector": "api_dispatcher",
            "status": "available",
            "requested_action": payload.get("action", "inspect"),
        }

    def _browser_adapter(self, _: ContextSnapshot, payload: dict) -> dict:
        return {
            "connector": "browser_adapter",
            "status": "available",
            "target": payload.get("target", "browser-session"),
        }

    def _os_adapter(self, _: ContextSnapshot, payload: dict) -> dict:
        return {
            "connector": "os_adapter",
            "status": "available",
            "operation": payload.get("operation", "workspace-inspection"),
        }

    def _database_adapter(self, _: ContextSnapshot, payload: dict) -> dict:
        return {
            "connector": "database_adapter",
            "status": "available",
            "database": payload.get("database", "configured-store"),
        }

    def _github_adapter(self, _: ContextSnapshot, payload: dict) -> dict:
        return {
            "connector": "github_adapter",
            "status": "available",
            "repository_action": payload.get("repository_action", "inspect"),
        }

    def _calendar_adapter(self, _: ContextSnapshot, payload: dict) -> dict:
        return {
            "connector": "calendar_adapter",
            "status": "available",
            "calendar_action": payload.get("calendar_action", "review"),
        }

    def _document_adapter(self, _: ContextSnapshot, payload: dict) -> dict:
        return {
            "connector": "document_adapter",
            "status": "available",
            "document_action": payload.get("document_action", "summarize"),
        }

    def _send_email_draft(self, context: ContextSnapshot, payload: dict) -> dict:
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
        }

    def _search_web(self, context: ContextSnapshot, payload: dict) -> dict:
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
        }

    def _browser_search(self, context: ContextSnapshot, payload: dict) -> dict:
        query = payload.get("query", context.latest_message)
        return {
            "query": query,
            "browser_session": payload.get("session", context.session_id),
            "results": [
                {"title": f"Browser search result for {query[:40]}", "url": "https://example.com/browser-result"},
            ],
            "status": "browser_search_prepared",
        }

    def _save_memory(self, context: ContextSnapshot, payload: dict) -> dict:
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
        }

    def _load_recent_memory(self, context: ContextSnapshot, payload: dict) -> dict:
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
        }

    def _summarize_file(self, _: ContextSnapshot, payload: dict) -> dict:
        raw_path = payload.get("path")
        if not raw_path:
            return {
                "status": "missing_path",
                "summary": "No file path was provided.",
            }
        path = Path(str(raw_path))
        if not path.exists() or not path.is_file():
            return {
                "status": "not_found",
                "summary": f"File '{path}' does not exist.",
            }
        if path.stat().st_size > MAX_SUMMARY_FILE_BYTES:
            return {
                "status": "too_large",
                "summary": (
                    f"File '{path}' is larger than the safe summary limit of "
                    f"{MAX_SUMMARY_FILE_BYTES} bytes."
                ),
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

    def _generate_code(self, context: ContextSnapshot, payload: dict) -> dict:
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
        }

    def _open_page(self, _: ContextSnapshot, payload: dict) -> dict:
        url = payload.get("url", "https://example.com")
        return {
            "url": url,
            "status": "opened_placeholder",
            "browser": payload.get("browser", "playwright"),
        }

    def _extract_page_text(self, _: ContextSnapshot, payload: dict) -> dict:
        html = payload.get("html") or payload.get("text") or ""
        condensed = " ".join(str(html).split())
        return {
            "text": condensed[:500],
            "length": len(condensed),
            "status": "extracted",
        }

    def _create_task_record(self, context: ContextSnapshot, payload: dict) -> dict:
        title = payload.get("title", context.latest_message[:80])
        status = payload.get("status", "planned")
        record = MemoryRecord(
            memory_type="semantic",
            content=f"Task record: {title} [{status}]",
            source="create_task_record",
            salience=0.7,
            tags=["task", status],
            attributes={"title": title, "status": status},
        )
        self._memory_system.append_memory_record(context.user_id, context.session_id, record)
        return {
            "task_id": record.memory_id,
            "title": title,
            "status": status,
            "stored": True,
        }
