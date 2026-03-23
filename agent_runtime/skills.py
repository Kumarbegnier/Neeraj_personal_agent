from __future__ import annotations

from .models import ContextSnapshot, SkillDescriptor, UserRequest


class SkillLibrary:
    def __init__(self) -> None:
        self._skills = [
            SkillDescriptor(
                name="communication-workflow",
                description="Draft, revise, and confirm outbound messages before sending.",
                triggers=["email", "message", "reply", "draft"],
                tools=["api_dispatcher", "risk_monitor", "skill_manifest"],
            ),
            SkillDescriptor(
                name="coding-debug-loop",
                description="Inspect, edit, verify, and reflect on implementation work.",
                triggers=["code", "debug", "fix", "implement", "build"],
                tools=["os_adapter", "semantic_memory", "verification_harness"],
            ),
            SkillDescriptor(
                name="system-architecture-loop",
                description="Design modular cognitive architectures with planning, memory, verification, and safety.",
                triggers=["architecture", "orchestrator", "agentic", "system", "cognitive"],
                tools=["working_memory", "plan_analyzer", "verification_harness", "risk_monitor"],
            ),
            SkillDescriptor(
                name="research-brief",
                description="Collect evidence, summarize findings, and preserve references.",
                triggers=["research", "search", "summarize", "compare"],
                tools=["browser_adapter", "document_adapter", "vector_memory"],
            ),
            SkillDescriptor(
                name="document-review",
                description="Open files, extract content, and organize document insights.",
                triggers=["file", "pdf", "document", "report"],
                tools=["document_adapter", "vector_memory", "working_memory"],
            ),
            SkillDescriptor(
                name="task-coordinator",
                description="Manage schedules, reminders, and execution checklists.",
                triggers=["schedule", "calendar", "task", "reminder"],
                tools=["calendar_adapter", "risk_monitor", "skill_manifest"],
            ),
        ]

    def recommend(
        self,
        request: UserRequest,
        context: ContextSnapshot,
        agent_name: str,
    ) -> list[SkillDescriptor]:
        lowered = request.message.lower()
        selected: list[SkillDescriptor] = []

        for skill in self._skills:
            if any(trigger in lowered for trigger in skill.triggers):
                selected.append(skill)

        agent_defaults = {
            "communication": "communication-workflow",
            "coding": "coding-debug-loop",
            "research": "research-brief",
            "file": "document-review",
            "task": "task-coordinator",
        }
        default_skill = agent_defaults.get(agent_name)
        if default_skill and all(skill.name != default_skill for skill in selected):
            selected.extend(skill for skill in self._skills if skill.name == default_skill)

        if "design_system" in context.active_goals and all(
            skill.name != "system-architecture-loop" for skill in selected
        ):
            selected.extend(skill for skill in self._skills if skill.name == "system-architecture-loop")

        if not selected and context.active_goals:
            goal_to_skill = {
                "ship_code": "coding-debug-loop",
                "research": "research-brief",
                "inspect_files": "document-review",
                "manage_tasks": "task-coordinator",
                "communicate": "communication-workflow",
                "design_system": "system-architecture-loop",
            }
            for goal in context.active_goals:
                skill_name = goal_to_skill.get(goal)
                if skill_name:
                    selected.extend(skill for skill in self._skills if skill.name == skill_name)

        deduped: list[SkillDescriptor] = []
        seen: set[str] = set()
        for skill in selected:
            if skill.name not in seen:
                deduped.append(skill)
                seen.add(skill.name)
        return deduped

    def catalog(self) -> list[SkillDescriptor]:
        return list(self._skills)
