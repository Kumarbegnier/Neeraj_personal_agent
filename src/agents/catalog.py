from __future__ import annotations

from src.schemas.catalog import AgentCatalog, AgentDescriptor


AGENT_DESCRIPTORS = [
    AgentDescriptor(
        key="communication",
        display_name="Communication Agent",
        role="communication",
        description="Handles messaging, replies, drafts, and outbound communication workflows.",
        responsibilities=[
            "Draft approval-aware communication",
            "Preserve recent conversation context",
            "Keep outbound actions reviewable before sending",
        ],
        default_tools=["session_history", "working_memory", "risk_monitor", "send_email_draft"],
    ),
    AgentDescriptor(
        key="coding",
        display_name="Coding Agent",
        role="coding",
        description="Handles architecture, implementation, debugging, repository inspection, and execution design.",
        responsibilities=[
            "Plan and inspect implementation work",
            "Generate and verify code-oriented outputs",
            "Use memory and verification to adapt execution",
        ],
        default_tools=["working_memory", "plan_analyzer", "semantic_memory", "generate_code", "github_adapter"],
    ),
    AgentDescriptor(
        key="research",
        display_name="Research Agent",
        role="research",
        description="Handles search, evidence collection, synthesis, and structured summary generation.",
        responsibilities=[
            "Collect evidence from retrieval and search surfaces",
            "Ground claims in retrieved context",
            "Attach verification before synthesis",
        ],
        default_tools=["working_memory", "vector_memory", "search_web", "load_recent_memory", "verification_harness"],
    ),
    AgentDescriptor(
        key="web",
        display_name="Browser Agent",
        role="browser",
        description="Handles browser-first or page interaction workflows.",
        responsibilities=[
            "Prepare browser-first evidence collection",
            "Open pages and extract structured text",
            "Verify each major browser step before continuing",
            "Keep browsing inside the shared execution loop",
        ],
        default_tools=[
            "working_memory",
            "browser_search",
            "open_page",
            "extract_page_text",
            "verify_browser_goal",
            "browser_adapter",
        ],
    ),
    AgentDescriptor(
        key="file",
        display_name="File Agent",
        role="file",
        description="Handles document, PDF, and local file analysis workflows.",
        responsibilities=[
            "Inspect local files safely",
            "Summarize documents into evidence",
            "Ground file analysis in memory and plan context",
        ],
        default_tools=["working_memory", "summarize_file", "document_adapter", "vector_memory"],
    ),
    AgentDescriptor(
        key="task",
        display_name="Task Agent",
        role="task",
        description="Handles schedules, reminders, checklists, and task coordination.",
        responsibilities=[
            "Track tasks and planning checkpoints",
            "Create task records for durable follow-up",
            "Preserve coordination context",
        ],
        default_tools=["working_memory", "create_task_record", "calendar_adapter", "risk_monitor"],
    ),
    AgentDescriptor(
        key="general",
        display_name="General Agent",
        role="general",
        description="Acts as the fallback specialist when no narrower lane is selected.",
        responsibilities=[
            "Handle mixed or ambiguous requests",
            "Fallback when specialist confidence is low",
            "Keep the loop moving when no narrower branch dominates",
        ],
        default_tools=["working_memory", "capability_map", "verification_harness"],
    ),
]


def get_agent_descriptors() -> list[AgentDescriptor]:
    return [descriptor.model_copy(deep=True) for descriptor in AGENT_DESCRIPTORS]


def get_agent_descriptor(key: str) -> AgentDescriptor:
    for descriptor in AGENT_DESCRIPTORS:
        if descriptor.key == key:
            return descriptor.model_copy(deep=True)
    raise KeyError(f"Unknown agent descriptor: {key}")


def get_agent_catalog() -> AgentCatalog:
    return AgentCatalog(agents=get_agent_descriptors())
