from __future__ import annotations

from src.schemas.catalog import ToolCatalog, ToolDescriptor


TOOL_DESCRIPTORS = [
    ToolDescriptor(name="session_history", category="memory", description="Load recent session history and summarize it."),
    ToolDescriptor(name="semantic_memory", category="memory", description="Retrieve semantic memory from the shared memory system."),
    ToolDescriptor(name="vector_memory", category="memory", description="Retrieve vector-style memory fragments for evidence grounding."),
    ToolDescriptor(name="working_memory", category="memory", description="Expose the current working memory summary and constraints."),
    ToolDescriptor(name="goal_stack", category="memory", description="Inspect active goals and the session goal stack."),
    ToolDescriptor(name="save_memory", category="memory", description="Persist a semantic memory record into shared memory."),
    ToolDescriptor(name="load_recent_memory", category="memory", description="Load recent episodic, semantic, and retrieved memory."),
    ToolDescriptor(name="capability_map", category="planning", description="Describe the current orchestration and execution surfaces."),
    ToolDescriptor(name="execution_catalog", category="planning", description="Enumerate currently available connectors and governance surfaces."),
    ToolDescriptor(name="skill_manifest", category="planning", description="List reusable skills available to specialists."),
    ToolDescriptor(name="plan_analyzer", category="planning", description="Inspect plan structure, constraints, and verification focus."),
    ToolDescriptor(name="verification_harness", category="verification", description="Prepare structured verification checks for current claims."),
    ToolDescriptor(name="risk_monitor", category="safety", description="Report risk posture and whether confirmation may be required.", risk_level="medium"),
    ToolDescriptor(name="api_dispatcher", category="integration", description="Represent the API integration surface available to the runtime."),
    ToolDescriptor(name="browser_adapter", category="integration", description="Represent the browser automation surface available to the runtime.", risk_level="medium"),
    ToolDescriptor(name="os_adapter", category="integration", description="Represent local operating system inspection capabilities.", risk_level="medium"),
    ToolDescriptor(name="database_adapter", category="integration", description="Represent configured database access surfaces.", risk_level="medium"),
    ToolDescriptor(name="github_adapter", category="integration", description="Represent repository inspection and code-host workflows.", risk_level="medium"),
    ToolDescriptor(name="calendar_adapter", category="integration", description="Represent calendar and schedule coordination workflows.", risk_level="medium"),
    ToolDescriptor(name="document_adapter", category="integration", description="Represent document ingestion and extraction workflows."),
    ToolDescriptor(name="send_email_draft", category="communication", description="Create a safe draft outbound message artifact.", risk_level="medium", side_effect="draft"),
    ToolDescriptor(name="search_web", category="research", description="Return structured research leads for a query."),
    ToolDescriptor(name="browser_search", category="research", description="Prepare a browser-first evidence collection flow.", risk_level="medium"),
    ToolDescriptor(name="summarize_file", category="file", description="Read and summarize a local file."),
    ToolDescriptor(name="generate_code", category="coding", description="Generate starter code or implementation sketches."),
    ToolDescriptor(name="open_page", category="browser", description="Open a page using a browser abstraction.", risk_level="medium"),
    ToolDescriptor(name="extract_page_text", category="browser", description="Extract textual content from a page or supplied markup."),
    ToolDescriptor(name="create_task_record", category="task", description="Store a structured task record in shared memory."),
]


def get_tool_descriptors() -> list[ToolDescriptor]:
    return [descriptor.model_copy(deep=True) for descriptor in TOOL_DESCRIPTORS]


def get_tool_descriptor_map() -> dict[str, ToolDescriptor]:
    return {descriptor.name: descriptor.model_copy(deep=True) for descriptor in TOOL_DESCRIPTORS}


def get_tool_catalog() -> ToolCatalog:
    return ToolCatalog(tools=get_tool_descriptors())


def tool_names_for_category(category: str) -> list[str]:
    return [descriptor.name for descriptor in TOOL_DESCRIPTORS if descriptor.category == category]
