from src.tools.catalog import tool_names_for_category


MEMORY_TOOL_NAMES = tool_names_for_category("memory")


def describe_memory_tools() -> dict[str, list[str]]:
    return {"memory_tools": MEMORY_TOOL_NAMES}
