from src.tools.catalog import tool_names_for_category


SEARCH_TOOL_NAMES = tool_names_for_category("research") + ["load_recent_memory"]


def describe_search_tools() -> dict[str, list[str]]:
    return {"search_tools": SEARCH_TOOL_NAMES}
