from src.tools.catalog import tool_names_for_category


BROWSER_TOOL_NAMES = tool_names_for_category("browser") + ["browser_search", "browser_adapter"]


def describe_browser_tools() -> dict[str, list[str]]:
    return {"browser_tools": BROWSER_TOOL_NAMES}
