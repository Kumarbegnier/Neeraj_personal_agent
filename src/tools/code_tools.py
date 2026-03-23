from src.tools.catalog import tool_names_for_category


CODE_TOOL_NAMES = tool_names_for_category("coding") + ["github_adapter"]


def describe_code_tools() -> dict[str, list[str]]:
    return {"code_tools": CODE_TOOL_NAMES}
