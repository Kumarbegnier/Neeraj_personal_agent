from src.tools.catalog import tool_names_for_category


FILE_TOOL_NAMES = tool_names_for_category("file")


def describe_file_tools() -> dict[str, list[str]]:
    return {"file_tools": FILE_TOOL_NAMES}
