from src.tools.catalog import tool_names_for_category


EMAIL_TOOL_NAMES = tool_names_for_category("communication")


def describe_email_tools() -> dict[str, list[str]]:
    return {"email_tools": EMAIL_TOOL_NAMES}
