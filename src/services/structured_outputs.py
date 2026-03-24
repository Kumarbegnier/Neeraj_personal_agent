from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def extract_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("The provider did not return valid JSON.") from None
        return json.loads(content[start : end + 1])


def parse_structured_response(content: str, output_type: type[BaseModel]) -> BaseModel:
    return output_type.model_validate(extract_json_object(content))


def estimate_response_completeness(output: BaseModel | None) -> float:
    if output is None:
        return 0.0

    data = output.model_dump(exclude_none=False)
    if not data:
        return 0.0

    populated = sum(1 for value in data.values() if _is_populated(value))
    return round(populated / len(data), 2)


def structured_task_success(output: BaseModel | None, structured_output_validity: bool) -> bool:
    if not structured_output_validity:
        return False
    return estimate_response_completeness(output) >= 0.35


def _is_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_is_populated(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_is_populated(item) for item in value)
    return True
