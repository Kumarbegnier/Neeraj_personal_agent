from __future__ import annotations

from typing import Any, MutableMapping

from .config import FrontendConfig
from .services.api_client import ApiClient, ApiClientError, RequestEnvelope
from .utils.state import (
    append_message,
    clear_error,
    record_activity,
    reset_workspace,
    set_error,
    sync_agent_catalog,
    sync_audit_events,
    sync_health,
    sync_interaction,
    sync_plan,
    sync_session_snapshot,
    sync_tool_catalog,
)


def build_api_client(state: MutableMapping[str, Any], config: FrontendConfig) -> ApiClient:
    return ApiClient(
        base_url=str(state["backend_url"]),
        timeout_seconds=config.request_timeout_seconds,
    )


def refresh_health(state: MutableMapping[str, Any], client: ApiClient) -> bool:
    clear_error(state)
    try:
        health = client.health()
        sync_health(state, health)
        record_activity(state, "health_refresh", "Fetched backend health details.")
        return True
    except ApiClientError as exc:
        set_error(state, str(exc))
        record_activity(state, "health_error", str(exc))
        return False


def refresh_architecture(state: MutableMapping[str, Any], client: ApiClient) -> bool:
    clear_error(state)
    try:
        state["architecture"] = client.architecture()
        record_activity(state, "architecture_refresh", "Fetched runtime architecture stages.")
        return True
    except ApiClientError as exc:
        set_error(state, str(exc))
        record_activity(state, "architecture_error", str(exc))
        return False


def refresh_runtime_catalogs(state: MutableMapping[str, Any], client: ApiClient) -> bool:
    clear_error(state)
    try:
        sync_agent_catalog(state, client.agents())
        sync_tool_catalog(state, client.tools())
        record_activity(state, "catalog_refresh", "Fetched agent and tool catalogs.")
        return True
    except ApiClientError as exc:
        set_error(state, str(exc))
        record_activity(state, "catalog_error", str(exc))
        return False


def refresh_audit_events(state: MutableMapping[str, Any], client: ApiClient, limit: int = 100) -> bool:
    clear_error(state)
    try:
        sync_audit_events(state, client.audit_logs(limit=limit).events)
        record_activity(state, "audit_refresh", "Fetched backend audit events.")
        return True
    except ApiClientError as exc:
        set_error(state, str(exc))
        record_activity(state, "audit_error", str(exc))
        return False


def refresh_session_snapshot(state: MutableMapping[str, Any], client: ApiClient) -> bool:
    clear_error(state)
    try:
        snapshot = client.session_state(
            user_id=str(state["user_id"]),
            session_id=str(state["session_id"]),
        )
        sync_session_snapshot(state, snapshot)
        record_activity(state, "session_refresh", "Fetched the latest session snapshot.")
        return True
    except ApiClientError as exc:
        set_error(state, str(exc))
        record_activity(state, "session_error", str(exc))
        return False


def preview_plan(
    state: MutableMapping[str, Any],
    client: ApiClient,
    message: str,
) -> bool:
    clear_error(state)
    try:
        plan = client.plan(_build_request(state, message))
        sync_plan(state, plan)
        record_activity(state, "plan_preview", f"Generated a plan preview for: {message[:80]}")
        refresh_audit_events(state, client)
        return True
    except ApiClientError as exc:
        set_error(state, str(exc))
        record_activity(state, "plan_error", str(exc))
        return False


def submit_chat_prompt(
    state: MutableMapping[str, Any],
    client: ApiClient,
    prompt: str,
) -> bool:
    clear_error(state)
    state["last_prompt"] = prompt
    append_message(state, "user", prompt)
    try:
        interaction = client.chat(_build_request(state, prompt))
        sync_interaction(state, interaction)
        append_message(
            state,
            "assistant",
            interaction.response,
            caption=f"Agent: {interaction.assigned_agent} | Risk: {interaction.safety.risk_level}",
        )
        record_activity(state, "chat_turn", f"Completed routed interaction via {interaction.assigned_agent}.")
        refresh_session_snapshot(state, client)
        refresh_audit_events(state, client)
        return True
    except ApiClientError as exc:
        set_error(state, str(exc))
        append_message(state, "assistant", f"Backend error: {exc}", caption="Execution interrupted")
        record_activity(state, "chat_error", str(exc))
        return False


def approve_pending_request(
    state: MutableMapping[str, Any],
    client: ApiClient,
) -> bool:
    clear_error(state)
    prompt = state.get("pending_prompt") or state.get("last_prompt")
    if not prompt:
        set_error(state, "There is no pending request to approve.")
        return False

    try:
        interaction = client.execute(_build_request(state, prompt, approval_granted=True))
        sync_interaction(state, interaction)
        append_message(
            state,
            "assistant",
            interaction.response,
            caption="Approved execution",
        )
        record_activity(state, "approval_execute", "Approved and re-executed the pending request.")
        refresh_session_snapshot(state, client)
        refresh_audit_events(state, client)
        return True
    except ApiClientError as exc:
        set_error(state, str(exc))
        record_activity(state, "approval_error", str(exc))
        return False


def start_new_session(state: MutableMapping[str, Any]) -> None:
    reset_workspace(state, new_session=True)
    record_activity(state, "session_reset", "Started a new local Streamlit session.")


def clear_workspace(state: MutableMapping[str, Any]) -> None:
    reset_workspace(state, new_session=False)
    record_activity(state, "workspace_reset", "Cleared the local workspace state.")


def _build_request(
    state: MutableMapping[str, Any],
    message: str,
    *,
    approval_granted: bool | None = None,
) -> RequestEnvelope:
    approved = bool(state["approval_toggle"]) if approval_granted is None else approval_granted
    return RequestEnvelope(
        user_id=str(state["user_id"]),
        session_id=str(state["session_id"]),
        message=message,
        selected_model=str(state["selected_model"]),
        approval_granted=approved,
    )
