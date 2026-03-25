from __future__ import annotations

from typing import Any, MutableMapping

from src.runtime.models import InteractionResponse

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
    sync_runtime_traces,
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
        _record_client_error(state, "health_error", exc)
        return False


def refresh_architecture(state: MutableMapping[str, Any], client: ApiClient) -> bool:
    clear_error(state)
    try:
        state["architecture"] = client.architecture()
        record_activity(state, "architecture_refresh", "Fetched runtime architecture stages.")
        return True
    except ApiClientError as exc:
        _record_client_error(state, "architecture_error", exc)
        return False


def refresh_runtime_catalogs(state: MutableMapping[str, Any], client: ApiClient) -> bool:
    clear_error(state)
    try:
        sync_agent_catalog(state, client.agents())
        sync_tool_catalog(state, client.tools())
        record_activity(state, "catalog_refresh", "Fetched agent and tool catalogs.")
        return True
    except ApiClientError as exc:
        _record_client_error(state, "catalog_error", exc)
        return False


def refresh_audit_events(state: MutableMapping[str, Any], client: ApiClient, limit: int = 100) -> bool:
    clear_error(state)
    try:
        sync_audit_events(state, client.audit_logs(limit=limit).events)
        record_activity(state, "audit_refresh", "Fetched backend audit events.")
        return True
    except ApiClientError as exc:
        _record_client_error(state, "audit_error", exc)
        return False


def refresh_runtime_traces(state: MutableMapping[str, Any], client: ApiClient, limit: int = 25) -> bool:
    clear_error(state)
    try:
        sync_runtime_traces(state, client.runtime_traces(limit=limit))
        record_activity(state, "observability_refresh", "Fetched recent runtime observability traces.")
        return True
    except ApiClientError as exc:
        _record_client_error(state, "observability_error", exc)
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
        _record_client_error(state, "session_error", exc)
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
        _record_client_error(state, "plan_error", exc)
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
        _finalize_interaction(
            state,
            client,
            interaction,
            assistant_caption=f"Agent: {interaction.assigned_agent} | Risk: {interaction.safety.risk_level}",
            activity_event="chat_turn",
            activity_detail=f"Completed routed interaction via {interaction.assigned_agent}.",
        )
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
        _finalize_interaction(
            state,
            client,
            interaction,
            assistant_caption="Approved execution",
            activity_event="approval_execute",
            activity_detail="Approved and re-executed the pending request.",
        )
        return True
    except ApiClientError as exc:
        _record_client_error(state, "approval_error", exc)
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


def _finalize_interaction(
    state: MutableMapping[str, Any],
    client: ApiClient,
    interaction: InteractionResponse,
    *,
    assistant_caption: str,
    activity_event: str,
    activity_detail: str,
) -> None:
    sync_interaction(state, interaction)
    append_message(
        state,
        "assistant",
        interaction.response,
        caption=assistant_caption,
    )
    record_activity(state, activity_event, activity_detail)
    refresh_session_snapshot(state, client)
    refresh_audit_events(state, client)
    refresh_runtime_traces(state, client)


def _record_client_error(
    state: MutableMapping[str, Any],
    activity_event: str,
    exc: ApiClientError,
) -> None:
    set_error(state, str(exc))
    record_activity(state, activity_event, str(exc))
