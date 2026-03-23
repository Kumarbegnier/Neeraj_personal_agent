from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import streamlit as st


@dataclass(frozen=True)
class MetricSpec:
    label: str
    value: str | int | float
    help: str = ""


def render_section_intro(title: str, detail: str) -> None:
    st.subheader(title)
    st.caption(detail)


def render_empty_state(title: str, detail: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(detail)


def render_metric_strip(metrics: Sequence[MetricSpec]) -> None:
    if not metrics:
        return

    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        column.metric(metric.label, metric.value, help=metric.help or None)


def render_dataframe_or_caption(
    rows: Sequence[dict[str, Any]],
    empty_message: str,
) -> None:
    if rows:
        st.dataframe(list(rows), use_container_width=True, hide_index=True)
    else:
        st.caption(empty_message)


def render_dataframe_card(
    title: str,
    rows: Sequence[dict[str, Any]],
    *,
    empty_message: str,
    caption: str = "",
) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if caption:
            st.caption(caption)
        render_dataframe_or_caption(rows, empty_message)


def render_text_card(
    title: str,
    body: str,
    *,
    empty_message: str,
    caption: str = "",
) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if caption:
            st.caption(caption)
        if body.strip():
            st.write(body)
        else:
            st.caption(empty_message)


def render_bullet_card(
    title: str,
    items: Sequence[str],
    *,
    empty_message: str,
    caption: str = "",
    limit: int | None = None,
) -> None:
    visible_items = list(items[:limit] if limit is not None else items)
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if caption:
            st.caption(caption)
        if visible_items:
            for item in visible_items:
                st.write(f"- {item}")
        else:
            st.caption(empty_message)


def render_json_card(
    title: str,
    payload: Any | None,
    *,
    empty_message: str,
    caption: str = "",
) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if caption:
            st.caption(caption)
        if payload is None:
            st.caption(empty_message)
        else:
            st.json(payload, expanded=True)
