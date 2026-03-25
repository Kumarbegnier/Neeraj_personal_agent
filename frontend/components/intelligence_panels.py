from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from frontend.components.primitives import (
    MetricSpec,
    render_dataframe_card,
    render_empty_state,
    render_metric_strip,
    render_section_intro,
    render_text_card,
)
from frontend.view_models import (
    architecture_path_rows,
    autonomy_metrics_rows,
    compact_text,
    evaluation_winner_rows,
    format_ratio,
    humanize_label,
    memory_preview_rows,
    provider_selection_rows,
    reflection_summary_lines,
    runtime_trace_rows,
    summarize_memory,
    tool_timeline_rows,
)
from src.runtime.models import InteractionResponse, MemorySnapshot, RuntimeTrace
from src.schemas.platform import HealthResponse
from src.schemas.routing import TaskFamilyRoutingWinner


def render_runtime_intelligence_dashboard(
    interaction: InteractionResponse | None,
    health: HealthResponse | None,
    memory: MemorySnapshot | None,
    evaluation_winners: Sequence[TaskFamilyRoutingWinner],
    runtime_traces: Sequence[RuntimeTrace],
) -> None:
    render_section_intro(
        "Runtime Intelligence",
        "A minimal research-facing view of provider routing, architecture choice, tool progress, reflection, autonomy, evaluations, and retrieved evidence.",
    )

    if interaction is None and health is None and not evaluation_winners and memory is None:
        render_empty_state(
            "No runtime intelligence is available yet.",
            "Refresh the page or complete a routed interaction to populate provider, architecture, memory, and evaluation signals.",
        )
        return

    render_metric_strip(
        [
            MetricSpec(
                "Architecture",
                humanize_label(interaction.architecture.mode.value) if interaction and interaction.architecture else "Unavailable",
            ),
            MetricSpec(
                "Provider routes",
                len(interaction.model_runs) if interaction and interaction.model_runs else len(provider_selection_rows(None, health)),
            ),
            MetricSpec(
                "Tool events",
                len(interaction.tool_results) if interaction is not None else 0,
            ),
            MetricSpec(
                "Autonomy",
                (
                    format_ratio(
                        interaction.autonomy_metrics.autonomous_steps_count / max(interaction.autonomy_metrics.total_steps, 1)
                    )
                    if interaction is not None and interaction.autonomy_metrics.total_steps
                    else "N/A"
                ),
            ),
            MetricSpec("Routing winners", len(evaluation_winners)),
            MetricSpec("Retrieved memory", len(memory.retrieved) if memory is not None else 0),
        ]
    )

    top_left, top_right = st.columns([1.1, 0.9], gap="large")
    with top_left:
        render_dataframe_card(
            "Current Provider Selection",
            provider_selection_rows(interaction, health),
            empty_message="No provider routing data is available yet.",
            caption="Shows the latest routed providers and models when an interaction exists, or the backend routing defaults otherwise.",
        )
        render_dataframe_card(
            "Current Architecture Path",
            architecture_path_rows(interaction),
            empty_message="Run a routed task to inspect the selected execution path.",
            caption="This is the current task-level architecture choice returned by the backend.",
        )
        if interaction is not None and interaction.architecture is not None:
            render_text_card(
                "Architecture Rationale",
                interaction.architecture.rationale,
                empty_message="No architecture rationale is available.",
                caption=interaction.architecture.reasoning.summary,
            )

    with top_right:
        reflection_body = "\n".join(reflection_summary_lines(interaction))
        render_text_card(
            "Reflection Summary",
            reflection_body,
            empty_message="No reflection summary is available yet.",
            caption=(
                f"Verification: {humanize_label(interaction.verification.status)}"
                if interaction is not None
                else "Reflection signals appear after a completed interaction."
            ),
        )
        render_dataframe_card(
            "Autonomy Metrics",
            autonomy_metrics_rows(interaction.autonomy_metrics if interaction is not None else None),
            empty_message="No autonomy metrics are available.",
            caption="These counters summarize how independently the loop moved through the latest run.",
        )

    lower_left, lower_right = st.columns([1.15, 0.85], gap="large")
    with lower_left:
        render_dataframe_card(
            "Tool Execution Timeline",
            tool_timeline_rows(interaction.tool_results if interaction is not None else []),
            empty_message="No tool execution timeline is available for this session yet.",
            caption="Ordered tool events from the latest routed interaction.",
        )
    with lower_right:
        render_dataframe_card(
            "Memory Retrieval Preview",
            memory_preview_rows(memory),
            empty_message="No retrieved memory records are available for preview.",
            caption=compact_text(summarize_memory(memory), limit=180),
        )

    render_dataframe_card(
        "Evaluation Winners By Task Family",
        evaluation_winner_rows(evaluation_winners),
        empty_message="No adaptive routing evaluation winners are available yet.",
        caption="Winners are computed on the backend from historical structured validity, latency, success, completeness, and retry behavior.",
    )

    render_dataframe_card(
        "Recent Runtime Runs",
        runtime_trace_rows(runtime_traces[:8]),
        empty_message="No persisted runtime traces are available yet.",
        caption="Recent run summaries help anchor the dashboard in actual execution activity.",
    )
