from __future__ import annotations

from typing import Any

from src.core.config import Settings
from src.schemas.evaluation import ProviderEvaluationResult
from src.schemas.routing import (
    AdaptiveRoutingDecision,
    HistoricalRoutingStats,
    ModelProvider,
    ModelTaskType,
    RoutingPolicyEntry,
    RoutingRequest,
    RoutingScore,
    TaskFamilyRoutingWinner,
)

from .routing_history import RoutingHistoryStore


class RoutingPolicyService:
    """Owns both the deterministic fallback table and adaptive history-aware routing policy."""

    def __init__(
        self,
        settings: Settings,
        history_store: RoutingHistoryStore | None = None,
    ) -> None:
        self._settings = settings
        self._history_store = history_store or RoutingHistoryStore(
            max_entries_per_family=getattr(settings, "routing_history_limit", 40)
        )
        self._min_adaptive_samples = max(1, int(getattr(settings, "adaptive_routing_min_samples", 3)))
        self._adaptive_margin = max(0.0, float(getattr(settings, "adaptive_routing_min_margin", 0.05)))
        self._routing_policy = {
            ModelTaskType.ORCHESTRATION: ModelProvider.OPENAI,
            ModelTaskType.TOOL_EXECUTION: ModelProvider.OPENAI,
            ModelTaskType.COMMUNICATION: ModelProvider.CLAUDE,
            ModelTaskType.REFLECTION: ModelProvider.CLAUDE,
            ModelTaskType.RESEARCH: ModelProvider.GEMINI,
            ModelTaskType.WEB_GROUNDING: ModelProvider.GEMINI,
            ModelTaskType.PLANNING: ModelProvider.DEEPSEEK,
            ModelTaskType.REASONING: ModelProvider.DEEPSEEK,
        }
        self._rationales = {
            ModelTaskType.ORCHESTRATION: "OpenAI remains the orchestration and tool-calling backbone.",
            ModelTaskType.TOOL_EXECUTION: "OpenAI is the default execution summarizer and tool-calling path.",
            ModelTaskType.COMMUNICATION: "Claude is preferred for polished communication-heavy outputs.",
            ModelTaskType.REFLECTION: "Claude is preferred for critique, reflection, and revision guidance.",
            ModelTaskType.RESEARCH: "Gemini is preferred for research-oriented synthesis tasks.",
            ModelTaskType.WEB_GROUNDING: "Gemini is preferred when web grounding is part of the task.",
            ModelTaskType.PLANNING: "DeepSeek is preferred for structured planning and decomposition.",
            ModelTaskType.REASONING: "DeepSeek is preferred for JSON-heavy reasoning and verification.",
        }

    def provider_for(self, task_type: ModelTaskType) -> ModelProvider:
        return self._routing_policy.get(task_type, ModelProvider.OPENAI)

    def default_model(self, provider: ModelProvider) -> str:
        return {
            ModelProvider.OPENAI: self._settings.openai_responses_model,
            ModelProvider.CLAUDE: self._settings.claude_model,
            ModelProvider.GEMINI: self._settings.gemini_model,
            ModelProvider.DEEPSEEK: self._settings.deepseek_model,
        }[provider]

    def candidate_models(self) -> list[str]:
        return [
            self._settings.openai_responses_model,
            self._settings.claude_model,
            self._settings.gemini_model,
            self._settings.deepseek_model,
        ]

    def entry(self, task_type: ModelTaskType) -> RoutingPolicyEntry:
        provider = self.provider_for(task_type)
        return RoutingPolicyEntry(
            task_type=task_type,
            provider=provider,
            default_model=self.default_model(provider),
            rationale=self._rationales.get(task_type, "Defaulted to the primary orchestration provider."),
        )

    def entries(self) -> list[RoutingPolicyEntry]:
        return [self.entry(task_type) for task_type in ModelTaskType]

    def routing_table(self) -> dict[str, str]:
        return {task_type.value: self.provider_for(task_type).value for task_type in ModelTaskType}

    def task_family_for(self, request: RoutingRequest) -> str:
        raw_family = (
            request.task_family
            or request.metadata.get("task_family")
            or request.metadata.get("routing_task_family")
            or request.task_type.value
        )
        normalized = str(raw_family).strip().lower().replace("-", "_").replace(" ", "_")
        return normalized or request.task_type.value

    def record_evaluation(self, result: ProviderEvaluationResult) -> None:
        normalized_family = result.task_family.strip().lower().replace("-", "_").replace(" ", "_") or result.task_type.value
        self._history_store.record(
            result.model_copy(update={"task_family": normalized_family})
        )

    def historical_stats(
        self,
        *,
        task_type: ModelTaskType,
        task_family: str,
    ) -> list[HistoricalRoutingStats]:
        return self._history_store.stats_for(
            task_type=task_type,
            task_family=task_family,
            providers=ModelProvider,
        )

    def decide(self, request: RoutingRequest) -> AdaptiveRoutingDecision:
        fallback_provider = self.provider_for(request.task_type)
        task_family = self.task_family_for(request)
        stats = self.historical_stats(task_type=request.task_type, task_family=task_family)
        scores = [self._score(stat) for stat in stats]
        score_by_provider = {score.provider: score for score in scores}
        eligible_scores = [score for score in scores if score.eligible]

        if not eligible_scores:
            return AdaptiveRoutingDecision(
                task_type=request.task_type,
                task_family=task_family,
                selected_provider=fallback_provider,
                fallback_provider=fallback_provider,
                used_history=False,
                minimum_samples_required=self._min_adaptive_samples,
                scores=scores,
                historical_stats=stats,
                reason=(
                    f"{self.entry(request.task_type).rationale} Historical evaluation data for task family "
                    f"'{task_family}' did not meet the minimum sample threshold of {self._min_adaptive_samples}."
                ),
            )

        best_score = self._best_score(eligible_scores, fallback_provider=fallback_provider)
        fallback_score = score_by_provider.get(fallback_provider)
        if (
            fallback_score is not None
            and best_score.provider != fallback_provider
            and best_score.total_score < fallback_score.total_score + self._adaptive_margin
        ):
            return AdaptiveRoutingDecision(
                task_type=request.task_type,
                task_family=task_family,
                selected_provider=fallback_provider,
                fallback_provider=fallback_provider,
                used_history=True,
                minimum_samples_required=self._min_adaptive_samples,
                scores=scores,
                historical_stats=stats,
                reason=(
                    f"Historical routing scores for task family '{task_family}' were considered, but "
                    f"the best alternative provider did not beat the deterministic fallback margin of "
                    f"{self._adaptive_margin:.2f}."
                ),
            )

        history_clause = (
            "Historical evaluation data reinforced the deterministic fallback route."
            if best_score.provider == fallback_provider
            else "Historical evaluation data overrode the deterministic fallback route."
        )
        return AdaptiveRoutingDecision(
            task_type=request.task_type,
            task_family=task_family,
            selected_provider=best_score.provider,
            fallback_provider=fallback_provider,
            used_history=True,
            minimum_samples_required=self._min_adaptive_samples,
            scores=scores,
            historical_stats=stats,
            reason=(
                f"{history_clause} Selected provider '{best_score.provider.value}' for task family "
                f"'{task_family}' with score {best_score.total_score:.3f}."
            ),
        )

    def health(self) -> dict[str, Any]:
        return {
            "adaptive_routing_enabled": True,
            "minimum_samples_required": self._min_adaptive_samples,
            "minimum_score_margin": self._adaptive_margin,
            **self._history_store.health(),
        }

    def evaluation_winners(self, *, limit: int = 12) -> list[TaskFamilyRoutingWinner]:
        winners: list[TaskFamilyRoutingWinner] = []
        for task_type, task_family in self._history_store.tracked_families():
            decision = self.decide(RoutingRequest(task_type=task_type, task_family=task_family))
            score_by_provider = {score.provider: score for score in decision.scores}
            stats_by_provider = {stats.provider: stats for stats in decision.historical_stats}
            winner_score = score_by_provider.get(decision.selected_provider, RoutingScore(
                task_type=task_type,
                task_family=task_family,
                provider=decision.selected_provider,
            ))
            winner_stats = stats_by_provider.get(decision.selected_provider, HistoricalRoutingStats(
                task_type=task_type,
                task_family=task_family,
                provider=decision.selected_provider,
            ))
            winners.append(
                TaskFamilyRoutingWinner(
                    task_type=task_type,
                    task_family=task_family,
                    selected_provider=decision.selected_provider,
                    fallback_provider=decision.fallback_provider,
                    winning_score=winner_score.total_score,
                    sample_count=winner_stats.sample_count,
                    structured_output_validity_rate=winner_stats.structured_output_validity_rate,
                    task_success_rate=winner_stats.task_success_rate,
                    average_completeness=winner_stats.average_completeness,
                    average_latency_ms=winner_stats.average_latency_ms,
                    retry_frequency=winner_stats.retry_frequency,
                    used_history=decision.used_history,
                    reason=decision.reason,
                    last_evaluated_at=winner_stats.last_evaluated_at,
                )
            )

        ordered = sorted(
            winners,
            key=lambda winner: (
                -winner.sample_count,
                -winner.winning_score,
                winner.task_type.value,
                winner.task_family,
            ),
        )
        return ordered[: max(1, limit)]

    def _score(self, stats: HistoricalRoutingStats) -> RoutingScore:
        latency_reference_ms = stats.average_latency_ms if stats.average_latency_ms is not None else 10000
        latency_component = max(0.0, 1.0 - min(latency_reference_ms / 10000, 1.0)) * 0.10
        validity_component = stats.structured_output_validity_rate * 0.30
        success_component = stats.task_success_rate * 0.30
        completeness_component = stats.average_completeness * 0.20
        retry_penalty = stats.retry_frequency * 0.15
        total_score = validity_component + success_component + completeness_component + latency_component - retry_penalty
        eligible = stats.sample_count >= self._min_adaptive_samples
        return RoutingScore(
            task_type=stats.task_type,
            task_family=stats.task_family,
            provider=stats.provider,
            sample_count=stats.sample_count,
            structured_output_validity_component=round(validity_component, 3),
            latency_component=round(latency_component, 3),
            task_success_component=round(success_component, 3),
            completeness_component=round(completeness_component, 3),
            retry_penalty=round(retry_penalty, 3),
            total_score=round(total_score, 3),
            eligible=eligible,
            rationale=(
                f"samples={stats.sample_count}, valid_rate={stats.structured_output_validity_rate:.2f}, "
                f"success_rate={stats.task_success_rate:.2f}, completeness={stats.average_completeness:.2f}, "
                f"retry_frequency={stats.retry_frequency:.2f}, avg_latency_ms={stats.average_latency_ms or 0}"
            ),
        )

    def _best_score(
        self,
        scores: list[RoutingScore],
        *,
        fallback_provider: ModelProvider,
    ) -> RoutingScore:
        return sorted(
            scores,
            key=lambda score: (
                -score.total_score,
                0 if score.provider == fallback_provider else 1,
                score.provider.value,
            ),
        )[0]
