from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from src.schemas.evaluation import ProviderEvaluationResult
from src.schemas.routing import HistoricalRoutingStats, ModelProvider, ModelTaskType


class RoutingHistoryStore:
    """Bounded in-memory evaluation history keyed by task family and provider."""

    def __init__(self, *, max_entries_per_family: int = 40) -> None:
        self._max_entries_per_family = max(1, max_entries_per_family)
        self._history: dict[
            tuple[ModelTaskType, str],
            dict[ModelProvider, deque[ProviderEvaluationResult]],
        ] = defaultdict(dict)

    def record(self, result: ProviderEvaluationResult) -> None:
        family_key = (result.task_type, result.task_family)
        provider_buckets = self._history[family_key]
        bucket = provider_buckets.setdefault(
            result.provider,
            deque(maxlen=self._max_entries_per_family),
        )
        bucket.append(result.model_copy(deep=True))

    def stats_for(
        self,
        *,
        task_type: ModelTaskType,
        task_family: str,
        providers: Iterable[ModelProvider] | None = None,
    ) -> list[HistoricalRoutingStats]:
        family_key = (task_type, task_family)
        provider_buckets = self._history.get(family_key, {})
        stats: list[HistoricalRoutingStats] = []
        for provider in providers or ModelProvider:
            samples = list(provider_buckets.get(provider, ()))
            sample_count = len(samples)
            stats.append(
                HistoricalRoutingStats(
                    task_type=task_type,
                    task_family=task_family,
                    provider=provider,
                    sample_count=sample_count,
                    structured_output_validity_rate=(
                        round(sum(1 for sample in samples if sample.structured_output_validity) / sample_count, 3)
                        if sample_count
                        else 0.0
                    ),
                    average_latency_ms=(
                        round(sum(sample.latency_ms for sample in samples) / sample_count)
                        if sample_count
                        else None
                    ),
                    task_success_rate=(
                        round(sum(1 for sample in samples if sample.task_success) / sample_count, 3)
                        if sample_count
                        else 0.0
                    ),
                    average_completeness=(
                        round(sum(sample.response_completeness for sample in samples) / sample_count, 3)
                        if sample_count
                        else 0.0
                    ),
                    retry_frequency=(
                        round(sum(1 for sample in samples if sample.retry_count > 0) / sample_count, 3)
                        if sample_count
                        else 0.0
                    ),
                    last_evaluated_at=samples[-1].recorded_at if sample_count else None,
                )
            )
        return stats

    def tracked_families(self) -> list[tuple[ModelTaskType, str]]:
        return sorted(
            self._history.keys(),
            key=lambda family_key: (family_key[0].value, family_key[1]),
        )

    def health(self) -> dict[str, int]:
        tracked_families = len(self._history)
        total_records = sum(
            len(records)
            for provider_buckets in self._history.values()
            for records in provider_buckets.values()
        )
        return {
            "tracked_task_families": tracked_families,
            "total_records": total_records,
            "max_entries_per_family": self._max_entries_per_family,
        }
