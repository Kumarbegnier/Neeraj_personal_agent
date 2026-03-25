from __future__ import annotations

import unittest

from src.core.config import get_settings
from src.schemas.evaluation import ProviderEvaluationResult
from src.schemas.routing import ModelProvider, ModelTaskType, RoutingRequest
from src.services.model_router import ModelRouter
from src.services.routing_history import RoutingHistoryStore
from src.services.routing_policy import RoutingPolicyService


def build_router() -> tuple[ModelRouter, RoutingPolicyService]:
    settings = get_settings()
    history_store = RoutingHistoryStore(max_entries_per_family=12)
    policy = RoutingPolicyService(settings, history_store=history_store)
    return ModelRouter(settings, policy), policy


class AdaptiveRoutingTests(unittest.TestCase):
    def test_router_uses_deterministic_fallback_without_history(self) -> None:
        router, _policy = build_router()

        decision = router.route(
            RoutingRequest(
                task_type=ModelTaskType.REASONING,
                task_family="json_reasoning",
            )
        )

        self.assertEqual(decision.provider, ModelProvider.DEEPSEEK)
        self.assertEqual(decision.fallback_provider, ModelProvider.DEEPSEEK)
        self.assertIsNotNone(decision.adaptive_decision)
        self.assertFalse(decision.adaptive_decision.used_history)

    def test_router_can_override_fallback_from_history(self) -> None:
        router, policy = build_router()
        self._seed_history(
            router,
            task_type=ModelTaskType.REASONING,
            task_family="json_reasoning",
            provider=ModelProvider.OPENAI,
            sample_count=4,
            latency_ms=420,
            valid=True,
            success=True,
            completeness=0.94,
            retry_count=0,
            model=policy.default_model(ModelProvider.OPENAI),
        )
        self._seed_history(
            router,
            task_type=ModelTaskType.REASONING,
            task_family="json_reasoning",
            provider=ModelProvider.DEEPSEEK,
            sample_count=4,
            latency_ms=2600,
            valid=False,
            success=False,
            completeness=0.18,
            retry_count=1,
            model=policy.default_model(ModelProvider.DEEPSEEK),
        )

        decision = router.route(
            RoutingRequest(
                task_type=ModelTaskType.REASONING,
                task_family="json_reasoning",
            )
        )

        self.assertEqual(decision.provider, ModelProvider.OPENAI)
        self.assertTrue(decision.adaptive_decision.used_history)
        self.assertEqual(decision.adaptive_decision.selected_provider, ModelProvider.OPENAI)
        self.assertEqual(decision.adaptive_decision.fallback_provider, ModelProvider.DEEPSEEK)

    def test_history_is_isolated_by_task_family(self) -> None:
        router, policy = build_router()
        self._seed_history(
            router,
            task_type=ModelTaskType.REASONING,
            task_family="json_reasoning",
            provider=ModelProvider.OPENAI,
            sample_count=4,
            latency_ms=400,
            valid=True,
            success=True,
            completeness=0.91,
            retry_count=0,
            model=policy.default_model(ModelProvider.OPENAI),
        )

        decision = router.route(
            RoutingRequest(
                task_type=ModelTaskType.REASONING,
                task_family="math_reasoning",
            )
        )

        self.assertEqual(decision.provider, ModelProvider.DEEPSEEK)
        self.assertFalse(decision.adaptive_decision.used_history)

    def test_stats_capture_retry_frequency_and_explicit_provider_override(self) -> None:
        router, policy = build_router()
        self._seed_history(
            router,
            task_type=ModelTaskType.RESEARCH,
            task_family="evidence_synthesis",
            provider=ModelProvider.GEMINI,
            sample_count=3,
            latency_ms=900,
            valid=True,
            success=False,
            completeness=0.52,
            retry_count=1,
            model=policy.default_model(ModelProvider.GEMINI),
        )

        stats = router.historical_stats(
            task_type=ModelTaskType.RESEARCH,
            task_family="evidence_synthesis",
        )
        gemini_stats = next(stat for stat in stats if stat.provider == ModelProvider.GEMINI)
        self.assertEqual(gemini_stats.retry_frequency, 1.0)

        decision = router.route(
            RoutingRequest(
                task_type=ModelTaskType.RESEARCH,
                task_family="evidence_synthesis",
                selected_provider=ModelProvider.CLAUDE,
            )
        )
        self.assertEqual(decision.provider, ModelProvider.CLAUDE)

    def test_evaluation_winners_return_best_provider_per_task_family(self) -> None:
        router, policy = build_router()
        self._seed_history(
            router,
            task_type=ModelTaskType.REASONING,
            task_family="json_reasoning",
            provider=ModelProvider.OPENAI,
            sample_count=4,
            latency_ms=350,
            valid=True,
            success=True,
            completeness=0.96,
            retry_count=0,
            model=policy.default_model(ModelProvider.OPENAI),
        )
        self._seed_history(
            router,
            task_type=ModelTaskType.REASONING,
            task_family="json_reasoning",
            provider=ModelProvider.DEEPSEEK,
            sample_count=4,
            latency_ms=1900,
            valid=False,
            success=False,
            completeness=0.21,
            retry_count=1,
            model=policy.default_model(ModelProvider.DEEPSEEK),
        )
        self._seed_history(
            router,
            task_type=ModelTaskType.RESEARCH,
            task_family="evidence_synthesis",
            provider=ModelProvider.GEMINI,
            sample_count=3,
            latency_ms=700,
            valid=True,
            success=True,
            completeness=0.88,
            retry_count=0,
            model=policy.default_model(ModelProvider.GEMINI),
        )

        winners = router.evaluation_winners(limit=10)

        by_family = {winner.task_family: winner for winner in winners}
        self.assertEqual(by_family["json_reasoning"].selected_provider, ModelProvider.OPENAI)
        self.assertEqual(by_family["evidence_synthesis"].selected_provider, ModelProvider.GEMINI)
        self.assertGreaterEqual(by_family["json_reasoning"].sample_count, 4)

    def _seed_history(
        self,
        router: ModelRouter,
        *,
        task_type: ModelTaskType,
        task_family: str,
        provider: ModelProvider,
        sample_count: int,
        latency_ms: int,
        valid: bool,
        success: bool,
        completeness: float,
        retry_count: int,
        model: str,
    ) -> None:
        for _ in range(sample_count):
            router.record_evaluation(
                ProviderEvaluationResult(
                    task_type=task_type,
                    task_family=task_family,
                    provider=provider,
                    model=model,
                    structured_output_validity=valid,
                    latency_ms=latency_ms,
                    task_success=success,
                    response_completeness=completeness,
                    retry_count=retry_count,
                )
            )


if __name__ == "__main__":
    unittest.main()
