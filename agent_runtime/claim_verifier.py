from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.schemas.claims import ClaimRecord, ClaimVerificationReport, EvidenceLink

from .models import AgentState
from .runtime_utils import dedupe_preserve_order, tokenize_words


@dataclass(slots=True)
class _EvidenceSnippet:
    source_type: str
    source_name: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = tokenize_words(self.text)


class ClaimVerifier:
    def verify(
        self,
        state: AgentState,
        *,
        candidate_response: str | None = None,
        source: str = "state_claims",
    ) -> ClaimVerificationReport:
        run_reason = self._run_reason(state)
        target = "candidate_response" if candidate_response else "state_claims"
        if not run_reason:
            return ClaimVerificationReport(
                enabled=False,
                status="skipped",
                summary="Claim verification was skipped because the task is not in a research or high-stakes mode.",
                confidence_summary="Claim verification is disabled for low-risk tasks unless explicitly requested.",
                confidence=0.0,
                target=target,
                claim_count=0,
                disabled_reason="low_risk_task",
                metadata={"source": source},
            )

        claims = self._claims_for_target(state, candidate_response)
        evidence_pool = self._build_evidence_pool(state)
        if not claims:
            return ClaimVerificationReport(
                enabled=True,
                status="passed",
                summary="No major claims required explicit verification.",
                confidence_summary="High confidence because no major unsupported claims were extracted.",
                confidence=1.0,
                target=target,
                claim_count=0,
                evidence_coverage=1.0,
                metadata={
                    "source": source,
                    "run_reason": run_reason,
                    "evidence_count": len(evidence_pool),
                },
            )

        records = [self._verify_claim(claim, source, evidence_pool) for claim in claims]
        supported = [record.claim_text for record in records if record.support_status == "supported"]
        weak = [record.claim_text for record in records if record.support_status == "weakly_supported"]
        unsupported = [record.claim_text for record in records if record.support_status == "unsupported"]
        coverage = round((len(supported) + (0.5 * len(weak))) / len(records), 2)
        confidence = round(sum(record.confidence for record in records) / len(records), 2)
        status = "passed" if not weak and not unsupported else "needs_attention"
        summary = (
            "Claim verification found every extracted claim strongly supported by runtime evidence."
            if status == "passed"
            else (
                f"Claim verification found {len(weak)} weakly supported and "
                f"{len(unsupported)} unsupported claim(s)."
            )
        )

        return ClaimVerificationReport(
            enabled=True,
            status=status,
            summary=summary,
            confidence_summary=self._confidence_summary(confidence, weak, unsupported),
            confidence=confidence,
            target=target,
            claim_count=len(records),
            claims=records,
            supported_claims=supported,
            weakly_supported_claims=weak,
            unsupported_claims=unsupported,
            evidence_coverage=coverage,
            metadata={
                "source": source,
                "run_reason": run_reason,
                "evidence_count": len(evidence_pool),
            },
        )

    def should_run(self, state: AgentState) -> bool:
        return bool(self._run_reason(state))

    def _run_reason(self, state: AgentState) -> str:
        metadata = state.request.metadata
        if bool(metadata.get("force_claim_verification")):
            return "forced_by_request"
        if bool(metadata.get("disable_claim_verification")):
            return ""
        if state.route and state.route.agent_name == "research":
            return "research_agent"
        if state.architecture and state.architecture.mode.value in {
            "multi_agent_research",
            "browser_heavy_verified",
        }:
            return state.architecture.mode.value
        if state.control and state.control.risk_level in {"high", "critical"}:
            return "control_high_risk"
        if state.context_packet and state.context_packet.approval_state.risk_level in {"high", "critical"}:
            return "context_high_risk"
        if state.safety and state.safety.risk_level == "high":
            return "safety_high_risk"
        return ""

    def _claims_for_target(
        self,
        state: AgentState,
        candidate_response: str | None,
    ) -> list[str]:
        if candidate_response:
            return self._extract_major_claims(candidate_response)

        explicit_claims = []
        if state.execution:
            explicit_claims.extend(state.execution.claims)
        if state.decision:
            explicit_claims.extend(state.decision.claims_to_verify)
        claims = dedupe_preserve_order(claim.strip() for claim in explicit_claims if claim.strip())
        if claims:
            return claims
        if state.execution and state.execution.summary:
            return self._extract_major_claims(state.execution.summary)
        return []

    def _extract_major_claims(self, text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []

        candidates = [
            self._strip_bullet_prefix(part)
            for part in re.split(r"(?<=[.!?])\s+|\s*;\s*", cleaned)
        ]
        filtered: list[str] = []
        ignored_prefixes = (
            "the general agent completed",
            "the research agent completed",
            "the coding agent completed",
            "verification passed",
            "verification still has gaps",
            "reflection found",
            "no additional approval gate",
            "additional confirmation is required",
        )
        for candidate in candidates:
            normalized = candidate.strip(" .")
            lowered = normalized.lower()
            tokens = tokenize_words(normalized)
            if len(tokens) < 4:
                continue
            if lowered.startswith(ignored_prefixes):
                continue
            filtered.append(normalized)
        return dedupe_preserve_order(filtered)

    def _build_evidence_pool(self, state: AgentState) -> list[_EvidenceSnippet]:
        snippets: list[_EvidenceSnippet] = []

        if state.execution:
            snippets.extend(
                self._snippet_batch(
                    "execution",
                    state.execution.agent_name,
                    [
                        *state.execution.observations,
                        *self._flatten_strings(state.execution.artifacts),
                    ],
                )
            )

        for observation in state.observations[-8:]:
            snippets.extend(
                self._snippet_batch(
                    "observation",
                    observation.source,
                    [
                        observation.summary,
                        *observation.evidence,
                        *self._flatten_strings(observation.payload),
                    ],
                    metadata={"step_index": observation.step_index},
                )
            )

        for tool_result in state.tool_history[-10:] + state.last_tool_results[-10:]:
            snippets.extend(
                self._snippet_batch(
                    "tool_evidence",
                    tool_result.tool_name,
                    [
                        *tool_result.evidence,
                        *self._flatten_strings(tool_result.output),
                    ],
                    metadata={
                        "status": tool_result.status,
                        "risk_level": tool_result.risk_level,
                    },
                )
            )

        for memory in state.memory.retrieved[:6]:
            snippets.extend(
                self._snippet_batch(
                    "memory",
                    memory.source or memory.memory_type,
                    [memory.content],
                    metadata={
                        "memory_type": memory.memory_type,
                        "score": memory.score,
                        "tags": list(memory.tags),
                    },
                )
            )

        if state.context_packet:
            snippets.extend(
                self._snippet_batch(
                    "context",
                    "retrieved_facts",
                    state.context_packet.memory.retrieved_facts[:6],
                )
            )

        deduped: list[_EvidenceSnippet] = []
        seen: set[tuple[str, str, str]] = set()
        for snippet in snippets:
            if len(snippet.tokens) < 2:
                continue
            key = (snippet.source_type, snippet.source_name, snippet.text)
            if key in seen:
                continue
            deduped.append(snippet)
            seen.add(key)
        return deduped

    def _verify_claim(
        self,
        claim: str,
        source: str,
        evidence_pool: list[_EvidenceSnippet],
    ) -> ClaimRecord:
        claim_tokens = tokenize_words(claim)
        if not claim_tokens:
            return ClaimRecord(
                claim_text=claim,
                source=source,
                support_status="unsupported",
                confidence=0.0,
                notes=["The extracted claim did not contain enough lexical signal to verify."],
            )

        links: list[EvidenceLink] = []
        lowered_claim = claim.lower()
        for snippet in evidence_pool:
            matched_terms = sorted(claim_tokens & snippet.tokens)
            if not matched_terms:
                continue
            coverage = len(matched_terms) / len(claim_tokens)
            density = len(matched_terms) / max(len(snippet.tokens), 1)
            exact_bonus = 0.25 if lowered_claim in snippet.text.lower() else 0.0
            summary_bonus = 0.08 if snippet.source_type in {"tool_evidence", "observation", "memory"} else 0.0
            score = round(min(1.0, (coverage * 0.75) + min(0.12, density * 0.4) + exact_bonus + summary_bonus), 2)
            if score < 0.18:
                continue
            links.append(
                EvidenceLink(
                    source_type=snippet.source_type,
                    source_name=snippet.source_name,
                    excerpt=self._excerpt(snippet.text),
                    relevance_score=score,
                    support_strength=self._support_strength(score),
                    matched_terms=matched_terms[:8],
                    metadata=snippet.metadata,
                )
            )

        links.sort(key=lambda link: link.relevance_score, reverse=True)
        selected_links = links[:3]
        best_score = selected_links[0].relevance_score if selected_links else 0.0
        strong_links = sum(1 for link in selected_links if link.relevance_score >= 0.45)

        if best_score >= 0.7 or (best_score >= 0.55 and strong_links >= 2):
            support_status = "supported"
            confidence = round(min(1.0, best_score + (0.08 if strong_links >= 2 else 0.0)), 2)
            notes = ["Evidence links strongly support this claim."]
        elif best_score >= 0.35:
            support_status = "weakly_supported"
            confidence = round(min(0.79, max(best_score, 0.4)), 2)
            notes = ["Evidence overlaps with the claim, but support is incomplete or indirect."]
        else:
            support_status = "unsupported"
            confidence = round(best_score, 2)
            notes = ["No sufficiently grounded evidence was found for this claim in the current runtime state."]

        return ClaimRecord(
            claim_text=claim,
            source=source,
            support_status=support_status,
            confidence=confidence,
            evidence_links=selected_links,
            notes=notes,
        )

    def _snippet_batch(
        self,
        source_type: str,
        source_name: str,
        texts: list[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[_EvidenceSnippet]:
        batch: list[_EvidenceSnippet] = []
        for text in texts:
            normalized = self._normalize_text(text)
            if normalized:
                batch.append(
                    _EvidenceSnippet(
                        source_type=source_type,
                        source_name=source_name,
                        text=normalized,
                        metadata=dict(metadata or {}),
                    )
                )
        return batch

    def _flatten_strings(self, value: Any) -> list[str]:
        flattened: list[str] = []
        self._walk_value(value, flattened, depth=0)
        return flattened[:8]

    def _walk_value(self, value: Any, flattened: list[str], *, depth: int) -> None:
        if depth > 3 or len(flattened) >= 8:
            return
        if isinstance(value, str):
            normalized = self._normalize_text(value)
            if normalized:
                flattened.append(normalized)
            return
        if isinstance(value, dict):
            for nested in value.values():
                self._walk_value(nested, flattened, depth=depth + 1)
            return
        if isinstance(value, list):
            for nested in value:
                self._walk_value(nested, flattened, depth=depth + 1)
            return
        if isinstance(value, (int, float, bool)):
            flattened.append(str(value))

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        return normalized[:320]

    def _strip_bullet_prefix(self, text: str) -> str:
        return re.sub(r"^(?:[-*]|\d+\.)\s*", "", text).strip()

    def _excerpt(self, text: str) -> str:
        return text if len(text) <= 180 else f"{text[:177].rstrip()}..."

    def _support_strength(self, score: float) -> str:
        if score >= 0.7:
            return "strong"
        if score >= 0.45:
            return "moderate"
        return "weak"

    def _confidence_summary(
        self,
        confidence: float,
        weak_claims: list[str],
        unsupported_claims: list[str],
    ) -> str:
        if unsupported_claims:
            return (
                f"Low confidence ({confidence:.2f}) because {len(unsupported_claims)} claim(s) remain unsupported."
            )
        if weak_claims:
            return (
                f"Moderate confidence ({confidence:.2f}) because some claims are only weakly supported."
            )
        return f"High confidence ({confidence:.2f}) because all extracted claims are strongly grounded."
