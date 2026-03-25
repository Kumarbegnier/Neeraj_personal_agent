from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ClaimVerificationStatus = Literal["passed", "needs_attention", "skipped"]
ClaimSupportStatus = Literal["supported", "weakly_supported", "unsupported"]
EvidenceStrength = Literal["strong", "moderate", "weak"]


class EvidenceLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: str
    source_name: str = ""
    excerpt: str
    relevance_score: float = 0.0
    support_strength: EvidenceStrength = "weak"
    matched_terms: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_text: str
    source: str = "candidate_response"
    support_status: ClaimSupportStatus = "unsupported"
    confidence: float = 0.0
    evidence_links: list[EvidenceLink] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ClaimVerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    status: ClaimVerificationStatus = "skipped"
    summary: str = ""
    confidence_summary: str = ""
    confidence: float = 0.0
    target: str = "state_claims"
    claim_count: int = 0
    claims: list[ClaimRecord] = Field(default_factory=list)
    supported_claims: list[str] = Field(default_factory=list)
    weakly_supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    evidence_coverage: float = 0.0
    disabled_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
