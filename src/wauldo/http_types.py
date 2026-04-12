"""HTTP API types for OpenAI-compatible endpoints."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel

# ── Chat Completions ─────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str
    content: Optional[str] = None
    name: Optional[str] = None

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls(role="assistant", content=content)


class ChatRequest(BaseModel):
    """Request body for POST /v1/chat/completions."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None

    @classmethod
    def quick(cls, model: str, message: str) -> ChatRequest:
        return cls(model=model, messages=[ChatMessage.user(message)])


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from POST /v1/chat/completions."""

    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

    def content(self) -> str:
        """Get the text content of the first choice, or empty string."""
        if self.choices and self.choices[0].message.content:
            return self.choices[0].message.content
        return ""


# ── Models ───────────────────────────────────────────────────────────────


class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class ModelList(BaseModel):
    """Response from GET /v1/models."""

    object: str
    data: List[Model]


# ── Embeddings ───────────────────────────────────────────────────────────


class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """Response from POST /v1/embeddings."""

    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# ── RAG ──────────────────────────────────────────────────────────────────


class RagUploadResponse(BaseModel):
    document_id: str
    chunks_count: int


class RagSource(BaseModel):
    document_id: str
    content: str
    score: float
    chunk_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class RagAuditInfo(BaseModel):
    """Audit trail for RAG responses — verification and accountability."""

    confidence: float
    retrieval_path: str
    sources_evaluated: int
    sources_used: int
    best_score: float
    grounded: bool
    confidence_label: str
    model: str
    latency_ms: int
    # Retrieval funnel diagnostics (v1.6.5+)
    candidates_found: Optional[int] = None
    candidates_after_tenant: Optional[int] = None
    candidates_after_score: Optional[int] = None
    query_type: Optional[str] = None


class RagQueryResponse(BaseModel):
    """Response from POST /v1/query with full audit trail."""

    answer: str
    sources: List[RagSource]
    audit: Optional[RagAuditInfo] = None
    # Legacy flat fields (servers < v1.6.5)
    confidence: Optional[float] = None
    grounded: Optional[bool] = None

    def get_confidence(self) -> Optional[float]:
        """Get confidence from audit (preferred) or legacy flat field."""
        if self.audit:
            return self.audit.confidence
        return self.confidence

    def get_grounded(self) -> Optional[bool]:
        """Get grounded from audit (preferred) or legacy flat field."""
        if self.audit:
            return self.audit.grounded
        return self.grounded


# ── Guard (Fact-Check) ──────────────────────────────────────────────────


class GuardClaim(BaseModel):
    """A single verified claim from the Guard response."""

    text: str
    claim_type: Optional[str] = None
    supported: bool
    confidence: float
    confidence_label: Optional[str] = None
    verdict: str
    action: str
    reason: Optional[str] = None
    evidence: Optional[str] = None


class GuardResponse(BaseModel):
    """Response from POST /v1/fact-check — the Guard verification API.

    Use ``is_safe`` to quickly check if the content passed verification.
    """

    verdict: str
    action: str
    hallucination_rate: float
    mode: str
    total_claims: int
    supported_claims: int
    confidence: float
    claims: List[GuardClaim]
    mode_warning: Optional[str] = None
    processing_time_ms: Optional[int] = None

    @property
    def is_safe(self) -> bool:
        """True if the verdict allows the content through."""
        return self.verdict == "verified"

    @property
    def is_blocked(self) -> bool:
        """True if the content should be blocked."""
        return self.action == "block"


# ── Orchestrator ─────────────────────────────────────────────────────────


class OrchestratorResponse(BaseModel):
    final_output: str


# ── Upload File ─────────────────────────────────────────────────────────


class UploadFileResponse(BaseModel):
    """Response from POST /v1/upload-file (multipart file upload)."""

    document_id: str
    chunks_count: int
    quality_label: Optional[str] = None
    quality_score: Optional[float] = None


# ── Verify Citation ─────────────────────────────────────────────────────


class CitationInfo(BaseModel):
    """A single citation extracted from the text."""

    marker: str
    source_name: Optional[str] = None
    is_valid: bool = False


class VerifyCitationResponse(BaseModel):
    """Response from POST /v1/verify — citation validation."""

    citation_ratio: float
    has_sufficient_citations: bool
    uncited_sentences: List[str] = []
    citations: List[CitationInfo] = []
    phantom_count: int = 0


# ── Fact-Check (alias) ──────────────────────────────────────────────────

# async_client uses FactCheckResponse; it's the same shape as GuardResponse.
FactCheckResponse = GuardResponse


# ── Analytics & Insights ────────────────────────────────────────────────


class TokenStats(BaseModel):
    """Token usage breakdown for InsightsResponse."""

    baseline_total: int
    real_total: int
    saved_total: int
    saved_percent_avg: float
    saved_percent_min: float = 0.0
    saved_percent_max: float = 0.0


class CostStats(BaseModel):
    """Cost savings for InsightsResponse."""

    estimated_usd_saved: float


class InsightsResponse(BaseModel):
    """Response from GET /v1/insights — ROI summary."""

    tig_key: Optional[str] = None
    total_requests: int = 0
    intelligence_requests: int = 0
    fallback_requests: int = 0
    tokens: Optional[TokenStats] = None
    cost: Optional[CostStats] = None


class CacheMetrics(BaseModel):
    """Cache performance in AnalyticsResponse."""

    total_requests: int = 0
    cache_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


class TokenSavings(BaseModel):
    """Token savings summary in AnalyticsResponse."""

    total_baseline: int = 0
    total_real: int = 0
    total_saved: int = 0
    avg_savings_percent: float = 0.0


class AnalyticsResponse(BaseModel):
    """Response from GET /v1/analytics — usage + cache metrics."""

    cache: Optional[CacheMetrics] = None
    tokens: Optional[TokenSavings] = None
    uptime_secs: int = 0


class TenantTraffic(BaseModel):
    """Per-tenant traffic entry in TrafficSummary."""

    tenant_id: str
    requests_today: int = 0
    tokens_used: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0


class TrafficSummary(BaseModel):
    """Response from GET /v1/analytics/traffic — per-tenant monitoring."""

    total_requests_today: int = 0
    total_tokens_today: int = 0
    top_tenants: List[TenantTraffic] = []
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    uptime_secs: int = 0
