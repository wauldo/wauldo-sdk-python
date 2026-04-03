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


# ── Orchestrator ─────────────────────────────────────────────────────────


class OrchestratorResponse(BaseModel):
    final_output: str


# ── Fact-Check ──────────────────────────────────────────────────────────


class ClaimResult(BaseModel):
    text: str
    claim_type: str
    supported: bool
    confidence: float
    confidence_label: str
    verdict: str
    action: str
    reason: Optional[str] = None
    evidence: Optional[str] = None


class FactCheckResponse(BaseModel):
    verdict: str
    action: str
    hallucination_rate: float
    mode: str
    total_claims: int
    supported_claims: int
    confidence: float
    claims: list[ClaimResult]
    mode_warning: Optional[str] = None
    processing_time_ms: int


# ── Citation Verify ────────────────────────────────────────────────────


class SourceChunk(BaseModel):
    name: str
    content: str


class CitationDetail(BaseModel):
    citation: str
    source_name: str
    is_valid: bool


class VerifyCitationRequest(BaseModel):
    text: str
    sources: Optional[list[SourceChunk]] = None
    threshold: Optional[float] = None


class VerifyCitationResponse(BaseModel):
    citation_ratio: float
    has_sufficient_citations: bool
    sentence_count: int
    citation_count: int
    uncited_sentences: list[str]
    citations: Optional[list[CitationDetail]] = None
    phantom_count: Optional[int] = None
    processing_time_ms: int
