"""HTTP client for Wauldo REST API (OpenAI-compatible)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from .exceptions import ValidationError
from .http_streaming import stream_chat_sse
from .http_transport import HttpTransport
from .http_types import (
    ChatRequest,
    ChatResponse,
    EmbeddingResponse,
    FactCheckResponse,
    ModelList,
    OrchestratorResponse,
    RagQueryResponse,
    RagUploadResponse,
    UploadFileResponse,
)

if TYPE_CHECKING:
    from .conversation import Conversation

logger = logging.getLogger("wauldo")


class HttpClient:
    """Synchronous HTTP client for the Wauldo REST API.

    Covers all OpenAI-compatible endpoints plus RAG and orchestrator.
    Uses only stdlib (no external HTTP dependency).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        headers: Optional[dict[str, str]] = None,
        on_request: Optional[Callable[[str, str], None]] = None,
        on_response: Optional[Callable[[int, float], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._extra_headers: dict[str, str] = headers or {}
        self._transport = HttpTransport(
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            headers_fn=self._headers,
            on_request=on_request,
            on_response=on_response,
            on_error=on_error,
        )

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self._extra_headers)
        return h

    def _request(
        self, method: str, path: str, body: Optional[dict[str, Any]] = None, timeout_ms: Optional[int] = None,
    ) -> bytes:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        return self._transport.execute(method, url, data=data, timeout_ms=timeout_ms)

    def _request_multipart(
        self, method: str, path: str, files: dict, form_data: dict, timeout_ms: Optional[int] = None,
    ) -> bytes:
        """Send multipart/form-data request (for file uploads)."""
        import io
        boundary = "----WauldoSDKBoundary"
        body_parts: list[bytes] = []
        for key, (filename, fileobj) in files.items():
            body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode())
            body_parts.append(fileobj.read() if hasattr(fileobj, "read") else fileobj)
            body_parts.append(b"\r\n")
        for key, value in form_data.items():
            body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
        body_parts.append(f"--{boundary}--\r\n".encode())
        data = b"".join(body_parts)

        url = f"{self.base_url}{path}"
        # Override Content-Type for multipart
        old_headers_fn = self._transport._headers_fn
        def multipart_headers() -> dict[str, str]:
            h = old_headers_fn()
            h["Content-Type"] = f"multipart/form-data; boundary={boundary}"
            return h
        self._transport._headers_fn = multipart_headers
        try:
            return self._transport.execute(method, url, data=data, timeout_ms=timeout_ms)
        finally:
            self._transport._headers_fn = old_headers_fn

    # ── OpenAI-compatible endpoints ──────────────────────────────────────

    def list_models(self) -> ModelList:
        """GET /v1/models -- List available LLM models."""
        data = self._request("GET", "/v1/models")
        return ModelList.model_validate_json(data)

    def chat(
        self, request: ChatRequest, timeout_ms: Optional[int] = None,
    ) -> ChatResponse:
        """POST /v1/chat/completions -- Chat completion (non-streaming).

        Args:
            request: The chat request containing model, messages, and options.
            timeout_ms: Per-request timeout in milliseconds. Overrides the
                client-level ``timeout`` for this single call.

        Returns:
            Validated ``ChatResponse`` with choices and usage stats.
        """
        if not request.messages:
            raise ValueError("messages cannot be empty")
        req = request.model_copy(update={"stream": False})
        data = self._request(
            "POST", "/v1/chat/completions", req.model_dump(exclude_none=True), timeout_ms=timeout_ms,
        )
        return ChatResponse.model_validate_json(data)

    def chat_simple(
        self, model: str, message: str, timeout_ms: Optional[int] = None, **kwargs: object,
    ) -> str:
        """Convenience: single message chat, returns content string.

        Args:
            model: LLM model identifier.
            message: The user message.
            timeout_ms: Per-request timeout in milliseconds.
            **kwargs: Extra fields forwarded to ``ChatRequest``.

        Returns:
            The assistant reply as a plain string.
        """
        req = ChatRequest.quick(model, message)
        if kwargs:
            req = req.model_copy(update=kwargs)
        resp = self.chat(req, timeout_ms=timeout_ms)
        return resp.choices[0].message.content or ""

    def chat_stream(self, request: ChatRequest) -> Iterator[str]:
        """POST /v1/chat/completions -- SSE streaming, yields content chunks."""
        req = request.model_copy(update={"stream": True})
        url = f"{self.base_url}/v1/chat/completions"
        data = json.dumps(req.model_dump(exclude_none=True)).encode()
        yield from stream_chat_sse(url, data, self._headers(), self.timeout)

    def embeddings(self, input: Union[str, List[str]], model: str) -> EmbeddingResponse:
        """POST /v1/embeddings -- Generate text embeddings."""
        body = {"input": input, "model": model}
        data = self._request("POST", "/v1/embeddings", body)
        return EmbeddingResponse.model_validate_json(data)

    # ── RAG endpoints ────────────────────────────────────────────────────

    _MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

    def rag_upload(
        self, content: str, filename: Optional[str] = None, timeout_ms: Optional[int] = None,
    ) -> RagUploadResponse:
        """POST /v1/upload -- Upload document for RAG indexing.

        Args:
            content: The document text to index (max 10 MB).
            filename: Optional source filename for metadata.
            timeout_ms: Per-request timeout in milliseconds.

        Returns:
            ``RagUploadResponse`` with document_id and chunks_count.

        Raises:
            ValidationError: If content is empty or exceeds 10 MB.
        """
        if not content.strip():
            raise ValidationError("Content cannot be empty", field="content")
        if len(content) > self._MAX_UPLOAD_SIZE:
            raise ValidationError(
                f"Content exceeds maximum size ({len(content)} > {self._MAX_UPLOAD_SIZE} bytes)",
                field="content",
            )
        body: dict[str, Any] = {"content": content}
        if filename:
            body["filename"] = filename
        data = self._request("POST", "/v1/upload", body, timeout_ms=timeout_ms)
        return RagUploadResponse.model_validate_json(data)

    def upload_file(
        self,
        file_path: str,
        title: Optional[str] = None,
        tags: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> UploadFileResponse:
        """POST /v1/upload-file -- Upload a file (PDF, DOCX, text, image) for RAG indexing.

        Args:
            file_path: Path to the file to upload (PDF, DOCX, TXT, etc.).
            title: Optional document title.
            tags: Optional comma-separated tags.
            timeout_ms: Per-request timeout in milliseconds.

        Returns:
            ``UploadFileResponse`` with document_id, chunks_count, quality scoring.
        """
        import os
        if not os.path.isfile(file_path):
            raise ValidationError(f"File not found: {file_path}", field="file_path")
        if os.path.getsize(file_path) > self._MAX_UPLOAD_SIZE:
            raise ValidationError("File exceeds 10 MB limit", field="file_path")

        files = {"file": (os.path.basename(file_path), open(file_path, "rb"))}
        form_data: dict[str, str] = {}
        if title:
            form_data["title"] = title
        if tags:
            form_data["tags"] = tags

        data = self._request_multipart("POST", "/v1/upload-file", files, form_data, timeout_ms)
        return UploadFileResponse.model_validate_json(data)

    def rag_query(
        self,
        query: str,
        top_k: int = 5,
        timeout_ms: Optional[int] = None,
        debug: bool = False,
        quality_mode: Optional[str] = None,
    ) -> RagQueryResponse:
        """POST /v1/query -- Query RAG knowledge base.

        Args:
            query: Search query for the RAG knowledge base.
            top_k: Number of sources to retrieve (1-100).
            timeout_ms: Per-request timeout in milliseconds.
            debug: Enable debug mode — returns retrieval funnel details.
            quality_mode: "fast", "balanced", or "premium".

        Raises:
            ValidationError: If query is empty or top_k is out of range (1-100).
        """
        if not query.strip():
            raise ValidationError("Query cannot be empty", field="query")
        if not 1 <= top_k <= 100:
            raise ValidationError("top_k must be between 1 and 100", field="top_k")
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if debug:
            body["debug"] = True
        if quality_mode:
            body["quality_mode"] = quality_mode
        data = self._request("POST", "/v1/query", body, timeout_ms=timeout_ms)
        return RagQueryResponse.model_validate_json(data)

    # ── Orchestrator endpoints ───────────────────────────────────────────

    def orchestrate(self, prompt: str) -> OrchestratorResponse:
        """POST /v1/orchestrator/execute -- Route to best specialist agent."""
        data = self._request("POST", "/v1/orchestrator/execute", {"prompt": prompt})
        return OrchestratorResponse.model_validate_json(data)

    def orchestrate_parallel(self, prompt: str) -> OrchestratorResponse:
        """POST /v1/orchestrator/parallel -- Run all 4 specialists in parallel."""
        data = self._request("POST", "/v1/orchestrator/parallel", {"prompt": prompt})
        return OrchestratorResponse.model_validate_json(data)

    # ── Fact-Check endpoints ────────────────────────────────────────────

    def fact_check(
        self,
        text: str,
        source_context: str,
        mode: str = "lexical",
    ) -> FactCheckResponse:
        """POST /v1/fact-check -- Verify claims against source context.

        Args:
            text: Text containing claims to verify.
            source_context: Source document to verify against.
            mode: Verification mode (lexical, hybrid, semantic).

        Returns:
            FactCheckResponse with verdict, action, and per-claim results.
        """
        body: dict = {"text": text, "source_context": source_context, "mode": mode}
        data = self._request("POST", "/v1/fact-check", body)
        return FactCheckResponse.model_validate_json(data)

    def verify_citation(
        self,
        text: str,
        sources: "list[dict] | None" = None,
        threshold: "float | None" = None,
    ) -> "VerifyCitationResponse":
        """POST /v1/verify -- Verify citations in AI-generated text.

        Args:
            text: AI-generated text with or without citations.
            sources: Optional source chunks [{"name": "...", "content": "..."}].
            threshold: Minimum citation ratio (0.0-1.0, default 0.5).

        Returns:
            VerifyCitationResponse with citation_ratio, uncited_sentences, phantom detection.
        """
        from .http_types import VerifyCitationResponse

        body: dict = {"text": text}
        if sources is not None:
            body["sources"] = sources
        if threshold is not None:
            body["threshold"] = threshold
        data = self._request("POST", "/v1/verify", body)
        return VerifyCitationResponse.model_validate_json(data)

    # ── Analytics & Insights endpoints ──────────────────────────────────

    def get_insights(self) -> "InsightsResponse":
        """GET /v1/insights -- ROI metrics for your API key."""
        from .http_types import InsightsResponse
        data = self._request("GET", "/v1/insights")
        return InsightsResponse.model_validate_json(data)

    def get_analytics(self, minutes: int = 60) -> "AnalyticsResponse":
        """GET /v1/analytics -- Usage analytics and cache performance."""
        from .http_types import AnalyticsResponse
        data = self._request("GET", f"/v1/analytics?minutes={minutes}")
        return AnalyticsResponse.model_validate_json(data)

    def get_analytics_traffic(self) -> "TrafficSummary":
        """GET /v1/analytics/traffic -- Per-tenant traffic monitoring."""
        from .http_types import TrafficSummary
        data = self._request("GET", "/v1/analytics/traffic")
        return TrafficSummary.model_validate_json(data)

    def guard(
        self,
        text: str,
        source: str,
        mode: str = "lexical",
    ) -> "GuardResult":
        """Verify an LLM output against a source document.

        Convenience wrapper around fact_check(). Returns a simple
        safe/unsafe result for use as a hallucination firewall.

        Args:
            text: The LLM-generated text to verify.
            source: The source document to verify against.
            mode: Verification mode — "lexical" (fast), "hybrid", or "semantic".

        Returns:
            GuardResult with safe, verdict, action, reason, confidence.
        """
        from .http_types import GuardResult
        result = self.fact_check(text=text, source_context=source, mode=mode)
        return GuardResult(
            safe=result.claims[0].verdict == "verified" if result.claims else False,
            verdict=result.claims[0].verdict if result.claims else "rejected",
            action=result.claims[0].action if result.claims else "block",
            reason=result.claims[0].reason if result.claims else "no_claims",
            confidence=result.claims[0].confidence if result.claims else 0.0,
        )

    # ── Convenience helpers ──────────────────────────────────────────────

    def conversation(
        self, system: Optional[str] = None, model: str = "default",
    ) -> Conversation:
        """Create a stateful ``Conversation`` with automatic history management.

        Args:
            system: Optional system prompt prepended to every request.
            model: LLM model identifier used for all turns.

        Returns:
            A ``Conversation`` instance bound to this client.
        """
        from .conversation import Conversation as Conv
        return Conv(self, system=system, model=model)

    def rag_ask(self, question: str, text: str, source: str = "document") -> str:
        """Upload text and query in one call. Returns answer string.

        Args:
            question: The question to ask about the document.
            text: The document content to upload and index.
            source: Filename label for the uploaded document.

        Returns:
            The answer string from the RAG pipeline.
        """
        self.rag_upload(content=text, filename=source)
        result = self.rag_query(question)
        return result.answer
