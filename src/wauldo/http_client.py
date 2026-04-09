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
    GuardResponse,
    ModelList,
    OrchestratorResponse,
    RagQueryResponse,
    RagUploadResponse,
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

    # ── Guard (Fact-Check) ───────────────────────────────────────────────

    def guard(
        self,
        text: str,
        source_context: str,
        mode: str = "lexical",
        timeout_ms: Optional[int] = None,
    ) -> GuardResponse:
        """POST /v1/fact-check -- Verify text claims against source context.

        Guard is a hallucination firewall: it checks whether an LLM's output
        is supported by the source documents. If claims don't match, Guard
        blocks the response before it reaches users.

        Args:
            text: The LLM-generated text to verify.
            source_context: The ground-truth source document(s).
            mode: Verification mode — "lexical" (<1ms, token overlap),
                "hybrid" (~50ms, token + BGE embeddings), or
                "semantic" (~500ms, embeddings only).
            timeout_ms: Per-request timeout in milliseconds.

        Returns:
            ``GuardResponse`` with verdict, action, claims, and confidence.

        Example::

            result = client.guard(
                text="Returns accepted within 60 days",
                source_context="Our return policy: 14 days.",
            )
            if result.is_blocked:
                print(f"Blocked: {result.claims[0].reason}")
        """
        if not text.strip():
            raise ValidationError("text cannot be empty", field="text")
        if not source_context.strip():
            raise ValidationError("source_context cannot be empty", field="source_context")
        body: dict[str, Any] = {
            "text": text,
            "source_context": source_context,
            "mode": mode,
        }
        data = self._request("POST", "/v1/fact-check", body, timeout_ms=timeout_ms)
        return GuardResponse.model_validate_json(data)

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
