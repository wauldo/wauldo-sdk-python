"""Async HTTP client for Wauldo REST API (OpenAI-compatible)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Callable, List, Optional, Union

from .async_transport import AsyncHttpTransport
from .exceptions import ValidationError
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
    VerifyCitationResponse,
)

logger = logging.getLogger("wauldo")


class AsyncHttpClient:
    """Async HTTP client for the Wauldo REST API.

    Covers all OpenAI-compatible endpoints plus RAG and orchestrator.
    Requires aiohttp: pip install wauldo[async]
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
        self._transport = AsyncHttpTransport(
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

    async def _request(
        self, method: str, path: str, body: Optional[dict[str, Any]] = None, timeout_ms: Optional[int] = None,
    ) -> bytes:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        return await self._transport.execute(method, url, data=data, timeout_ms=timeout_ms)

    async def _request_multipart(
        self, method: str, path: str, files: dict, form_data: dict, timeout_ms: Optional[int] = None,
    ) -> bytes:
        import aiohttp

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
        old_headers_fn = self._transport._headers_fn
        def multipart_headers() -> dict[str, str]:
            h = old_headers_fn()
            h["Content-Type"] = f"multipart/form-data; boundary={boundary}"
            return h
        self._transport._headers_fn = multipart_headers
        try:
            return await self._transport.execute(method, url, data=data, timeout_ms=timeout_ms)
        finally:
            self._transport._headers_fn = old_headers_fn

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def list_models(self) -> ModelList:
        """GET /v1/models"""
        data = await self._request("GET", "/v1/models")
        return ModelList.model_validate_json(data)

    async def chat(
        self, request: ChatRequest, timeout_ms: Optional[int] = None,
    ) -> ChatResponse:
        """POST /v1/chat/completions (non-streaming)."""
        if not request.messages:
            raise ValueError("messages cannot be empty")
        req = request.model_copy(update={"stream": False})
        data = await self._request(
            "POST", "/v1/chat/completions", req.model_dump(exclude_none=True), timeout_ms=timeout_ms,
        )
        return ChatResponse.model_validate_json(data)

    async def chat_simple(
        self, model: str, message: str, timeout_ms: Optional[int] = None, **kwargs: object,
    ) -> str:
        """Single message chat, returns content string."""
        req = ChatRequest.quick(model, message)
        if kwargs:
            req = req.model_copy(update=kwargs)
        resp = await self.chat(req, timeout_ms=timeout_ms)
        return resp.choices[0].message.content or ""

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """POST /v1/chat/completions -- SSE streaming, yields content chunks."""
        import aiohttp

        req = request.model_copy(update={"stream": True})
        url = f"{self.base_url}/v1/chat/completions"
        data = json.dumps(req.model_dump(exclude_none=True)).encode()

        session = await self._transport._get_session()
        ct = aiohttp.ClientTimeout(total=self.timeout)
        async with session.post(url, data=data, headers=self._headers(), timeout=ct) as resp:
            if resp.status >= 400:
                body = await resp.read()
                raise Exception(f"HTTP {resp.status}: {body.decode(errors='replace')}")
            buf = ""
            async for chunk in resp.content.iter_any():
                buf += chunk.decode(errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    try:
                        parsed = json.loads(payload)
                        choices = parsed.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        logger.warning("Malformed SSE chunk: %s", payload[:100])

    async def embeddings(self, input: Union[str, List[str]], model: str) -> EmbeddingResponse:
        """POST /v1/embeddings"""
        body = {"input": input, "model": model}
        data = await self._request("POST", "/v1/embeddings", body)
        return EmbeddingResponse.model_validate_json(data)

    _MAX_UPLOAD_SIZE = 10 * 1024 * 1024

    async def rag_upload(
        self, content: str, filename: Optional[str] = None, timeout_ms: Optional[int] = None,
    ) -> RagUploadResponse:
        """POST /v1/upload"""
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
        data = await self._request("POST", "/v1/upload", body, timeout_ms=timeout_ms)
        return RagUploadResponse.model_validate_json(data)

    async def upload_file(
        self,
        file_path: str,
        title: Optional[str] = None,
        tags: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> UploadFileResponse:
        """POST /v1/upload-file"""
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

        data = await self._request_multipart("POST", "/v1/upload-file", files, form_data, timeout_ms)
        return UploadFileResponse.model_validate_json(data)

    async def rag_query(
        self,
        query: str,
        top_k: int = 5,
        timeout_ms: Optional[int] = None,
        debug: bool = False,
        quality_mode: Optional[str] = None,
    ) -> RagQueryResponse:
        """POST /v1/query"""
        if not query.strip():
            raise ValidationError("Query cannot be empty", field="query")
        if not 1 <= top_k <= 100:
            raise ValidationError("top_k must be between 1 and 100", field="top_k")
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if debug:
            body["debug"] = True
        if quality_mode:
            body["quality_mode"] = quality_mode
        data = await self._request("POST", "/v1/query", body, timeout_ms=timeout_ms)
        return RagQueryResponse.model_validate_json(data)

    async def orchestrate(self, prompt: str) -> OrchestratorResponse:
        """POST /v1/orchestrator/execute"""
        data = await self._request("POST", "/v1/orchestrator/execute", {"prompt": prompt})
        return OrchestratorResponse.model_validate_json(data)

    async def orchestrate_parallel(self, prompt: str) -> OrchestratorResponse:
        """POST /v1/orchestrator/parallel"""
        data = await self._request("POST", "/v1/orchestrator/parallel", {"prompt": prompt})
        return OrchestratorResponse.model_validate_json(data)

    async def fact_check(
        self,
        text: str,
        source_context: str,
        mode: str = "lexical",
    ) -> FactCheckResponse:
        """POST /v1/fact-check"""
        body: dict = {"text": text, "source_context": source_context, "mode": mode}
        data = await self._request("POST", "/v1/fact-check", body)
        return FactCheckResponse.model_validate_json(data)

    async def verify_citation(
        self,
        text: str,
        sources: "list[dict] | None" = None,
        threshold: "float | None" = None,
    ) -> VerifyCitationResponse:
        """POST /v1/verify"""
        body: dict = {"text": text}
        if sources is not None:
            body["sources"] = sources
        if threshold is not None:
            body["threshold"] = threshold
        data = await self._request("POST", "/v1/verify", body)
        return VerifyCitationResponse.model_validate_json(data)

    def conversation(
        self, system: Optional[str] = None, model: str = "default",
    ):
        """Create a stateful Conversation bound to this async client."""
        from .conversation import Conversation as Conv
        return Conv(self, system=system, model=model)

    async def get_insights_async(self) -> "InsightsResponse":
        """GET /v1/insights"""
        from .http_types import InsightsResponse
        data = await self._request("GET", "/v1/insights")
        return InsightsResponse.model_validate_json(data)

    async def get_analytics_async(self, minutes: int = 60) -> "AnalyticsResponse":
        """GET /v1/analytics"""
        from .http_types import AnalyticsResponse
        data = await self._request("GET", f"/v1/analytics?minutes={minutes}")
        return AnalyticsResponse.model_validate_json(data)

    async def get_analytics_traffic_async(self) -> "TrafficSummary":
        """GET /v1/analytics/traffic"""
        from .http_types import TrafficSummary
        data = await self._request("GET", "/v1/analytics/traffic")
        return TrafficSummary.model_validate_json(data)

    async def rag_ask(self, question: str, text: str, source: str = "document") -> str:
        """Upload text and query in one call. Returns answer string."""
        await self.rag_upload(content=text, filename=source)
        result = await self.rag_query(question)
        return result.answer
