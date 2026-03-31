"""Mock HTTP client for testing without a running server."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from .conversation import Conversation

from .http_types import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    Model,
    ModelList,
    OrchestratorResponse,
    RagQueryResponse,
    RagSource,
    RagUploadResponse,
    Usage,
)


class MockHttpClient:
    """Drop-in replacement for HttpClient that returns deterministic data.

    Useful for unit tests and local development without a running server.
    """

    def __init__(
        self,
        chat_response: str = "This is a mock response.",
        models: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Create a mock client with deterministic responses.

        Args:
            chat_response: Fixed string returned by ``chat()`` and ``chat_simple()``.
            models: List of model IDs returned by ``list_models()``.
                Defaults to ``["mock-model"]``.
            embeddings: Embedding vectors cycled through by ``embeddings()``.
                Defaults to ``[[0.1, 0.2, 0.3]]``.

        Example::

            mock = MockHttpClient(chat_response="42")
            assert mock.chat_simple("m", "question") == "42"
        """
        self._chat_response = chat_response
        self._models = models or ["mock-model"]
        self._embeddings = embeddings or [[0.1, 0.2, 0.3]]

    def list_models(self) -> ModelList:
        """Return mocked model list.

        Returns:
            A ``ModelList`` containing the model IDs provided at construction.
        """
        data = [Model(id=m, object="model", created=0, owned_by="mock") for m in self._models]
        return ModelList(object="list", data=data)

    def chat(self, request: ChatRequest, timeout_ms: Optional[int] = None) -> ChatResponse:
        """Return a mocked chat response.

        Args:
            request: The chat request (model and messages are read but ignored).
            timeout_ms: Ignored in mock — accepted for API compatibility.

        Returns:
            A ``ChatResponse`` with the fixed ``chat_response`` string.
        """
        return ChatResponse(
            id="mock-id",
            object="chat.completion",
            created=0,
            model=request.model,
            choices=[ChatChoice(index=0, message=ChatMessage.assistant(self._chat_response), finish_reason="stop")],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def chat_simple(self, model: str, message: str, timeout_ms: Optional[int] = None, **kwargs: object) -> str:
        """Return the mocked chat content string.

        Args:
            model: Model name (ignored in mock).
            message: User message (ignored in mock).
            **kwargs: Extra options (ignored).

        Returns:
            The fixed ``chat_response`` string provided at construction.
        """
        return self._chat_response

    def chat_stream(self, request: ChatRequest) -> Iterator[str]:
        """Yield mocked content chunks (one word at a time)."""
        for word in self._chat_response.split():
            yield word + " "

    def embeddings(self, input: Union[str, List[str]], model: str) -> EmbeddingResponse:
        """Return mocked embedding vectors."""
        inputs = [input] if isinstance(input, str) else input
        data = [EmbeddingData(embedding=self._embeddings[i % len(self._embeddings)], index=i) for i in range(len(inputs))]
        return EmbeddingResponse(data=data, model=model, usage=EmbeddingUsage(prompt_tokens=len(inputs), total_tokens=len(inputs)))

    def rag_upload(self, content: str, filename: Optional[str] = None, timeout_ms: Optional[int] = None) -> RagUploadResponse:
        """Return a mocked upload response."""
        return RagUploadResponse(document_id="mock-doc-001", chunks_count=3)

    def rag_query(
        self, query: str, top_k: int = 5, timeout_ms: Optional[int] = None,
        debug: bool = False, quality_mode: Optional[str] = None,
    ) -> RagQueryResponse:
        """Return a mocked RAG query response."""
        source = RagSource(document_id="mock-doc-001", content="Mock source content.", score=0.95)
        return RagQueryResponse(answer=self._chat_response, sources=[source])

    def orchestrate(self, prompt: str) -> OrchestratorResponse:
        """Return a mocked orchestrator response."""
        return OrchestratorResponse(final_output=self._chat_response)

    def orchestrate_parallel(self, prompt: str) -> OrchestratorResponse:
        """Return a mocked parallel orchestrator response."""
        return OrchestratorResponse(final_output=self._chat_response)

    def conversation(
        self, system: Optional[str] = None, model: str = "default",
    ) -> Conversation:
        """Create a mock Conversation bound to this mock client."""
        from .conversation import Conversation
        return Conversation(self, system=system, model=model)  # type: ignore[arg-type]

    def rag_ask(self, question: str, text: str, source: str = "document") -> str:
        """Upload text and query in one call. Returns mocked answer."""
        self.rag_upload(content=text, filename=source)
        result = self.rag_query(question)
        return result.answer
