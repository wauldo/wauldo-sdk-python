"""Tests for Wauldo HTTP Client (REST API)."""

import json

from wauldo.http_client import HttpClient
from wauldo.http_types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingResponse,
    Model,
    ModelList,
    OrchestratorResponse,
    RagQueryResponse,
    RagSource,
    RagUploadResponse,
    Usage,
)


# ============================================================================
# Client Construction
# ============================================================================


def test_http_client_default():
    client = HttpClient()
    assert client.base_url == "http://localhost:3000"
    assert client.api_key is None
    assert client.timeout == 120


def test_http_client_custom():
    client = HttpClient(
        base_url="http://example.com:8080/",
        api_key="sk-test",
        timeout=30,
    )
    assert client.base_url == "http://example.com:8080"
    assert client.api_key == "sk-test"


def test_http_client_headers_no_key():
    client = HttpClient()
    headers = client._headers()
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" not in headers


def test_http_client_headers_with_key():
    client = HttpClient(api_key="sk-test-123")
    headers = client._headers()
    assert headers["Authorization"] == "Bearer sk-test-123"


# ============================================================================
# Type Builders
# ============================================================================


def test_chat_message_user():
    msg = ChatMessage.user("Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_chat_message_system():
    msg = ChatMessage.system("Be concise")
    assert msg.role == "system"
    assert msg.content == "Be concise"


def test_chat_message_assistant():
    msg = ChatMessage.assistant("Sure!")
    assert msg.role == "assistant"
    assert msg.content == "Sure!"


def test_chat_request_quick():
    req = ChatRequest.quick("qwen2.5:7b", "What is Rust?")
    assert req.model == "qwen2.5:7b"
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "What is Rust?"
    assert req.temperature is None
    assert req.stream is None


def test_chat_request_serialization():
    req = ChatRequest(
        model="gpt-4",
        messages=[ChatMessage.user("Hi")],
        temperature=0.7,
        max_tokens=100,
    )
    data = req.model_dump(exclude_none=True)
    assert data["model"] == "gpt-4"
    assert data["temperature"] == 0.7
    assert data["max_tokens"] == 100
    assert "stream" not in data  # None excluded


# ============================================================================
# Response Deserialization
# ============================================================================


def test_chat_response_parsing():
    raw = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1709900000,
        "model": "qwen2.5:7b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    resp = ChatResponse.model_validate(raw)
    assert resp.id == "chatcmpl-123"
    assert resp.choices[0].message.content == "Hello!"
    assert resp.usage.total_tokens == 15


def test_model_list_parsing():
    raw = {
        "object": "list",
        "data": [
            {
                "id": "qwen2.5:7b",
                "object": "model",
                "created": 1709000000,
                "owned_by": "ollama",
            },
            {
                "id": "llama3:8b",
                "object": "model",
                "created": 1709000001,
                "owned_by": "ollama",
            },
        ],
    }
    models = ModelList.model_validate(raw)
    assert len(models.data) == 2
    assert models.data[0].id == "qwen2.5:7b"


def test_embedding_response_parsing():
    raw = {
        "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "bge-small-en",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    resp = EmbeddingResponse.model_validate(raw)
    assert len(resp.data[0].embedding) == 3
    assert resp.model == "bge-small-en"


def test_rag_upload_response_parsing():
    raw = {"document_id": "doc-123", "chunks_count": 5}
    resp = RagUploadResponse.model_validate(raw)
    assert resp.document_id == "doc-123"
    assert resp.chunks_count == 5


def test_rag_query_response_parsing():
    raw = {
        "answer": "Rust is a systems language",
        "sources": [
            {"document_id": "doc-1", "content": "Rust...", "score": 0.95}
        ],
    }
    resp = RagQueryResponse.model_validate(raw)
    assert resp.answer == "Rust is a systems language"
    assert len(resp.sources) == 1
    assert abs(resp.sources[0].score - 0.95) < 0.001


def test_orchestrator_response_parsing():
    raw = {"final_output": "The code looks good"}
    resp = OrchestratorResponse.model_validate(raw)
    assert resp.final_output == "The code looks good"


# ============================================================================
# JSON round-trip
# ============================================================================


def test_chat_request_json_roundtrip():
    req = ChatRequest.quick("model-1", "test")
    json_str = req.model_dump_json(exclude_none=True)
    parsed = json.loads(json_str)
    assert parsed["model"] == "model-1"
    assert parsed["messages"][0]["content"] == "test"
