"""Mock HTTP tests for HttpClient — validates real HTTP behavior via urllib patches."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from wauldo.exceptions import WauldoError
from wauldo.http_client import HttpClient
from wauldo.http_types import ChatRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_urlopen_response(body: dict, status: int = 200) -> MagicMock:
    """Create a mock urllib response context-manager with JSON body."""
    raw = json.dumps(body).encode()
    resp = MagicMock()
    resp.read.return_value = raw
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


_CHAT_BODY = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from mock!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
}


# ---------------------------------------------------------------------------
# 1. test_chat_returns_response
# ---------------------------------------------------------------------------


@patch("wauldo.http_transport.urlopen")
def test_chat_returns_response(mock_urlopen: MagicMock) -> None:
    mock_urlopen.return_value = _mock_urlopen_response(_CHAT_BODY)

    client = HttpClient(base_url="http://fake:3000", max_retries=1)
    req = ChatRequest.quick("test-model", "Hi")
    resp = client.chat(req)

    assert resp.id == "chatcmpl-mock"
    assert resp.choices[0].message.content == "Hello from mock!"
    assert resp.usage.total_tokens == 12
    mock_urlopen.assert_called_once()


# ---------------------------------------------------------------------------
# 2. test_chat_simple_shortcut
# ---------------------------------------------------------------------------


@patch("wauldo.http_transport.urlopen")
def test_chat_simple_shortcut(mock_urlopen: MagicMock) -> None:
    mock_urlopen.return_value = _mock_urlopen_response(_CHAT_BODY)

    client = HttpClient(base_url="http://fake:3000", max_retries=1)
    content = client.chat_simple("test-model", "Hi")

    assert content == "Hello from mock!"


# ---------------------------------------------------------------------------
# 3. test_chat_stream_yields_content
# ---------------------------------------------------------------------------


@patch("wauldo.http_streaming.urlopen")
def test_chat_stream_yields_content(mock_urlopen: MagicMock) -> None:
    sse_payload = (
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n'
        b"\n"
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n'
        b"\n"
        b"data: [DONE]\n"
    )
    mock_resp = BytesIO(sse_payload)
    mock_urlopen.return_value = mock_resp

    client = HttpClient(base_url="http://fake:3000", max_retries=1)
    req = ChatRequest.quick("test-model", "Hi")
    chunks = list(client.chat_stream(req))

    assert chunks == ["hello", " world"]


# ---------------------------------------------------------------------------
# 4. test_chat_stream_skips_malformed_json
# ---------------------------------------------------------------------------


@patch("wauldo.http_streaming.urlopen")
def test_chat_stream_skips_malformed_json(mock_urlopen: MagicMock) -> None:
    sse_payload = (
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
        b"\n"
        b"data: {INVALID JSON}\n"
        b"\n"
        b'data: {"choices":[{"delta":{"content":"!"}}]}\n'
        b"\n"
        b"data: [DONE]\n"
    )
    mock_resp = BytesIO(sse_payload)
    mock_urlopen.return_value = mock_resp

    client = HttpClient(base_url="http://fake:3000", max_retries=1)
    req = ChatRequest.quick("test-model", "Hi")
    chunks = list(client.chat_stream(req))

    assert chunks == ["ok", "!"]


# ---------------------------------------------------------------------------
# 5. test_embeddings
# ---------------------------------------------------------------------------


@patch("wauldo.http_transport.urlopen")
def test_embeddings(mock_urlopen: MagicMock) -> None:
    body = {
        "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "bge-small-en",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    mock_urlopen.return_value = _mock_urlopen_response(body)

    client = HttpClient(base_url="http://fake:3000", max_retries=1)
    resp = client.embeddings("Hello world", model="bge-small-en")

    assert len(resp.data[0].embedding) == 3
    assert resp.model == "bge-small-en"


# ---------------------------------------------------------------------------
# 6. test_rag_upload
# ---------------------------------------------------------------------------


@patch("wauldo.http_transport.urlopen")
def test_rag_upload(mock_urlopen: MagicMock) -> None:
    body = {"document_id": "doc-abc", "chunks_count": 7}
    mock_urlopen.return_value = _mock_urlopen_response(body)

    client = HttpClient(base_url="http://fake:3000", max_retries=1)
    resp = client.rag_upload(content="Some document text", filename="doc.txt")

    assert resp.document_id == "doc-abc"
    assert resp.chunks_count == 7


# ---------------------------------------------------------------------------
# 7. test_rag_query
# ---------------------------------------------------------------------------


@patch("wauldo.http_transport.urlopen")
def test_rag_query(mock_urlopen: MagicMock) -> None:
    body = {
        "answer": "Rust is a systems language",
        "sources": [
            {"document_id": "doc-1", "content": "Rust...", "score": 0.95}
        ],
    }
    mock_urlopen.return_value = _mock_urlopen_response(body)

    client = HttpClient(base_url="http://fake:3000", max_retries=1)
    resp = client.rag_query("What is Rust?", top_k=3)

    assert resp.answer == "Rust is a systems language"
    assert len(resp.sources) == 1
    assert abs(resp.sources[0].score - 0.95) < 0.001


# ---------------------------------------------------------------------------
# 8. test_http_error_raises
# ---------------------------------------------------------------------------


@patch("wauldo.http_transport.urlopen")
def test_http_error_raises(mock_urlopen: MagicMock) -> None:
    from urllib.error import HTTPError

    err_resp = BytesIO(b"Internal Server Error")
    exc = HTTPError(
        url="http://fake:3000/v1/models",
        code=500,
        msg="Internal Server Error",
        hdrs=None,  # type: ignore[arg-type]
        fp=err_resp,
    )
    mock_urlopen.side_effect = exc

    client = HttpClient(base_url="http://fake:3000", max_retries=1)

    with pytest.raises(WauldoError, match="HTTP 500"):
        client.list_models()


# ---------------------------------------------------------------------------
# 9. test_models_list
# ---------------------------------------------------------------------------


@patch("wauldo.http_transport.urlopen")
def test_models_list(mock_urlopen: MagicMock) -> None:
    body = {
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
    mock_urlopen.return_value = _mock_urlopen_response(body)

    client = HttpClient(base_url="http://fake:3000", max_retries=1)
    models = client.list_models()

    assert len(models.data) == 2
    assert models.data[0].id == "qwen2.5:7b"
    assert models.data[1].id == "llama3:8b"


# ---------------------------------------------------------------------------
# Guard (fact-check) tests
# ---------------------------------------------------------------------------

_GUARD_REJECTED = {
    "verdict": "rejected",
    "action": "block",
    "hallucination_rate": 1.0,
    "mode": "lexical",
    "total_claims": 1,
    "supported_claims": 0,
    "confidence": 0.0,
    "claims": [
        {
            "text": "Returns are accepted within 60 days of purchase",
            "claim_type": "Fact",
            "supported": False,
            "confidence": 0.3,
            "confidence_label": "very_low",
            "verdict": "rejected",
            "action": "block",
            "reason": "numerical_mismatch",
            "evidence": "Our return policy allows returns within 14 days of purchase.",
        }
    ],
    "processing_time_ms": 0,
}

_GUARD_VERIFIED = {
    "verdict": "verified",
    "action": "allow",
    "hallucination_rate": 0.0,
    "mode": "lexical",
    "total_claims": 1,
    "supported_claims": 1,
    "confidence": 0.85,
    "claims": [
        {
            "text": "Rust was released in 2010",
            "claim_type": "Fact",
            "supported": True,
            "confidence": 0.85,
            "confidence_label": "high",
            "verdict": "verified",
            "action": "allow",
            "reason": None,
            "evidence": "Rust was released in 2010 by Mozilla Research.",
        }
    ],
    "processing_time_ms": 0,
}


@patch("wauldo.http_transport.urlopen")
def test_guard_catches_numerical_mismatch(mock_urlopen: MagicMock) -> None:
    """Guard blocks '60 days' when source says '14 days'."""
    mock_urlopen.return_value = _mock_urlopen_response(_GUARD_REJECTED)
    client = HttpClient(base_url="http://fake:3000", max_retries=1)

    result = client.guard(
        text="Returns are accepted within 60 days of purchase",
        source_context="Our return policy allows returns within 14 days of purchase.",
    )

    assert result.verdict == "rejected"
    assert result.action == "block"
    assert result.is_blocked is True
    assert result.is_safe is False
    assert result.hallucination_rate == 1.0
    assert result.total_claims == 1
    assert result.supported_claims == 0
    assert result.claims[0].reason == "numerical_mismatch"
    assert result.claims[0].supported is False


@patch("wauldo.http_transport.urlopen")
def test_guard_verifies_correct_claim(mock_urlopen: MagicMock) -> None:
    """Guard passes through correct claims."""
    mock_urlopen.return_value = _mock_urlopen_response(_GUARD_VERIFIED)
    client = HttpClient(base_url="http://fake:3000", max_retries=1)

    result = client.guard(
        text="Rust was released in 2010",
        source_context="Rust was released in 2010 by Mozilla Research.",
    )

    assert result.verdict == "verified"
    assert result.action == "allow"
    assert result.is_safe is True
    assert result.is_blocked is False
    assert result.hallucination_rate == 0.0
    assert result.confidence == 0.85
    assert result.claims[0].supported is True


def test_guard_rejects_empty_text() -> None:
    """Guard raises ValidationError on empty text."""
    client = HttpClient(base_url="http://fake:3000", max_retries=1)
    with pytest.raises(Exception, match="empty"):
        client.guard(text="", source_context="some context")


def test_guard_rejects_empty_context() -> None:
    """Guard raises ValidationError on empty source_context."""
    client = HttpClient(base_url="http://fake:3000", max_retries=1)
    with pytest.raises(Exception, match="empty"):
        client.guard(text="some claim", source_context="")


@patch("wauldo.http_transport.urlopen")
def test_guard_mode_parameter(mock_urlopen: MagicMock) -> None:
    """Guard sends the correct mode parameter."""
    mock_urlopen.return_value = _mock_urlopen_response(_GUARD_VERIFIED)
    client = HttpClient(base_url="http://fake:3000", max_retries=1)

    client.guard(text="claim", source_context="ctx", mode="hybrid")

    call_args = mock_urlopen.call_args
    request_obj = call_args[0][0]
    sent_body = json.loads(request_obj.data.decode())
    assert sent_body["mode"] == "hybrid"
    assert sent_body["text"] == "claim"
    assert sent_body["source_context"] == "ctx"
