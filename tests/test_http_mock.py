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
