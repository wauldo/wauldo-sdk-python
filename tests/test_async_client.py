import pytest

from wauldo.async_client import AsyncHttpClient
from wauldo.exceptions import ValidationError
from wauldo.http_types import ChatRequest, ChatMessage


def test_async_client_default():
    client = AsyncHttpClient()
    assert client.base_url == "http://localhost:3000"
    assert client.api_key is None
    assert client.timeout == 120


def test_async_client_custom():
    client = AsyncHttpClient(
        base_url="http://example.com:8080/",
        api_key="sk-test",
        timeout=30,
    )
    assert client.base_url == "http://example.com:8080"
    assert client.api_key == "sk-test"


def test_async_client_headers_no_key():
    client = AsyncHttpClient()
    headers = client._headers()
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" not in headers


def test_async_client_headers_with_key():
    client = AsyncHttpClient(api_key="sk-abc")
    headers = client._headers()
    assert headers["Authorization"] == "Bearer sk-abc"


def test_async_client_has_methods():
    client = AsyncHttpClient()
    for name in [
        "list_models", "chat", "chat_simple", "chat_stream",
        "embeddings", "rag_upload", "upload_file", "rag_query",
        "orchestrate", "orchestrate_parallel", "fact_check",
        "verify_citation", "conversation", "rag_ask",
        "get_insights_async", "get_analytics_async", "get_analytics_traffic_async",
    ]:
        assert hasattr(client, name), f"missing method: {name}"
        assert callable(getattr(client, name))


@pytest.mark.asyncio
async def test_chat_empty_messages_raises():
    client = AsyncHttpClient()
    req = ChatRequest(model="test", messages=[])
    with pytest.raises(ValueError, match="messages cannot be empty"):
        await client.chat(req)


@pytest.mark.asyncio
async def test_rag_query_empty_raises():
    client = AsyncHttpClient()
    with pytest.raises(ValidationError):
        await client.rag_query("   ")


@pytest.mark.asyncio
async def test_rag_query_topk_out_of_range():
    client = AsyncHttpClient()
    with pytest.raises(ValidationError):
        await client.rag_query("test", top_k=0)


@pytest.mark.asyncio
async def test_rag_upload_empty_raises():
    client = AsyncHttpClient()
    with pytest.raises(ValidationError):
        await client.rag_upload("   ")


@pytest.mark.asyncio
async def test_context_manager():
    async with AsyncHttpClient() as client:
        assert client.base_url == "http://localhost:3000"
