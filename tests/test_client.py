"""Tests for SDK client."""

import pytest
from wauldo.client import AgentClient, AsyncAgentClient
from wauldo.exceptions import ValidationError


class TestAgentClientValidation:
    """Test client input validation without server."""

    def test_reason_empty_problem(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.reason("")
        assert exc.value.field == "problem"

    def test_reason_invalid_depth(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.reason("test", depth=0)
        assert exc.value.field == "depth"

        with pytest.raises(ValidationError) as exc:
            client.reason("test", depth=11)
        assert exc.value.field == "depth"

    def test_reason_invalid_branches(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.reason("test", branches=0)
        assert exc.value.field == "branches"

    def test_extract_concepts_empty(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.extract_concepts("")
        assert exc.value.field == "text"

    def test_chunk_document_empty(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.chunk_document("")
        assert exc.value.field == "content"

    def test_retrieve_context_empty(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.retrieve_context("")
        assert exc.value.field == "query"

    def test_summarize_empty(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.summarize("")
        assert exc.value.field == "content"

    def test_search_knowledge_empty(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.search_knowledge("")
        assert exc.value.field == "query"

    def test_add_to_knowledge_empty(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.add_to_knowledge("")
        assert exc.value.field == "text"

    def test_plan_task_empty(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.plan_task("")
        assert exc.value.field == "task"

    def test_plan_task_invalid_max_steps(self):
        client = AgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            client.plan_task("test", max_steps=0)
        assert exc.value.field == "max_steps"

        with pytest.raises(ValidationError) as exc:
            client.plan_task("test", max_steps=21)
        assert exc.value.field == "max_steps"


class TestAsyncAgentClientValidation:
    """Test async client input validation without server."""

    @pytest.mark.asyncio
    async def test_reason_empty_problem(self):
        client = AsyncAgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            await client.reason("")
        assert exc.value.field == "problem"

    @pytest.mark.asyncio
    async def test_extract_concepts_empty(self):
        client = AsyncAgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            await client.extract_concepts("")
        assert exc.value.field == "text"

    @pytest.mark.asyncio
    async def test_plan_task_empty(self):
        client = AsyncAgentClient(auto_connect=False)
        with pytest.raises(ValidationError) as exc:
            await client.plan_task("")
        assert exc.value.field == "task"


class TestClientInitialization:
    """Test client initialization."""

    def test_default_initialization(self):
        client = AgentClient(auto_connect=False)
        assert client._auto_connect is False
        assert client._connected is False

    def test_custom_timeout(self):
        client = AgentClient(timeout=60.0, auto_connect=False)
        assert client._transport.timeout == 60.0

    def test_async_default_initialization(self):
        client = AsyncAgentClient(auto_connect=False)
        assert client._auto_connect is False
        assert client._connected is False
