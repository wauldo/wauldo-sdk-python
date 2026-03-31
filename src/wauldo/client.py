"""
Wauldo client implementations.
"""

import json
from typing import cast, Any, Dict, List, Literal, Optional

from .exceptions import AgentConnectionError, ValidationError
from .models import (
    Chunk,
    ChunkResult,
    ConceptResult,
    KnowledgeGraphResult,
    PlanResult,
    ReasoningResult,
    RetrievalResult,
    ToolDefinition,
)
from .transport import AsyncStdioTransport, StdioTransport


def _parse_chunks(raw: str) -> list[Chunk]:
    """Parse chunk results from raw API response."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            raw_chunks = data.get("chunks", data.get("results", []))
            if isinstance(raw_chunks, list):
                return [
                    Chunk(
                        id=c.get("id", ""),
                        content=c.get("content", ""),
                        position=c.get("position", i),
                        priority=c.get("priority", "medium"),
                    )
                    for i, c in enumerate(raw_chunks)
                    if isinstance(c, dict)
                ]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return []


def _parse_retrieval_results(raw: str) -> list[Chunk]:
    """Parse retrieval results from raw API response."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            raw_results = data.get("results", data.get("chunks", []))
            if isinstance(raw_results, list):
                return [
                    Chunk(
                        id=c.get("id", ""),
                        content=c.get("content", ""),
                        position=c.get("position", i),
                        priority=c.get("priority", "medium"),
                    )
                    for i, c in enumerate(raw_results)
                    if isinstance(c, dict)
                ]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return []


class AgentClient:
    """
    Synchronous client for Wauldo MCP Server.

    Example:
        ```python
        with AgentClient() as client:
            result = client.reason("How to optimize this algorithm?")
            print(result.solution)
        ```
    """

    def __init__(
        self,
        server_path: Optional[str] = None,
        timeout: float = 30.0,
        auto_connect: bool = True,
    ) -> None:
        """
        Initialize client.

        Args:
            server_path: Path to MCP server binary
            timeout: Default timeout for operations
            auto_connect: Automatically connect on first operation
        """
        self._transport = StdioTransport(server_path, timeout)
        self._auto_connect = auto_connect
        self._connected = False

    def connect(self) -> "AgentClient":
        """Connect to MCP server."""
        self._transport.connect()
        self._connected = True
        return self

    def disconnect(self) -> None:
        """Disconnect from MCP server."""
        self._transport.disconnect()
        self._connected = False

    def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if not self._connected:
            if self._auto_connect:
                self.connect()
            else:
                raise AgentConnectionError("Not connected. Call connect() first.")

    def __enter__(self) -> "AgentClient":
        return self.connect()

    def __exit__(self, *args: Any) -> None:
        self.disconnect()

    # Tool discovery
    def list_tools(self) -> List[ToolDefinition]:
        """
        List all available tools.

        Returns:
            List of tool definitions
        """
        self._ensure_connected()
        result = self._transport.request("tools/list")
        return [ToolDefinition(**tool) for tool in result.get("tools", [])]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Call a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result content
        """
        self._ensure_connected()
        result = self._transport.request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        content = result.get("content", [])
        if content and isinstance(content[0], dict):
            text: str = content[0].get("text", "")
            return text
        return ""

    # Reasoning
    def reason(
        self,
        problem: str,
        depth: int = 3,
        branches: int = 3,
    ) -> ReasoningResult:
        """
        Perform Tree-of-Thought reasoning on a problem.

        Args:
            problem: The problem or question to reason about
            depth: Depth of the thought tree (1-10)
            branches: Number of branches at each level (1-10)

        Returns:
            ReasoningResult with solution and thought tree

        Example:
            ```python
            result = client.reason(
                "What's the best sorting algorithm for nearly sorted data?",
                depth=4,
                branches=3
            )
            print(result.solution)
            ```
        """
        if not problem.strip():
            raise ValidationError("Problem cannot be empty", field="problem")
        if not 1 <= depth <= 10:
            raise ValidationError("Depth must be between 1 and 10", field="depth")
        if not 1 <= branches <= 10:
            raise ValidationError("Branches must be between 1 and 10", field="branches")

        content = self.call_tool(
            "reason_tree_of_thought",
            {"problem": problem, "depth": depth, "branches": branches},
        )
        return ReasoningResult.from_content(content, problem, depth, branches)

    # Concept extraction
    def extract_concepts(
        self,
        text: str,
        source_type: Literal["text", "code"] = "text",
    ) -> ConceptResult:
        """
        Extract concepts from text or code.

        Args:
            text: The text or code to analyze
            source_type: Type of input ("text" or "code")

        Returns:
            ConceptResult with extracted concepts

        Example:
            ```python
            result = client.extract_concepts(
                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                source_type="code"
            )
            for concept in result.concepts:
                print(f"{concept.name}: {concept.weight}")
            ```
        """
        if not text.strip():
            raise ValidationError("Text cannot be empty", field="text")

        content = self.call_tool(
            "extract_concepts",
            {"text": text, "source_type": source_type},
        )
        return ConceptResult.from_content(content, source_type)

    # Long context management
    def chunk_document(
        self,
        content: str,
        chunk_size: int = 512,
    ) -> ChunkResult:
        """
        Split a document into manageable chunks.

        Args:
            content: Document content to chunk
            chunk_size: Target size for each chunk

        Returns:
            ChunkResult with list of chunks
        """
        if not content.strip():
            raise ValidationError("Content cannot be empty", field="content")

        result = self.call_tool(
            "manage_long_context",
            {"operation": "chunk", "content": content, "chunk_size": chunk_size},
        )
        chunks = _parse_chunks(result)
        return ChunkResult(chunks=chunks, total_chunks=len(chunks), raw_content=result)

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve relevant context for a query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            RetrievalResult with matching chunks
        """
        if not query.strip():
            raise ValidationError("Query cannot be empty", field="query")

        result = self.call_tool(
            "manage_long_context",
            {"operation": "retrieve", "query": query, "top_k": top_k},
        )
        results = _parse_retrieval_results(result)
        return RetrievalResult(query=query, results=results, raw_content=result)

    def summarize(self, content: str) -> str:
        """
        Summarize document content.

        Args:
            content: Content to summarize

        Returns:
            Summary text
        """
        if not content.strip():
            raise ValidationError("Content cannot be empty", field="content")

        return self.call_tool(
            "manage_long_context",
            {"operation": "summarize", "content": content},
        )

    # Knowledge graph
    def search_knowledge(
        self,
        query: str,
        limit: int = 10,
    ) -> KnowledgeGraphResult:
        """
        Search the knowledge graph.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            KnowledgeGraphResult with matching nodes
        """
        if not query.strip():
            raise ValidationError("Query cannot be empty", field="query")

        result = self.call_tool(
            "query_knowledge_graph",
            {"operation": "search", "query": query, "limit": limit},
        )
        return KnowledgeGraphResult(operation="search", raw_content=result)

    def add_to_knowledge(self, text: str) -> KnowledgeGraphResult:
        """
        Add concepts from text to knowledge graph.

        Args:
            text: Text to extract and add concepts from

        Returns:
            KnowledgeGraphResult with added nodes
        """
        if not text.strip():
            raise ValidationError("Text cannot be empty", field="text")

        result = self.call_tool(
            "query_knowledge_graph",
            {"operation": "add", "text": text},
        )
        return KnowledgeGraphResult(operation="add", raw_content=result)

    def knowledge_stats(self) -> KnowledgeGraphResult:
        """
        Get knowledge graph statistics.

        Returns:
            KnowledgeGraphResult with stats
        """
        result = self.call_tool(
            "query_knowledge_graph",
            {"operation": "stats"},
        )
        return KnowledgeGraphResult(operation="stats", raw_content=result)

    # Task planning
    def plan_task(
        self,
        task: str,
        context: str = "",
        max_steps: int = 10,
        detail_level: Literal["brief", "normal", "detailed"] = "normal",
    ) -> PlanResult:
        """
        Break down a task into actionable steps.

        Args:
            task: The task or goal to plan
            context: Additional context or constraints
            max_steps: Maximum number of steps
            detail_level: Level of detail for each step

        Returns:
            PlanResult with steps and effort estimates

        Example:
            ```python
            plan = client.plan_task(
                "Implement user authentication",
                context="Using JWT tokens",
                detail_level="detailed"
            )
            for step in plan.steps:
                print(f"{step.number}. {step.title}")
            ```
        """
        if not task.strip():
            raise ValidationError("Task cannot be empty", field="task")
        if not 1 <= max_steps <= 20:
            raise ValidationError("max_steps must be between 1 and 20", field="max_steps")

        content = self.call_tool(
            "plan_task",
            {
                "task": task,
                "context": context,
                "max_steps": max_steps,
                "detail_level": detail_level,
            },
        )
        return PlanResult.from_content(content, task)


class AsyncAgentClient:
    """
    Asynchronous client for Wauldo MCP Server.

    Example:
        ```python
        async with AsyncAgentClient() as client:
            result = await client.reason("How to optimize this algorithm?")
            print(result.solution)
        ```
    """

    def __init__(
        self,
        server_path: Optional[str] = None,
        timeout: float = 30.0,
        auto_connect: bool = True,
    ) -> None:
        """
        Initialize async client.

        Args:
            server_path: Path to MCP server binary
            timeout: Default timeout for operations
            auto_connect: Automatically connect on first operation
        """
        self._transport = AsyncStdioTransport(server_path, timeout)
        self._auto_connect = auto_connect
        self._connected = False

    async def connect(self) -> "AsyncAgentClient":
        """Connect to MCP server."""
        await self._transport.connect()
        self._connected = True
        return self

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        await self._transport.disconnect()
        self._connected = False

    async def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if not self._connected:
            if self._auto_connect:
                await self.connect()
            else:
                raise AgentConnectionError("Not connected. Call connect() first.")

    async def __aenter__(self) -> "AsyncAgentClient":
        return await self.connect()

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()

    # Tool discovery
    async def list_tools(self) -> List[ToolDefinition]:
        """List all available tools."""
        await self._ensure_connected()
        result = await self._transport.request("tools/list")
        return [ToolDefinition(**tool) for tool in result.get("tools", [])]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool by name."""
        await self._ensure_connected()
        result = await self._transport.request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        content = result.get("content", [])
        if content and isinstance(content[0], dict):
            text: str = content[0].get("text", "")
            return text
        return ""

    # Reasoning
    async def reason(
        self,
        problem: str,
        depth: int = 3,
        branches: int = 3,
    ) -> ReasoningResult:
        """Perform Tree-of-Thought reasoning on a problem."""
        if not problem.strip():
            raise ValidationError("Problem cannot be empty", field="problem")
        if not 1 <= depth <= 10:
            raise ValidationError("Depth must be between 1 and 10", field="depth")
        if not 1 <= branches <= 10:
            raise ValidationError("Branches must be between 1 and 10", field="branches")

        content = await self.call_tool(
            "reason_tree_of_thought",
            {"problem": problem, "depth": depth, "branches": branches},
        )
        return ReasoningResult.from_content(content, problem, depth, branches)

    # Concept extraction
    async def extract_concepts(
        self,
        text: str,
        source_type: Literal["text", "code"] = "text",
    ) -> ConceptResult:
        """Extract concepts from text or code."""
        if not text.strip():
            raise ValidationError("Text cannot be empty", field="text")

        content = await self.call_tool(
            "extract_concepts",
            {"text": text, "source_type": source_type},
        )
        return ConceptResult.from_content(content, source_type)

    # Long context management
    async def chunk_document(
        self,
        content: str,
        chunk_size: int = 512,
    ) -> ChunkResult:
        """Split a document into manageable chunks."""
        if not content.strip():
            raise ValidationError("Content cannot be empty", field="content")

        result = await self.call_tool(
            "manage_long_context",
            {"operation": "chunk", "content": content, "chunk_size": chunk_size},
        )
        chunks = _parse_chunks(result)
        return ChunkResult(chunks=chunks, total_chunks=len(chunks), raw_content=result)

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve relevant context for a query."""
        if not query.strip():
            raise ValidationError("Query cannot be empty", field="query")

        result = await self.call_tool(
            "manage_long_context",
            {"operation": "retrieve", "query": query, "top_k": top_k},
        )
        results = _parse_retrieval_results(result)
        return RetrievalResult(query=query, results=results, raw_content=result)

    async def summarize(self, content: str) -> str:
        """Summarize document content."""
        if not content.strip():
            raise ValidationError("Content cannot be empty", field="content")

        return await self.call_tool(
            "manage_long_context",
            {"operation": "summarize", "content": content},
        )

    # Knowledge graph
    async def search_knowledge(
        self,
        query: str,
        limit: int = 10,
    ) -> KnowledgeGraphResult:
        """Search the knowledge graph."""
        if not query.strip():
            raise ValidationError("Query cannot be empty", field="query")

        result = await self.call_tool(
            "query_knowledge_graph",
            {"operation": "search", "query": query, "limit": limit},
        )
        return KnowledgeGraphResult(operation="search", raw_content=result)

    async def add_to_knowledge(self, text: str) -> KnowledgeGraphResult:
        """Add concepts from text to knowledge graph."""
        if not text.strip():
            raise ValidationError("Text cannot be empty", field="text")

        result = await self.call_tool(
            "query_knowledge_graph",
            {"operation": "add", "text": text},
        )
        return KnowledgeGraphResult(operation="add", raw_content=result)

    async def knowledge_stats(self) -> KnowledgeGraphResult:
        """Get knowledge graph statistics."""
        result = await self.call_tool(
            "query_knowledge_graph",
            {"operation": "stats"},
        )
        return KnowledgeGraphResult(operation="stats", raw_content=result)

    # Task planning
    async def plan_task(
        self,
        task: str,
        context: str = "",
        max_steps: int = 10,
        detail_level: Literal["brief", "normal", "detailed"] = "normal",
    ) -> PlanResult:
        """Break down a task into actionable steps."""
        if not task.strip():
            raise ValidationError("Task cannot be empty", field="task")
        if not 1 <= max_steps <= 20:
            raise ValidationError("max_steps must be between 1 and 20", field="max_steps")

        content = await self.call_tool(
            "plan_task",
            {
                "task": task,
                "context": context,
                "max_steps": max_steps,
                "detail_level": detail_level,
            },
        )
        return PlanResult.from_content(content, task)
