"""
Wauldo Python SDK

Two client interfaces:
- AgentClient / AsyncAgentClient — MCP server (stdio JSON-RPC) for reasoning, planning, tools
- HttpClient — REST API (OpenAI-compatible) for chat, embeddings, RAG, orchestrator
"""

from .client import AgentClient, AsyncAgentClient
from .conversation import Conversation
from .exceptions import (
    AgentConnectionError,
    AgentTimeoutError,
    ServerError,
    ToolNotFoundError,
    ValidationError,
    WauldoError,
)
from .async_client import AsyncHttpClient
from .http_client import HttpClient
from .http_types import (
    ChatMessage as HttpChatMessage,
)
from .http_types import (
    ChatRequest,
    ChatResponse,
    CitationDetail,
    ClaimResult,
    EmbeddingResponse,
    FactCheckResponse,
    SourceChunk,
    VerifyCitationResponse,
    ModelList,
    OrchestratorResponse,
    RagAuditInfo,
    RagQueryResponse,
    RagSource,
    RagUploadResponse,
    UploadFileResponse,
    DocumentQuality,
)
from .mock_client import MockHttpClient
from .models import (
    Chunk,
    ChunkResult,
    Concept,
    ConceptResult,
    GraphNode,
    KnowledgeGraphResult,
    PlanResult,
    PlanStep,
    ReasoningResult,
    RetrievalResult,
    ToolDefinition,
    ToolsListResult,
)

__version__ = "0.1.0"
__all__ = [
    # MCP Clients
    "AgentClient",
    "AsyncAgentClient",
    # HTTP Client
    "AsyncHttpClient",
    "HttpClient",
    "MockHttpClient",
    "Conversation",
    "ChatRequest",
    "ChatResponse",
    "HttpChatMessage",
    "EmbeddingResponse",
    "ModelList",
    "OrchestratorResponse",
    "RagAuditInfo",
    "RagQueryResponse",
    "RagSource",
    "RagUploadResponse",
    "UploadFileResponse",
    "DocumentQuality",
    "ClaimResult",
    "FactCheckResponse",
    "CitationDetail",
    "SourceChunk",
    "VerifyCitationResponse",
    # Exceptions
    "WauldoError",
    "AgentConnectionError",
    "AgentTimeoutError",
    "ServerError",
    "ToolNotFoundError",
    "ValidationError",
    # MCP Models
    "ReasoningResult",
    "ConceptResult",
    "Concept",
    "ChunkResult",
    "Chunk",
    "RetrievalResult",
    "KnowledgeGraphResult",
    "PlanResult",
    "PlanStep",
    "GraphNode",
    "ToolDefinition",
    "ToolsListResult",
]
