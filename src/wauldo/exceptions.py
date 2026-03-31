"""
Custom exceptions for Wauldo SDK.
"""

from typing import Any, Optional


class WauldoError(Exception):
    """Base exception for all Wauldo errors."""

    def __init__(self, message: str, code: Optional[int] = None, data: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class AgentConnectionError(WauldoError):
    """Raised when connection to MCP server fails."""

    def __init__(self, message: str = "Failed to connect to MCP server"):
        super().__init__(message, code=-32000)


class ServerError(WauldoError):
    """Raised when server returns an error response."""

    pass


class ValidationError(WauldoError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, code=-32602)
        self.field = field


class AgentTimeoutError(WauldoError):
    """Raised when operation times out."""

    def __init__(self, message: str = "Operation timed out", timeout: Optional[float] = None):
        super().__init__(message, code=-32001)
        self.timeout = timeout


class ToolNotFoundError(WauldoError):
    """Raised when requested tool is not available."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool not found: {tool_name}", code=-32601)
        self.tool_name = tool_name
