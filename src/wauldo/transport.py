"""
Transport layer for MCP communication.
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import cast, Any, Dict, Optional

from .exceptions import AgentConnectionError, AgentTimeoutError, ServerError


def _find_mcp_server() -> str:
    """Find MCP server binary in common locations."""
    search_paths = [
        Path.cwd() / "target" / "release" / "wauldo-mcp",
        Path.cwd() / "target" / "debug" / "wauldo-mcp",
        Path.cwd().parent / "target" / "release" / "wauldo-mcp",
        Path.home() / ".cargo" / "bin" / "wauldo-mcp",
    ]
    for path in search_paths:
        if path.exists():
            return str(path)
    raise AgentConnectionError(
        "MCP server binary not found. Please provide server_path or install with 'cargo install'."
    )


class StdioTransport:
    """Synchronous stdio transport for MCP server communication."""

    def __init__(
        self,
        server_path: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize transport.

        Args:
            server_path: Path to MCP server binary. If None, searches in common locations.
            timeout: Default timeout for operations in seconds.
        """
        self._server_path = server_path
        self.timeout = timeout
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._request_id = 0

    @property
    def server_path(self) -> str:
        """Get server path, finding it lazily if needed."""
        if self._server_path is None:
            self._server_path = self._find_server()
        return self._server_path

    def _find_server(self) -> str:
        return _find_mcp_server()

    def connect(self) -> None:
        """Start MCP server process."""
        if self._process is not None:
            return

        try:
            self._process = subprocess.Popen(
                [self.server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise AgentConnectionError(f"Server binary not found: {self.server_path}") from e
        except Exception as e:
            raise AgentConnectionError(f"Failed to start server: {e}") from e

        # Initialize MCP connection — cleanup process on failure
        try:
            self._initialize()
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        """Stop MCP server process."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def _initialize(self) -> Dict[str, Any]:
        """Send MCP initialize request."""
        return self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "wauldo-python", "version": "0.1.0"},
            },
        )

    def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send JSON-RPC request and wait for response.

        Args:
            method: RPC method name
            params: Method parameters
            timeout: Request timeout in seconds

        Returns:
            Response result

        Raises:
            AgentConnectionError: If not connected
            ServerError: If server returns error
            AgentTimeoutError: If request times out
        """
        if self._process is None:
            raise AgentConnectionError("Not connected. Call connect() first.")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        request_data = json.dumps(request) + "\n"

        try:
            if self._process.stdin is None or self._process.stdout is None:
                raise AgentConnectionError("MCP server process has no stdin/stdout")

            self._process.stdin.write(request_data.encode())
            self._process.stdin.flush()

            # Read response with timeout (cross-platform: threading for Windows compat)
            import threading
            timeout_val = timeout or self.timeout
            response_line_holder: list[bytes] = [b""]
            error_holder: list[Optional[Exception]] = [None]

            def _read_line() -> None:
                try:
                    response_line_holder[0] = self._process.stdout.readline()  # type: ignore[union-attr]
                except Exception as exc:
                    error_holder[0] = exc

            reader_thread = threading.Thread(target=_read_line, daemon=True)
            reader_thread.start()
            reader_thread.join(timeout=timeout_val)

            if reader_thread.is_alive():
                raise AgentTimeoutError(f"Request timed out after {timeout_val}s", timeout_val)
            if error_holder[0] is not None:
                raise AgentConnectionError(f"Read error: {error_holder[0]}") from error_holder[0]

            response_line = response_line_holder[0]
            if not response_line:
                raise AgentConnectionError("Server closed connection")

            response = json.loads(response_line.decode())

        except json.JSONDecodeError as e:
            raise ServerError(f"Invalid JSON response: {e}") from e
        except Exception as e:
            if isinstance(e, (AgentConnectionError, ServerError, AgentTimeoutError)):
                raise
            raise AgentConnectionError(f"Communication error: {e}") from e

        # Check for error response
        if "error" in response:
            error = response["error"]
            raise ServerError(
                error.get("message", "Unknown error"),
                code=error.get("code"),
                data=error.get("data"),
            )

        result: dict[str, Any] = response.get("result", {})
        return result

    def __enter__(self) -> "StdioTransport":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()


class AsyncStdioTransport:
    """Asynchronous stdio transport for MCP server communication."""

    def __init__(
        self,
        server_path: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize async transport.

        Args:
            server_path: Path to MCP server binary.
            timeout: Default timeout for operations in seconds.
        """
        self._server_path = server_path
        self.timeout = timeout
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._lock = asyncio.Lock()

    @property
    def server_path(self) -> str:
        """Get server path, finding it lazily if needed."""
        if self._server_path is None:
            self._server_path = self._find_server()
        return self._server_path

    def _find_server(self) -> str:
        return _find_mcp_server()

    async def connect(self) -> None:
        """Start MCP server process."""
        if self._process is not None:
            return

        try:
            self._process = await asyncio.create_subprocess_exec(
                self.server_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise AgentConnectionError(f"Server binary not found: {self.server_path}") from e
        except Exception as e:
            raise AgentConnectionError(f"Failed to start server: {e}") from e

        # Initialize MCP connection — cleanup process on failure
        try:
            await self._initialize()
        except Exception:
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Stop MCP server process."""
        if self._process is not None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None

    async def _initialize(self) -> Dict[str, Any]:
        """Send MCP initialize request."""
        return await self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "wauldo-python", "version": "0.1.0"},
            },
        )

    async def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send JSON-RPC request and wait for response.

        Args:
            method: RPC method name
            params: Method parameters
            timeout: Request timeout in seconds

        Returns:
            Response result
        """
        if self._process is None:
            raise AgentConnectionError("Not connected. Call connect() first.")

        async with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
            }
            if params:
                request["params"] = params

            request_data = json.dumps(request) + "\n"

            try:
                if self._process.stdin is None or self._process.stdout is None:
                    raise AgentConnectionError("Server process stdin/stdout not available")

                self._process.stdin.write(request_data.encode())
                await self._process.stdin.drain()

                # Read response with timeout
                timeout_val = timeout or self.timeout
                response_line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=timeout_val,
                )

                if not response_line:
                    raise AgentConnectionError("Server closed connection")

                response = json.loads(response_line.decode())

            except asyncio.TimeoutError as e:
                raise AgentTimeoutError(f"Request timed out after {timeout_val}s", timeout_val) from e
            except json.JSONDecodeError as e:
                raise ServerError(f"Invalid JSON response: {e}") from e
            except Exception as e:
                if isinstance(e, (AgentConnectionError, ServerError, AgentTimeoutError)):
                    raise
                raise AgentConnectionError(f"Communication error: {e}") from e

        # Check for error response
        if "error" in response:
            error = response["error"]
            raise ServerError(
                error.get("message", "Unknown error"),
                code=error.get("code"),
                data=error.get("data"),
            )

        result: dict[str, Any] = response.get("result", {})
        return result

    async def __aenter__(self) -> "AsyncStdioTransport":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()
