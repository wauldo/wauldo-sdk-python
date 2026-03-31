"""SSE streaming helpers for the HTTP client."""

from __future__ import annotations

import codecs
import json
import logging
from collections.abc import Iterator
from typing import Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .exceptions import AgentConnectionError, ServerError

logger = logging.getLogger("wauldo")


def stream_chat_sse(
    url: str,
    data: bytes,
    headers: dict[str, str],
    timeout: int,
) -> Iterator[str]:
    """Open an SSE connection and yield content chunks.

    Args:
        url: Full URL to POST /v1/chat/completions.
        data: JSON-encoded request body.
        headers: HTTP headers including auth.
        timeout: Socket timeout in seconds.

    Yields:
        Content strings from each SSE delta event.
    """
    http_req = Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urlopen(http_req, timeout=timeout)
    except HTTPError as e:
        raise ServerError(f"HTTP {e.code}: {e.read().decode()}", code=e.code) from e
    except Exception as e:
        raise AgentConnectionError(f"Streaming request failed: {e}") from e

    try:
        yield from _parse_sse_stream(resp)
    finally:
        resp.close()


def _parse_sse_stream(resp: object) -> Iterator[str]:
    """Parse SSE lines from an HTTP response, yielding content deltas.

    Uses an incremental UTF-8 decoder to handle multi-byte characters
    split across network chunk boundaries.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    for raw_chunk in iter(lambda: resp.read(4096), b""):  # type: ignore[attr-defined]
        buffer += decoder.decode(raw_chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                return
            content = _extract_delta_content(payload)
            if content:
                yield content
    # Flush any remaining bytes in the decoder
    remaining = decoder.decode(b"", final=True)
    if remaining:
        buffer += remaining


def _extract_delta_content(payload: str) -> Optional[str]:
    """Extract content string from a single SSE JSON payload."""
    try:
        parsed = json.loads(payload)
        choices = parsed.get("choices", [])
        if choices:
            content: str | None = choices[0].get("delta", {}).get("content")
            return content
    except json.JSONDecodeError:
        logger.warning("Malformed SSE chunk skipped: %s", payload[:100])
    return None
