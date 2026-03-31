"""Low-level HTTP transport with retry and exponential backoff."""

from __future__ import annotations

import logging
import socket
import time
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .exceptions import AgentConnectionError, AgentTimeoutError, ServerError, WauldoError

logger = logging.getLogger("wauldo")


class HttpTransport:
    """Handles HTTP requests with retry, backoff, and event hooks."""

    def __init__(
        self,
        timeout: int,
        max_retries: int,
        retry_backoff: float,
        headers_fn: Callable[[], dict[str, str]],
        on_request: Optional[Callable[[str, str], None]] = None,
        on_response: Optional[Callable[[int, float], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._headers_fn = headers_fn
        self._on_request = on_request
        self._on_response = on_response
        self._on_error = on_error

    def execute(
        self, method: str, url: str, data: Optional[bytes] = None, timeout_ms: Optional[int] = None,
    ) -> bytes:
        """Execute HTTP request with exponential backoff on retryable errors."""
        effective_timeout = timeout_ms / 1000.0 if timeout_ms else self.timeout
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            if self._on_request:
                self._on_request(method, url)
            logger.debug("Request: %s %s", method, url)
            start = time.monotonic()
            req = Request(url, data=data, headers=self._headers_fn(), method=method)
            try:
                with urlopen(req, timeout=effective_timeout) as resp:
                    body = resp.read()
                    elapsed = (time.monotonic() - start) * 1000
                    logger.debug("Response: %s in %.0fms", resp.status, elapsed)
                    if self._on_response:
                        self._on_response(resp.status, elapsed)
                    return bytes(body)
            except HTTPError as e:
                last_error = e
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    delay = self._backoff_delay(attempt, e)
                    logger.warning(
                        "Retry %d/%d: HTTP %d — backoff %.1fs",
                        attempt + 1, self.max_retries, e.code, delay,
                    )
                    time.sleep(delay)
                    continue
                if self._on_error:
                    self._on_error(e)
                body_text = e.read().decode() if e.fp else ""
                import json as _json
                try:
                    msg = _json.loads(body_text).get("error", {}).get("message", body_text)
                except Exception:
                    msg = body_text
                raise ServerError(f"HTTP {e.code}: {msg}", code=e.code) from e
            except socket.timeout as e:
                if self._on_error:
                    self._on_error(e)
                raise AgentTimeoutError(f"Request timed out: {e}") from e
            except URLError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Retry %d/%d: connection error — backoff %.1fs",
                        attempt + 1, self.max_retries, delay,
                    )
                    time.sleep(delay)
                    continue
                if self._on_error:
                    self._on_error(e)
                if isinstance(e.reason, socket.timeout):
                    raise AgentTimeoutError(f"Request timed out: {e}") from e
                raise AgentConnectionError(f"Request failed: {e}") from e
            except Exception as e:
                if self._on_error:
                    self._on_error(e)
                raise WauldoError(f"Request failed: {e}") from e
        raise AgentConnectionError(
            f"Request failed after {self.max_retries} retries: {last_error}"
        )

    def _backoff_delay(self, attempt: int, error: Optional[HTTPError] = None) -> float:
        """Calculate backoff delay, respecting Retry-After header on 429."""
        if error and error.code == 429:
            retry_after = error.headers.get("Retry-After") if error.headers else None
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return float(self.retry_backoff * (2 ** attempt))
