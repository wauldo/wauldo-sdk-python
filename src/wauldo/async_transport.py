from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import Callable, Optional

from .exceptions import AgentConnectionError, AgentTimeoutError, ServerError, WauldoError

logger = logging.getLogger("wauldo")


class AsyncHttpTransport:

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
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def execute(
        self, method: str, url: str, data: Optional[bytes] = None, timeout_ms: Optional[int] = None,
    ) -> bytes:
        import aiohttp

        effective_timeout = timeout_ms / 1000.0 if timeout_ms else self.timeout
        ct = aiohttp.ClientTimeout(total=effective_timeout)
        last_error: Optional[Exception] = None
        session = await self._get_session()

        for attempt in range(self.max_retries):
            if self._on_request:
                self._on_request(method, url)
            logger.debug("Request: %s %s", method, url)

            import time
            start = time.monotonic()
            try:
                async with session.request(
                    method, url, data=data, headers=self._headers_fn(), timeout=ct,
                ) as resp:
                    body = await resp.read()
                    elapsed = (time.monotonic() - start) * 1000
                    logger.debug("Response: %s in %.0fms", resp.status, elapsed)
                    if self._on_response:
                        self._on_response(resp.status, elapsed)

                    if resp.status >= 400:
                        if resp.status in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                            delay = self._backoff_delay(attempt, resp.status, resp.headers)
                            logger.warning(
                                "Retry %d/%d: HTTP %d — backoff %.1fs",
                                attempt + 1, self.max_retries, resp.status, delay,
                            )
                            await asyncio.sleep(delay)
                            continue

                        body_text = body.decode(errors="replace")
                        try:
                            msg = _json.loads(body_text).get("error", {}).get("message", body_text)
                        except Exception:
                            msg = body_text
                        err = ServerError(f"HTTP {resp.status}: {msg}", code=resp.status)
                        if self._on_error:
                            self._on_error(err)
                        raise err

                    return bytes(body)

            except ServerError:
                raise
            except asyncio.TimeoutError as e:
                if self._on_error:
                    self._on_error(e)
                raise AgentTimeoutError(f"Request timed out: {e}") from e
            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Retry %d/%d: connection error — backoff %.1fs",
                        attempt + 1, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                if self._on_error:
                    self._on_error(e)
                raise AgentConnectionError(f"Request failed: {e}") from e
            except Exception as e:
                if self._on_error:
                    self._on_error(e)
                raise WauldoError(f"Request failed: {e}") from e

        raise AgentConnectionError(
            f"Request failed after {self.max_retries} retries: {last_error}"
        )

    def _backoff_delay(self, attempt: int, status: Optional[int] = None, headers=None) -> float:
        if status == 429 and headers:
            retry_after = headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return float(self.retry_backoff * (2 ** attempt))
