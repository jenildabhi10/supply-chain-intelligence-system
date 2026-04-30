"""
BaseClient — shared HTTP plumbing for all ingestion clients.

Every source-specific client subclasses this and gets:
  - Rate limiting (token-bucket)
  - Retry with exponential backoff (transient failures only)
  - Structured logging on every request
  - Graceful degradation: failed requests return None, not exceptions
    (callers decide whether to halt or continue)
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

# Transient errors that warrant a retry
_RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)


class RateLimiter:
    """
    Simple token-bucket rate limiter.
    Sleeps the calling thread to respect calls_per_second.
    """

    def __init__(self, calls_per_second: float) -> None:
        self._min_interval = 1.0 / max(calls_per_second, 0.001)
        self._last_call    = 0.0

    def acquire(self) -> None:
        now     = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


class BaseClient:
    """
    Reusable sync HTTP client.

    Subclass it and call self._get() / self._get_bytes() in your methods.
    All retries, rate-limiting, and logging are handled here.
    """

    # Subclasses override these
    BASE_URL:          str   = ""
    CALLS_PER_SECOND:  float = 2.0   # conservative default
    TIMEOUT_SECONDS:   float = 30.0
    MAX_RETRIES:       int   = 3

    def __init__(self) -> None:
        self._rate_limiter = RateLimiter(self.CALLS_PER_SECOND)
        self._http = httpx.Client(
            headers={
                "User-Agent": (
                    "SupplyChainIntelligence/1.0 "
                    "(research demo; contact: jenildabhi10@gmail.com)"
                ),
                "Accept": "application/json",
            },
            timeout=self.TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        self.log = structlog.get_logger(client=self.__class__.__name__)

    # ── Core request methods ───────────────────────────────────────────────────

    def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """
        GET request → parsed JSON dict (or None on failure).
        Applies rate limiting + retry automatically.
        Non-JSON responses raise ValueError which is caught here.
        """
        try:
            response = self._raw_get(url, params=params, headers=headers)
            return response.json()
        except RetryError as exc:
            self.log.error("get_failed_after_retries", url=url, error=str(exc))
            return None
        except httpx.HTTPStatusError as exc:
            self.log.warning(
                "http_error",
                url=url,
                status=exc.response.status_code,
                body=exc.response.text[:200],
            )
            return None
        except Exception as exc:
            self.log.error("get_exception", url=url, error=str(exc))
            return None

    def _get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> str | None:
        """GET → raw text (useful for GDELT lastupdate.txt, CSV lines, etc.)"""
        try:
            response = self._raw_get(url, params=params)
            return response.text
        except RetryError as exc:
            self.log.error("get_text_failed", url=url, error=str(exc))
            return None
        except Exception as exc:
            self.log.error("get_text_exception", url=url, error=str(exc))
            return None

    def _get_bytes(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> bytes | None:
        """GET → raw bytes (useful for downloading compressed GDELT files)."""
        try:
            response = self._raw_get(url, params=params)
            return response.content
        except RetryError as exc:
            self.log.error("get_bytes_failed", url=url, error=str(exc))
            return None
        except Exception as exc:
            self.log.error("get_bytes_exception", url=url, error=str(exc))
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def _raw_get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Low-level GET with rate limiting. Decorated with retry."""
        self._rate_limiter.acquire()
        self.log.debug("http_get", url=url, params=params)
        resp = self._http.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp

    # ── Context manager / cleanup ──────────────────────────────────────────────

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> BaseClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
