"""EverSpring MCP - Async browser for Spring documentation scraping.

This module provides SpringBrowser, an async Playwright-based browser
for scraping Spring documentation with:
- User-agent rotation to avoid rate limits
- Network idle waiting for JS-rendered content
- Exponential backoff with tenacity library
- Comprehensive error handling
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import Self

import httpx
from bs4 import BeautifulSoup, Tag
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Response,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from everspring_mcp.models.base import compute_hash
from everspring_mcp.scraper.exceptions import (
    BrowserLaunchError,
    ContentExtractionError,
    NavigationError,
    NavigationTimeoutError,
    RateLimitError,
)
from everspring_mcp.utils.logging import get_logger

logger = get_logger("scraper.browser")

# User agents for rotation - modern browsers on various platforms
USER_AGENTS: list[str] = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

STATIC_FETCH_BASE_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Fast-path regex extraction for core content blocks before Playwright rendering.
MAIN_CONTENT_REGEX = re.compile(
    r"<main\b[^>]*>(?P<content>.*?)</main>",
    re.IGNORECASE | re.DOTALL,
)
DOC_CONTENT_REGEX = re.compile(
    r"<div\b[^>]*class=(?P<quote>['\"])[^'\"]*\bdoc\b[^'\"]*(?P=quote)[^>]*>(?P<content>.*?)</div>",
    re.IGNORECASE | re.DOTALL,
)


class BrowserConfig(BaseModel):
    """Configuration for SpringBrowser.

    Attributes:
        headless: Run browser in headless mode (default: True)
        timeout_ms: Default timeout for operations in milliseconds
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height
        max_retries: Maximum retry attempts for failed operations
        base_retry_delay: Base delay for exponential backoff (seconds)
        rate_limit_delay: Delay when rate limited (seconds)
    """

    model_config = ConfigDict(frozen=True)

    headless: bool = Field(
        default=True,
        description="Run browser in headless mode",
    )
    timeout_ms: int = Field(
        default=30000,
        ge=5000,
        le=120000,
        description="Default timeout in milliseconds",
    )
    viewport_width: int = Field(
        default=1920,
        ge=800,
        le=3840,
        description="Viewport width in pixels",
    )
    viewport_height: int = Field(
        default=1080,
        ge=600,
        le=2160,
        description="Viewport height in pixels",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts",
    )
    base_retry_delay: float = Field(
        default=1.0,
        ge=0.5,
        le=10.0,
        description="Base delay for exponential backoff (seconds)",
    )
    rate_limit_delay: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Delay when rate limited (seconds)",
    )
    enable_fast_precheck: bool = Field(
        default=True,
        description="Enable lightweight HTTP hash precheck before Playwright rendering",
    )
    fast_precheck_timeout_ms: int = Field(
        default=5000,
        ge=500,
        le=30000,
        description="HTTP precheck timeout in milliseconds",
    )
    static_fetch_timeout_ms: int = Field(
        default=8000,
        ge=500,
        le=45000,
        description="Timeout for static HTML fetch before Playwright fallback",
    )


class NotModifiedSignal(BaseModel):
    """Signal returned when fast precheck detects unchanged content."""

    model_config = ConfigDict(frozen=True)

    signal: str = Field(
        default="NotModified",
        description="Signal marker for unchanged content",
    )
    url: str = Field(
        description="Checked URL",
    )
    content_hash: str = Field(
        description="SHA-256 hash from precheck content extraction",
    )


class SpringBrowser:
    """Async browser for scraping Spring documentation.

    Uses Playwright to handle JavaScript-rendered content with
    user-agent rotation, retry logic, and comprehensive error handling.

    Usage:
        async with SpringBrowser() as browser:
            await browser.navigate("https://docs.spring.io/...")
            html = await browser.get_html()

    Attributes:
        config: Browser configuration settings
    """

    def __init__(self, config: BrowserConfig | None = None) -> None:
        """Initialize SpringBrowser.

        Args:
            config: Browser configuration. Uses defaults if not provided.
        """
        self.config = config or BrowserConfig()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._current_url: str | None = None
        self._last_precheck_hash: str | None = None

    async def __aenter__(self) -> Self:
        """Enter async context.

        Playwright launch is deferred until a rendered-navigation path is needed.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context and close browser."""
        await self._close()

    async def _launch(self) -> None:
        """Launch browser and create initial context lazily."""
        if self._browser is not None and self._page is not None:
            return
        try:
            start = time.perf_counter()
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            if self._browser is None:
                self._browser = await self._playwright.chromium.launch(
                    headless=self.config.headless,
                )
            await self._create_context()
            logger.info(f"Browser launched in {time.perf_counter() - start:.2f}s")
        except PlaywrightError as e:
            raise BrowserLaunchError(f"Failed to launch browser: {e}") from e

    async def _ensure_playwright_ready(self) -> None:
        """Ensure Playwright browser/page is available for rendered operations."""
        if self._browser is not None and self._page is not None:
            return
        await self._launch()

    async def _close(self) -> None:
        """Close browser and cleanup resources."""
        if self._page:
            await self._page.close()
            self._page = None
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Browser closed")

    async def _create_context(self) -> None:
        """Create browser context with rotated user agent."""
        if not self._browser:
            raise BrowserLaunchError("Browser not launched")

        user_agent = self._get_user_agent()
        self._context = await self._browser.new_context(
            user_agent=user_agent,
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
        )
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.config.timeout_ms)
        logger.debug(f"Created context with user agent: {user_agent[:50]}...")

    async def _rotate_context(self) -> None:
        """Rotate to new browser context with different user agent.

        Useful when encountering rate limits or for stealth.
        """
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        await self._create_context()
        logger.info("Rotated browser context with new user agent")

    def _get_user_agent(self) -> str:
        """Get random user agent from pool."""
        return random.choice(USER_AGENTS)

    @staticmethod
    def _unique_selectors(selectors: list[str]) -> list[str]:
        """Preserve selector order while removing duplicates/empties."""
        seen: set[str] = set()
        normalized: list[str] = []
        for selector in selectors:
            clean = selector.strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        return normalized

    async def _fetch_html_httpx(self, url: str, timeout_ms: int) -> str | None:
        """Fetch HTML via httpx as the fast path before Playwright fallback."""
        timeout_seconds = timeout_ms / 1000
        headers = {
            **STATIC_FETCH_BASE_HEADERS,
            "User-Agent": self._get_user_agent(),
        }
        timeout = httpx.Timeout(timeout_seconds)

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
            ) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                html = response.text
                if not html.strip():
                    logger.debug("Static fetch returned empty HTML for %s", url)
                    return None
                return html
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
            httpx.RequestError,
        ) as exc:
            logger.debug("Static fetch failed for %s: %s", url, exc)
            return None

    @classmethod
    def _is_static_html_sufficient(
        cls,
        html: str,
        content_selectors: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Validate whether static HTML likely contains parseable main content."""
        soup = BeautifulSoup(html, "lxml")
        selectors = cls._unique_selectors(
            (content_selectors or [])
            + [
                "main",
                "article",
                "div.content",
                ".content",
            ]
        )
        for selector in selectors:
            node = soup.select_one(selector)
            if node and isinstance(node, Tag):
                text = cls._normalize_extracted_content(node.get_text(" ", strip=True))
                if text:
                    return True, f"selector '{selector}' matched"

        return False, "no strong content selector matched"

    @staticmethod
    def _normalize_extracted_content(content: str) -> str:
        """Normalize whitespace for stable hash comparisons."""
        return re.sub(r"\s+", " ", content).strip()

    @classmethod
    def extract_core_content(cls, html: str) -> str | None:
        """Extract core page content quickly using regex then minimal BeautifulSoup."""
        for pattern in (MAIN_CONTENT_REGEX, DOC_CONTENT_REGEX):
            match = pattern.search(html)
            if not match:
                continue
            extracted = cls._normalize_extracted_content(match.group("content"))
            if extracted:
                return extracted

        soup = BeautifulSoup(html, "lxml")
        node = soup.select_one("main, div.doc, article.doc, main.content, #content")
        if node and isinstance(node, Tag):
            extracted = cls._normalize_extracted_content(node.get_text(" ", strip=True))
            if extracted:
                return extracted

        body = soup.body
        if body and isinstance(body, Tag):
            extracted = cls._normalize_extracted_content(body.get_text(" ", strip=True))
            return extracted or None

        return None

    @classmethod
    def compute_core_content_hash(cls, html: str) -> str | None:
        """Compute SHA-256 hash from extracted core content."""
        extracted = cls.extract_core_content(html)
        if not extracted:
            return None
        return compute_hash(extracted)

    async def fast_precheck(
        self,
        url: str,
        stored_hash: str | None = None,
    ) -> NotModifiedSignal | None:
        """Run lightweight async HTTP precheck before Playwright navigation.

        This fetches the URL via httpx, extracts the main content area, computes
        SHA-256, and compares against a stored hash. If equal, returns a
        NotModified signal so callers can skip expensive Playwright rendering.
        """
        self._last_precheck_hash = None
        if not self.config.enable_fast_precheck:
            return None

        html = await self._fetch_html_httpx(url, timeout_ms=self.config.fast_precheck_timeout_ms)
        if html is None:
            return None

        content_hash = self.compute_core_content_hash(html)
        self._last_precheck_hash = content_hash
        if not content_hash:
            return None

        if stored_hash and content_hash == stored_hash:
            logger.info(
                "Fast precheck matched stored hash; skipping Playwright for %s",
                url,
            )
            return NotModifiedSignal(url=url, content_hash=content_hash)

        return None

    @retry(
        retry=retry_if_exception_type(
            (NavigationTimeoutError, PlaywrightTimeoutError, PlaywrightError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def navigate(self, url: str) -> Response:
        """Navigate to URL and wait for network idle.

        Uses tenacity for automatic retry with exponential backoff on:
        - Playwright timeouts
        - Network errors
        - Navigation failures

        Args:
            url: URL to navigate to

        Returns:
            Response object from navigation

        Raises:
            NavigationTimeoutError: If navigation times out after retries
            RateLimitError: If rate limited (HTTP 429) - no retry
            NavigationError: For other navigation failures after retries
        """
        await self._ensure_playwright_ready()
        if not self._page:
            raise BrowserLaunchError("Browser not launched")

        self._current_url = url
        logger.debug(f"Navigating to: {url}")

        try:
            response = await self._page.goto(
                url,
                wait_until="networkidle",
                timeout=self.config.timeout_ms,
            )

            if response is None:
                raise NavigationError("No response received", url=url)

            # Check for rate limiting (don't retry, raise immediately)
            if response.status == 429:
                retry_after = response.headers.get("retry-after")
                retry_seconds = int(retry_after) if retry_after else None
                raise RateLimitError(
                    "Rate limited by server",
                    url=url,
                    retry_after=retry_seconds,
                )

            # Check for other HTTP errors
            if response.status >= 400:
                raise NavigationError(
                    f"HTTP {response.status}: {response.status_text}",
                    url=url,
                    status_code=response.status,
                )

            logger.info(f"Navigation successful: {response.status}")
            return response

        except PlaywrightTimeoutError as e:
            raise NavigationTimeoutError(
                f"Timeout navigating to {url}",
                url=url,
                timeout_ms=self.config.timeout_ms,
            ) from e
        except PlaywrightError as e:
            raise NavigationError(f"Navigation failed: {e}", url=url) from e

    async def navigate_with_retry(
        self,
        url: str,
        max_retries: int | None = None,
    ) -> Response:
        """Navigate to URL with retry, exponential backoff, and rate limit handling.

        This method adds special handling for rate limits on top of the base
        navigate() method's retry logic. When rate limited:
        - Waits for server-specified retry-after time
        - Rotates browser context (new user agent)
        - Retries the request

        Args:
            url: URL to navigate to
            max_retries: Override default max retries (for rate limits)

        Returns:
            Response object from navigation

        Raises:
            RateLimitError: If rate limited after all retries
            NavigationTimeoutError: If navigation times out after retries
            NavigationError: For other persistent failures
        """
        retries = max_retries or self.config.max_retries
        attempt = 0
        while attempt < retries:
            attempt += 1
            try:
                return await self.navigate(url)
            except RateLimitError as e:
                if attempt >= retries:
                    raise
                wait_time = e.retry_after or self.config.rate_limit_delay
                logger.warning(
                    "Rate limited, rotating context and waiting %.1fs before retry (%d/%d)",
                    wait_time,
                    attempt,
                    retries,
                )
                await asyncio.sleep(wait_time)
                await self._rotate_context()
            except NavigationError as e:
                if not self._is_transient_navigation_error(e) or attempt >= retries:
                    raise
                wait_time = self._transient_backoff_delay(attempt)
                logger.warning(
                    "Transient navigation failure, rotating context and retrying in %.2fs (%d/%d): %s",
                    wait_time,
                    attempt,
                    retries,
                    e,
                )
                await asyncio.sleep(wait_time)
                await self._rotate_context()

        raise NavigationError("Unexpected retry state", url=url)

    def _is_transient_navigation_error(self, error: NavigationError) -> bool:
        """Detect whether a navigation error is likely transient and retryable."""
        if error.status_code in {408, 425, 429, 500, 502, 503, 504, 522, 523, 524}:
            return True

        message = str(error).upper()
        transient_markers = (
            "ERR_NAME_NOT_RESOLVED",
            "ERR_CONNECTION_RESET",
            "ERR_CONNECTION_CLOSED",
            "ERR_CONNECTION_TIMED_OUT",
            "ERR_CONNECTION_REFUSED",
            "ERR_NETWORK_CHANGED",
            "ERR_INTERNET_DISCONNECTED",
            "DNS_PROBE_FINISHED",
            "TEMPORARY_FAILURE",
        )
        return any(marker in message for marker in transient_markers)

    def _transient_backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff with jitter for transient navigation retries."""
        base = self.config.base_retry_delay * (2 ** max(0, attempt - 1))
        jitter = random.uniform(0.0, self.config.base_retry_delay)
        return min(60.0, base + jitter)

    async def get_html(self) -> str:
        """Get full page HTML content.

        Returns:
            Complete HTML content of current page

        Raises:
            ContentExtractionError: If content cannot be extracted
        """
        if not self._page:
            raise BrowserLaunchError("Browser not launched")

        try:
            content = await self._page.content()
            if not content:
                raise ContentExtractionError(
                    "Empty page content",
                    url=self._current_url,
                )
            logger.debug(f"Extracted HTML: {len(content)} chars")
            return content
        except PlaywrightError as e:
            raise ContentExtractionError(
                f"Failed to extract HTML: {e}",
                url=self._current_url,
            ) from e

    async def get_html_with_fallback(
        self,
        url: str,
        content_selectors: list[str] | None = None,
    ) -> str:
        """Fetch HTML through static-first strategy with Playwright fallback."""
        self._current_url = url
        static_html = await self._fetch_html_httpx(
            url,
            timeout_ms=self.config.static_fetch_timeout_ms,
        )
        if static_html is not None:
            sufficient, reason = self._is_static_html_sufficient(
                static_html,
                content_selectors=content_selectors,
            )
            if sufficient:
                logger.debug("Static HTML accepted for %s (%s)", url, reason)
                return static_html
            logger.debug("Static HTML insufficient for %s (%s)", url, reason)
        else:
            logger.debug("Static HTML unavailable for %s, falling back to Playwright", url)

        await self.navigate_with_retry(url)
        return await self.get_html()

    async def get_content(self, selector: str) -> str:
        """Get text content of element matching selector.

        Args:
            selector: CSS selector for element

        Returns:
            Text content of matched element

        Raises:
            ContentExtractionError: If element not found or extraction fails
        """
        if not self._page:
            raise BrowserLaunchError("Browser not launched")

        try:
            element = await self._page.query_selector(selector)
            if not element:
                raise ContentExtractionError(
                    f"Element not found: {selector}",
                    url=self._current_url,
                    selector=selector,
                )

            content = await element.text_content()
            if content is None:
                raise ContentExtractionError(
                    f"Empty content for selector: {selector}",
                    url=self._current_url,
                    selector=selector,
                )

            return content

        except PlaywrightError as e:
            raise ContentExtractionError(
                f"Failed to extract content: {e}",
                url=self._current_url,
                selector=selector,
            ) from e

    async def get_inner_html(self, selector: str) -> str:
        """Get inner HTML of element matching selector.

        Args:
            selector: CSS selector for element

        Returns:
            Inner HTML of matched element

        Raises:
            ContentExtractionError: If element not found or extraction fails
        """
        if not self._page:
            raise BrowserLaunchError("Browser not launched")

        try:
            element = await self._page.query_selector(selector)
            if not element:
                raise ContentExtractionError(
                    f"Element not found: {selector}",
                    url=self._current_url,
                    selector=selector,
                )

            html = await element.inner_html()
            return html

        except PlaywrightError as e:
            raise ContentExtractionError(
                f"Failed to extract inner HTML: {e}",
                url=self._current_url,
                selector=selector,
            ) from e

    async def wait_for_selector(
        self,
        selector: str,
        timeout_ms: int | None = None,
    ) -> None:
        """Wait for element matching selector to appear.

        Args:
            selector: CSS selector to wait for
            timeout_ms: Override default timeout

        Raises:
            NavigationTimeoutError: If element doesn't appear in time
        """
        if not self._page:
            raise BrowserLaunchError("Browser not launched")

        try:
            await self._page.wait_for_selector(
                selector,
                timeout=timeout_ms or self.config.timeout_ms,
            )
        except PlaywrightTimeoutError as e:
            raise NavigationTimeoutError(
                f"Timeout waiting for selector: {selector}",
                url=self._current_url,
                timeout_ms=timeout_ms or self.config.timeout_ms,
            ) from e

    @property
    def current_url(self) -> str | None:
        """Get current page URL."""
        return self._current_url

    @property
    def last_precheck_hash(self) -> str | None:
        """Get the latest hash produced by fast_precheck (if any)."""
        return self._last_precheck_hash

    @property
    def is_launched(self) -> bool:
        """Check if browser is launched and ready."""
        return self._browser is not None and self._page is not None


__all__ = [
    "BrowserConfig",
    "NotModifiedSignal",
    "SpringBrowser",
    "USER_AGENTS",
]
