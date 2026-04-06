"""EverSpring MCP - Async browser for Spring documentation scraping.

This module provides SpringBrowser, an async Playwright-based browser
for scraping Spring documentation with:
- User-agent rotation to avoid rate limits
- Network idle waiting for JS-rendered content
- Retry logic with exponential backoff
- Comprehensive error handling
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Self

from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    Playwright,
    Response,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)
from pydantic import BaseModel, ConfigDict, Field

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
    
    async def __aenter__(self) -> Self:
        """Enter async context and launch browser."""
        await self._launch()
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
        """Launch browser and create initial context."""
        try:
            start = time.perf_counter()
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless,
            )
            await self._create_context()
            logger.info(f"Browser launched in {time.perf_counter() - start:.2f}s")
        except PlaywrightError as e:
            raise BrowserLaunchError(f"Failed to launch browser: {e}") from e
    
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
    
    async def navigate(self, url: str) -> Response:
        """Navigate to URL and wait for network idle.
        
        Args:
            url: URL to navigate to
            
        Returns:
            Response object from navigation
            
        Raises:
            NavigationTimeoutError: If navigation times out
            RateLimitError: If rate limited (HTTP 429)
            NavigationError: For other navigation failures
        """
        if not self._page:
            raise BrowserLaunchError("Browser not launched")
        
        self._current_url = url
        logger.info(f"Navigating to: {url}")
        
        try:
            response = await self._page.goto(
                url,
                wait_until="networkidle",
                timeout=self.config.timeout_ms,
            )
            
            if response is None:
                raise NavigationError("No response received", url=url)
            
            # Check for rate limiting
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
        """Navigate to URL with retry and exponential backoff.
        
        Args:
            url: URL to navigate to
            max_retries: Override default max retries
            
        Returns:
            Response object from navigation
            
        Raises:
            NavigationTimeoutError: If all retries exhausted due to timeout
            RateLimitError: If rate limited after retries
            NavigationError: For other persistent failures
        """
        retries = max_retries or self.config.max_retries
        last_error: Exception | None = None
        
        for attempt in range(retries):
            try:
                return await self.navigate(url)
                
            except RateLimitError as e:
                last_error = e
                wait_time = e.retry_after or self.config.rate_limit_delay
                logger.warning(
                    f"Rate limited on attempt {attempt + 1}/{retries}, "
                    f"waiting {wait_time}s"
                )
                await asyncio.sleep(wait_time)
                await self._rotate_context()
                
            except (NavigationError, NavigationTimeoutError) as e:
                last_error = e
                if attempt == retries - 1:
                    break
                
                # Exponential backoff with jitter
                wait_time = (
                    self.config.base_retry_delay * (2 ** attempt)
                    + random.uniform(0, 1)
                )
                logger.warning(
                    f"Navigation failed on attempt {attempt + 1}/{retries}, "
                    f"retrying in {wait_time:.2f}s: {e}"
                )
                await asyncio.sleep(wait_time)
        
        # Re-raise last error after all retries exhausted
        if last_error:
            raise last_error
        raise NavigationError("Navigation failed with unknown error", url=url)
    
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
    def is_launched(self) -> bool:
        """Check if browser is launched and ready."""
        return self._browser is not None and self._page is not None


__all__ = [
    "BrowserConfig",
    "SpringBrowser",
    "USER_AGENTS",
]
