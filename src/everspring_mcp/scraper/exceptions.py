"""EverSpring MCP - Custom scraper exceptions.

This module provides exception hierarchy for scraper operations:
- ScraperError: Base exception for all scraper errors
- NavigationError: Page navigation failures
- NavigationTimeoutError: Timeout during navigation
- RateLimitError: Rate limiting detected
- ContentExtractionError: HTML content extraction failures
"""

from __future__ import annotations


class ScraperError(Exception):
    """Base exception for all scraper-related errors."""
    
    def __init__(self, message: str, url: str | None = None) -> None:
        self.url = url
        super().__init__(message)
    
    def __str__(self) -> str:
        if self.url:
            return f"{self.args[0]} (URL: {self.url})"
        return str(self.args[0])


class NavigationError(ScraperError):
    """Raised when page navigation fails.
    
    This includes DNS failures, connection errors, and HTTP errors
    (except rate limiting which has its own exception).
    """
    
    def __init__(
        self,
        message: str,
        url: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(message, url)


class NavigationTimeoutError(ScraperError):
    """Raised when page navigation times out.
    
    This occurs when the page doesn't reach the expected state
    (e.g., networkidle) within the configured timeout.
    """
    
    def __init__(
        self,
        message: str,
        url: str | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self.timeout_ms = timeout_ms
        super().__init__(message, url)


class RateLimitError(ScraperError):
    """Raised when rate limiting is detected.
    
    This occurs when the server returns HTTP 429 or other
    indicators of rate limiting.
    """
    
    def __init__(
        self,
        message: str,
        url: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, url)


class ContentExtractionError(ScraperError):
    """Raised when content extraction from HTML fails.
    
    This includes selector not found, empty content,
    or malformed HTML structure.
    """
    
    def __init__(
        self,
        message: str,
        url: str | None = None,
        selector: str | None = None,
    ) -> None:
        self.selector = selector
        super().__init__(message, url)


class BrowserLaunchError(ScraperError):
    """Raised when browser fails to launch.
    
    This typically indicates Playwright installation issues
    or system resource constraints.
    """
    pass


__all__ = [
    "ScraperError",
    "NavigationError",
    "NavigationTimeoutError",
    "RateLimitError",
    "ContentExtractionError",
    "BrowserLaunchError",
]
