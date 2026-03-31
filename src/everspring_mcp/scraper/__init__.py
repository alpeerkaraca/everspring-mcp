"""EverSpring MCP - Scraper package for Spring documentation.

This package provides tools for scraping Spring documentation:
- SpringBrowser: Async Playwright browser with retry and rate limiting
- Custom exceptions for error handling
"""

from .browser import BrowserConfig, SpringBrowser, USER_AGENTS
from .exceptions import (
    BrowserLaunchError,
    ContentExtractionError,
    NavigationError,
    NavigationTimeoutError,
    RateLimitError,
    ScraperError,
)

__all__ = [
    # Browser
    "SpringBrowser",
    "BrowserConfig",
    "USER_AGENTS",
    # Exceptions
    "ScraperError",
    "NavigationError",
    "NavigationTimeoutError",
    "RateLimitError",
    "ContentExtractionError",
    "BrowserLaunchError",
]
