"""EverSpring MCP - Scraper package for Spring documentation.

This package provides tools for scraping Spring documentation:
- SpringBrowser: Async Playwright browser with retry and rate limiting
- SpringDocParser: HTML parsing and Markdown conversion
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
from .parser import (
    LANGUAGE_PATTERNS,
    SPRING_DOC_SELECTORS,
    ParserConfig,
    SpringDocParser,
    SpringMarkdownConverter,
)

__all__ = [
    # Browser
    "SpringBrowser",
    "BrowserConfig",
    "USER_AGENTS",
    # Parser
    "SpringDocParser",
    "ParserConfig",
    "SpringMarkdownConverter",
    "SPRING_DOC_SELECTORS",
    "LANGUAGE_PATTERNS",
    # Exceptions
    "ScraperError",
    "NavigationError",
    "NavigationTimeoutError",
    "RateLimitError",
    "ContentExtractionError",
    "BrowserLaunchError",
]
