"""EverSpring MCP - Scraper package for Spring documentation.

This package provides tools for scraping Spring documentation:
- SpringBrowser: Async Playwright browser with retry and rate limiting
- SpringDocParser: HTML parsing and Markdown conversion
- ScraperPipeline: AWS Lambda handler for orchestrated scraping
- SpringDocDiscovery: URL discovery for automatic crawling
- Custom exceptions for error handling
"""

from .browser import BrowserConfig, SpringBrowser, USER_AGENTS
from .discovery import (
    DiscoveredLink,
    DiscoveryConfig,
    DiscoveryResult,
    DiscoveryStatus,
    SpringDocDiscovery,
    extract_base_path,
    flatten_nav_items,
    is_internal_link,
    normalize_url,
    should_skip_url,
    SKIP_EXTENSIONS,
    SKIP_PATTERNS,
)
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
from .pipeline import (
    PipelineConfig,
    PipelineStatus,
    S3Client,
    ScrapeResult,
    ScrapeTarget,
    ScraperPipeline,
    lambda_handler,
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
    # Discovery
    "SpringDocDiscovery",
    "DiscoveryConfig",
    "DiscoveryStatus",
    "DiscoveredLink",
    "DiscoveryResult",
    "normalize_url",
    "extract_base_path",
    "is_internal_link",
    "should_skip_url",
    "flatten_nav_items",
    "SKIP_EXTENSIONS",
    "SKIP_PATTERNS",
    # Pipeline
    "ScraperPipeline",
    "PipelineConfig",
    "PipelineStatus",
    "S3Client",
    "ScrapeResult",
    "ScrapeTarget",
    "lambda_handler",
    # Exceptions
    "ScraperError",
    "NavigationError",
    "NavigationTimeoutError",
    "RateLimitError",
    "ContentExtractionError",
    "BrowserLaunchError",
]
