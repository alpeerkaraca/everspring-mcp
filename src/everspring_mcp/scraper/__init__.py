"""EverSpring MCP - Scraper package for Spring documentation.

This package provides tools for scraping Spring documentation:
- SpringBrowser: Async Playwright browser with retry and rate limiting
- SpringDocParser: HTML parsing and Markdown conversion
- ScraperPipeline: AWS Lambda handler for orchestrated scraping
- SpringDocDiscovery: URL discovery for automatic crawling
- Custom exceptions for error handling
"""

from everspring_mcp.scraper.browser import (
    USER_AGENTS,
    BrowserConfig,
    NotModifiedSignal,
    SpringBrowser,
)
from everspring_mcp.scraper.discovery import (
    BASE_SKIP_PATTERNS,
    REFERENCE_ONLY_SKIP_PATTERNS,
    SKIP_EXTENSIONS,
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
)
from everspring_mcp.scraper.exceptions import (
    BrowserLaunchError,
    ContentExtractionError,
    NavigationError,
    NavigationTimeoutError,
    RateLimitError,
    ScraperError,
)
from everspring_mcp.scraper.parser import (
    LANGUAGE_PATTERNS,
    SPRING_DOC_SELECTORS,
    ParserConfig,
    SpringDocParser,
    SpringMarkdownConverter,
)
from everspring_mcp.scraper.pipeline import (
    PipelineConfig,
    PipelineStatus,
    S3Client,
    ScrapeResult,
    ScraperPipeline,
    ScrapeTarget,
    lambda_handler,
)
from everspring_mcp.scraper.registry import SubmoduleRegistry, SubmoduleTarget

__all__ = [
    # Browser
    "SpringBrowser",
    "BrowserConfig",
    "NotModifiedSignal",
    "USER_AGENTS",
    # Parser
    "SpringDocParser",
    "ParserConfig",
    "SpringMarkdownConverter",
    "SPRING_DOC_SELECTORS",
    "LANGUAGE_PATTERNS",
    "SubmoduleRegistry",
    "SubmoduleTarget",
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
    "BASE_SKIP_PATTERNS",
    "REFERENCE_ONLY_SKIP_PATTERNS",
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
