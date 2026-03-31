"""EverSpring MCP - Real-time Spring documentation for LLMs.

This package provides:
- Pydantic models for Spring documentation data
- Async browser for scraping Spring docs
"""

__version__ = "0.1.0"

from .models import (
    # Base
    HashableContent,
    TimestampedModel,
    VersionedModel,
    # Spring
    SpringModule,
    SpringVersion,
    VersionRange,
    # Content
    ScrapedPage,
)
from .scraper import (
    BrowserConfig,
    SpringBrowser,
)

__all__ = [
    "__version__",
    # Models
    "VersionedModel",
    "TimestampedModel",
    "HashableContent",
    "SpringModule",
    "SpringVersion",
    "VersionRange",
    "ScrapedPage",
    # Scraper
    "SpringBrowser",
    "BrowserConfig",
]
