"""EverSpring MCP - Real-time Spring documentation for LLMs.

This package provides:
- Pydantic models for Spring documentation data
- Async browser for scraping Spring docs
"""

import os

# Disable tokenizer parallelism globally to prevent deadlocks in async/worker loops
os.environ["TOKENIZERS_PARALLELISM"] = "false"

__version__ = "0.1.0"

from everspring_mcp.models import (
    # Base
    HashableContent,
    # Content
    ScrapedPage,
    # Spring
    SpringModule,
    SpringVersion,
    TimestampedModel,
    VersionedModel,
    VersionRange,
)
from everspring_mcp.scraper import (
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
