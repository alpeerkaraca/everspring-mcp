"""EverSpring MCP - Spring documentation discovery logic.

This module provides URL discovery for Spring documentation:
- DiscoveryConfig: Configuration for discovery behavior
- DiscoveredLink: Model for a discovered URL with metadata
- DiscoveryResult: Collection of discovered links with statistics
- SpringDocDiscovery: Async class for discovering documentation URLs
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Self
from urllib.parse import urljoin, urlparse, urlunparse

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from ..models.base import VersionedModel, compute_hash
from ..models.content import ContentType
from ..models.spring import SpringModule, SpringVersion
from ..utils.logging import get_logger
from .browser import BrowserConfig, SpringBrowser
from .parser import ParserConfig, SpringDocParser, SPRING_DOC_SELECTORS
from .exceptions import ContentExtractionError, NavigationError, ScraperError
from .pipeline import ScrapeTarget


logger = get_logger("scraper.discovery")


# File extensions to skip during discovery
SKIP_EXTENSIONS: set[str] = {
    ".pdf", ".zip", ".tar", ".gz", ".jar",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
}

# URL patterns to skip for all content types
BASE_SKIP_PATTERNS: list[re.Pattern] = [
    re.compile(r"github\.com"),
    re.compile(r"stackoverflow\.com"),
    re.compile(r"mailto:"),
]

# URL patterns to skip for non-API documentation
REFERENCE_ONLY_SKIP_PATTERNS: list[re.Pattern] = [
    re.compile(r"/api/"),
    re.compile(r"/javadoc/"),
    re.compile(r"/javadoc-api/"),
]


class DiscoveryStatus(str, Enum):
    """Status of discovery operation."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DiscoveryConfig(BaseModel):
    """Configuration for SpringDocDiscovery.
    
    Attributes:
        max_depth: Maximum link depth from entry URL (0 = entry only)
        max_links: Maximum number of links to discover
        follow_external: Whether to follow external links (always False for safety)
        skip_extensions: File extensions to skip
        skip_patterns: URL patterns to skip
        browser_config: Configuration for SpringBrowser
        parser_config: Configuration for SpringDocParser
    """
    
    model_config = ConfigDict(frozen=True)
    
    max_depth: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum link depth from entry URL",
    )
    max_links: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of links to discover",
    )
    follow_external: bool = Field(
        default=False,
        description="Whether to follow external links (always False)",
    )
    skip_extensions: set[str] = Field(
        default_factory=lambda: SKIP_EXTENSIONS.copy(),
        description="File extensions to skip",
    )
    suppress_http_404: bool = Field(
        default=True,
        description="Drop links that return HTTP 404 during discovery traversal",
    )
    browser_config: BrowserConfig = Field(
        default_factory=BrowserConfig,
        description="Browser configuration",
    )
    parser_config: ParserConfig = Field(
        default_factory=ParserConfig,
        description="Parser configuration",
    )
    
    @model_validator(mode="after")
    def enforce_no_external(self) -> Self:
        """Ensure external links are never followed."""
        if self.follow_external:
            # Override to False - security requirement
            object.__setattr__(self, "follow_external", False)
        return self


class DiscoveredLink(VersionedModel):
    """A discovered documentation URL with metadata.
    
    Attributes:
        url: Normalized absolute URL
        depth: Distance from entry URL (0 = entry)
        source_url: URL where this link was found
        title: Link text/title if available
        discovered_at: When link was discovered
    """
    
    url: str = Field(
        pattern=r"^https?://[^\s]+$",
        description="Normalized absolute URL",
    )
    depth: int = Field(
        ge=0,
        description="Distance from entry URL",
    )
    source_url: str | None = Field(
        default=None,
        description="URL where this link was found",
    )
    title: str | None = Field(
        default=None,
        description="Link text/title if available",
    )
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When link was discovered",
    )
    
    @computed_field
    @property
    def url_hash(self) -> str:
        """SHA-256 hash of the URL for deduplication."""
        return compute_hash(self.url)[:16]
    
    def to_scrape_target(
        self,
        module: SpringModule,
        version: SpringVersion,
        submodule: str | None = None,
        content_type: ContentType = ContentType.REFERENCE,
    ) -> ScrapeTarget:
        """Convert to ScrapeTarget for pipeline processing.
        
        Args:
            module: Spring module
            version: Spring version
            submodule: Optional submodule key
            content_type: Documentation type
            
        Returns:
            ScrapeTarget for this URL
        """
        return ScrapeTarget(
            url=self.url,
            module=module,
            submodule=submodule,
            major=version.major,
            minor=version.minor,
            patch=version.patch,
            content_type=content_type,
        )


class DiscoveryResult(VersionedModel):
    """Result of a discovery operation.
    
    Attributes:
        entry_url: Starting URL for discovery
        module: Spring module being discovered
        version: Spring version
        links: List of discovered links
        total_found: Total links found before filtering
        duplicates_removed: Count of duplicate links removed
        filtered_out: Count of links filtered (external, wrong path, etc.)
        status: Discovery status
        error_message: Error message if failed
        started_at: When discovery started
        completed_at: When discovery completed
    """
    
    entry_url: str = Field(
        description="Starting URL for discovery",
    )
    module: SpringModule = Field(
        description="Spring module being discovered",
    )
    version: SpringVersion = Field(
        description="Spring version",
    )
    links: list[DiscoveredLink] = Field(
        default_factory=list,
        description="List of discovered links",
    )
    total_found: int = Field(
        default=0,
        ge=0,
        description="Total links found before filtering",
    )
    duplicates_removed: int = Field(
        default=0,
        ge=0,
        description="Count of duplicate links removed",
    )
    filtered_out: int = Field(
        default=0,
        ge=0,
        description="Count of links filtered out",
    )
    status: DiscoveryStatus = Field(
        default=DiscoveryStatus.PENDING,
        description="Discovery status",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if failed",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When discovery started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When discovery completed",
    )
    
    @computed_field
    @property
    def link_count(self) -> int:
        """Number of unique links discovered."""
        return len(self.links)
    
    @computed_field
    @property
    def duration_seconds(self) -> float | None:
        """Discovery duration in seconds."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds()
    
    def to_scrape_targets(
        self,
        content_type: ContentType = ContentType.REFERENCE,
        submodule: str | None = None,
        auto_version: bool = False,
    ) -> list[ScrapeTarget]:
        """Convert all links to ScrapeTargets.
        
        Args:
            content_type: Documentation type for all targets
            submodule: Optional submodule key
            auto_version: When True, leave version fields unset for auto-detection
            
        Returns:
            List of ScrapeTarget objects
        """
        if auto_version:
            return [
                ScrapeTarget(
                    url=link.url,
                    module=self.module,
                    submodule=submodule,
                    major=None,
                    minor=0,
                    patch=0,
                    content_type=content_type,
                )
                for link in self.links
            ]
        return [
            link.to_scrape_target(self.module, self.version, submodule, content_type)
            for link in self.links
        ]


# =============================================================================
# URL Utilities
# =============================================================================


def normalize_url(url: str, base_url: str | None = None) -> str:
    """Normalize URL by resolving relative paths and cleaning.
    
    - Resolves relative URLs against base
    - Removes anchors (#section)
    - Removes query parameters
    - Ensures trailing slash consistency
    
    Args:
        url: URL to normalize (can be relative)
        base_url: Base URL for resolving relative paths
        
    Returns:
        Normalized absolute URL
    """
    # Handle empty or anchor-only URLs
    if not url or url.startswith("#"):
        return ""
    
    # Resolve relative URLs
    if base_url and not url.startswith(("http://", "https://")):
        url = urljoin(base_url, url)
    
    # Parse URL
    parsed = urlparse(url)
    
    # Skip non-HTTP schemes
    if parsed.scheme not in ("http", "https"):
        return ""
    
    # Remove anchor and query params
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        "",  # params
        "",  # query
        "",  # fragment
    ))
    
    # Normalize trailing slash for directories
    # Keep trailing slash if path doesn't have extension
    path = parsed.path
    if path and not path.endswith("/"):
        # Check if it looks like a file
        last_segment = path.split("/")[-1]
        if "." not in last_segment:
            # Looks like a directory, add trailing slash
            normalized = normalized.rstrip("/") + "/"
    
    return normalized


def extract_base_path(entry_url: str) -> str:
    """Extract base path from entry URL for filtering.
    
    Example:
        entry_url = "https://docs.spring.io/spring-boot/reference/index.html"
        returns = "https://docs.spring.io/spring-boot/reference/"
    
    Args:
        entry_url: Entry point URL
        
    Returns:
        Base path for filtering internal links
    """
    parsed = urlparse(entry_url)
    path = parsed.path
    
    # Remove filename if present
    if "/" in path:
        # Check if last segment looks like a file
        last_segment = path.rsplit("/", 1)[-1]
        if "." in last_segment:
            # Remove filename
            path = path.rsplit("/", 1)[0] + "/"
        elif not path.endswith("/"):
            path = path + "/"
    
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        path,
        "", "", "",
    ))


def is_internal_link(url: str, base_path: str) -> bool:
    """Check if URL is internal to the base path.
    
    Args:
        url: URL to check
        base_path: Base path URL
        
    Returns:
        True if URL is within base path
    """
    if not url or not base_path:
        return False
    
    # Must start with base path
    return url.startswith(base_path)


def should_skip_url(
    url: str,
    skip_extensions: set[str],
    content_type: ContentType = ContentType.REFERENCE,
) -> bool:
    """Check if URL should be skipped.
    
    Args:
        url: URL to check
        skip_extensions: File extensions to skip
        content_type: Discovery content type
        
    Returns:
        True if URL should be skipped
    """
    if not url:
        return True
    
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    # Check file extensions
    for ext in skip_extensions:
        if path.endswith(ext):
            return True
    
    # Check base patterns
    for pattern in BASE_SKIP_PATTERNS:
        if pattern.search(url):
            return True

    # API/javadoc links are only allowed in api-doc mode
    if content_type != ContentType.API_DOC:
        for pattern in REFERENCE_ONLY_SKIP_PATTERNS:
            if pattern.search(url):
                return True
    
    return False


def flatten_nav_items(items: list[dict]) -> list[tuple[str, str | None]]:
    """Recursively flatten navigation items to (url, title) pairs.
    
    Args:
        items: List of navigation items from parser.extract_sidebar()
        
    Returns:
        List of (url, title) tuples
    """
    results: list[tuple[str, str | None]] = []
    
    for item in items:
        url = item.get("url", "")
        title = item.get("title")
        
        if url:
            results.append((url, title))
        
        children = item.get("children", [])
        if children:
            results.extend(flatten_nav_items(children))
    
    return results


# =============================================================================
# Discovery Class
# =============================================================================


class SpringDocDiscovery:
    """Async discovery class for Spring documentation.
    
    Discovers all documentation URLs within a Spring module/version
    by extracting links from navigation sidebars.
    
    Usage:
        discovery = SpringDocDiscovery()
        result = await discovery.discover(
            entry_url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            version=SpringVersion(module=SpringModule.BOOT, major=4),
        )
        targets = result.to_scrape_targets()
    """
    
    def __init__(self, config: DiscoveryConfig | None = None) -> None:
        """Initialize discovery with configuration.
        
        Args:
            config: Discovery configuration. Uses defaults if not provided.
        """
        self.config = config or DiscoveryConfig()
        self._parser = SpringDocParser(self.config.parser_config)
        
        # State for discovery
        self._visited: set[str] = set()
        self._discovered: list[DiscoveredLink] = []
        self._total_found = 0
        self._duplicates = 0
        self._filtered = 0
    
    def _reset_state(self) -> None:
        """Reset discovery state for new run."""
        self._visited.clear()
        self._discovered.clear()
        self._total_found = 0
        self._duplicates = 0
        self._filtered = 0
    
    async def _extract_links_from_page(
        self,
        html: str,
        page_url: str,
        base_path: str,
        content_type: ContentType = ContentType.REFERENCE,
    ) -> list[tuple[str, str | None]]:
        """Extract links from a page's sidebar navigation.
        
        Args:
            html: Page HTML content
            page_url: URL of the page
            base_path: Base path for filtering
            
        Returns:
            List of (normalized_url, title) tuples
        """
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, "lxml")
        
        # Extract sidebar navigation
        sidebar_items = self._parser.extract_sidebar(soup)
        raw_links = flatten_nav_items(sidebar_items)
        
        # Also extract links from main content for additional coverage
        content_links: list[tuple[str, str | None]] = []
        
        # Find main content
        try:
            content = self._parser.extract_main_content(soup)
            for anchor in content.find_all("a", href=True):
                href = anchor.get("href", "")
                title = anchor.get_text(strip=True)
                if href:
                    content_links.append((href, title or None))
        except ContentExtractionError:
            pass
        
        # Combine and normalize
        all_links: list[tuple[str, str | None]] = []
        seen_urls: set[str] = set()
        
        for href, title in raw_links + content_links:
            # Normalize URL
            normalized = normalize_url(href, page_url)
            
            if not normalized:
                continue
            
            # Skip if already seen in this extraction
            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)
            
            # Check if internal and should not be skipped
            if not is_internal_link(normalized, base_path):
                continue
            
            if should_skip_url(normalized, self.config.skip_extensions, content_type):
                continue
            
            all_links.append((normalized, title))
        
        return all_links
    
    async def discover(
        self,
        entry_url: str,
        module: SpringModule,
        version: SpringVersion,
        content_type: ContentType = ContentType.REFERENCE,
    ) -> DiscoveryResult:
        """Discover all documentation URLs from entry point.
        
        Navigates to entry URL, extracts sidebar links, and recursively
        discovers all internal documentation pages.
        
        Args:
            entry_url: Starting URL for discovery
            module: Spring module being discovered
            version: Spring version
            content_type: Discovery content type
            
        Returns:
            DiscoveryResult with all discovered links
        """
        logger.info(f"Starting discovery from: {entry_url}")
        
        # Reset state
        self._reset_state()
        
        # Create result object
        result = DiscoveryResult(
            entry_url=entry_url,
            module=module,
            version=version,
            status=DiscoveryStatus.IN_PROGRESS,
        )
        
        # Normalize entry URL and extract base path
        entry_url = normalize_url(entry_url) or entry_url
        base_path = extract_base_path(entry_url)
        
        logger.debug(f"Base path for filtering: {base_path}")
        
        try:
            # Queue for BFS traversal: (url, depth, source_url, title)
            queue: list[tuple[str, int, str | None, str | None]] = [
                (entry_url, 0, None, None)
            ]
            
            async with SpringBrowser(self.config.browser_config) as browser:
                while queue and len(self._discovered) < self.config.max_links:
                    url, depth, source_url, title = queue.pop(0)
                    
                    # Skip if already visited
                    if url in self._visited:
                        self._duplicates += 1
                        continue
                    
                    # Mark as visited
                    self._visited.add(url)
                    
                    # Create discovered link
                    link = DiscoveredLink(
                        url=url,
                        depth=depth,
                        source_url=source_url,
                        title=title,
                    )
                    self._discovered.append(link)
                    
                    logger.debug(
                        f"Discovered [{depth}]: {url[:80]}..."
                        if len(url) > 80 else f"Discovered [{depth}]: {url}"
                    )
                    
                    # Check depth limit for further exploration
                    if depth >= self.config.max_depth:
                        continue
                    
                    # Navigate and extract more links
                    try:
                        await browser.navigate_with_retry(url)
                        html = await browser.get_html()
                        
                        new_links = await self._extract_links_from_page(
                            html,
                            url,
                            base_path,
                            content_type=content_type,
                        )
                        
                        self._total_found += len(new_links)
                        
                        # Add new links to queue
                        for link_url, link_title in new_links:
                            if link_url not in self._visited:
                                queue.append((
                                    link_url,
                                    depth + 1,
                                    url,
                                    link_title,
                                ))
                            else:
                                self._duplicates += 1
                                
                    except (NavigationError, ContentExtractionError) as e:
                        if (
                            isinstance(e, NavigationError)
                            and self.config.suppress_http_404
                            and e.status_code == 404
                        ):
                            if self._discovered and self._discovered[-1].url == url:
                                self._discovered.pop()
                            else:
                                self._discovered = [
                                    link for link in self._discovered if link.url != url
                                ]
                            logger.info("Suppressed 404 discovery link: %s", url)
                            continue
                        logger.warning(f"Failed to extract links from {url}: {e}")
                        continue
            
            # Calculate filtered count
            self._filtered = self._total_found - len(self._discovered) - self._duplicates
            if self._filtered < 0:
                self._filtered = 0
            
            # Build final result
            result = DiscoveryResult(
                entry_url=entry_url,
                module=module,
                version=version,
                links=self._discovered,
                total_found=self._total_found,
                duplicates_removed=self._duplicates,
                filtered_out=self._filtered,
                status=DiscoveryStatus.COMPLETED,
                started_at=result.started_at,
                completed_at=datetime.now(timezone.utc),
            )
            
            logger.info(
                f"Discovery completed: {result.link_count} links found "
                f"({self._duplicates} duplicates, {self._filtered} filtered)"
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Discovery failed: {e}")
            return DiscoveryResult(
                entry_url=entry_url,
                module=module,
                version=version,
                links=self._discovered,
                total_found=self._total_found,
                duplicates_removed=self._duplicates,
                filtered_out=self._filtered,
                status=DiscoveryStatus.FAILED,
                error_message=str(e),
                started_at=result.started_at,
                completed_at=datetime.now(timezone.utc),
            )


__all__ = [
    # Config
    "DiscoveryConfig",
    "DiscoveryStatus",
    # Models
    "DiscoveredLink",
    "DiscoveryResult",
    # Class
    "SpringDocDiscovery",
    # Utilities
    "normalize_url",
    "extract_base_path",
    "is_internal_link",
    "should_skip_url",
    "flatten_nav_items",
    # Constants
    "SKIP_EXTENSIONS",
    "BASE_SKIP_PATTERNS",
    "REFERENCE_ONLY_SKIP_PATTERNS",
]
