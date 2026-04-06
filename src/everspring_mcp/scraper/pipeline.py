"""EverSpring MCP - AWS Lambda scraper pipeline.

This module provides the scraping pipeline for AWS Lambda execution:
- PipelineConfig: Configuration with environment variable support
- ScrapeTarget: Target URL specification
- S3Client: Secure S3 client wrapper with IAM authentication
- ScraperPipeline: Orchestration class for browse → parse → upload
- lambda_handler: AWS Lambda entry point

Also integrates with discovery module for automatic URL discovery.
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Self

from everspring_mcp.models.base import SHA256Hash, VersionedModel, compute_hash
from everspring_mcp.models.content import ContentType, ScrapedPage
from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.models.sync import S3ObjectRef, SyncManifest
from everspring_mcp.utils.logging import get_logger, setup_logging
from .browser import BrowserConfig, SpringBrowser
from .parser import ParserConfig, SpringDocParser
from .registry import SubmoduleRegistry, SubmoduleTarget
from .exceptions import (
    ContentExtractionError,
    NavigationError,
    RateLimitError,
    ScraperError,
)

if TYPE_CHECKING:
    from .discovery import DiscoveryResult

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logger = get_logger("scraper.pipeline")


class PipelineStatus(str, Enum):
    """Status of a pipeline operation."""
    
    SUCCESS = "success"
    SKIPPED = "skipped"  # Content unchanged (hash match)
    FAILED = "failed"


class ScrapeTarget(VersionedModel):
    """Target URL specification for scraping.
    
    Attributes:
        url: URL to scrape
        module: Spring module (BOOT, FRAMEWORK, etc.)
        submodule: Optional submodule key (e.g., redis)
        major: Major version number (optional if auto-detected)
        minor: Minor version number (optional)
        patch: Patch version number (optional)
        content_type: Type of documentation
        version_selector: CSS selector for version extraction
    """
    
    url: str = Field(
        pattern=r"^https?://[^\s]+$",
        description="URL to scrape",
    )
    module: SpringModule = Field(
        description="Spring module this page belongs to",
    )
    submodule: str | None = Field(
        default=None,
        description="Optional submodule key",
    )
    major: int | None = Field(
        default=None,
        ge=1,
        description="Major version number",
    )
    minor: int = Field(
        default=0,
        ge=0,
        description="Minor version number",
    )
    patch: int = Field(
        default=0,
        ge=0,
        description="Patch version number",
    )
    content_type: ContentType = Field(
        default=ContentType.REFERENCE,
        description="Type of documentation",
    )
    version_selector: str = Field(
        default="span.version",
        description="CSS selector for version extraction",
    )
    
    @field_validator("module", mode="before")
    @classmethod
    def coerce_module(cls, v: Any) -> SpringModule:
        """Convert string module name to SpringModule enum."""
        if isinstance(v, SpringModule):
            return v
        if isinstance(v, str):
            # Try direct value match (e.g., "spring-boot")
            for member in SpringModule:
                if member.value == v:
                    return member
            # Try name match (e.g., "BOOT")
            try:
                return SpringModule[v.upper().replace("-", "_").replace("SPRING_", "")]
            except KeyError:
                pass
            raise ValueError(f"Invalid Spring module: {v}")
        raise TypeError(f"Expected string or SpringModule, got {type(v)}")
    
    @field_validator("content_type", mode="before")
    @classmethod
    def coerce_content_type(cls, v: Any) -> ContentType:
        """Convert string content type to ContentType enum."""
        if isinstance(v, ContentType):
            return v
        if isinstance(v, str):
            # Try direct value match
            for member in ContentType:
                if member.value == v:
                    return member
            # Try name match
            try:
                return ContentType[v.upper().replace("-", "_")]
            except KeyError:
                pass
            raise ValueError(f"Invalid content type: {v}")
        raise TypeError(f"Expected string or ContentType, got {type(v)}")
    
    @model_validator(mode="after")
    def validate_version(self) -> Self:
        """Ensure version meets minimum requirements."""
        if self.major is None:
            return self
        # Create SpringVersion to validate minimum version
        SpringVersion(
            module=self.module,
            major=self.major,
            minor=self.minor,
            patch=self.patch,
        )
        return self
    
    @property
    def version(self) -> SpringVersion | None:
        """Get SpringVersion from target specification."""
        if self.major is None:
            return None
        return SpringVersion(
            module=self.module,
            major=self.major,
            minor=self.minor,
            patch=self.patch,
        )
    
    def s3_key_prefix_for(self, version: SpringVersion) -> str:
        """Generate S3 key prefix for this target and version."""
        version_str = f"{version.major}.{version.minor}.{version.patch}"
        if self.submodule:
            return f"{self.module.value}/{self.submodule}/{version_str}"
        return f"{self.module.value}/{version_str}"


class PipelineConfig(BaseModel):
    """Configuration for ScraperPipeline.
    
    Supports loading from environment variables for AWS Lambda.
    
    Attributes:
        s3_bucket: S3 bucket name for content storage
        s3_prefix: Prefix for all S3 keys
        aws_region: AWS region for S3 client
        enable_hash_check: Enable incremental sync via hash comparison
        browser_config: Configuration for SpringBrowser
        parser_config: Configuration for SpringDocParser
    """
    
    model_config = ConfigDict(frozen=True)
    
    # Required environment variables for Lambda
    ENV_BUCKET: ClassVar[str] = "EVERSPRING_S3_BUCKET"
    ENV_PREFIX: ClassVar[str] = "EVERSPRING_S3_PREFIX"
    ENV_REGION: ClassVar[str] = "AWS_REGION"
    
    s3_bucket: str = Field(
        pattern=r"^[a-z0-9][a-z0-9\-\.]{1,61}[a-z0-9]$",
        description="S3 bucket name",
    )
    s3_prefix: str = Field(
        default="docs",
        description="S3 key prefix",
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region",
    )
    enable_hash_check: bool = Field(
        default=True,
        description="Enable incremental sync via hash comparison",
    )
    browser_config: BrowserConfig = Field(
        default_factory=BrowserConfig,
        description="Browser configuration",
    )
    parser_config: ParserConfig = Field(
        default_factory=ParserConfig,
        description="Parser configuration",
    )
    
    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Create configuration from environment variables.
        
        Required environment variables:
        - EVERSPRING_S3_BUCKET: S3 bucket name
        
        Optional environment variables:
        - EVERSPRING_S3_PREFIX: S3 key prefix (default: "docs")
        - AWS_REGION: AWS region (default: "us-east-1")
        
        Returns:
            PipelineConfig from environment
            
        Raises:
            ValueError: If required environment variables are missing
        """
        bucket = os.environ.get(cls.ENV_BUCKET)
        if not bucket:
            raise ValueError(
                f"Missing required environment variable: {cls.ENV_BUCKET}"
            )
        
        return cls(
            s3_bucket=bucket,
            s3_prefix=os.environ.get(cls.ENV_PREFIX, "docs"),
            aws_region=os.environ.get(cls.ENV_REGION, "us-east-1"),
        )


class ScrapeResult(VersionedModel):
    """Result of a single scrape operation.
    
    Attributes:
        target: Original scrape target
        status: Operation status
        s3_ref: S3 reference if uploaded
        content_hash: SHA-256 hash of content
        error_message: Error message if failed
        scraped_at: Timestamp of scraping
    """
    
    target: ScrapeTarget = Field(
        description="Original scrape target",
    )
    status: PipelineStatus = Field(
        description="Operation status",
    )
    s3_ref: S3ObjectRef | None = Field(
        default=None,
        description="S3 reference if uploaded",
    )
    content_hash: SHA256Hash | None = Field(
        default=None,
        description="SHA-256 hash of content",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if failed",
    )
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of scraping",
    )
    
    @classmethod
    def success(
        cls,
        target: ScrapeTarget,
        s3_ref: S3ObjectRef,
        content_hash: str,
    ) -> ScrapeResult:
        """Create success result."""
        return cls(
            target=target,
            status=PipelineStatus.SUCCESS,
            s3_ref=s3_ref,
            content_hash=content_hash,
        )
    
    @classmethod
    def skipped(
        cls,
        target: ScrapeTarget,
        content_hash: str,
    ) -> ScrapeResult:
        """Create skipped result (content unchanged)."""
        return cls(
            target=target,
            status=PipelineStatus.SKIPPED,
            content_hash=content_hash,
        )
    
    @classmethod
    def failed(
        cls,
        target: ScrapeTarget,
        error: str,
    ) -> ScrapeResult:
        """Create failed result."""
        return cls(
            target=target,
            status=PipelineStatus.FAILED,
            error_message=error,
        )


class S3Client:
    """Secure S3 client wrapper with IAM authentication.
    
    Uses IAM role-based authentication (no hardcoded credentials).
    Lambda execution role provides necessary permissions.
    
    Attributes:
        bucket: S3 bucket name
        prefix: Key prefix for all operations
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
    ) -> None:
        """Initialize S3 client.
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all operations
            region: AWS region
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        
        # Use IAM role - Lambda automatically uses execution role
        # No hardcoded credentials
        self._client = boto3.client("s3", region_name=region)
        logger.info(f"S3 client initialized for bucket: {bucket}")
    
    def _full_key(self, key: str) -> str:
        """Get full S3 key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{key.lstrip('/')}"
        return key.lstrip("/")
    
    def upload_content(
        self,
        content: str,
        key: str,
        content_hash: str,
        metadata: dict[str, str] | None = None,
    ) -> S3ObjectRef:
        """Upload content to S3 with SHA-256 hash verification.
        
        Args:
            content: Content to upload
            key: S3 object key (prefix will be added)
            content_hash: Pre-computed SHA-256 hash for verification
            metadata: Additional metadata to store
            
        Returns:
            S3ObjectRef for uploaded object
            
        Raises:
            ValueError: If content hash doesn't match computed hash
            ClientError: If S3 upload fails
        """
        # Verify content hash before upload
        computed_hash = compute_hash(content)
        if computed_hash != content_hash:
            raise ValueError(
                f"Content hash mismatch: expected {content_hash}, got {computed_hash}"
            )
        
        full_key = self._full_key(key)
        content_bytes = content.encode("utf-8")
        
        # Build metadata
        s3_metadata = {
            "content-hash": content_hash,
            "schema-version": str(VersionedModel.CURRENT_SCHEMA_VERSION),
            "uploaded-at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            s3_metadata.update(metadata)
        
        try:
            response = self._client.put_object(
                Bucket=self.bucket,
                Key=full_key,
                Body=content_bytes,
                ContentType="text/markdown",
                Metadata=s3_metadata,
            )
            
            logger.info(f"Uploaded to s3://{self.bucket}/{full_key}")
            
            return S3ObjectRef(
                bucket=self.bucket,
                key=full_key,
                etag=response.get("ETag", "").strip('"'),
                size_bytes=len(content_bytes),
                content_hash=content_hash,
                last_modified=datetime.now(timezone.utc),
            )
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    def upload_json(
        self,
        data: dict[str, Any] | BaseModel,
        key: str,
        content_hash: str | None = None,
    ) -> S3ObjectRef:
        """Upload JSON data to S3.
        
        Args:
            data: Dictionary or Pydantic model to upload
            key: S3 object key
            content_hash: Optional pre-computed hash
            
        Returns:
            S3ObjectRef for uploaded object
        """
        if isinstance(data, BaseModel):
            json_str = data.model_dump_json(indent=2)
        else:
            json_str = json.dumps(data, indent=2, default=str)
        
        if content_hash is None:
            content_hash = compute_hash(json_str)
        
        full_key = self._full_key(key)
        content_bytes = json_str.encode("utf-8")
        
        s3_metadata = {
            "content-hash": content_hash,
            "content-type": "application/json",
            "schema-version": str(VersionedModel.CURRENT_SCHEMA_VERSION),
        }
        
        try:
            response = self._client.put_object(
                Bucket=self.bucket,
                Key=full_key,
                Body=content_bytes,
                ContentType="application/json",
                Metadata=s3_metadata,
            )
            
            logger.info(f"Uploaded JSON to s3://{self.bucket}/{full_key}")
            
            return S3ObjectRef(
                bucket=self.bucket,
                key=full_key,
                etag=response.get("ETag", "").strip('"'),
                size_bytes=len(content_bytes),
                content_hash=content_hash,
                last_modified=datetime.now(timezone.utc),
            )
            
        except ClientError as e:
            logger.error(f"S3 JSON upload failed: {e}")
            raise
    
    def get_content_hash(self, key: str) -> str | None:
        """Get content hash from S3 object metadata.
        
        Used for incremental sync - skip upload if hash matches.
        
        Args:
            key: S3 object key
            
        Returns:
            Content hash if exists, None otherwise
        """
        full_key = self._full_key(key)
        
        try:
            response = self._client.head_object(
                Bucket=self.bucket,
                Key=full_key,
            )
            return response.get("Metadata", {}).get("content-hash")
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                return None
            logger.error(f"Failed to get object metadata: {e}")
            raise
    
    def check_exists(self, key: str) -> bool:
        """Check if S3 object exists.
        
        Args:
            key: S3 object key
            
        Returns:
            True if object exists
        """
        return self.get_content_hash(key) is not None
    
    def should_upload(self, key: str, content_hash: str) -> bool:
        """Check if content should be uploaded (hash changed).
        
        Implements incremental sync logic to minimize S3 operations.
        
        Args:
            key: S3 object key
            content_hash: New content hash
            
        Returns:
            True if upload needed (new or changed content)
        """
        existing_hash = self.get_content_hash(key)
        if existing_hash is None:
            logger.debug(f"Object doesn't exist: {key}")
            return True
        if existing_hash != content_hash:
            logger.debug(f"Hash changed for {key}: {existing_hash} -> {content_hash}")
            return True
        logger.debug(f"Hash unchanged for {key}, skipping upload")
        return False
    
    def download_manifest(self, key: str) -> SyncManifest | None:
        """Download sync manifest from S3.
        
        Args:
            key: S3 object key for manifest
            
        Returns:
            SyncManifest if exists, None otherwise
        """
        full_key = self._full_key(key)
        
        try:
            response = self._client.get_object(
                Bucket=self.bucket,
                Key=full_key,
            )
            body = response["Body"].read().decode("utf-8")
            return SyncManifest.model_validate_json(body)
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "NoSuchKey":
                return None
            logger.error(f"Failed to download manifest: {e}")
            raise


class ScraperPipeline:
    """Orchestration class for scraping pipeline.
    
    Coordinates browser navigation, HTML parsing, hash checking,
    and S3 upload for Spring documentation scraping.
    
    Usage:
        config = PipelineConfig.from_env()
        pipeline = ScraperPipeline(config)
        
        target = ScrapeTarget(
            url="https://docs.spring.io/spring-boot/docs/current/reference/html/",
            module=SpringModule.BOOT,
            major=4,
        )
        result = await pipeline.scrape_url(target)
    """
    
    def __init__(self, config: PipelineConfig) -> None:
        """Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self._s3 = S3Client(
            bucket=config.s3_bucket,
            prefix=config.s3_prefix,
            region=config.aws_region,
        )
        self._parser = SpringDocParser(config.parser_config)
        logger.info("ScraperPipeline initialized")
    
    def _generate_s3_key(self, target: ScrapeTarget, content_hash: str) -> str:
        """Generate S3 key for scraped content.
        
        Format: {module}/{submodule?}/{version}/{url_hash}.md
        
        Args:
            target: Scrape target
            content_hash: Content hash (first 8 chars used in key)
            
        Returns:
            S3 object key
        """
        # Use URL hash for unique identification
        url_hash = compute_hash(target.url)[:16]
        if not target.version:
            raise ValueError("Target version is required for S3 key generation")
        return f"{target.s3_key_prefix_for(target.version)}/{url_hash}.md"
    
    def _generate_metadata_key(self, target: ScrapeTarget) -> str:
        """Generate S3 key for scraped page metadata.
        
        Args:
            target: Scrape target
            
        Returns:
            S3 object key for metadata JSON
        """
        url_hash = compute_hash(target.url)[:16]
        if not target.version:
            raise ValueError("Target version is required for metadata key generation")
        return f"{target.s3_key_prefix_for(target.version)}/metadata/{url_hash}.json"
    
    async def scrape_url(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape single URL and upload to S3.
        
        Pipeline: Browse → Parse → Hash Check → S3 Upload
        
        Args:
            target: Scrape target specification
            
        Returns:
            ScrapeResult with operation status
        """
        logger.info(f"Scraping: {target.url}")
        
        try:
            # Step 1: Browse - Navigate and get HTML
            async with SpringBrowser(self.config.browser_config) as browser:
                await browser.navigate_with_retry(target.url)
                raw_html = await browser.get_html()
            
            logger.debug(f"Retrieved HTML: {len(raw_html)} chars")
            
            # Step 2: Parse - Convert to ScrapedPage model
            parser = self._parser
            if target.version_selector != self.config.parser_config.version_selectors[0]:
                parser = SpringDocParser(
                    self.config.parser_config.model_copy(
                        update={"version_selectors": [target.version_selector]}
                    )
                )
            scraped_page = parser.parse(
                html=raw_html,
                url=target.url,
                module=target.module,
                version=target.version,
                submodule=target.submodule,
                content_type=target.content_type,
            )
            if target.version_selector != self.config.parser_config.version_selectors[0]:
                logger.debug(
                    "Target version selector override in use: %s",
                    target.version_selector,
                )
            if target.version is None:
                target = target.model_copy(
                    update={
                        "major": scraped_page.version.major,
                        "minor": scraped_page.version.minor,
                        "patch": scraped_page.version.patch,
                    }
                )
            elif scraped_page.version != target.version:
                raise ContentExtractionError(
                    f"Version mismatch: expected {target.version.version_string}, got {scraped_page.version.version_string}",
                    url=target.url,
                )
            
            content_hash = scraped_page.content_hash
            s3_key = self._generate_s3_key(target, content_hash)
            
            # Step 3: Hash Check - Skip if unchanged
            if self.config.enable_hash_check:
                if not self._s3.should_upload(s3_key, content_hash):
                    logger.info(f"Content unchanged, skipping: {target.url}")
                    return ScrapeResult.skipped(target, content_hash)
            
            # Step 4: S3 Upload - Upload markdown content
            s3_ref = self._s3.upload_content(
                content=scraped_page.markdown_content,
                key=s3_key,
                content_hash=content_hash,
                metadata={
                    "source-url": target.url,
                    "module": target.module.value,
                    "version": scraped_page.version.version_string,
                    "title": scraped_page.title[:256],  # Limit metadata size
                    "submodule": target.submodule or "",
                },
            )
            
            # Upload metadata JSON
            metadata_key = self._generate_metadata_key(target)
            self._s3.upload_json(
                data=scraped_page.model_dump(
                    exclude={"raw_html"},  # Exclude large raw HTML
                    mode="json",
                ),
                key=metadata_key,
            )
            
            logger.info(f"Successfully scraped and uploaded: {target.url}")
            return ScrapeResult.success(target, s3_ref, content_hash)
            
        except (NavigationError, RateLimitError, ContentExtractionError) as e:
            logger.error(f"Scraping failed for {target.url}: {e}")
            return ScrapeResult.failed(target, str(e))
        except ClientError as e:
            logger.error(f"S3 operation failed for {target.url}: {e}")
            return ScrapeResult.failed(target, f"S3 error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error scraping {target.url}")
            return ScrapeResult.failed(target, f"Unexpected error: {e}")
    
    async def process_batch(
        self,
        targets: list[ScrapeTarget],
        concurrency: int = 3,
    ) -> list[ScrapeResult]:
        """Process multiple scrape targets with controlled concurrency.
        
        Args:
            targets: List of scrape targets
            concurrency: Maximum concurrent scrapes
            
        Returns:
            List of ScrapeResult for each target
        """
        if not targets:
            return []
        
        logger.info(f"Processing batch of {len(targets)} targets")
        
        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        
        async def scrape_with_semaphore(target: ScrapeTarget) -> ScrapeResult:
            async with semaphore:
                return await self.scrape_url(target)
        
        tasks = [scrape_with_semaphore(target) for target in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results: list[ScrapeResult] = []
        for target, result in zip(targets, results):
            if isinstance(result, Exception):
                processed_results.append(
                    ScrapeResult.failed(target, str(result))
                )
            else:
                processed_results.append(result)
        
        # Log summary
        success_count = sum(1 for r in processed_results if r.status == PipelineStatus.SUCCESS)
        skipped_count = sum(1 for r in processed_results if r.status == PipelineStatus.SKIPPED)
        failed_count = sum(1 for r in processed_results if r.status == PipelineStatus.FAILED)
        
        logger.info(
            f"Batch complete: {success_count} success, "
            f"{skipped_count} skipped, {failed_count} failed"
        )
        
        return processed_results
    
    async def discover_and_scrape(
        self,
        entry_url: str | None,
        module: SpringModule | None,
        version: SpringVersion | None,
        content_type: ContentType = ContentType.REFERENCE,
        concurrency: int = 3,
        submodule: str | None = None,
    ) -> tuple[DiscoveryResult, list[ScrapeResult]]:
        """Discover and scrape all documentation pages.
        
        Combines discovery and scraping into a single operation:
        1. Discover all URLs from entry point
        2. Convert to ScrapeTargets
        3. Process batch scraping
        
        Args:
            entry_url: Starting URL for discovery
            module: Spring module
            version: Spring version
            content_type: Documentation type for all pages
            concurrency: Maximum concurrent scrapes
            
        Returns:
            Tuple of (DiscoveryResult, list of ScrapeResult)
        """
        from .discovery import SpringDocDiscovery, DiscoveryConfig
        
        if entry_url and module and version:
            return await self._discover_and_scrape_target(
                entry_url=entry_url,
                module=module,
                version=version,
                content_type=content_type,
                concurrency=concurrency,
                submodule=submodule,
            )

        if entry_url or module or version:
            raise ValueError("entry_url, module, and version must be provided together")

        return await self._discover_and_scrape_registry(
            content_type=content_type,
            concurrency=concurrency,
        )

    async def _discover_and_scrape_target(
        self,
        entry_url: str,
        module: SpringModule,
        version: SpringVersion,
        content_type: ContentType,
        concurrency: int,
        submodule: str | None,
    ) -> tuple[DiscoveryResult, list[ScrapeResult]]:
        from .discovery import SpringDocDiscovery, DiscoveryConfig
        
        logger.info(f"Starting discover_and_scrape from: {entry_url}")
        
        # Create discovery with same browser config
        discovery_config = DiscoveryConfig(
            browser_config=self.config.browser_config,
            parser_config=self.config.parser_config,
        )
        discovery = SpringDocDiscovery(discovery_config)
        
        # Run discovery
        discovery_result = await discovery.discover(
            entry_url,
            module,
            version,
            content_type=content_type,
        )
        
        if discovery_result.link_count == 0:
            logger.warning("No links discovered, nothing to scrape")
            return discovery_result, []
        
        logger.info(f"Discovered {discovery_result.link_count} links, starting batch scrape")
        
        # Convert to targets
        targets = discovery_result.to_scrape_targets(
            content_type,
            submodule=submodule,
            auto_version=content_type != ContentType.API_DOC,
        )
        
        # Run batch scraping
        scrape_results = await self.process_batch(targets, concurrency)
        
        return discovery_result, scrape_results

    async def _discover_and_scrape_registry(
        self,
        content_type: ContentType,
        concurrency: int,
    ) -> tuple[DiscoveryResult, list[ScrapeResult]]:
        registry_path = Path("config") / "submodules.json"
        registry = SubmoduleRegistry.load(registry_path)
        return await self.discover_and_scrape_registry(
            registry=registry,
            content_type=content_type,
            concurrency=concurrency,
        )

    async def discover_and_scrape_registry(
        self,
        registry: SubmoduleRegistry,
        content_type: ContentType = ContentType.REFERENCE,
        concurrency: int = 3,
    ) -> tuple[DiscoveryResult, list[ScrapeResult]]:
        """Discover and scrape using an explicit submodule registry."""
        all_results: list[ScrapeResult] = []
        last_discovery: DiscoveryResult | None = None

        for target in registry.targets:
            discovery_result, scrape_results = await self._discover_for_registry_target(
                target=target,
                content_type=content_type,
                concurrency=concurrency,
            )
            last_discovery = discovery_result
            all_results.extend(scrape_results)

        if not last_discovery:
            raise ValueError("Submodule registry is empty")
        return last_discovery, all_results

    async def _discover_for_registry_target(
        self,
        target: SubmoduleTarget,
        content_type: ContentType,
        concurrency: int,
    ) -> tuple[DiscoveryResult, list[ScrapeResult]]:
        discovery_config = DiscoveryConfig(
            browser_config=self.config.browser_config,
            parser_config=self.config.parser_config.model_copy(
                update={"version_selectors": [target.version_selector]}
            ),
        )
        discovery = SpringDocDiscovery(discovery_config)

        entry_version = SpringVersion(
            module=target.module_key,
            major=target.module_key.minimum_supported_version,
        )
        discovery_result = await discovery.discover(
            target.base_url,
            target.module_key,
            entry_version,
            content_type=content_type,
        )
        targets = [
            ScrapeTarget(
                url=link.url,
                module=target.module_key,
                submodule=target.submodule_key,
                major=None,
                minor=0,
                patch=0,
                content_type=content_type,
                version_selector=target.version_selector,
            )
            for link in discovery_result.links
        ]
        scrape_results = await self.process_batch(targets, concurrency)
        return discovery_result, scrape_results


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entry point for scraper pipeline.
    
    Event format:
    {
        "targets": [
            {
                "url": "https://docs.spring.io/spring-boot/everspring_mcp..",
                "module": "spring-boot",
                "major": 4,
                "minor": 0,
                "patch": 0
            }
        ],
        "concurrency": 3  // optional
    }
    
    Response format:
    {
        "statusCode": 200,
        "body": {
            "processed": 5,
            "results": [
                {
                    "url": "everspring_mcp..",
                    "status": "success",
                    "s3_key": "everspring_mcp..",
                    "content_hash": "everspring_mcp.."
                }
            ]
        }
    }
    
    Args:
        event: Lambda event with targets array
        context: Lambda context (unused)
        
    Returns:
        Lambda response with status and results
    """
    # Configure centralized logging for Lambda entrypoint usage.
    setup_logging(level="INFO", console=True, file=True, name="scraper")
    
    logger.info(f"Lambda invoked with event: {json.dumps(event, default=str)[:500]}")
    
    try:
        # Load configuration from environment
        config = PipelineConfig.from_env()
        
        # Parse targets from event
        raw_targets = event.get("targets", [])
        if not raw_targets:
            return {
                "statusCode": 400,
                "body": {"error": "No targets provided in event"},
            }
        
        targets: list[ScrapeTarget] = []
        for raw in raw_targets:
            try:
                targets.append(ScrapeTarget(**raw))
            except Exception as e:
                logger.warning(f"Invalid target: {raw} - {e}")
        
        if not targets:
            return {
                "statusCode": 400,
                "body": {"error": "No valid targets after parsing"},
            }
        
        # Get concurrency from event (default: 3)
        concurrency = event.get("concurrency", 3)
        
        # Run pipeline
        pipeline = ScraperPipeline(config)
        results = asyncio.run(pipeline.process_batch(targets, concurrency))
        
        # Format response
        response_results = []
        for result in results:
            response_results.append({
                "url": result.target.url,
                "status": result.status.value,
                "s3_key": result.s3_ref.key if result.s3_ref else None,
                "content_hash": result.content_hash,
                "error": result.error_message,
            })
        
        # Count statuses
        status_counts = {
            "success": sum(1 for r in results if r.status == PipelineStatus.SUCCESS),
            "skipped": sum(1 for r in results if r.status == PipelineStatus.SKIPPED),
            "failed": sum(1 for r in results if r.status == PipelineStatus.FAILED),
        }
        
        logger.info(f"Pipeline complete: {status_counts}")
        
        return {
            "statusCode": 200,
            "body": {
                "processed": len(results),
                "status_counts": status_counts,
                "results": response_results,
            },
        }
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {
            "statusCode": 500,
            "body": {"error": f"Configuration error: {e}"},
        }
    except Exception as e:
        logger.exception("Pipeline failed with unexpected error")
        return {
            "statusCode": 500,
            "body": {"error": f"Pipeline failed: {e}"},
        }


# Allow running as standalone script for testing
if __name__ == "__main__":
    # Example usage for local testing
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler("everspring_scraper.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Test event
    test_event = {
        "targets": [
            {
                "url": "https://docs.spring.io/spring-boot/reference/",
                "module": "spring-boot",
                "major": 4,
            }
        ]
    }
    
    # Ensure environment variables are set
    if not os.environ.get("EVERSPRING_S3_BUCKET"):
        print("Error: EVERSPRING_S3_BUCKET environment variable not set")
        sys.exit(1)
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2, default=str))


__all__ = [
    # Config
    "PipelineConfig",
    "ScrapeTarget",
    "ScrapeResult",
    "PipelineStatus",
    # Clients
    "S3Client",
    # Pipeline
    "ScraperPipeline",
    # Lambda
    "lambda_handler",
]
