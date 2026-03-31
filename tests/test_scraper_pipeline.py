"""EverSpring MCP - Pipeline and S3 interaction tests.

Tests for:
- ScraperPipeline async operations
- S3Client with SHA-256 metadata
- Incremental sync (hash comparison)
- lambda_handler event parsing
- ScrapeTarget validation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import pytest
from moto import mock_aws
from pydantic import ValidationError

from everspring_mcp.models.base import compute_hash
from everspring_mcp.models.content import ContentType
from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.scraper.pipeline import (
    PipelineConfig,
    PipelineStatus,
    S3Client,
    ScrapeResult,
    ScrapeTarget,
    ScraperPipeline,
    lambda_handler,
)


# =============================================================================
# ScrapeTarget Validation Tests
# =============================================================================


class TestScrapeTarget:
    """Tests for ScrapeTarget model validation."""

    def test_valid_scrape_target(self) -> None:
        """Test creating valid ScrapeTarget."""
        target = ScrapeTarget(
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            major=4,
            minor=0,
            patch=0,
        )
        
        assert target.url == "https://docs.spring.io/spring-boot/reference/"
        assert target.module == SpringModule.BOOT
        assert target.major == 4

    def test_scrape_target_version_property(self) -> None:
        """Test ScrapeTarget.version property returns SpringVersion."""
        target = ScrapeTarget(
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            major=4,
            minor=1,
            patch=2,
        )
        
        version = target.version
        assert isinstance(version, SpringVersion)
        assert version.major == 4
        assert version.minor == 1
        assert version.patch == 2

    def test_scrape_target_s3_key_prefix(self) -> None:
        """Test ScrapeTarget.s3_key_prefix generation."""
        target = ScrapeTarget(
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            major=4,
            minor=1,
            patch=0,
        )
        
        assert target.s3_key_prefix == "spring-boot/4.1.0"

    def test_scrape_target_invalid_version(self) -> None:
        """Test ScrapeTarget rejects invalid Spring versions."""
        with pytest.raises(ValidationError):
            ScrapeTarget(
                url="https://docs.spring.io/spring-boot/reference/",
                module=SpringModule.BOOT,
                major=3,  # Boot requires 4+
            )

    def test_scrape_target_invalid_url(self) -> None:
        """Test ScrapeTarget rejects invalid URLs."""
        with pytest.raises(ValidationError):
            ScrapeTarget(
                url="not-a-valid-url",
                module=SpringModule.BOOT,
                major=4,
            )

    def test_scrape_target_content_type_default(self) -> None:
        """Test ScrapeTarget defaults to REFERENCE content type."""
        target = ScrapeTarget(
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            major=4,
        )
        
        assert target.content_type == ContentType.REFERENCE


# =============================================================================
# PipelineConfig Tests
# =============================================================================


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_valid_config(self) -> None:
        """Test creating valid PipelineConfig."""
        config = PipelineConfig(
            s3_bucket="my-bucket",
            s3_prefix="docs",
            aws_region="us-west-2",
        )
        
        assert config.s3_bucket == "my-bucket"
        assert config.s3_prefix == "docs"
        assert config.aws_region == "us-west-2"

    def test_config_defaults(self) -> None:
        """Test PipelineConfig default values."""
        config = PipelineConfig(s3_bucket="my-bucket")
        
        assert config.s3_prefix == "docs"
        assert config.aws_region == "us-east-1"
        assert config.enable_hash_check is True

    def test_config_from_env(self, pipeline_env_vars: None) -> None:
        """Test PipelineConfig.from_env() loads from environment."""
        config = PipelineConfig.from_env()
        
        assert config.s3_bucket == "test-bucket"
        assert config.s3_prefix == "test-docs"
        assert config.aws_region == "us-east-1"

    def test_config_from_env_missing_bucket(self) -> None:
        """Test from_env raises when bucket not set."""
        import os
        
        # Ensure bucket is not set
        os.environ.pop("EVERSPRING_S3_BUCKET", None)
        
        with pytest.raises(ValueError, match="EVERSPRING_S3_BUCKET"):
            PipelineConfig.from_env()

    def test_config_invalid_bucket_name(self) -> None:
        """Test PipelineConfig rejects invalid bucket names."""
        with pytest.raises(ValidationError):
            PipelineConfig(s3_bucket="Invalid_Bucket_Name")


# =============================================================================
# S3Client Tests
# =============================================================================


class TestS3Client:
    """Tests for S3Client operations."""

    def test_s3_client_initialization(self, mock_s3: Any) -> None:
        """Test S3Client initializes correctly."""
        client = S3Client(
            bucket="test-bucket",
            prefix="docs",
            region="us-east-1",
        )
        
        assert client.bucket == "test-bucket"
        assert client.prefix == "docs"

    def test_full_key_with_prefix(self, mock_s3: Any) -> None:
        """Test S3Client._full_key adds prefix."""
        client = S3Client(bucket="test-bucket", prefix="docs")
        
        full_key = client._full_key("spring-boot/4.0.0/page.md")
        assert full_key == "docs/spring-boot/4.0.0/page.md"

    def test_full_key_no_prefix(self, mock_s3: Any) -> None:
        """Test S3Client._full_key without prefix."""
        client = S3Client(bucket="test-bucket", prefix="")
        
        full_key = client._full_key("spring-boot/4.0.0/page.md")
        assert full_key == "spring-boot/4.0.0/page.md"

    def test_upload_content_with_hash_verification(self, mock_s3: Any) -> None:
        """Test upload_content verifies SHA-256 hash."""
        client = S3Client(bucket="test-bucket", prefix="docs")
        
        content = "# Test Content\n\nThis is test content."
        content_hash = compute_hash(content)
        
        s3_ref = client.upload_content(
            content=content,
            key="spring-boot/4.0.0/test.md",
            content_hash=content_hash,
            metadata={"source-url": "https://example.com"},
        )
        
        assert s3_ref.bucket == "test-bucket"
        assert s3_ref.key == "docs/spring-boot/4.0.0/test.md"
        assert s3_ref.content_hash == content_hash

    def test_upload_content_hash_mismatch_raises(self, mock_s3: Any) -> None:
        """Test upload_content raises on hash mismatch."""
        client = S3Client(bucket="test-bucket", prefix="docs")
        
        content = "# Test Content"
        wrong_hash = "0" * 64  # Wrong hash
        
        with pytest.raises(ValueError, match="hash mismatch"):
            client.upload_content(
                content=content,
                key="test.md",
                content_hash=wrong_hash,
            )

    def test_upload_content_stores_metadata(self, mock_s3: Any) -> None:
        """Test upload_content stores SHA-256 hash in S3 metadata."""
        client = S3Client(bucket="test-bucket", prefix="docs")
        
        content = "# Test"
        content_hash = compute_hash(content)
        
        client.upload_content(
            content=content,
            key="test.md",
            content_hash=content_hash,
        )
        
        # Verify metadata was stored
        response = mock_s3.head_object(
            Bucket="test-bucket",
            Key="docs/test.md",
        )
        
        assert response["Metadata"]["content-hash"] == content_hash
        assert "schema-version" in response["Metadata"]

    def test_get_content_hash_existing_object(self, mock_s3: Any) -> None:
        """Test get_content_hash retrieves hash from existing object."""
        # Upload object with hash metadata
        mock_s3.put_object(
            Bucket="test-bucket",
            Key="docs/existing.md",
            Body=b"content",
            Metadata={"content-hash": "abc123"},
        )
        
        client = S3Client(bucket="test-bucket", prefix="docs")
        hash_value = client.get_content_hash("existing.md")
        
        assert hash_value == "abc123"

    def test_get_content_hash_nonexistent_object(self, mock_s3: Any) -> None:
        """Test get_content_hash returns None for nonexistent object."""
        client = S3Client(bucket="test-bucket", prefix="docs")
        hash_value = client.get_content_hash("nonexistent.md")
        
        assert hash_value is None

    def test_should_upload_new_content(self, mock_s3: Any) -> None:
        """Test should_upload returns True for new content."""
        client = S3Client(bucket="test-bucket", prefix="docs")
        
        should = client.should_upload("new-file.md", compute_hash("new content"))
        assert should is True

    def test_should_upload_changed_content(self, mock_s3: Any) -> None:
        """Test should_upload returns True when hash changed."""
        # Upload existing content
        mock_s3.put_object(
            Bucket="test-bucket",
            Key="docs/existing.md",
            Body=b"old content",
            Metadata={"content-hash": compute_hash("old content")},
        )
        
        client = S3Client(bucket="test-bucket", prefix="docs")
        should = client.should_upload("existing.md", compute_hash("new content"))
        
        assert should is True

    def test_should_upload_unchanged_content(self, mock_s3: Any) -> None:
        """Test should_upload returns False when hash unchanged."""
        content = "unchanged content"
        content_hash = compute_hash(content)
        
        # Upload existing content with same hash
        mock_s3.put_object(
            Bucket="test-bucket",
            Key="docs/existing.md",
            Body=content.encode(),
            Metadata={"content-hash": content_hash},
        )
        
        client = S3Client(bucket="test-bucket", prefix="docs")
        should = client.should_upload("existing.md", content_hash)
        
        assert should is False

    def test_upload_json(self, mock_s3: Any) -> None:
        """Test upload_json uploads JSON data."""
        client = S3Client(bucket="test-bucket", prefix="docs")
        
        data = {"title": "Test", "version": "4.0.0"}
        
        s3_ref = client.upload_json(data, "metadata/test.json")
        
        assert s3_ref.key == "docs/metadata/test.json"
        
        # Verify content
        response = mock_s3.get_object(
            Bucket="test-bucket",
            Key="docs/metadata/test.json",
        )
        body = json.loads(response["Body"].read().decode())
        assert body["title"] == "Test"


# =============================================================================
# ScraperPipeline Tests
# =============================================================================


class TestScraperPipeline:
    """Tests for ScraperPipeline operations."""

    @pytest.mark.asyncio
    async def test_scrape_url_success(
        self,
        mock_s3: Any,
        pipeline_config: PipelineConfig,
        mock_browser: AsyncMock,
        sample_spring_html: str,
        spring_boot_version: SpringVersion,
    ) -> None:
        """Test successful URL scraping."""
        target = ScrapeTarget(
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            major=4,
        )
        
        with patch(
            "everspring_mcp.scraper.pipeline.SpringBrowser",
            return_value=mock_browser,
        ):
            pipeline = ScraperPipeline(pipeline_config)
            result = await pipeline.scrape_url(target)
        
        assert result.status == PipelineStatus.SUCCESS
        assert result.content_hash is not None
        assert result.s3_ref is not None
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_scrape_url_skipped_unchanged(
        self,
        mock_s3: Any,
        pipeline_config: PipelineConfig,
        mock_browser: AsyncMock,
        sample_spring_html: str,
    ) -> None:
        """Test scraping skips when content unchanged."""
        from everspring_mcp.scraper.parser import SpringDocParser
        
        # Pre-compute the hash that will be generated
        parser = SpringDocParser()
        version = SpringVersion(module=SpringModule.BOOT, major=4)
        scraped = parser.parse(
            html=sample_spring_html,
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            version=version,
        )
        
        # Upload existing content with same hash
        url_hash = compute_hash("https://docs.spring.io/spring-boot/reference/")[:16]
        key = f"test-docs/spring-boot/4.0.0/{url_hash}.md"
        
        mock_s3.put_object(
            Bucket="test-bucket",
            Key=key,
            Body=scraped.markdown_content.encode(),
            Metadata={"content-hash": scraped.content_hash},
        )
        
        target = ScrapeTarget(
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            major=4,
        )
        
        with patch(
            "everspring_mcp.scraper.pipeline.SpringBrowser",
            return_value=mock_browser,
        ):
            pipeline = ScraperPipeline(pipeline_config)
            result = await pipeline.scrape_url(target)
        
        assert result.status == PipelineStatus.SKIPPED
        assert result.content_hash == scraped.content_hash

    @pytest.mark.asyncio
    async def test_scrape_url_failure(
        self,
        mock_s3: Any,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test scrape_url handles navigation failures."""
        from everspring_mcp.scraper.exceptions import NavigationError
        
        # Mock browser that raises error
        mock_browser = AsyncMock()
        mock_browser.__aenter__ = AsyncMock(return_value=mock_browser)
        mock_browser.__aexit__ = AsyncMock(return_value=None)
        mock_browser.navigate_with_retry = AsyncMock(
            side_effect=NavigationError("Connection failed")
        )
        
        target = ScrapeTarget(
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            major=4,
        )
        
        with patch(
            "everspring_mcp.scraper.pipeline.SpringBrowser",
            return_value=mock_browser,
        ):
            pipeline = ScraperPipeline(pipeline_config)
            result = await pipeline.scrape_url(target)
        
        assert result.status == PipelineStatus.FAILED
        assert result.error_message is not None
        assert "Connection failed" in result.error_message

    @pytest.mark.asyncio
    async def test_process_batch(
        self,
        mock_s3: Any,
        pipeline_config: PipelineConfig,
        mock_browser: AsyncMock,
    ) -> None:
        """Test batch processing multiple targets."""
        targets = [
            ScrapeTarget(
                url=f"https://docs.spring.io/spring-boot/reference/page{i}/",
                module=SpringModule.BOOT,
                major=4,
            )
            for i in range(3)
        ]
        
        with patch(
            "everspring_mcp.scraper.pipeline.SpringBrowser",
            return_value=mock_browser,
        ):
            pipeline = ScraperPipeline(pipeline_config)
            results = await pipeline.process_batch(targets, concurrency=2)
        
        assert len(results) == 3
        # All should succeed (using mock browser)
        assert all(r.status == PipelineStatus.SUCCESS for r in results)

    @pytest.mark.asyncio
    async def test_process_batch_empty(
        self,
        mock_s3: Any,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test batch processing with empty targets."""
        pipeline = ScraperPipeline(pipeline_config)
        results = await pipeline.process_batch([], concurrency=2)
        
        assert results == []


# =============================================================================
# ScrapeResult Tests
# =============================================================================


class TestScrapeResult:
    """Tests for ScrapeResult model."""

    def test_scrape_result_success(self) -> None:
        """Test creating success ScrapeResult."""
        from everspring_mcp.models.sync import S3ObjectRef
        
        target = ScrapeTarget(
            url="https://example.com/",
            module=SpringModule.BOOT,
            major=4,
        )
        s3_ref = S3ObjectRef(
            bucket="test-bucket",
            key="docs/test.md",
        )
        
        result = ScrapeResult.success(
            target=target,
            s3_ref=s3_ref,
            content_hash="a" * 64,
        )
        
        assert result.status == PipelineStatus.SUCCESS
        assert result.s3_ref == s3_ref
        assert result.error_message is None

    def test_scrape_result_skipped(self) -> None:
        """Test creating skipped ScrapeResult."""
        target = ScrapeTarget(
            url="https://example.com/",
            module=SpringModule.BOOT,
            major=4,
        )
        
        result = ScrapeResult.skipped(
            target=target,
            content_hash="a" * 64,
        )
        
        assert result.status == PipelineStatus.SKIPPED
        assert result.s3_ref is None

    def test_scrape_result_failed(self) -> None:
        """Test creating failed ScrapeResult."""
        target = ScrapeTarget(
            url="https://example.com/",
            module=SpringModule.BOOT,
            major=4,
        )
        
        result = ScrapeResult.failed(
            target=target,
            error="Connection timeout",
        )
        
        assert result.status == PipelineStatus.FAILED
        assert result.error_message == "Connection timeout"


# =============================================================================
# Lambda Handler Tests
# =============================================================================


class TestLambdaHandler:
    """Tests for lambda_handler function."""

    def test_lambda_handler_success(
        self,
        mock_s3: Any,
        pipeline_env_vars: None,
        mock_browser: AsyncMock,
        sample_spring_html: str,
    ) -> None:
        """Test lambda_handler processes targets successfully."""
        event = {
            "targets": [
                {
                    "url": "https://docs.spring.io/spring-boot/reference/",
                    "module": "spring-boot",
                    "major": 4,
                }
            ],
            "concurrency": 1,
        }
        
        with patch(
            "everspring_mcp.scraper.pipeline.SpringBrowser",
            return_value=mock_browser,
        ):
            response = lambda_handler(event, None)
        
        assert response["statusCode"] == 200
        assert response["body"]["processed"] == 1
        assert response["body"]["status_counts"]["success"] == 1

    def test_lambda_handler_no_targets(
        self,
        mock_s3: Any,
        pipeline_env_vars: None,
    ) -> None:
        """Test lambda_handler returns error when no targets."""
        event = {"targets": []}
        
        response = lambda_handler(event, None)
        
        assert response["statusCode"] == 400
        assert "error" in response["body"]

    def test_lambda_handler_invalid_target(
        self,
        mock_s3: Any,
        pipeline_env_vars: None,
    ) -> None:
        """Test lambda_handler handles invalid targets gracefully."""
        event = {
            "targets": [
                {
                    "url": "not-a-url",  # Invalid
                    "module": "spring-boot",
                    "major": 4,
                },
            ],
        }
        
        response = lambda_handler(event, None)
        
        # Should return 400 because no valid targets after parsing
        assert response["statusCode"] == 400

    def test_lambda_handler_missing_bucket_env(self) -> None:
        """Test lambda_handler returns error when bucket not configured."""
        import os
        
        # Ensure bucket is not set
        os.environ.pop("EVERSPRING_S3_BUCKET", None)
        
        event = {
            "targets": [
                {
                    "url": "https://example.com/",
                    "module": "spring-boot",
                    "major": 4,
                }
            ],
        }
        
        response = lambda_handler(event, None)
        
        assert response["statusCode"] == 500
        assert "Configuration error" in response["body"]["error"]

    def test_lambda_handler_response_format(
        self,
        mock_s3: Any,
        pipeline_env_vars: None,
        mock_browser: AsyncMock,
    ) -> None:
        """Test lambda_handler response format."""
        event = {
            "targets": [
                {
                    "url": "https://docs.spring.io/spring-boot/reference/",
                    "module": "spring-boot",
                    "major": 4,
                }
            ],
        }
        
        with patch(
            "everspring_mcp.scraper.pipeline.SpringBrowser",
            return_value=mock_browser,
        ):
            response = lambda_handler(event, None)
        
        # Check response structure
        assert "statusCode" in response
        assert "body" in response
        assert "processed" in response["body"]
        assert "status_counts" in response["body"]
        assert "results" in response["body"]
        
        # Check result structure
        result = response["body"]["results"][0]
        assert "url" in result
        assert "status" in result
        assert "content_hash" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(
        self,
        mock_s3: Any,
        pipeline_config: PipelineConfig,
        mock_browser: AsyncMock,
        sample_spring_html: str,
    ) -> None:
        """Test complete pipeline flow: scrape -> parse -> upload."""
        target = ScrapeTarget(
            url="https://docs.spring.io/spring-boot/reference/",
            module=SpringModule.BOOT,
            major=4,
        )
        
        with patch(
            "everspring_mcp.scraper.pipeline.SpringBrowser",
            return_value=mock_browser,
        ):
            pipeline = ScraperPipeline(pipeline_config)
            result = await pipeline.scrape_url(target)
        
        # Verify result
        assert result.status == PipelineStatus.SUCCESS
        assert result.s3_ref is not None
        
        # Verify S3 content
        s3_response = mock_s3.get_object(
            Bucket="test-bucket",
            Key=result.s3_ref.key,
        )
        content = s3_response["Body"].read().decode()
        
        # Content should be markdown
        assert "Spring Boot Reference Documentation" in content
        
        # Verify metadata
        head_response = mock_s3.head_object(
            Bucket="test-bucket",
            Key=result.s3_ref.key,
        )
        metadata = head_response["Metadata"]
        
        assert metadata["content-hash"] == result.content_hash
        assert "schema-version" in metadata
        assert metadata["source-url"] == target.url
