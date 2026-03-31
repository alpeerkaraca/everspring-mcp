"""EverSpring MCP - S3 sync service.

This module provides the S3SyncService for downloading
documents from S3 with incremental sync support.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

from ..models.base import compute_hash
from ..models.sync import (
    ChangeType,
    FileChange,
    FileEntry,
    SyncDelta,
    SyncManifest,
    SyncStatus,
)
from ..models.spring import SpringModule
from .config import SyncConfig

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)


class DownloadResult(BaseModel):
    """Result of a file download operation."""
    
    s3_key: str = Field(description="S3 object key")
    local_path: Path = Field(description="Local file path")
    size_bytes: int = Field(description="Downloaded size")
    content_hash: str = Field(description="SHA256 of content")
    success: bool = Field(default=True)
    error: str | None = Field(default=None)


class SyncResult(BaseModel):
    """Result of a sync operation."""
    
    module: str = Field(description="Spring module")
    version: str = Field(description="Version string")
    status: SyncStatus = Field(description="Final status")
    files_added: int = Field(default=0)
    files_modified: int = Field(default=0)
    files_removed: int = Field(default=0)
    bytes_downloaded: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = Field(default=None)
    
    @property
    def total_changes(self) -> int:
        """Total number of changes."""
        return self.files_added + self.files_modified + self.files_removed
    
    @property
    def has_errors(self) -> bool:
        """Check if sync had errors."""
        return len(self.errors) > 0


class S3SyncService:
    """Service for syncing documents from S3.
    
    Handles:
    - Fetching and parsing manifests
    - Computing deltas between local and remote
    - Downloading changed files
    - Hash verification
    
    Example:
        config = SyncConfig()
        service = S3SyncService(config)
        
        manifest = await service.fetch_manifest("spring-boot", "4.0.5")
        delta = service.compute_delta(local_manifest, manifest, SpringModule.BOOT)
        
        for change in delta.changes:
            if change.change_type in (ChangeType.ADDED, ChangeType.MODIFIED):
                result = await service.download_file(change.path)
    """
    
    def __init__(
        self,
        config: SyncConfig,
        s3_client: S3Client | None = None,
    ) -> None:
        """Initialize sync service.
        
        Args:
            config: Sync configuration
            s3_client: Optional S3 client (creates one if not provided)
        """
        self.config = config
        self._s3: S3Client = s3_client or boto3.client(
            "s3",
            region_name=config.s3_region,
        )
        self._semaphore = asyncio.Semaphore(config.download_concurrency)
    
    async def fetch_manifest(
        self,
        module: str,
        version: str,
    ) -> SyncManifest | None:
        """Fetch manifest from S3.
        
        Args:
            module: Spring module (e.g., "spring-boot")
            version: Version string (e.g., "4.0.5")
            
        Returns:
            Parsed manifest or None if not found
        """
        key = self.config.get_manifest_key(module, version)
        
        try:
            response = await asyncio.to_thread(
                self._s3.get_object,
                Bucket=self.config.s3_bucket,
                Key=key,
            )
            
            body = response["Body"].read().decode("utf-8")
            data = json.loads(body)
            
            manifest = SyncManifest.model_validate(data)
            logger.info(
                "Fetched manifest for %s/%s: %d files",
                module, version, manifest.file_count,
            )
            return manifest
            
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning("Manifest not found: %s", key)
                return None
            raise
    
    def compute_manifest_hash(self, manifest: SyncManifest) -> str:
        """Compute SHA256 hash of manifest content.
        
        Args:
            manifest: Manifest to hash
            
        Returns:
            SHA256 hash string
        """
        content = manifest.model_dump_json(exclude={"created_at", "updated_at"})
        return compute_hash(content)
    
    def compute_delta(
        self,
        local_manifest: SyncManifest | None,
        remote_manifest: SyncManifest,
        module: SpringModule,
    ) -> SyncDelta:
        """Compute changes between local and remote manifests.
        
        Args:
            local_manifest: Current local manifest (None if fresh sync)
            remote_manifest: Remote manifest from S3
            module: Spring module
            
        Returns:
            SyncDelta with list of changes
        """
        if local_manifest is None:
            # Fresh sync - all files are new
            changes = [
                FileChange(
                    path=f.path,
                    change_type=ChangeType.ADDED,
                    new_hash=f.content_hash,
                    size_bytes=f.size_bytes,
                )
                for f in remote_manifest.files
            ]
            
            return SyncDelta(
                from_version="0.0.0",
                to_version=remote_manifest.version,
                module=module,
                changes=changes,
            )
        
        # Use existing delta computation
        return SyncDelta.compute(local_manifest, remote_manifest, module)
    
    async def download_file(
        self,
        s3_key: str,
        expected_hash: str | None = None,
    ) -> DownloadResult:
        """Download a single file from S3.
        
        Args:
            s3_key: S3 object key
            expected_hash: Expected SHA256 hash for verification
            
        Returns:
            Download result with status
        """
        local_path = self.config.get_local_path(s3_key)
        
        async with self._semaphore:
            try:
                # Ensure parent directory exists
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                response = await asyncio.to_thread(
                    self._s3.get_object,
                    Bucket=self.config.s3_bucket,
                    Key=s3_key,
                )
                
                content = response["Body"].read()
                size_bytes = len(content)
                
                # Compute hash
                content_hash = compute_hash(content)
                
                # Verify hash if provided
                if expected_hash and content_hash != expected_hash:
                    return DownloadResult(
                        s3_key=s3_key,
                        local_path=local_path,
                        size_bytes=size_bytes,
                        content_hash=content_hash,
                        success=False,
                        error=f"Hash mismatch: expected {expected_hash}, got {content_hash}",
                    )
                
                # Write to local file
                await asyncio.to_thread(local_path.write_bytes, content)
                
                logger.debug(
                    "Downloaded %s (%d bytes)",
                    s3_key, size_bytes,
                )
                
                return DownloadResult(
                    s3_key=s3_key,
                    local_path=local_path,
                    size_bytes=size_bytes,
                    content_hash=content_hash,
                    success=True,
                )
                
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_msg = f"S3 error ({error_code}): {e.response['Error']['Message']}"
                logger.error("Failed to download %s: %s", s3_key, error_msg)
                
                return DownloadResult(
                    s3_key=s3_key,
                    local_path=local_path,
                    size_bytes=0,
                    content_hash="",
                    success=False,
                    error=error_msg,
                )
            except Exception as e:
                logger.exception("Unexpected error downloading %s", s3_key)
                return DownloadResult(
                    s3_key=s3_key,
                    local_path=local_path,
                    size_bytes=0,
                    content_hash="",
                    success=False,
                    error=str(e),
                )
    
    async def download_changes(
        self,
        delta: SyncDelta,
        module: str,
        version: str,
    ) -> list[DownloadResult]:
        """Download all changed files from a delta.
        
        Args:
            delta: Computed delta with changes
            module: Spring module
            version: Version string
            
        Returns:
            List of download results
        """
        results: list[DownloadResult] = []
        
        # Filter to only additions and modifications
        to_download = [
            change for change in delta.changes
            if change.change_type in (ChangeType.ADDED, ChangeType.MODIFIED)
        ]
        
        if not to_download:
            logger.info("No files to download")
            return results
        
        logger.info(
            "Downloading %d files for %s/%s",
            len(to_download), module, version,
        )
        
        # Download concurrently
        tasks = [
            self.download_file(
                s3_key=self.config.get_s3_key(module, version, change.path),
                expected_hash=change.new_hash,
            )
            for change in to_download
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Log summary
        success_count = sum(1 for r in results if r.success)
        total_bytes = sum(r.size_bytes for r in results if r.success)
        
        logger.info(
            "Downloaded %d/%d files (%d bytes)",
            success_count, len(results), total_bytes,
        )
        
        return results
    
    async def delete_local_file(self, s3_key: str) -> bool:
        """Delete a local file.
        
        Args:
            s3_key: S3 key (used to determine local path)
            
        Returns:
            True if file was deleted
        """
        local_path = self.config.get_local_path(s3_key)
        
        try:
            if local_path.exists():
                await asyncio.to_thread(local_path.unlink)
                logger.debug("Deleted local file: %s", local_path)
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete %s: %s", local_path, e)
            return False
    
    async def list_s3_files(
        self,
        module: str,
        version: str,
    ) -> list[dict[str, Any]]:
        """List all files in S3 for a module/version.
        
        Args:
            module: Spring module
            version: Version string
            
        Returns:
            List of S3 object metadata dicts
        """
        prefix = f"{self.config.s3_prefix}/{module}/{version}/"
        files: list[dict[str, Any]] = []
        
        paginator = self._s3.get_paginator("list_objects_v2")
        
        async for page in self._async_paginate(paginator, self.config.s3_bucket, prefix):
            for obj in page.get("Contents", []):
                # Skip manifest file
                if obj["Key"].endswith("manifest.json"):
                    continue
                files.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                    "etag": obj.get("ETag", "").strip('"'),
                })
        
        logger.debug("Found %d files in S3 for %s/%s", len(files), module, version)
        return files
    
    async def _async_paginate(
        self,
        paginator: Any,
        bucket: str,
        prefix: str,
    ):
        """Async wrapper for boto3 paginator."""
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        for page in page_iterator:
            yield page
    
    async def build_manifest_from_s3(
        self,
        module: str,
        version: str,
    ) -> SyncManifest:
        """Build a manifest by scanning S3 objects.
        
        Use this when no manifest.json exists in S3.
        
        Args:
            module: Spring module
            version: Version string
            
        Returns:
            Generated manifest
        """
        files = await self.list_s3_files(module, version)
        
        entries: list[FileEntry] = []
        total_size = 0
        
        for f in files:
            # Need to download each file to get content hash
            # This is expensive - better to have manifest.json
            key = f["key"]
            
            try:
                response = await asyncio.to_thread(
                    self._s3.head_object,
                    Bucket=self.config.s3_bucket,
                    Key=key,
                )
                
                # Try to get hash from metadata
                content_hash = response.get("Metadata", {}).get("content-hash", "")
                
                if not content_hash:
                    # Download to compute hash
                    obj_response = await asyncio.to_thread(
                        self._s3.get_object,
                        Bucket=self.config.s3_bucket,
                        Key=key,
                    )
                    content = obj_response["Body"].read()
                    content_hash = compute_hash(content)
                
                # Extract filename from key
                filename = key.split("/")[-1]
                
                entries.append(FileEntry(
                    path=filename,
                    content_hash=content_hash,
                    size_bytes=f["size"],
                ))
                total_size += f["size"]
                
            except Exception as e:
                logger.warning("Failed to process %s: %s", key, e)
        
        # Compute pack hash from all file hashes
        combined = "".join(sorted(e.content_hash for e in entries))
        pack_hash = compute_hash(combined)
        
        manifest = SyncManifest(
            version=version,
            pack_hash=pack_hash,
            file_count=len(entries),
            total_size_bytes=total_size,
            files=entries,
        )
        
        logger.info(
            "Built manifest for %s/%s: %d files, %d bytes",
            module, version, len(entries), total_size,
        )
        
        return manifest


__all__ = [
    "DownloadResult",
    "SyncResult",
    "S3SyncService",
]
