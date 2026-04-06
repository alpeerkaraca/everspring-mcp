"""EverSpring MCP - S3 sync service.

This module provides the S3SyncService for synchronizing
local knowledge stores with S3, including separated
SQLite/Chroma snapshot uploads and retry logic using tenacity.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

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
from ..utils.logging import get_logger
from .config import SyncConfig

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = get_logger("sync.s3")


class DownloadResult(BaseModel):
    """Result of a file download operation."""
    
    s3_key: str = Field(description="S3 object key")
    local_path: Path = Field(description="Local file path")
    size_bytes: int = Field(description="Downloaded size")
    content_hash: str = Field(description="SHA256 of content")
    success: bool = Field(default=True)
    error: str | None = Field(default=None)


class KnowledgePackTransferResult(BaseModel):
    """Result of a knowledge pack transfer operation."""

    s3_key: str = Field(description="S3 object key")
    archive_path: Path = Field(description="Local archive path")
    size_bytes: int = Field(description="Archive size")
    content_hash: str = Field(description="SHA256 hash of archive")
    extracted_to: Path | None = Field(
        default=None,
        description="Extraction destination when applicable",
    )
    success: bool = Field(default=True)
    error: str | None = Field(default=None)


class SnapshotTransferResult(BaseModel):
    """Result of a DB snapshot upload operation."""

    snapshot_name: str = Field(description="Snapshot file name")
    s3_key: str = Field(description="S3 object key")
    archive_path: Path = Field(description="Local archive path")
    size_bytes: int = Field(description="Archive size")
    content_hash: str = Field(description="SHA256 hash of archive")
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
    """Service for syncing knowledge packs and manifests from S3.
    
    Handles:
    - Uploading separate DB snapshots (ChromaDB + SQLite)
    - Uploading local SQLite + ChromaDB as a knowledge-pack archive
    - Downloading and extracting knowledge-pack archives
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
        start = time.perf_counter()
        self.config = config
        self._s3: S3Client = s3_client or boto3.client(
            "s3",
            region_name=config.s3_region,
        )
        self._semaphore = asyncio.Semaphore(config.download_concurrency)
        logger.info(f"S3SyncService initialized in {time.perf_counter() - start:.2f}s")

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash for a local file without loading all bytes at once."""
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _create_knowledge_pack_archive(self, archive_path: Path) -> tuple[int, str]:
        """Create a zipped knowledge pack from local SQLite + ChromaDB persistence."""
        db_path = self.config.db_path
        chroma_dir = self.config.chroma_dir

        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        if not chroma_dir.exists() or not chroma_dir.is_dir():
            raise FileNotFoundError(f"ChromaDB directory not found: {chroma_dir}")

        archive_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as bundle:
            bundle.write(db_path, arcname=self.config.db_filename)

            chroma_files = sorted(p for p in chroma_dir.rglob("*") if p.is_file())
            if chroma_files:
                for file_path in chroma_files:
                    relative = file_path.relative_to(chroma_dir).as_posix()
                    bundle.write(file_path, arcname=f"{self.config.chroma_subdir}/{relative}")
            else:
                # Preserve directory structure even when Chroma is currently empty.
                bundle.writestr(f"{self.config.chroma_subdir}/", "")

        archive_hash = self._compute_file_hash(archive_path)
        archive_size = archive_path.stat().st_size
        return archive_size, archive_hash

    def _create_sqlite_snapshot_archive(self, archive_path: Path) -> tuple[int, str]:
        """Create a SQLite snapshot zip archive."""
        db_path = self.config.db_path
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as bundle:
            bundle.write(db_path, arcname=self.config.db_filename)

        archive_hash = self._compute_file_hash(archive_path)
        archive_size = archive_path.stat().st_size
        return archive_size, archive_hash

    def _create_chroma_snapshot_archive(self, archive_path: Path) -> tuple[int, str]:
        """Create a ChromaDB snapshot zip archive."""
        chroma_dir = self.config.chroma_dir
        if not chroma_dir.exists() or not chroma_dir.is_dir():
            raise FileNotFoundError(f"ChromaDB directory not found: {chroma_dir}")

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as bundle:
            chroma_files = sorted(p for p in chroma_dir.rglob("*") if p.is_file())
            if chroma_files:
                for file_path in chroma_files:
                    relative = file_path.relative_to(chroma_dir).as_posix()
                    bundle.write(file_path, arcname=relative)
            else:
                bundle.writestr(".chroma-empty", "")

        archive_hash = self._compute_file_hash(archive_path)
        archive_size = archive_path.stat().st_size
        return archive_size, archive_hash

    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def upload_db_snapshots(
        self,
        snapshot_date: date | datetime | None = None,
        cleanup_local_archives: bool = False,
    ) -> list[SnapshotTransferResult]:
        """Upload separate SQLite and ChromaDB snapshot archives to db-snapshots."""
        chroma_archive = self.config.get_chroma_snapshot_local_path(snapshot_date)
        sqlite_archive = self.config.get_sqlite_snapshot_local_path(snapshot_date)
        chroma_key = self.config.get_chroma_snapshot_key(snapshot_date)
        sqlite_key = self.config.get_sqlite_snapshot_key(snapshot_date)

        jobs: list[tuple[str, Path, str, Any]] = [
            ("chroma_db", chroma_archive, chroma_key, self._create_chroma_snapshot_archive),
            ("sqlite_metadata", sqlite_archive, sqlite_key, self._create_sqlite_snapshot_archive),
        ]

        results: list[SnapshotTransferResult] = []

        for snapshot_type, archive_path, s3_key, create_archive in jobs:
            try:
                archive_size, archive_hash = await asyncio.to_thread(create_archive, archive_path)

                await asyncio.to_thread(
                    self._s3.upload_file,
                    str(archive_path),
                    self.config.s3_bucket,
                    s3_key,
                    ExtraArgs={
                        "ContentType": "application/zip",
                        "ServerSideEncryption": "AES256",
                        "Metadata": {
                            "content-hash": archive_hash,
                            "snapshot-type": snapshot_type,
                        },
                    },
                )

                if cleanup_local_archives and archive_path.exists():
                    await asyncio.to_thread(archive_path.unlink)

                logger.info(
                    "Uploaded %s snapshot %s (%d bytes) to s3://%s/%s",
                    snapshot_type,
                    archive_path,
                    archive_size,
                    self.config.s3_bucket,
                    s3_key,
                )
                results.append(
                    SnapshotTransferResult(
                        snapshot_name=Path(s3_key).name,
                        s3_key=s3_key,
                        archive_path=archive_path,
                        size_bytes=archive_size,
                        content_hash=archive_hash,
                        success=True,
                    )
                )

            except (ClientError, BotoCoreError) as e:
                if isinstance(e, ClientError):
                    error_code = e.response["Error"]["Code"]
                    logger.warning(
                        "S3 error uploading %s snapshot (%s), will retry if attempts remain",
                        snapshot_type,
                        error_code,
                    )
                else:
                    logger.warning(
                        "BotoCore error uploading %s snapshot, will retry if attempts remain",
                        snapshot_type,
                    )
                raise
            except Exception as e:
                logger.exception("Unexpected error uploading %s snapshot", snapshot_type)
                results.append(
                    SnapshotTransferResult(
                        snapshot_name=Path(s3_key).name,
                        s3_key=s3_key,
                        archive_path=archive_path,
                        size_bytes=0,
                        content_hash="",
                        success=False,
                        error=str(e),
                    )
                )

        return results

    def _safe_extract_archive(self, archive_path: Path, destination_dir: Path) -> None:
        """Extract archive with path traversal protection."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_root = destination_dir.resolve()

        db_target = self.config.db_path
        chroma_target = self.config.chroma_dir

        with zipfile.ZipFile(archive_path, mode="r") as bundle:
            members = bundle.infolist()
            has_db = False
            has_chroma = False

            for member in members:
                member_name = member.filename
                member_path = Path(member_name)
                if member_path.is_absolute():
                    raise ValueError(f"Unsafe archive path: {member_name}")

                target_path = (destination_dir / member_name).resolve()
                if not target_path.is_relative_to(destination_root):
                    raise ValueError(f"Unsafe archive path traversal: {member_name}")

                normalized_name = member_name.rstrip("/")
                if normalized_name == self.config.db_filename:
                    has_db = True
                if normalized_name.startswith(f"{self.config.chroma_subdir}/"):
                    has_chroma = True

            if not has_db:
                raise ValueError("Knowledge pack archive missing SQLite database")
            if not has_chroma:
                raise ValueError("Knowledge pack archive missing ChromaDB data")

            if db_target.exists():
                db_target.unlink()
            if chroma_target.exists():
                shutil.rmtree(chroma_target)

            bundle.extractall(path=destination_dir)

    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def upload_knowledge_pack(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
        archive_path: Path | None = None,
    ) -> KnowledgePackTransferResult:
        """Zip local SQLite+ChromaDB data and upload as a single S3 object."""
        local_archive = archive_path or self.config.get_local_pack_path(
            module=module,
            version=version,
            submodule=submodule,
        )
        s3_key = self.config.get_knowledge_pack_key(
            module=module,
            version=version,
            submodule=submodule,
        )

        try:
            archive_size, archive_hash = await asyncio.to_thread(
                self._create_knowledge_pack_archive,
                local_archive,
            )

            await asyncio.to_thread(
                self._s3.upload_file,
                str(local_archive),
                self.config.s3_bucket,
                s3_key,
                ExtraArgs={
                    "ContentType": "application/zip",
                    "ServerSideEncryption": "AES256",
                    "Metadata": {
                        "content-hash": archive_hash,
                        "pack-type": "sqlite-chromadb",
                    },
                },
            )

            logger.info(
                "Uploaded knowledge pack %s (%d bytes) to s3://%s/%s",
                local_archive,
                archive_size,
                self.config.s3_bucket,
                s3_key,
            )
            return KnowledgePackTransferResult(
                s3_key=s3_key,
                archive_path=local_archive,
                size_bytes=archive_size,
                content_hash=archive_hash,
                success=True,
            )

        except (ClientError, BotoCoreError) as e:
            if isinstance(e, ClientError):
                error_code = e.response["Error"]["Code"]
                logger.warning(
                    "S3 error uploading knowledge pack (%s), will retry if attempts remain",
                    error_code,
                )
            else:
                logger.warning("BotoCore error uploading knowledge pack, will retry if attempts remain")
            raise
        except Exception as e:
            logger.exception("Unexpected error uploading knowledge pack to %s", s3_key)
            return KnowledgePackTransferResult(
                s3_key=s3_key,
                archive_path=local_archive,
                size_bytes=0,
                content_hash="",
                success=False,
                error=str(e),
            )

    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def download_knowledge_pack(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
        cleanup_archive: bool = True,
    ) -> KnowledgePackTransferResult:
        """Download and extract a zipped knowledge pack into local data directories."""
        s3_key = self.config.get_knowledge_pack_key(
            module=module,
            version=version,
            submodule=submodule,
        )
        local_archive = self.config.get_local_pack_path(
            module=module,
            version=version,
            submodule=submodule,
        )
        local_archive.parent.mkdir(parents=True, exist_ok=True)

        try:
            def _download_blob() -> tuple[bytes, dict[str, str]]:
                response = self._s3.get_object(
                    Bucket=self.config.s3_bucket,
                    Key=s3_key,
                )
                content = response["Body"].read()
                metadata = response.get("Metadata", {})
                return content, metadata

            content, metadata = await asyncio.to_thread(_download_blob)
            await asyncio.to_thread(local_archive.write_bytes, content)

            archive_hash = compute_hash(content)
            archive_size = len(content)
            expected_hash = metadata.get("content-hash")

            if expected_hash and expected_hash != archive_hash:
                raise ValueError(
                    f"Knowledge pack hash mismatch: expected {expected_hash}, got {archive_hash}",
                )

            await asyncio.to_thread(
                self._safe_extract_archive,
                local_archive,
                self.config.local_data_dir,
            )

            if cleanup_archive and local_archive.exists():
                await asyncio.to_thread(local_archive.unlink)

            logger.info(
                "Downloaded and extracted knowledge pack from s3://%s/%s",
                self.config.s3_bucket,
                s3_key,
            )
            return KnowledgePackTransferResult(
                s3_key=s3_key,
                archive_path=local_archive,
                size_bytes=archive_size,
                content_hash=archive_hash,
                extracted_to=self.config.local_data_dir,
                success=True,
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                logger.warning("Knowledge pack not found: %s", s3_key)
                return KnowledgePackTransferResult(
                    s3_key=s3_key,
                    archive_path=local_archive,
                    size_bytes=0,
                    content_hash="",
                    success=False,
                    error=f"Knowledge pack not found: {s3_key}",
                )

            logger.warning(
                "S3 error downloading knowledge pack (%s), will retry if attempts remain",
                error_code,
            )
            raise
        except BotoCoreError:
            logger.warning("BotoCore error downloading knowledge pack, will retry if attempts remain")
            raise
        except Exception as e:
            logger.exception("Unexpected error downloading knowledge pack from %s", s3_key)
            return KnowledgePackTransferResult(
                s3_key=s3_key,
                archive_path=local_archive,
                size_bytes=0,
                content_hash="",
                success=False,
                error=str(e),
            )
    
    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def fetch_manifest(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> SyncManifest | None:
        """Fetch manifest from S3 with exponential backoff retry.
        
        Automatically retries on:
        - AWS throttling (rate limiting)
        - Transient network errors
        - Temporary service unavailability
        
        Args:
            module: Spring module (e.g., "spring-boot")
            version: Version string (e.g., "4.0.5")
            submodule: Optional submodule name
            
        Returns:
            Parsed manifest or None if not found
            
        Raises:
            ClientError: After all retries exhausted
        """
        key = self.config.get_manifest_key(module, version, submodule=submodule)
        
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
            error_code = e.response["Error"]["Code"]
            # Don't retry on NoSuchKey - it's not a transient error
            if error_code == "NoSuchKey":
                logger.warning("Manifest not found: %s", key)
                return None
            # Retry on throttling and server errors
            logger.warning(f"S3 error fetching manifest ({error_code}), will retry if attempts remain")
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
    
    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def download_file(
        self,
        s3_key: str,
        expected_hash: str | None = None,
    ) -> DownloadResult:
        """Download a single file from S3 with exponential backoff retry.
        
        Automatically retries on:
        - AWS throttling (rate limiting)
        - Transient network errors
        - Temporary service unavailability
        
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
                    logger.error(
                        "Hash mismatch for %s: expected %s, got %s",
                        s3_key, expected_hash, content_hash
                    )
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
                
            except (ClientError, BotoCoreError) as e:
                # These will trigger tenacity retry
                if isinstance(e, ClientError):
                    error_code = e.response["Error"]["Code"]
                    logger.warning(
                        "S3 error downloading %s (%s), will retry if attempts remain",
                        s3_key, error_code
                    )
                else:
                    logger.warning(
                        "BotoCore error downloading %s, will retry if attempts remain",
                        s3_key
                    )
                raise
            except Exception as e:
                # Non-retryable errors
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
        submodule: str | None = None,
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
                s3_key=self.config.get_s3_key(module, version, change.path, submodule=submodule),
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
        submodule: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all files in S3 for a module/version.
        
        Args:
            module: Spring module
            version: Version string
            
        Returns:
            List of S3 object metadata dicts
        """
        if submodule:
            prefix = f"{self.config.s3_prefix}/{module}/{submodule}/{version}/"
        else:
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
        submodule: str | None = None,
    ) -> SyncManifest:
        """Build a manifest by scanning S3 objects.
        
        Use this when no manifest.json exists in S3.
        
        Args:
            module: Spring module
            version: Version string
            
        Returns:
            Generated manifest
        """
        files = await self.list_s3_files(module, version, submodule=submodule)
        if submodule:
            prefix = f"{self.config.s3_prefix}/{module}/{submodule}/{version}/"
        else:
            prefix = f"{self.config.s3_prefix}/{module}/{version}/"
        
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
                
                if not key.startswith(prefix):
                    logger.warning("Skipping unexpected key: %s", key)
                    continue

                # Store path relative to module/version (preserves subdirectories like metadata/)
                relative_path = key[len(prefix):]
                
                entries.append(FileEntry(
                    path=relative_path,
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
    "KnowledgePackTransferResult",
    "SnapshotTransferResult",
    "SyncResult",
    "S3SyncService",
]
