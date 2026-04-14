"""EverSpring MCP - S3 sync service.

This module provides the S3SyncService for synchronizing
local knowledge stores with S3, including separated
SQLite/Chroma snapshot uploads and retry logic using tenacity.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoCoreConfig
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm

from everspring_mcp.models.base import compute_hash
from everspring_mcp.models.spring import SpringModule
from everspring_mcp.models.sync import (
    ChangeType,
    FileChange,
    FileEntry,
    SyncDelta,
    SyncManifest,
    SyncStatus,
)
from everspring_mcp.sync.config import SyncConfig
from everspring_mcp.utils.logging import get_logger

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = get_logger("sync.s3")
SNAPSHOT_NAME_PATTERN = re.compile(
    r"^(?P<kind>chroma_db|sqlite_metadata)_(?P<token>\d{4}_\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2})?)\.zip$"
)
BM25_INDEX_FILENAME = "bm25_index.pkl"
MAX_ARCHIVE_MEMBERS = 200_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB


class _TransferProgressCallback:
    """Thread-safe tqdm callback adapter for boto3 transfers."""

    def __init__(self, description: str, total_bytes: int) -> None:
        self._lock = threading.Lock()
        self._bar = tqdm(
            total=max(total_bytes, 0),
            desc=description,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )

    def __call__(self, bytes_transferred: int) -> None:
        if bytes_transferred <= 0:
            return
        with self._lock:
            self._bar.update(bytes_transferred)

    def close(self) -> None:
        with self._lock:
            self._bar.close()


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
    snapshot_namespace: str = Field(
        default="bge-m3-main",
        description="Model+tier namespace for snapshot storage",
    )
    success: bool = Field(default=True)
    error: str | None = Field(default=None)


class ManifestBuildResult(BaseModel):
    """Result of generating/uploading a raw-data manifest."""

    module: str = Field(description="Spring module")
    version: str = Field(description="Version string")
    submodule: str | None = Field(default=None, description="Optional submodule")
    manifest_key: str = Field(description="S3 manifest key")
    status: str = Field(description="uploaded, skipped, or failed")
    file_count: int = Field(default=0, ge=0)
    total_size_bytes: int = Field(default=0, ge=0)
    error: str | None = Field(default=None)


class SnapshotSelection(BaseModel):
    """Latest matching snapshot pair selected from S3 listing."""

    snapshot_token: str = Field(description="Shared snapshot date/token")
    chroma_key: str = Field(description="S3 key for Chroma snapshot")
    sqlite_key: str = Field(description="S3 key for SQLite snapshot")
    chroma_name: str = Field(description="Chroma snapshot file name")
    sqlite_name: str = Field(description="SQLite snapshot file name")
    snapshot_namespace: str = Field(
        default="bge-m3-main",
        description="Model+tier namespace for snapshot storage",
    )
    chroma_size_bytes: int = Field(default=0, ge=0)
    sqlite_size_bytes: int = Field(default=0, ge=0)


class SnapshotDownloadResult(BaseModel):
    """Result of downloading and activating latest DB snapshots."""

    snapshot_token: str | None = Field(default=None)
    chroma_snapshot: str | None = Field(default=None)
    sqlite_snapshot: str | None = Field(default=None)
    snapshot_namespace: str = Field(default="bge-m3-main")
    bytes_downloaded: int = Field(default=0, ge=0)
    extracted_to: Path | None = Field(default=None)
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
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
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

        if s3_client:
            self._s3: S3Client = s3_client
        else:
            session = boto3.Session()
            credentials = session.get_credentials()

            boto_kwargs = {"max_pool_connections": config.parallel_jobs}

            if credentials is None:
                boto_kwargs["signature_version"] = UNSIGNED

            self._s3: S3Client = boto3.client(
                "s3",
                region_name=config.s3_region,
                config=BotoCoreConfig(**boto_kwargs),
            )

        self._semaphore = asyncio.Semaphore(config.parallel_jobs)
        logger.info(f"S3SyncService initialized in {time.perf_counter() - start:.2f}s")

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash for a local file without loading all bytes at once."""
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _create_transfer_callback(
        description: str, total_bytes: int
    ) -> _TransferProgressCallback:
        """Create a transfer progress callback for boto3 upload/download operations."""
        return _TransferProgressCallback(
            description=description, total_bytes=total_bytes
        )

    @staticmethod
    def _parse_snapshot_token(token: str) -> datetime:
        """Parse supported snapshot date tokens to a comparable datetime."""
        for fmt in ("%Y_%m_%d_%H_%M_%S", "%Y_%m_%d"):
            try:
                return datetime.strptime(token, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        raise ValueError(f"Unsupported snapshot token format: {token}")

    @staticmethod
    def _remove_path(path: Path) -> None:
        """Remove file or directory when it exists."""
        if path.is_dir():
            shutil.rmtree(path)
            return
        if path.exists():
            path.unlink()

    @staticmethod
    def _is_zip_symlink(member: zipfile.ZipInfo) -> bool:
        unix_mode = (member.external_attr >> 16) & 0o170000
        return unix_mode == 0o120000

    def _validate_archive_member_paths(
        self,
        members: list[zipfile.ZipInfo],
        destination_dir: Path,
    ) -> None:
        destination_root = destination_dir.resolve()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(
                f"Archive has too many entries ({len(members)} > {MAX_ARCHIVE_MEMBERS})"
            )

        total_uncompressed = 0
        for member in members:
            member_name = member.filename
            member_path = PurePosixPath(member_name)
            if member_path.is_absolute():
                raise ValueError(f"Unsafe archive path: {member_name}")
            if any(part == ".." for part in member_path.parts):
                raise ValueError(f"Unsafe archive path traversal: {member_name}")
            if self._is_zip_symlink(member):
                raise ValueError(f"Archive contains symlink entry: {member_name}")

            target_path = (destination_dir / member_name).resolve()
            if not target_path.is_relative_to(destination_root):
                raise ValueError(f"Unsafe archive path traversal: {member_name}")

            if not member.is_dir():
                total_uncompressed += max(member.file_size, 0)
                if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        "Archive exceeds uncompressed size limit "
                        f"({MAX_ARCHIVE_UNCOMPRESSED_BYTES} bytes)",
                    )

    def _extract_zip_safely(self, archive_path: Path, destination_dir: Path) -> None:
        """Extract zip archive with path traversal protection."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, mode="r") as bundle:
            members = bundle.infolist()
            self._validate_archive_member_paths(members, destination_dir)
            bundle.extractall(path=destination_dir)

    def _extract_sqlite_snapshot(
        self,
        archive_path: Path,
        destination_dir: Path,
    ) -> tuple[Path, Path | None]:
        """Extract sqlite snapshot archive and return staged DB/BM25 paths."""
        self._extract_zip_safely(archive_path, destination_dir)
        candidates = sorted(
            p for p in destination_dir.rglob(self.config.db_filename) if p.is_file()
        )
        if not candidates:
            raise ValueError(
                f"SQLite snapshot archive missing {self.config.db_filename}: {archive_path}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"SQLite snapshot archive contains multiple {self.config.db_filename} files: {archive_path}"
            )
        bm25_candidates = sorted(
            p for p in destination_dir.rglob(BM25_INDEX_FILENAME) if p.is_file()
        )
        if len(bm25_candidates) > 1:
            raise ValueError(
                f"SQLite snapshot archive contains multiple {BM25_INDEX_FILENAME} files: {archive_path}"
            )
        bm25_staged_path = bm25_candidates[0] if bm25_candidates else None
        return candidates[0], bm25_staged_path

    def _extract_chroma_snapshot(
        self, archive_path: Path, destination_dir: Path
    ) -> None:
        """Extract chroma snapshot archive to destination dir."""
        self._extract_zip_safely(archive_path, destination_dir)
        has_files = any(p.is_file() for p in destination_dir.rglob("*"))
        if not has_files:
            raise ValueError(f"Chroma snapshot archive is empty: {archive_path}")

    def _apply_snapshot_staging(
        self,
        sqlite_staged_path: Path,
        chroma_staged_dir: Path,
        bm25_staged_path: Path | None = None,
    ) -> None:
        """Atomically activate staged snapshot data with rollback protection."""
        live_db = self.config.db_path
        live_chroma = self.config.chroma_dir
        live_bm25 = self.config.local_data_dir / BM25_INDEX_FILENAME
        db_backup = live_db.with_suffix(f"{live_db.suffix}.bak")
        chroma_backup = live_chroma.parent / f"{live_chroma.name}.bak"
        bm25_backup = live_bm25.with_suffix(f"{live_bm25.suffix}.bak")

        self._remove_path(db_backup)
        self._remove_path(chroma_backup)
        self._remove_path(bm25_backup)
        live_db.parent.mkdir(parents=True, exist_ok=True)
        live_chroma.parent.mkdir(parents=True, exist_ok=True)
        live_bm25.parent.mkdir(parents=True, exist_ok=True)

        db_backed_up = False
        chroma_backed_up = False
        bm25_backed_up = False
        bm25_replaced = False

        try:
            if live_db.exists():
                live_db.replace(db_backup)
                db_backed_up = True
            if live_chroma.exists():
                shutil.move(str(live_chroma), str(chroma_backup))
                chroma_backed_up = True
            if bm25_staged_path is not None and live_bm25.exists():
                live_bm25.replace(bm25_backup)
                bm25_backed_up = True

            sqlite_staged_path.replace(live_db)
            shutil.move(str(chroma_staged_dir), str(live_chroma))
            if bm25_staged_path is not None:
                bm25_staged_path.replace(live_bm25)
                bm25_replaced = True

        except Exception:
            # Remove partially activated new data before restoring backups.
            cleanup_failures: list[str] = []
            for path in (live_db, live_chroma):
                try:
                    self._remove_path(path)
                except Exception as cleanup_exc:  # pragma: no cover - rare lock/path failures
                    cleanup_failures.append(f"cleanup {path}: {cleanup_exc}")
            if bm25_replaced:
                try:
                    self._remove_path(live_bm25)
                except Exception as cleanup_exc:  # pragma: no cover - rare lock/path failures
                    cleanup_failures.append(f"cleanup {live_bm25}: {cleanup_exc}")

            rollback_failures: list[str] = []
            if db_backed_up and db_backup.exists():
                try:
                    db_backup.replace(live_db)
                except Exception as rollback_exc:
                    rollback_failures.append(f"restore {db_backup} -> {live_db}: {rollback_exc}")
            if chroma_backed_up and chroma_backup.exists():
                try:
                    shutil.move(str(chroma_backup), str(live_chroma))
                except Exception as rollback_exc:
                    rollback_failures.append(
                        f"restore {chroma_backup} -> {live_chroma}: {rollback_exc}"
                    )
            if bm25_backed_up and bm25_backup.exists():
                try:
                    bm25_backup.replace(live_bm25)
                except Exception as rollback_exc:
                    rollback_failures.append(
                        f"restore {bm25_backup} -> {live_bm25}: {rollback_exc}"
                    )

            if cleanup_failures:
                logger.warning(
                    "Snapshot activation cleanup had %d failure(s): %s",
                    len(cleanup_failures),
                    "; ".join(cleanup_failures),
                )
            if rollback_failures:
                logger.error(
                    "Snapshot rollback had %d failure(s): %s",
                    len(rollback_failures),
                    "; ".join(rollback_failures),
                )
            raise
        else:
            self._remove_path(db_backup)
            self._remove_path(chroma_backup)
            self._remove_path(bm25_backup)

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
                    bundle.write(
                        file_path, arcname=f"{self.config.chroma_subdir}/{relative}"
                    )
            else:
                # Preserve directory structure even when Chroma is currently empty.
                bundle.writestr(f"{self.config.chroma_subdir}/", "")

        archive_hash = self._compute_file_hash(archive_path)
        archive_size = archive_path.stat().st_size
        return archive_size, archive_hash

    def _create_sqlite_snapshot_archive(self, archive_path: Path) -> tuple[int, str]:
        """Create a SQLite snapshot zip archive."""
        db_path = self.config.db_path
        bm25_path = self.config.local_data_dir / BM25_INDEX_FILENAME
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as bundle:
            bundle.write(db_path, arcname=self.config.db_filename)
            if bm25_path.exists() and bm25_path.is_file():
                bundle.write(bm25_path, arcname=BM25_INDEX_FILENAME)

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
            if not chroma_files:
                raise ValueError(f"ChromaDB directory is empty: {chroma_dir}")
            for file_path in chroma_files:
                relative = file_path.relative_to(chroma_dir).as_posix()
                bundle.write(file_path, arcname=relative)

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
        model_name: str | None = None,
        tier: str | None = None,
    ) -> list[SnapshotTransferResult]:
        """Upload separate SQLite and ChromaDB snapshot archives to db-snapshots."""
        effective_tier = (tier or self.config.model_tier).strip().lower()
        effective_model_name = (model_name or self.config.model_name).strip()
        snapshot_namespace = self.config.get_snapshot_namespace(
            model_name=effective_model_name,
            tier=effective_tier,
        )
        chroma_archive = self.config.get_chroma_snapshot_local_path(snapshot_date)
        sqlite_archive = self.config.get_sqlite_snapshot_local_path(snapshot_date)
        chroma_key = (
            f"{self.config.s3_prefix}/{self.config.db_snapshots_subprefix}/"
            f"{snapshot_namespace}/{Path(self.config.get_chroma_snapshot_key(snapshot_date)).name}"
        )
        sqlite_key = (
            f"{self.config.s3_prefix}/{self.config.db_snapshots_subprefix}/"
            f"{snapshot_namespace}/{Path(self.config.get_sqlite_snapshot_key(snapshot_date)).name}"
        )

        jobs: list[tuple[str, Path, str, Any]] = [
            (
                "chroma_db",
                chroma_archive,
                chroma_key,
                self._create_chroma_snapshot_archive,
            ),
            (
                "sqlite_metadata",
                sqlite_archive,
                sqlite_key,
                self._create_sqlite_snapshot_archive,
            ),
        ]

        results: list[SnapshotTransferResult] = []

        for snapshot_type, archive_path, s3_key, create_archive in jobs:
            try:
                archive_size, archive_hash = await asyncio.to_thread(
                    create_archive, archive_path
                )
                transfer_callback = self._create_transfer_callback(
                    description=f"Upload {Path(s3_key).name}",
                    total_bytes=archive_size,
                )

                try:
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
                                "tier": effective_tier,
                                "model-name": effective_model_name,
                                "snapshot-namespace": snapshot_namespace,
                            },
                        },
                        Callback=transfer_callback,
                    )
                finally:
                    transfer_callback.close()

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
                        snapshot_namespace=snapshot_namespace,
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
                logger.exception(
                    "Unexpected error uploading %s snapshot", snapshot_type
                )
                results.append(
                    SnapshotTransferResult(
                        snapshot_name=Path(s3_key).name,
                        s3_key=s3_key,
                        archive_path=archive_path,
                        size_bytes=0,
                        content_hash="",
                        snapshot_namespace=snapshot_namespace,
                        success=False,
                        error=str(e),
                    )
                )

        return results

    async def _list_db_snapshot_objects(
        self,
        snapshot_namespace: str,
        tier: str,
    ) -> list[dict[str, Any]]:
        """List all objects under the db-snapshots prefix."""
        paginator = self._s3.get_paginator("list_objects_v2")
        objects: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        prefixes = self._snapshot_prefix_candidates(
            snapshot_namespace=snapshot_namespace,
            tier=tier,
        )
        for base_prefix in prefixes:
            prefix = f"{base_prefix.rstrip('/')}/"
            try:
                async for page in self._async_paginate(
                    paginator, self.config.s3_bucket, prefix
                ):
                    for obj in page.get("Contents", []):
                        key = obj.get("Key")
                        if not key or key in seen_keys:
                            continue
                        seen_keys.add(key)
                        objects.append(obj)
            except (ClientError) as e:
                error_code = e.response.get("Error", {}).get("Code","")
                if error_code == "AccessDenied":
                    logger.warning("Access denied listing prefix %s, skipping...", prefix)
                    continue
                raise
        if not objects:
            logger.warning(
                "No snapshot objects found under prefixes: %s",
                ", ".join(f"{p.rstrip('/')}/" for p in prefixes),
            )
        return objects

    async def find_latest_snapshot_pair(
        self,
        model_name: str | None = None,
        tier: str | None = None,
    ) -> SnapshotSelection:
        """Find the latest date token that has both chroma and sqlite snapshots."""
        effective_tier = (tier or self.config.model_tier).strip().lower()
        effective_model_name = (model_name or self.config.model_name).strip()
        snapshot_namespace = self.config.get_snapshot_namespace(
            model_name=effective_model_name,
            tier=effective_tier,
        )
        objects = await self._list_db_snapshot_objects(
            snapshot_namespace=snapshot_namespace,
            tier=effective_tier,
        )
        if objects:
            scored_objects: list[tuple[dict[str, Any], int]] = [
                (
                    obj,
                    self._snapshot_scope_rank(
                        key=str(obj.get("Key", "")),
                        snapshot_namespace=snapshot_namespace,
                        tier=effective_tier,
                    ),
                )
                for obj in objects
            ]
            preferred_rank = max((rank for _, rank in scored_objects), default=0)
            if preferred_rank > 0:
                filtered_objects = [obj for obj, rank in scored_objects if rank == preferred_rank]
                if len(filtered_objects) < len(objects):
                    logger.info(
                        "Snapshot discovery narrowed candidates to %d/%d objects "
                        "(namespace=%s, rank=%d)",
                        len(filtered_objects),
                        len(objects),
                        snapshot_namespace,
                        preferred_rank,
                    )
                objects = filtered_objects
        chroma_by_token: dict[str, dict[str, Any]] = {}
        sqlite_by_token: dict[str, dict[str, Any]] = {}

        for obj in objects:
            key = obj.get("Key", "")
            file_name = Path(key).name
            match = SNAPSHOT_NAME_PATTERN.match(file_name)
            if not match:
                continue

            kind = match.group("kind")
            token = match.group("token")
            try:
                self._parse_snapshot_token(token)
            except ValueError:
                logger.warning(
                    "Skipping snapshot with invalid token %s: %s", token, key
                )
                continue

            target = chroma_by_token if kind == "chroma_db" else sqlite_by_token
            existing = target.get(token)
            if existing is None:
                target[token] = obj
                continue

            existing_modified = existing.get("LastModified")
            current_modified = obj.get("LastModified")
            if (
                current_modified
                and existing_modified
                and current_modified > existing_modified
            ):
                target[token] = obj

        common_tokens = set(chroma_by_token) & set(sqlite_by_token)
        if not common_tokens:
            latest_chroma = (
                max(chroma_by_token.keys(), key=self._parse_snapshot_token)
                if chroma_by_token
                else None
            )
            latest_sqlite = (
                max(sqlite_by_token.keys(), key=self._parse_snapshot_token)
                if sqlite_by_token
                else None
            )
            raise ValueError(
                "No common snapshot date found for chroma/sqlite archives "
                f"(latest chroma={latest_chroma}, latest sqlite={latest_sqlite})"
            )

        selected_token = max(common_tokens, key=self._parse_snapshot_token)
        chroma_obj = chroma_by_token[selected_token]
        sqlite_obj = sqlite_by_token[selected_token]
        chroma_key = chroma_obj["Key"]
        sqlite_key = sqlite_obj["Key"]

        selection = SnapshotSelection(
            snapshot_token=selected_token,
            chroma_key=chroma_key,
            sqlite_key=sqlite_key,
            chroma_name=Path(chroma_key).name,
            sqlite_name=Path(sqlite_key).name,
            snapshot_namespace=snapshot_namespace,
            chroma_size_bytes=int(chroma_obj.get("Size", 0)),
            sqlite_size_bytes=int(sqlite_obj.get("Size", 0)),
        )
        logger.info(
            "Selected snapshot pair %s (%s, %s) in namespace %s",
            selection.snapshot_token,
            selection.chroma_name,
            selection.sqlite_name,
            selection.snapshot_namespace,
        )
        return selection

    def _snapshot_scope_rank(
        self,
        key: str,
        snapshot_namespace: str,
        tier: str,
    ) -> int:
        """Rank snapshot object key scope for namespace-safe selection.

        Returns:
            2 if key is under .../db-snapshots/{snapshot_namespace}/...
            1 if key is under .../db-snapshots/{tier}/... (legacy tier namespace)
            0 otherwise (root/no-namespace legacy fallback)
        """
        normalized_key = key.strip().strip("/")
        if not normalized_key:
            return 0
        parts = [part.lower() for part in PurePosixPath(normalized_key).parts]
        snapshots_segment = self.config.db_snapshots_subprefix.strip().lower()
        namespace_segment = snapshot_namespace.strip().lower()
        tier_segment = tier.strip().lower()
        for idx, part in enumerate(parts):
            if part != snapshots_segment:
                continue
            if idx + 1 >= len(parts):
                return 0
            scope_segment = parts[idx + 1]
            if scope_segment == namespace_segment:
                return 2
            if scope_segment == tier_segment:
                return 1
            return 0
        return 0

    def _snapshot_prefix_candidates(self, snapshot_namespace: str, tier: str) -> list[str]:
        """Build candidate snapshot prefixes, including compatibility fallbacks."""
        candidates: list[str] = []

        def _add(prefix: str) -> None:
            normalized = prefix.strip().strip("/")
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        tier_segment = tier.strip().lower()
        namespace_segment = snapshot_namespace.strip().lower()

        # Primary configured location with namespace segment.
        _add(f"{self.config.s3_prefix}/{self.config.db_snapshots_subprefix}/{namespace_segment}")

        # Common canonical location used by production uploads (with namespace).
        _add(f"spring-docs/{self.config.db_snapshots_subprefix}/{namespace_segment}")

        # If s3_prefix accidentally points at raw-data, recover to root prefix.
        raw_suffix = f"/{self.config.raw_data_subprefix}"
        if self.config.s3_prefix.endswith(raw_suffix):
            root_prefix = self.config.s3_prefix[: -len(raw_suffix)]
            _add(f"{root_prefix}/{self.config.db_snapshots_subprefix}/{namespace_segment}")

        # Backward compatibility with legacy tier-only namespace.
        _add(f"{self.config.s3_prefix}/{self.config.db_snapshots_subprefix}/{tier_segment}")
        _add(f"spring-docs/{self.config.db_snapshots_subprefix}/{tier_segment}")

        # Legacy fallbacks without tier segment.
        _add(f"{self.config.s3_prefix}/{self.config.db_snapshots_subprefix}")
        _add(f"spring-docs/{self.config.db_snapshots_subprefix}")
        _add(self.config.db_snapshots_subprefix)
        return candidates

    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _download_snapshot_archive(
        self,
        s3_key: str,
        local_path: Path,
    ) -> tuple[int, str]:
        """Download a snapshot archive with hash verification and progress."""
        local_path.parent.mkdir(parents=True, exist_ok=True)

        head_response = await asyncio.to_thread(
            self._s3.head_object,
            Bucket=self.config.s3_bucket,
            Key=s3_key,
        )
        expected_size = int(head_response.get("ContentLength", 0))
        expected_hash = head_response.get("Metadata", {}).get("content-hash")

        callback = self._create_transfer_callback(
            description=f"Download {Path(s3_key).name}",
            total_bytes=expected_size,
        )
        try:
            await asyncio.to_thread(
                self._s3.download_file,
                self.config.s3_bucket,
                s3_key,
                str(local_path),
                Callback=callback,
            )
        finally:
            callback.close()

        archive_size = local_path.stat().st_size
        archive_hash = await asyncio.to_thread(self._compute_file_hash, local_path)
        if expected_size and archive_size != expected_size:
            raise ValueError(
                f"Snapshot size mismatch for {s3_key}: expected {expected_size}, got {archive_size}"
            )
        if expected_hash and archive_hash != expected_hash:
            raise ValueError(
                f"Snapshot hash mismatch for {s3_key}: expected {expected_hash}, got {archive_hash}"
            )
        return archive_size, archive_hash

    async def download_latest_db_snapshots(
        self,
        cleanup_local_archives: bool = True,
        model_name: str | None = None,
        tier: str | None = None,
    ) -> SnapshotDownloadResult:
        """Download and atomically activate the latest matching db snapshot pair."""
        effective_tier = (tier or self.config.model_tier).strip().lower()
        effective_model_name = (model_name or self.config.model_name).strip()
        snapshot_namespace = self.config.get_snapshot_namespace(
            model_name=effective_model_name,
            tier=effective_tier,
        )
        selection: SnapshotSelection | None = None
        bytes_downloaded = 0
        chroma_archive: Path | None = None
        sqlite_archive: Path | None = None
        staging_root: Path | None = None
        activation_success = False

        try:
            selection = await self.find_latest_snapshot_pair(
                model_name=effective_model_name,
                tier=effective_tier,
            )
            chroma_archive = self.config.packs_dir / selection.chroma_name
            sqlite_archive = self.config.packs_dir / selection.sqlite_name
            chroma_archive.parent.mkdir(parents=True, exist_ok=True)

            chroma_size, _ = await self._download_snapshot_archive(
                selection.chroma_key,
                chroma_archive,
            )
            sqlite_size, _ = await self._download_snapshot_archive(
                selection.sqlite_key,
                sqlite_archive,
            )
            bytes_downloaded = chroma_size + sqlite_size

            staging_root = Path(
                tempfile.mkdtemp(
                    prefix="snapshot-download-",
                    dir=str(self.config.local_data_dir),
                )
            )
            sqlite_staging_dir = staging_root / "sqlite"
            chroma_staging_dir = staging_root / "chroma"

            sqlite_staged_path, bm25_staged_path = await asyncio.to_thread(
                self._extract_sqlite_snapshot,
                sqlite_archive,
                sqlite_staging_dir,
            )
            await asyncio.to_thread(
                self._extract_chroma_snapshot,
                chroma_archive,
                chroma_staging_dir,
            )
            await asyncio.to_thread(
                self._apply_snapshot_staging,
                sqlite_staged_path,
                chroma_staging_dir,
                bm25_staged_path,
            )
            activation_success = True

            logger.info(
                "Activated snapshot pair %s (%d bytes)",
                selection.snapshot_token,
                bytes_downloaded,
            )
            return SnapshotDownloadResult(
                snapshot_token=selection.snapshot_token,
                chroma_snapshot=selection.chroma_name,
                sqlite_snapshot=selection.sqlite_name,
                snapshot_namespace=snapshot_namespace,
                bytes_downloaded=bytes_downloaded,
                extracted_to=self.config.local_data_dir,
                success=True,
            )

        except (
            ClientError,
            BotoCoreError,
            ValueError,
            RuntimeError,
            OSError,
            zipfile.BadZipFile,
        ) as e:
            logger.exception("Snapshot download/apply failed")
            return SnapshotDownloadResult(
                snapshot_token=selection.snapshot_token if selection else None,
                chroma_snapshot=selection.chroma_name if selection else None,
                sqlite_snapshot=selection.sqlite_name if selection else None,
                snapshot_namespace=snapshot_namespace,
                bytes_downloaded=bytes_downloaded,
                extracted_to=None,
                success=False,
                error=str(e),
            )
        finally:
            if staging_root and staging_root.exists():
                await asyncio.to_thread(shutil.rmtree, staging_root, True)
            if activation_success and cleanup_local_archives:
                for archive_path in (chroma_archive, sqlite_archive):
                    if archive_path and archive_path.exists():
                        await asyncio.to_thread(archive_path.unlink)

    def _model_prefix_candidates(self, tier: str) -> list[str]:
        """Build candidate model-artifact prefixes for a model tier."""
        candidates: list[str] = []

        def _add(prefix: str) -> None:
            normalized = prefix.strip().strip("/")
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        clean_tier = tier.strip().lower()
        _add(self.config.get_models_prefix(clean_tier))
        _add(f"spring-docs/{self.config.models_subprefix}/{clean_tier}")
        _add(f"{self.config.models_subprefix}/{clean_tier}")
        return candidates

    async def _download_object_to_path(
        self,
        s3_key: str,
        local_path: Path,
    ) -> DownloadResult:
        """Download one S3 object to a custom local path with hash verification."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            head = await asyncio.to_thread(
                self._s3.head_object,
                Bucket=self.config.s3_bucket,
                Key=s3_key,
            )
            expected_size = int(head.get("ContentLength", 0))
            expected_hash = head.get("Metadata", {}).get("content-hash")
            callback = self._create_transfer_callback(
                description=f"Download {Path(s3_key).name}",
                total_bytes=expected_size,
            )
            try:
                await asyncio.to_thread(
                    self._s3.download_file,
                    self.config.s3_bucket,
                    s3_key,
                    str(local_path),
                    Callback=callback,
                )
            finally:
                callback.close()

            size_bytes = local_path.stat().st_size
            content_hash = await asyncio.to_thread(self._compute_file_hash, local_path)
            if expected_size and size_bytes != expected_size:
                raise ValueError(
                    f"Model artifact size mismatch for {s3_key}: expected {expected_size}, got {size_bytes}"
                )
            if expected_hash and content_hash != expected_hash:
                raise ValueError(
                    f"Model artifact hash mismatch for {s3_key}: expected {expected_hash}, got {content_hash}"
                )
            return DownloadResult(
                s3_key=s3_key,
                local_path=local_path,
                size_bytes=size_bytes,
                content_hash=content_hash,
                success=True,
            )
        except (ClientError, BotoCoreError, OSError, ValueError) as exc:
            logger.exception("Model artifact download failed for %s", s3_key)
            return DownloadResult(
                s3_key=s3_key,
                local_path=local_path,
                size_bytes=0,
                content_hash="",
                success=False,
                error=str(exc),
            )

    async def download_model_artifacts(
        self,
        tier: str = "slim",
        destination_dir: Path | None = None,
    ) -> list[DownloadResult]:
        """Download prebuilt model artifacts for a tier into local model directory."""
        effective_tier = tier.strip().lower()
        target_root = destination_dir or self.config.get_local_model_dir(effective_tier)
        target_root.mkdir(parents=True, exist_ok=True)

        paginator = self._s3.get_paginator("list_objects_v2")
        candidates = self._model_prefix_candidates(effective_tier)
        objects: list[tuple[str, str]] = []
        seen_keys: set[str] = set()
        for base_prefix in candidates:
            prefix = f"{base_prefix.rstrip('/')}/"
            async for page in self._async_paginate(
                paginator, self.config.s3_bucket, prefix
            ):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if (
                        not key
                        or key.endswith("/")
                        or key in seen_keys
                        or not key.startswith(prefix)
                    ):
                        continue
                    seen_keys.add(key)
                    objects.append((key, prefix))

        if not objects:
            raise FileNotFoundError(
                "No model artifacts found under prefixes: "
                + ", ".join(f"{p.rstrip('/')}/" for p in candidates)
            )

        tasks = []
        for s3_key, prefix in objects:
            relative = s3_key.removeprefix(prefix).lstrip("/")
            local_path = target_root / relative
            tasks.append(self._download_object_to_path(s3_key=s3_key, local_path=local_path))
        return await asyncio.gather(*tasks)

    def _safe_extract_archive(self, archive_path: Path, destination_dir: Path) -> None:
        """Extract archive with path traversal protection."""
        destination_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path, mode="r") as bundle:
            members = bundle.infolist()
            self._validate_archive_member_paths(members, destination_dir)
            has_db = False
            has_chroma = False

            for member in members:
                member_name = member.filename
                normalized_name = member_name.rstrip("/")
                if normalized_name == self.config.db_filename:
                    has_db = True
                if normalized_name.startswith(f"{self.config.chroma_subdir}/"):
                    has_chroma = True

            if not has_db:
                raise ValueError("Knowledge pack archive missing SQLite database")
            if not has_chroma:
                raise ValueError("Knowledge pack archive missing ChromaDB data")

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
                logger.warning(
                    "BotoCore error uploading knowledge pack, will retry if attempts remain"
                )
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
        staging_root: Path | None = None

        try:
            def _download_blob() -> tuple[str, int, dict[str, str]]:
                response = self._s3.get_object(
                    Bucket=self.config.s3_bucket,
                    Key=s3_key,
                )
                body = response["Body"]
                metadata = response.get("Metadata", {})
                digest = hashlib.sha256()
                size_bytes = 0
                try:
                    with local_archive.open("wb") as archive_handle:
                        while True:
                            chunk = body.read(1024 * 1024)
                            if not chunk:
                                break
                            archive_handle.write(chunk)
                            digest.update(chunk)
                            size_bytes += len(chunk)
                finally:
                    body.close()
                return digest.hexdigest(), size_bytes, metadata

            archive_hash, archive_size, metadata = await asyncio.to_thread(_download_blob)
            expected_hash = metadata.get("content-hash")

            if expected_hash and expected_hash != archive_hash:
                raise ValueError(
                    f"Knowledge pack hash mismatch: expected {expected_hash}, got {archive_hash}",
                )

            staging_root = Path(
                tempfile.mkdtemp(
                    prefix="knowledge-pack-download-",
                    dir=str(self.config.local_data_dir),
                )
            )
            staging_extract_dir = staging_root / "extract"
            await asyncio.to_thread(
                self._safe_extract_archive,
                local_archive,
                staging_extract_dir,
            )
            sqlite_staged_path = staging_extract_dir / self.config.db_filename
            chroma_staged_dir = staging_extract_dir / self.config.chroma_subdir
            bm25_staged_path = staging_extract_dir / BM25_INDEX_FILENAME
            if not sqlite_staged_path.exists():
                raise ValueError(
                    "Knowledge pack archive extraction did not produce SQLite database",
                )
            if not chroma_staged_dir.exists() or not any(
                p.is_file() for p in chroma_staged_dir.rglob("*")
            ):
                raise ValueError(
                    "Knowledge pack archive extraction did not produce ChromaDB data",
                )
            await asyncio.to_thread(
                self._apply_snapshot_staging,
                sqlite_staged_path,
                chroma_staged_dir,
                bm25_staged_path if bm25_staged_path.exists() else None,
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
            logger.warning(
                "BotoCore error downloading knowledge pack, will retry if attempts remain"
            )
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error downloading knowledge pack from %s", s3_key
            )
            return KnowledgePackTransferResult(
                s3_key=s3_key,
                archive_path=local_archive,
                size_bytes=0,
                content_hash="",
                success=False,
                error=str(e),
            )
        finally:
            if staging_root and staging_root.exists():
                await asyncio.to_thread(shutil.rmtree, staging_root, True)

    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def upload_manifest(
        self,
        module: str,
        version: str,
        manifest: SyncManifest,
        submodule: str | None = None,
    ) -> str:
        """Upload manifest JSON to S3 for future incremental sync runs."""
        key = self.config.get_manifest_key(module, version, submodule=submodule)
        payload = manifest.model_dump_json(indent=2, exclude_computed_fields=True)
        payload_bytes = payload.encode("utf-8")
        content_hash = compute_hash(payload)

        await asyncio.to_thread(
            self._s3.put_object,
            Bucket=self.config.s3_bucket,
            Key=key,
            Body=payload_bytes,
            ContentType="application/json",
            ServerSideEncryption="AES256",
            Metadata={
                "content-hash": content_hash,
                "manifest-version": manifest.version,
            },
        )

        logger.info(
            "Uploaded manifest to s3://%s/%s (%d files)",
            self.config.s3_bucket,
            key,
            manifest.file_count,
        )
        return key

    async def ensure_manifest(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
        force: bool = False,
    ) -> ManifestBuildResult:
        """Ensure remote manifest exists, generating/uploading if needed."""
        manifest_key = self.config.get_manifest_key(
            module, version, submodule=submodule
        )

        try:
            if not force:
                existing_manifest = await self.fetch_manifest(
                    module, version, submodule=submodule
                )
                if existing_manifest is not None:
                    return ManifestBuildResult(
                        module=module,
                        version=version,
                        submodule=submodule,
                        manifest_key=manifest_key,
                        status="skipped",
                        file_count=existing_manifest.file_count,
                        total_size_bytes=existing_manifest.total_size_bytes,
                    )

            built_manifest = await self.build_manifest_from_s3(
                module, version, submodule=submodule
            )
            uploaded_key = await self.upload_manifest(
                module=module,
                version=version,
                manifest=built_manifest,
                submodule=submodule,
            )
            return ManifestBuildResult(
                module=module,
                version=version,
                submodule=submodule,
                manifest_key=uploaded_key,
                status="uploaded",
                file_count=built_manifest.file_count,
                total_size_bytes=built_manifest.total_size_bytes,
            )

        except (ClientError, BotoCoreError, ValueError, RuntimeError) as e:
            logger.warning(
                "Failed to ensure manifest for %s/%s%s: %s",
                module,
                version,
                f" ({submodule})" if submodule else "",
                e,
            )
            return ManifestBuildResult(
                module=module,
                version=version,
                submodule=submodule,
                manifest_key=manifest_key,
                status="failed",
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
            manifest = SyncManifest.model_validate_json(body)
            logger.info(
                "Fetched manifest for %s/%s: %d files",
                module,
                version,
                manifest.file_count,
            )
            return manifest

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            # Don't retry on NoSuchKey - it's not a transient error
            if error_code == "NoSuchKey":
                logger.warning("Manifest not found: %s", key)
                return None
            # Retry on throttling and server errors
            logger.warning(
                f"S3 error fetching manifest ({error_code}), will retry if attempts remain"
            )
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
                        s3_key,
                        expected_hash,
                        content_hash,
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
                    s3_key,
                    size_bytes,
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
                        s3_key,
                        error_code,
                    )
                else:
                    logger.warning(
                        "BotoCore error downloading %s, will retry if attempts remain",
                        s3_key,
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
            change
            for change in delta.changes
            if change.change_type in (ChangeType.ADDED, ChangeType.MODIFIED)
        ]

        if not to_download:
            logger.info("No files to download")
            return results

        logger.info(
            "Downloading %d files for %s/%s",
            len(to_download),
            module,
            version,
        )

        # Download concurrently
        task_specs = [
            (
                self.config.get_s3_key(
                    module, version, change.path, submodule=submodule
                ),
                change.new_hash,
            )
            for change in to_download
        ]
        tasks = [
            self.download_file(s3_key=s3_key, expected_hash=expected_hash)
            for s3_key, expected_hash in task_specs
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[DownloadResult] = []
        for (s3_key, _), item in zip(task_specs, gathered, strict=False):
            if isinstance(item, Exception):
                logger.error("Download task failed for %s: %s", s3_key, item)
                results.append(
                    DownloadResult(
                        s3_key=s3_key,
                        local_path=self.config.get_local_path(s3_key),
                        size_bytes=0,
                        content_hash="",
                        success=False,
                        error=str(item),
                    )
                )
                continue
            results.append(item)

        # Log summary
        success_count = sum(1 for r in results if r.success)
        total_bytes = sum(r.size_bytes for r in results if r.success)

        logger.info(
            "Downloaded %d/%d files (%d bytes)",
            success_count,
            len(results),
            total_bytes,
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
        files: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        paginator = self._s3.get_paginator("list_objects_v2")
        for prefix in self._manifest_prefix_candidates(
            module, version, submodule=submodule
        ):
            async for page in self._async_paginate(
                paginator, self.config.s3_bucket, prefix
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # Skip manifest file and avoid duplicate keys when prefixes overlap.
                    if key.endswith("manifest.json") or key in seen_keys:
                        continue
                    seen_keys.add(key)
                    files.append(
                        {
                            "key": key,
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                            "etag": obj.get("ETag", "").strip('"'),
                        }
                    )

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
        prefix_candidates = self._manifest_prefix_candidates(
            module, version, submodule=submodule
        )

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

                matching_prefix = next(
                    (prefix for prefix in prefix_candidates if key.startswith(prefix)),
                    None,
                )
                if matching_prefix is None:
                    logger.warning("Skipping unexpected key: %s", key)
                    continue

                # Store path relative to module/version (preserves subdirectories like metadata/)
                relative_path = key[len(matching_prefix) :]

                entries.append(
                    FileEntry(
                        path=relative_path,
                        content_hash=content_hash,
                        size_bytes=f["size"],
                    )
                )
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
            module,
            version,
            len(entries),
            total_size,
        )

        return manifest

    def _manifest_prefix_candidates(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> list[str]:
        """Build current and legacy S3 prefixes for manifest/listing lookups."""
        current_prefix = (
            f"{self.config.get_raw_data_prefix(module, version, submodule=submodule)}/"
        )
        if submodule:
            legacy_prefix = f"{self.config.s3_prefix}/{module}/{submodule}/{version}/"
        else:
            legacy_prefix = f"{self.config.s3_prefix}/{module}/{version}/"

        prefixes = [current_prefix]
        if legacy_prefix != current_prefix:
            prefixes.append(legacy_prefix)
        return prefixes


__all__ = [
    "DownloadResult",
    "KnowledgePackTransferResult",
    "ManifestBuildResult",
    "SnapshotDownloadResult",
    "SnapshotSelection",
    "SnapshotTransferResult",
    "SyncResult",
    "S3SyncService",
]
