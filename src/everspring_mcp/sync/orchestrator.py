"""EverSpring MCP - Sync orchestrator.

This module coordinates the full S3 → Local → SQLite sync flow:
1. Fetch manifest from S3
2. Compare with local state
3. Download changed files
4. Update SQLite database
5. Clean up removed files
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..models.base import compute_hash
from ..models.spring import SpringModule, SpringVersion
from ..models.sync import ChangeType, SyncDelta, SyncStatus
from ..storage.repository import (
    DocumentRecord,
    LocalManifestRecord,
    StorageManager,
)
from ..utils.logging import get_logger
from .config import SyncConfig
from .s3_sync import DownloadResult, S3SyncService, SyncResult

if TYPE_CHECKING:
    pass

logger = get_logger("sync.orchestrator")


# Progress callback type
ProgressCallback = Callable[[str, int, int], Awaitable[None] | None]


class SyncOrchestrator:
    """Orchestrates S3 to local sync operations.
    
    Coordinates:
    - S3SyncService for downloading
    - StorageManager for database operations
    - Progress reporting
    - Error handling and resume
    
    Example:
        config = SyncConfig()
        async with SyncOrchestrator(config) as orchestrator:
            result = await orchestrator.sync_module("spring-boot", "4.0.5")
            print(f"Synced {result.files_added} new files")
    """
    
    def __init__(
        self,
        config: SyncConfig | None = None,
        storage: StorageManager | None = None,
        s3_service: S3SyncService | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Initialize orchestrator.
        
        Args:
            config: Sync configuration (uses defaults if not provided)
            storage: Storage manager (creates one if not provided)
            s3_service: S3 sync service (creates one if not provided)
            progress_callback: Optional callback for progress updates
        """
        start = time.perf_counter()
        self.config = config or SyncConfig.from_env()
        self._storage = storage
        self._s3_service = s3_service
        self._owns_storage = storage is None
        self._progress_callback = progress_callback
        logger.info(f"SyncOrchestrator initialized in {time.perf_counter() - start:.3f}s")
    
    async def __aenter__(self) -> SyncOrchestrator:
        """Async context manager entry."""
        self.config.ensure_directories()
        
        if self._storage is None:
            self._storage = StorageManager(self.config.db_path)
            await self._storage.connect()
        
        if self._s3_service is None:
            self._s3_service = S3SyncService(self.config)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._owns_storage and self._storage:
            await self._storage.close()
    
    @property
    def storage(self) -> StorageManager:
        """Get storage manager."""
        if not self._storage:
            raise RuntimeError("Orchestrator not initialized")
        return self._storage
    
    @property
    def s3(self) -> S3SyncService:
        """Get S3 service."""
        if not self._s3_service:
            raise RuntimeError("Orchestrator not initialized")
        return self._s3_service
    
    async def _report_progress(
        self,
        message: str,
        current: int = 0,
        total: int = 0,
    ) -> None:
        """Report progress to callback."""
        if self._progress_callback:
            result = self._progress_callback(message, current, total)
            if hasattr(result, "__await__"):
                await result
    
    async def sync_module(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
        force: bool = False,
    ) -> SyncResult:
        """Sync a Spring module version from S3.
        
        Args:
            module: Spring module (e.g., "spring-boot")
            version: Version string (e.g., "4.0.5")
            force: Force full sync even if manifest unchanged
            
        Returns:
            SyncResult with operation summary
        """
        result = SyncResult(
            module=module,
            version=version,
            status=SyncStatus.IN_PROGRESS,
        )
        
        # Start sync history record
        sync_id = await self.storage.sync_history.start_sync(
            module=module,
            version=version,
            submodule=submodule,
        )
        
        try:
            await self._report_progress(f"Fetching manifest for {module}/{version}")
            
            # Step 1: Fetch remote manifest
            remote_manifest = await self.s3.fetch_manifest(module, version, submodule=submodule)
            
            if remote_manifest is None:
                # No manifest - try to build from S3 listing
                logger.info("No manifest found, building from S3 listing")
                await self._report_progress("Building manifest from S3")
                remote_manifest = await self.s3.build_manifest_from_s3(module, version, submodule=submodule)
            
            remote_hash = self.s3.compute_manifest_hash(remote_manifest)
            
            # Step 2: Get local manifest
            local_record = await self.storage.manifests.get(module, version, submodule=submodule)
            local_manifest = local_record.get_manifest() if local_record else None
            
            # Check if manifest changed
            if not force and local_record and local_record.manifest_hash == remote_hash:
                logger.info("Manifest unchanged, skipping sync")
                result.status = SyncStatus.COMPLETED
                result.completed_at = datetime.now(timezone.utc)
                
                await self.storage.sync_history.complete_sync(sync_id)
                return result
            
            # Step 3: Compute delta
            await self._report_progress("Computing changes")
            
            spring_module = self._parse_spring_module(module)
            delta = self.s3.compute_delta(local_manifest, remote_manifest, spring_module)
            
            logger.info(
                "Delta computed: +%d ~%d -%d",
                delta.added_count,
                delta.modified_count,
                delta.removed_count,
            )
            
            if not delta.has_changes():
                logger.info("No changes to sync")
                result.status = SyncStatus.COMPLETED
                result.completed_at = datetime.now(timezone.utc)
                
                await self.storage.sync_history.complete_sync(sync_id)
                return result
            
            # Step 4: Download changed files
            await self._report_progress(
                "Downloading files",
                0,
                delta.added_count + delta.modified_count,
            )
            
            download_results = await self.s3.download_changes(delta, module, version, submodule=submodule)
            
            # Separate metadata and markdown downloads
            metadata_cache: dict[str, dict] = {}
            markdown_downloads: list[DownloadResult] = []
            
            for download in download_results:
                relative_path = self._relative_path(download.s3_key, module, version, submodule)
                if self._is_metadata_path(relative_path):
                    if download.success:
                        metadata = self._load_metadata(download.local_path)
                        url_hash = self._extract_url_hash(relative_path)
                        metadata_cache[url_hash] = metadata
                    else:
                        result.errors.append(download.error or f"Failed: {download.s3_key}")
                    continue
                markdown_downloads.append(download)
            
            # Process markdown downloads
            for i, download in enumerate(markdown_downloads):
                await self._report_progress(
                    f"Processing {download.s3_key}",
                    i + 1,
                    len(markdown_downloads),
                )
                
                if download.success:
                    url_hash = self._extract_url_hash(
                        self._relative_path(download.s3_key, module, version, submodule)
                    )
                    metadata = metadata_cache.get(url_hash)
                    await self._process_downloaded_file(
                        download, module, version, metadata, submodule=submodule,
                    )
                    result.bytes_downloaded += download.size_bytes
                else:
                    result.errors.append(download.error or f"Failed: {download.s3_key}")
            
            # Count results (only markdown files)
            result.files_added = sum(
                1 for d in markdown_downloads
                if d.success and self._is_added(d.s3_key, delta, module, version, submodule=submodule)
            )
            result.files_modified = sum(
                1 for d in markdown_downloads
                if d.success and self._is_modified(d.s3_key, delta, module, version, submodule=submodule)
            )
            
            # Step 5: Handle removals
            removed_changes = [
                c for c in delta.changes
                if c.change_type == ChangeType.REMOVED
            ]

            for change in removed_changes:
                s3_key = self.config.get_s3_key(module, version, change.path, submodule=submodule)
                deleted = await self.s3.delete_local_file(s3_key)
                
                if deleted and not self._is_metadata_path(change.path):
                    await self.storage.documents.delete_by_s3_key(s3_key)
                    result.files_removed += 1
            
            # Step 6: Update local manifest cache
            await self.storage.manifests.save(
                module=module,
                version=version,
                manifest=remote_manifest,
                manifest_hash=remote_hash,
                submodule=submodule,
            )
            
            # Finalize
            result.status = SyncStatus.COMPLETED if not result.errors else SyncStatus.FAILED
            result.completed_at = datetime.now(timezone.utc)
            
            await self.storage.sync_history.complete_sync(
                sync_id=sync_id,
                files_added=result.files_added,
                files_modified=result.files_modified,
                files_removed=result.files_removed,
                bytes_downloaded=result.bytes_downloaded,
            )
            
            await self._report_progress(
                f"Sync complete: +{result.files_added} ~{result.files_modified} -{result.files_removed}",
            )
            
            return result
            
        except Exception as e:
            logger.exception("Sync failed for %s/%s", module, version)
            
            result.status = SyncStatus.FAILED
            result.errors.append(str(e))
            result.completed_at = datetime.now(timezone.utc)
            
            await self.storage.sync_history.fail_sync(sync_id, str(e))
            
            return result
    
    async def _process_downloaded_file(
        self,
        download: DownloadResult,
        module: str,
        version: str,
        metadata: dict | None,
        submodule: str | None = None,
    ) -> None:
        """Process a successfully downloaded file.
        
        Args:
            download: Download result
            module: Spring module
            version: Version string
            metadata: Optional metadata JSON for this document
        """
        # Parse version parts
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        
        # Extract relative path within module/version
        relative_path = self._relative_path(download.s3_key, module, version, submodule)
        url_hash = self._extract_url_hash(relative_path)
        
        # Read file to extract title (first heading)
        title = await self._extract_title(download.local_path)
        source_url = None
        submodule = None
        scraped_at = datetime.now(timezone.utc)
        
        if metadata:
            source_url = metadata.get("url")
            title = metadata.get("title") or title
            submodule = metadata.get("submodule") or submodule
            if metadata.get("scraped_at"):
                scraped_at = self._parse_iso_datetime(metadata["scraped_at"])
        
        # Generate document ID from URL hash
        doc_id = url_hash or compute_hash(download.s3_key)[:16]
        
        # Create/update document record
        doc = DocumentRecord(
            id=doc_id,
            url=source_url or f"s3://{self.config.s3_bucket}/{download.s3_key}",
            title=title,
            module=module,
            submodule=submodule,
            major_version=major,
            minor_version=minor,
            patch_version=patch,
            content_hash=download.content_hash,
            file_path=relative_path,
            s3_key=download.s3_key,
            size_bytes=download.size_bytes,
            scraped_at=scraped_at,
            synced_at=datetime.now(timezone.utc),
            schema_version="1.0.0",
            is_indexed=False,  # Will be indexed by ChromaDB later
        )
        
        await self.storage.documents.upsert(doc)
    
    async def _extract_title(self, file_path: Path) -> str:
        """Extract title from markdown file.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            Extracted title or filename
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            
            # Look for first heading
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("# "):
                    return line[2:].strip()
            
            # Fallback to filename
            return file_path.stem.replace("-", " ").replace("_", " ").title()
            
        except Exception:
            return file_path.stem
    
    def _parse_spring_module(self, module: str) -> SpringModule:
        """Parse module string to SpringModule enum.
        
        Args:
            module: Module string (e.g., "spring-boot")
            
        Returns:
            SpringModule enum value
        """
        # Map module strings to enum
        mapping = {
            "spring-boot": SpringModule.BOOT,
            "spring-framework": SpringModule.FRAMEWORK,
            "spring-security": SpringModule.SECURITY,
            "spring-data": SpringModule.DATA,
            "spring-cloud": SpringModule.CLOUD,
        }
        
        return mapping.get(module, SpringModule.BOOT)
    
    def _is_added(
        self,
        s3_key: str,
        delta: SyncDelta,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> bool:
        """Check if S3 key was an addition."""
        filename = self._relative_path(s3_key, module, version, submodule)
        return any(
            c.path == filename and c.change_type == ChangeType.ADDED
            for c in delta.changes
        )
    
    def _is_modified(
        self,
        s3_key: str,
        delta: SyncDelta,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> bool:
        """Check if S3 key was a modification."""
        filename = self._relative_path(s3_key, module, version, submodule)
        return any(
            c.path == filename and c.change_type == ChangeType.MODIFIED
            for c in delta.changes
        )

    def _relative_path(
        self,
        s3_key: str,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> str:
        """Get relative path within module/version."""
        prefix = f"{self.config.get_raw_data_prefix(module, version, submodule=submodule)}/"
        if submodule:
            legacy_prefix = f"{self.config.s3_prefix}/{module}/{submodule}/{version}/"
        else:
            legacy_prefix = f"{self.config.s3_prefix}/{module}/{version}/"
        if s3_key.startswith(prefix):
            return s3_key[len(prefix):]
        if s3_key.startswith(legacy_prefix):
            return s3_key[len(legacy_prefix):]
        return s3_key.split("/")[-1]

    @staticmethod
    def _is_metadata_path(relative_path: str) -> bool:
        """Check if relative path is metadata JSON."""
        normalized = relative_path.replace("\\", "/")
        if normalized.startswith("metadata/") and normalized.endswith(".json"):
            return True
        return Path(normalized).name == "metadata.json"

    @staticmethod
    def _extract_url_hash(relative_path: str) -> str:
        """Extract stable URL hash from relative object path."""
        normalized = relative_path.replace("\\", "/")
        path = Path(normalized)
        if path.name in {"document.md", "metadata.json"} and path.parent.name:
            return path.parent.name
        return path.stem

    @staticmethod
    def _load_metadata(path: Path) -> dict:
        """Load metadata JSON from file."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse metadata JSON: %s", path)
            return {}

    @staticmethod
    def _parse_iso_datetime(value: str) -> datetime:
        """Parse ISO datetime with Z support."""
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    
    async def get_sync_status(
        self,
        module: SpringModule | str,
        version: SpringVersion | str,
        submodule: str | None = None,
    ) -> dict:
        """Get current sync status for a module/version.
        
        Args:
            module: Spring module
            version: Version string
            
        Returns:
            Status dict with document counts and last sync info
        """
        # Get document count
        docs = await self.storage.documents.get_by_module_version(
            module=module,
            major=int(version.major) if isinstance(version, SpringVersion) else int(version.split(".")[0]),
            submodule=submodule,
        )
        
        # Get last sync
        last_sync = await self.storage.sync_history.get_latest(module, version, submodule=submodule)
        
        # Get unindexed count
        unindexed = [d for d in docs if not d.is_indexed]
        
        return {
            "module": module,
            "submodule": submodule,
            "version": version,
            "document_count": len(docs),
            "indexed_count": len(docs) - len(unindexed),
            "unindexed_count": len(unindexed),
            "last_sync": {
                "status": last_sync.status.value if last_sync else None,
                "started_at": last_sync.started_at.isoformat() if last_sync else None,
                "completed_at": last_sync.completed_at.isoformat() if last_sync and last_sync.completed_at else None,
                "files_added": last_sync.files_added if last_sync else 0,
                "files_modified": last_sync.files_modified if last_sync else 0,
                "files_removed": last_sync.files_removed if last_sync else 0,
            } if last_sync else None,
        }

    async def list_manifest_targets(self) -> list[dict[str, str | None]]:
        """List all cached manifest targets (module/submodule/version)."""
        cursor = await self.storage.db.execute(
            """
            SELECT module, submodule, version
            FROM local_manifest
            ORDER BY module, submodule, version
            """
        )
        rows = await cursor.fetchall()
        return [
            {
                "module": row["module"],
                "submodule": row["submodule"],
                "version": row["version"],
            }
            for row in rows
        ]
    
    async def list_modules(self) -> list[dict]:
        """List all synced modules with their versions.
        
        Returns:
            List of module info dicts
        """
        # Query distinct module/versions from documents
        cursor = await self.storage.db.execute(
            """
            SELECT module, submodule, major_version, minor_version, patch_version, COUNT(*) as doc_count
            FROM documents
            GROUP BY module, submodule, major_version, minor_version, patch_version
            ORDER BY module, submodule, major_version DESC, minor_version DESC, patch_version DESC
            """
        )
        rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            version = f"{row['major_version']}.{row['minor_version']}.{row['patch_version']}"
            results.append({
                "module": row["module"],
                "submodule": row["submodule"],
                "version": version,
                "document_count": row["doc_count"],
            })
        
        return results


__all__ = [
    "SyncOrchestrator",
    "ProgressCallback",
]
