"""EverSpring MCP - SQLite repository classes.

This module provides repository classes for database operations:
- DocumentRepository: CRUD for document metadata
- SyncHistoryRepository: Sync operation tracking
- LocalManifestRepository: Manifest cache management
- StorageManager: Unified access to all repositories
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
from everspring_mcp.models.content import ScrapedPage
from pydantic import BaseModel, ConfigDict, Field

from everspring_mcp.models.spring import SpringModule
from everspring_mcp.models.sync import SyncManifest, SyncStatus
from everspring_mcp.storage.schema import create_database
from everspring_mcp.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger("storage.repository")


# ============================================================================
# Pydantic Models for Database Records
# ============================================================================


class DocumentRecord(BaseModel):
    """Database record for a document."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="SHA256 hash of URL")
    url: str = Field(description="Original source URL")
    title: str = Field(description="Page title")
    module: str = Field(description="Spring module name")
    submodule: str | None = Field(default=None, description="Submodule key")
    major_version: int = Field(description="Major version")
    minor_version: int = Field(default=0, description="Minor version")
    patch_version: int = Field(default=0, description="Patch version")
    content_hash: str = Field(description="SHA256 of content")
    file_path: str = Field(description="Local relative path")
    s3_key: str = Field(description="S3 object key")
    size_bytes: int = Field(description="File size")
    scraped_at: datetime = Field(description="When scraped")
    synced_at: datetime | None = Field(default=None, description="When synced locally")
    schema_version: str = Field(description="Schema version")
    is_indexed: bool = Field(default=False, description="Has been vectorized")

    @property
    def version_string(self) -> str:
        """Get version as string."""
        return f"{self.major_version}.{self.minor_version}.{self.patch_version}"

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> DocumentRecord:
        """Create from database row."""
        return cls(
            id=row["id"],
            url=row["url"],
            title=row["title"],
            module=row["module"],
            submodule=row["submodule"],
            major_version=row["major_version"],
            minor_version=row["minor_version"],
            patch_version=row["patch_version"],
            content_hash=row["content_hash"],
            file_path=row["file_path"],
            s3_key=row["s3_key"],
            size_bytes=row["size_bytes"],
            scraped_at=datetime.fromisoformat(row["scraped_at"]),
            synced_at=datetime.fromisoformat(row["synced_at"])
            if row["synced_at"]
            else None,
            schema_version=row["schema_version"],
            is_indexed=bool(row["is_indexed"]),
        )


class SyncHistoryRecord(BaseModel):
    """Database record for sync history."""

    model_config = ConfigDict(frozen=True)

    id: int | None = Field(default=None, description="Auto-increment ID")
    module: str = Field(description="Spring module")
    submodule: str | None = Field(default=None, description="Submodule key")
    version: str = Field(description="Version string")
    status: SyncStatus = Field(description="Sync status")
    started_at: datetime = Field(description="Start time")
    completed_at: datetime | None = Field(default=None, description="End time")
    files_added: int = Field(default=0)
    files_modified: int = Field(default=0)
    files_removed: int = Field(default=0)
    bytes_downloaded: int = Field(default=0)
    error_message: str | None = Field(default=None)
    manifest_version: str | None = Field(default=None)

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> SyncHistoryRecord:
        """Create from database row."""
        return cls(
            id=row["id"],
            module=row["module"],
            submodule=row["submodule"],
            version=row["version"],
            status=SyncStatus(row["status"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            files_added=row["files_added"],
            files_modified=row["files_modified"],
            files_removed=row["files_removed"],
            bytes_downloaded=row["bytes_downloaded"],
            error_message=row["error_message"],
            manifest_version=row["manifest_version"],
        )


class LocalManifestRecord(BaseModel):
    """Database record for local manifest cache."""

    model_config = ConfigDict(frozen=True)

    module: str = Field(description="Spring module")
    submodule: str | None = Field(default=None, description="Submodule key")
    version: str = Field(description="Version string")
    manifest_hash: str = Field(description="SHA256 of manifest")
    manifest_json: str = Field(description="Full manifest JSON")
    fetched_at: datetime = Field(description="When fetched")

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> LocalManifestRecord:
        """Create from database row."""
        return cls(
            module=row["module"],
            submodule=row["submodule"],
            version=row["version"],
            manifest_hash=row["manifest_hash"],
            manifest_json=row["manifest_json"],
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
        )

    def get_manifest(self) -> SyncManifest:
        """Parse manifest JSON to SyncManifest model."""
        return SyncManifest.model_validate_json(self.manifest_json)


# ============================================================================
# Repository Classes
# ============================================================================


class DocumentRepository:
    """Repository for document metadata operations.

    Provides CRUD operations for the documents table.

    Example:
        async with aiosqlite.connect(db_path) as db:
            repo = DocumentRepository(db)
            doc = await repo.get_by_id("abc123")
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        """Initialize repository.

        Args:
            db: SQLite connection
        """
        self.db = db

    def _to_row_tuple(self, doc: DocumentRecord) -> tuple[Any, ...]:
        """Convert a DocumentRecord to a tuple for sqlite insertion."""
        return (
            doc.id,
            doc.url,
            doc.title,
            doc.module,
            doc.submodule,
            doc.major_version,
            doc.minor_version,
            doc.patch_version,
            doc.content_hash,
            doc.file_path,
            doc.s3_key,
            doc.size_bytes,
            doc.scraped_at.isoformat(),
            doc.synced_at.isoformat() if doc.synced_at else None,
            doc.schema_version,
            int(doc.is_indexed),
        )

    async def insert(self, doc: DocumentRecord) -> None:
        """Insert a new document.

        Args:
            doc: Document to insert

        Raises:
            aiosqlite.IntegrityError: If document already exists
        """
        await self.db.execute(
            """
            INSERT INTO documents (
                id, url, title, module, submodule, major_version, minor_version, patch_version,
                content_hash, file_path, s3_key, size_bytes, scraped_at, synced_at,
                schema_version, is_indexed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._to_row_tuple(doc),
        )
        await self.db.commit()
        logger.debug("Inserted document: %s", doc.id)

    async def upsert(self, doc: DocumentRecord) -> None:
        """Insert or update a document.

        Args:
            doc: Document to upsert
        """
        await self.db.execute(
            """
            INSERT INTO documents (
                id, url, title, module, submodule, major_version, minor_version, patch_version,
                content_hash, file_path, s3_key, size_bytes, scraped_at, synced_at,
                schema_version, is_indexed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                content_hash = excluded.content_hash,
                file_path = excluded.file_path,
                s3_key = excluded.s3_key,
                size_bytes = excluded.size_bytes,
                scraped_at = excluded.scraped_at,
                synced_at = excluded.synced_at,
                schema_version = excluded.schema_version,
                is_indexed = CASE 
                    WHEN excluded.content_hash != documents.content_hash THEN 0 
                    ELSE documents.is_indexed 
                END
            """,
            self._to_row_tuple(doc),
        )
        await self.db.commit()
        logger.debug("Upserted document: %s", doc.id)

    async def upsert_many(self, docs: list[DocumentRecord]) -> None:
        """Insert or update multiple documents in a single transaction.

        Args:
            docs: List of documents to upsert
        """
        if not docs:
            return

        await self.db.executemany(
            """
            INSERT INTO documents (
                id, url, title, module, submodule, major_version, minor_version, patch_version,
                content_hash, file_path, s3_key, size_bytes, scraped_at, synced_at,
                schema_version, is_indexed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                content_hash = excluded.content_hash,
                file_path = excluded.file_path,
                s3_key = excluded.s3_key,
                size_bytes = excluded.size_bytes,
                scraped_at = excluded.scraped_at,
                synced_at = excluded.synced_at,
                schema_version = excluded.schema_version,
                is_indexed = CASE 
                    WHEN excluded.content_hash != documents.content_hash THEN 0 
                    ELSE documents.is_indexed 
                END
            """,
            (self._to_row_tuple(doc) for doc in docs),
        )
        await self.db.commit()
        logger.debug("Upserted %d documents in batch", len(docs))

    async def save(self, page: ScrapedPage, s3_key: str = "", file_path: str = "") -> None:
        """Convert ScrapedPage to DocumentRecord and upsert.

        Args:
            page: Scraped page model
            s3_key: Optional S3 key
            file_path: Optional local file path
        """
        from everspring_mcp.models.base import compute_hash

        doc = DocumentRecord(
            id=compute_hash(str(page.url)),
            url=str(page.url),
            title=page.title,
            module=page.module.value,
            submodule=page.submodule,
            major_version=page.version.major,
            minor_version=page.version.minor,
            patch_version=page.version.patch,
            content_hash=page.content_hash,
            file_path=file_path or s3_key,  # Fallback
            s3_key=s3_key,
            size_bytes=len(page.markdown_content.encode("utf-8")),
            scraped_at=page.scraped_at,
            synced_at=datetime.now(UTC),  # Consider it synced if we are saving it manually
            schema_version="1.0",
            is_indexed=False,
        )
        await self.upsert(doc)

    async def get_by_id(self, doc_id: str) -> DocumentRecord | None:
        """Get document by ID.

        Args:
            doc_id: Document ID (SHA256 hash)

        Returns:
            Document record or None if not found
        """
        cursor = await self.db.execute(
            "SELECT * FROM documents WHERE id = ?",
            (doc_id,),
        )
        row = await cursor.fetchone()
        return DocumentRecord.from_row(row) if row else None

    async def get_by_url(self, url: str) -> DocumentRecord | None:
        """Get document by URL.

        Args:
            url: Source URL

        Returns:
            Document record or None if not found
        """
        cursor = await self.db.execute(
            "SELECT * FROM documents WHERE url = ?",
            (url,),
        )
        row = await cursor.fetchone()
        return DocumentRecord.from_row(row) if row else None

    async def get_by_s3_key(self, s3_key: str) -> DocumentRecord | None:
        """Get document by S3 key.

        Args:
            s3_key: S3 object key

        Returns:
            Document record or None if not found
        """
        cursor = await self.db.execute(
            "SELECT * FROM documents WHERE s3_key = ?",
            (s3_key,),
        )
        row = await cursor.fetchone()
        return DocumentRecord.from_row(row) if row else None

    async def get_by_module_version(
        self,
        module: str | SpringModule,
        major: int,
        minor: int | None = None,
        patch: int | None = None,
        submodule: str | None = None,
    ) -> list[DocumentRecord]:
        """Get documents by module and version.

        Args:
            module: Spring module
            major: Major version
            minor: Minor version (optional)

        Returns:
            List of matching documents
        """
        module_str = module.value if isinstance(module, SpringModule) else module

        filters = ["module = ?", "major_version = ?", "submodule IS ?"]
        params: list[object] = [module_str, major, submodule]

        if minor is not None:
            filters.append("minor_version = ?")
            params.append(minor)
        if patch is not None:
            filters.append("patch_version = ?")
            params.append(patch)

        if minor is not None and patch is not None:
            order_by = "file_path"
        else:
            order_by = "minor_version DESC, patch_version DESC, file_path"
        sql = f"""
            SELECT * FROM documents
            WHERE {" AND ".join(filters)}
            ORDER BY {order_by}
        """

        cursor = await self.db.execute(sql, tuple(params))

        rows = await cursor.fetchall()
        return [DocumentRecord.from_row(row) for row in rows]

    async def get_unindexed(self, limit: int = 100) -> list[DocumentRecord]:
        """Get documents that haven't been vectorized.

        Args:
            limit: Maximum number to return

        Returns:
            List of unindexed documents
        """
        cursor = await self.db.execute(
            """
            SELECT * FROM documents 
            WHERE is_indexed = 0 AND synced_at IS NOT NULL
            ORDER BY synced_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [DocumentRecord.from_row(row) for row in rows]

    async def mark_indexed(self, doc_ids: Sequence[str]) -> int:
        """Mark documents as indexed.

        Args:
            doc_ids: List of document IDs

        Returns:
            Number of documents updated
        """
        if not doc_ids:
            return 0

        placeholders = ",".join("?" * len(doc_ids))
        cursor = await self.db.execute(
            f"UPDATE documents SET is_indexed = 1 WHERE id IN ({placeholders})",
            tuple(doc_ids),
        )
        await self.db.commit()
        return cursor.rowcount

    async def reset_indexed(
        self,
        module: str | SpringModule | None = None,
        major: int | None = None,
        submodule: str | None = None,
    ) -> int:
        """Reset is_indexed flag back to 0.

        Args:
            module: Optional module filter
            major: Optional major version filter
            submodule: Optional submodule filter. Ignored unless module is set.

        Returns:
            Number of documents updated
        """
        filters: list[str] = ["is_indexed = 1"]
        params: list[object] = []

        if module is not None:
            module_str = module.value if isinstance(module, SpringModule) else module
            filters.extend(["module = ?", "submodule IS ?"])
            params.extend([module_str, submodule])
        if major is not None:
            filters.append("major_version = ?")
            params.append(major)

        where_clause = " AND ".join(filters)
        cursor = await self.db.execute(
            f"UPDATE documents SET is_indexed = 0 WHERE {where_clause}",
            tuple(params),
        )
        await self.db.commit()
        return cursor.rowcount

    async def mark_synced(self, doc_id: str, synced_at: datetime | None = None) -> bool:
        """Mark a document as synced.

        Args:
            doc_id: Document ID
            synced_at: Sync timestamp (defaults to now)

        Returns:
            True if document was updated
        """
        synced_at = synced_at or datetime.now(UTC)
        cursor = await self.db.execute(
            "UPDATE documents SET synced_at = ? WHERE id = ?",
            (synced_at.isoformat(), doc_id),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def delete(self, doc_id: str) -> bool:
        """Delete a document.

        Args:
            doc_id: Document ID

        Returns:
            True if document was deleted
        """
        cursor = await self.db.execute(
            "DELETE FROM documents WHERE id = ?",
            (doc_id,),
        )
        await self.db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("Deleted document: %s", doc_id)
        return deleted

    async def delete_by_s3_key(self, s3_key: str) -> bool:
        """Delete a document by S3 key.

        Args:
            s3_key: S3 object key

        Returns:
            True if document was deleted
        """
        cursor = await self.db.execute(
            "DELETE FROM documents WHERE s3_key = ?",
            (s3_key,),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def count(self, module: str | None = None) -> int:
        """Count documents.

        Args:
            module: Filter by module (optional)

        Returns:
            Document count
        """
        if module:
            cursor = await self.db.execute(
                "SELECT COUNT(*) FROM documents WHERE module = ?",
                (module,),
            )
        else:
            cursor = await self.db.execute("SELECT COUNT(*) FROM documents")

        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_content_hashes(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> dict[str, str]:
        """Get mapping of S3 keys to content hashes for a module/version.

        Args:
            module: Spring module
            version: Version string (e.g., "4.0.5")

        Returns:
            Dict mapping s3_key to content_hash
        """
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

        cursor = await self.db.execute(
            """
            SELECT s3_key, content_hash FROM documents
            WHERE module = ? AND submodule IS ?
            AND major_version = ? AND minor_version = ? AND patch_version = ?
            """,
            (module, submodule, major, minor, patch),
        )
        rows = await cursor.fetchall()
        return {row["s3_key"]: row["content_hash"] for row in rows}


class SyncHistoryRepository:
    """Repository for sync history operations.

    Tracks sync operations for auditing and resume support.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        """Initialize repository."""
        self.db = db

    async def start_sync(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
        manifest_version: str | None = None,
    ) -> int:
        """Record start of a sync operation.

        Args:
            module: Spring module
            version: Version string
            manifest_version: S3 manifest version

        Returns:
            Sync history ID
        """
        cursor = await self.db.execute(
            """
            INSERT INTO sync_history (
                module, submodule, version, status, started_at, manifest_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                module,
                submodule,
                version,
                SyncStatus.IN_PROGRESS.value,
                datetime.now(UTC).isoformat(),
                manifest_version,
            ),
        )
        await self.db.commit()
        return cursor.lastrowid or 0

    async def complete_sync(
        self,
        sync_id: int,
        files_added: int = 0,
        files_modified: int = 0,
        files_removed: int = 0,
        bytes_downloaded: int = 0,
    ) -> None:
        """Record successful completion of a sync.

        Args:
            sync_id: Sync history ID
            files_added: Number of files added
            files_modified: Number of files modified
            files_removed: Number of files removed
            bytes_downloaded: Total bytes downloaded
        """
        await self.db.execute(
            """
            UPDATE sync_history SET
                status = ?,
                completed_at = ?,
                files_added = ?,
                files_modified = ?,
                files_removed = ?,
                bytes_downloaded = ?
            WHERE id = ?
            """,
            (
                SyncStatus.COMPLETED.value,
                datetime.now(UTC).isoformat(),
                files_added,
                files_modified,
                files_removed,
                bytes_downloaded,
                sync_id,
            ),
        )
        await self.db.commit()
        logger.info(
            "Sync %d completed: +%d ~%d -%d (%d bytes)",
            sync_id,
            files_added,
            files_modified,
            files_removed,
            bytes_downloaded,
        )

    async def fail_sync(self, sync_id: int, error_message: str) -> None:
        """Record sync failure.

        Args:
            sync_id: Sync history ID
            error_message: Error description
        """
        await self.db.execute(
            """
            UPDATE sync_history SET
                status = ?,
                completed_at = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                SyncStatus.FAILED.value,
                datetime.now(UTC).isoformat(),
                error_message,
                sync_id,
            ),
        )
        await self.db.commit()
        logger.error("Sync %d failed: %s", sync_id, error_message)

    async def get_latest(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> SyncHistoryRecord | None:
        """Get most recent sync for a module/version.

        Args:
            module: Spring module
            version: Version string

        Returns:
            Most recent sync record or None
        """
        cursor = await self.db.execute(
            """
            SELECT * FROM sync_history
            WHERE module = ? AND submodule IS ? AND version = ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (module, submodule, version),
        )
        row = await cursor.fetchone()
        return SyncHistoryRecord.from_row(row) if row else None

    async def get_in_progress(self) -> list[SyncHistoryRecord]:
        """Get all in-progress syncs (for resume).

        Returns:
            List of in-progress sync records
        """
        cursor = await self.db.execute(
            "SELECT * FROM sync_history WHERE status = ?",
            (SyncStatus.IN_PROGRESS.value,),
        )
        rows = await cursor.fetchall()
        return [SyncHistoryRecord.from_row(row) for row in rows]

    async def get_history(
        self,
        module: str | None = None,
        submodule: str | None = None,
        limit: int = 50,
    ) -> list[SyncHistoryRecord]:
        """Get sync history.

        Args:
            module: Filter by module (optional)
            limit: Maximum records to return

        Returns:
            List of sync history records
        """
        if module:
            cursor = await self.db.execute(
                """
                SELECT * FROM sync_history
                WHERE module = ? AND submodule IS ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (module, submodule, limit),
            )
        else:
            cursor = await self.db.execute(
                """
                SELECT * FROM sync_history
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = await cursor.fetchall()
        return [SyncHistoryRecord.from_row(row) for row in rows]


class LocalManifestRepository:
    """Repository for local manifest cache.

    Caches S3 manifests locally for incremental sync comparison.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        """Initialize repository."""
        self.db = db

    async def get(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> LocalManifestRecord | None:
        """Get cached manifest.

        Args:
            module: Spring module
            version: Version string

        Returns:
            Cached manifest or None
        """
        cursor = await self.db.execute(
            "SELECT * FROM local_manifest WHERE module = ? AND submodule IS ? AND version = ?",
            (module, submodule, version),
        )
        row = await cursor.fetchone()
        return LocalManifestRecord.from_row(row) if row else None

    async def save(
        self,
        module: str,
        version: str,
        manifest: SyncManifest,
        manifest_hash: str,
        submodule: str | None = None,
    ) -> None:
        """Save or update cached manifest.

        Args:
            module: Spring module
            version: Version string
            manifest: Manifest to cache
            manifest_hash: SHA256 of manifest content
        """
        manifest_json = manifest.model_dump_json(
            exclude={"computed_total_size"},
        )

        fetched_at = datetime.now(UTC).isoformat()
        cursor = await self.db.execute(
            """
            UPDATE local_manifest
            SET manifest_hash = ?, manifest_json = ?, fetched_at = ?
            WHERE module = ? AND submodule IS ? AND version = ?
            """,
            (manifest_hash, manifest_json, fetched_at, module, submodule, version),
        )
        if cursor.rowcount == 0:
            await self.db.execute(
                """
                INSERT INTO local_manifest (
                    module, submodule, version, manifest_hash, manifest_json, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    module,
                    submodule,
                    version,
                    manifest_hash,
                    manifest_json,
                    fetched_at,
                ),
            )
        await self.db.commit()
        logger.debug("Saved manifest for %s/%s", module, version)

    async def delete(
        self, module: str, version: str, submodule: str | None = None
    ) -> bool:
        """Delete cached manifest.

        Args:
            module: Spring module
            version: Version string

        Returns:
            True if manifest was deleted
        """
        cursor = await self.db.execute(
            "DELETE FROM local_manifest WHERE module = ? AND submodule IS ? AND version = ?",
            (module, submodule, version),
        )
        await self.db.commit()
        return cursor.rowcount > 0


class StorageManager:
    """Unified access to all storage repositories.

    Provides a single entry point for database operations
    with automatic schema initialization.

    Example:
        async with StorageManager(db_path) as storage:
            doc = await storage.documents.get_by_id("abc123")
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialize storage manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        self._documents: DocumentRepository | None = None
        self._sync_history: SyncHistoryRepository | None = None
        self._manifests: LocalManifestRepository | None = None

    async def __aenter__(self) -> StorageManager:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Connect to database and initialize schema."""
        start = time.perf_counter()
        self._db = await create_database(self.db_path)
        self._documents = DocumentRepository(self._db)
        self._sync_history = SyncHistoryRepository(self._db)
        self._manifests = LocalManifestRepository(self._db)
        logger.info(
            f"Connected to database: {self.db_path} in {time.perf_counter() - start:.2f}s"
        )

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.debug("Closed database connection")

    @property
    def documents(self) -> DocumentRepository:
        """Get document repository."""
        if not self._documents:
            raise RuntimeError("StorageManager not connected")
        return self._documents

    @property
    def sync_history(self) -> SyncHistoryRepository:
        """Get sync history repository."""
        if not self._sync_history:
            raise RuntimeError("StorageManager not connected")
        return self._sync_history

    @property
    def manifests(self) -> LocalManifestRepository:
        """Get manifest repository."""
        if not self._manifests:
            raise RuntimeError("StorageManager not connected")
        return self._manifests

    @property
    def db(self) -> aiosqlite.Connection:
        """Get raw database connection."""
        if not self._db:
            raise RuntimeError("StorageManager not connected")
        return self._db


__all__ = [
    # Models
    "DocumentRecord",
    "SyncHistoryRecord",
    "LocalManifestRecord",
    # Repositories
    "DocumentRepository",
    "SyncHistoryRepository",
    "LocalManifestRepository",
    "StorageManager",
]
