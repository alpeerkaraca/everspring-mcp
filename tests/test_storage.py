"""Tests for storage module - SQLite schema and repositories."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from everspring_mcp.models.sync import FileEntry, SyncManifest, SyncStatus
from everspring_mcp.storage.schema import (
    CURRENT_SCHEMA_VERSION,
    SchemaManager,
    TableName,
    create_database,
    get_schema_sql,
)
from everspring_mcp.storage.repository import (
    DocumentRecord,
    DocumentRepository,
    LocalManifestRecord,
    LocalManifestRepository,
    StorageManager,
    SyncHistoryRecord,
    SyncHistoryRepository,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
async def db(temp_db_path: Path) -> aiosqlite.Connection:
    """Create initialized database connection."""
    conn = await create_database(temp_db_path)
    yield conn
    await conn.close()


@pytest.fixture
async def storage_manager(temp_db_path: Path) -> StorageManager:
    """Create storage manager."""
    async with StorageManager(temp_db_path) as storage:
        yield storage


@pytest.fixture
def sample_document() -> DocumentRecord:
    """Create sample document record."""
    return DocumentRecord(
        id="abc123def456",
        url="https://docs.spring.io/spring-boot/reference/getting-started.html",
        title="Getting Started",
        module="spring-boot",
        submodule=None,
        major_version=4,
        minor_version=0,
        patch_version=5,
        content_hash="a" * 64,
        file_path="getting-started.md",
        s3_key="docs/spring-boot/4.0.5/getting-started.md",
        size_bytes=12345,
        scraped_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        schema_version="1.0.0",
    )


@pytest.fixture
def sample_manifest() -> SyncManifest:
    """Create sample sync manifest."""
    return SyncManifest(
        version="1.0.0",
        pack_hash="b" * 64,
        file_count=2,
        total_size_bytes=25000,
        files=[
            FileEntry(
                path="getting-started.md",
                content_hash="a" * 64,
                size_bytes=12345,
            ),
            FileEntry(
                path="configuration.md",
                content_hash="c" * 64,
                size_bytes=12655,
            ),
        ],
    )


# ============================================================================
# Schema Tests
# ============================================================================


class TestSchema:
    """Tests for database schema."""
    
    def test_get_schema_sql_v1(self) -> None:
        """Test getting v1 schema SQL."""
        sql = get_schema_sql(1)
        
        assert "CREATE TABLE" in sql
        assert "documents" in sql
        assert "sync_history" in sql
        assert "local_manifest" in sql
        assert "schema_version" in sql
    
    def test_get_schema_sql_invalid_version(self) -> None:
        """Test getting invalid schema version raises error."""
        with pytest.raises(ValueError, match="Unsupported schema version"):
            get_schema_sql(999)
    
    def test_current_schema_version(self) -> None:
        """Test current schema version is defined."""
        assert CURRENT_SCHEMA_VERSION >= 1
    
    async def test_schema_manager_initialize(self, temp_db_path: Path) -> None:
        """Test schema manager creates tables."""
        async with aiosqlite.connect(temp_db_path) as db:
            manager = SchemaManager(db)
            await manager.initialize()
            
            # Verify tables exist
            assert await manager.verify_schema()
    
    async def test_schema_manager_get_version(self, db: aiosqlite.Connection) -> None:
        """Test getting current schema version."""
        manager = SchemaManager(db)
        version = await manager.get_current_version()
        
        assert version == CURRENT_SCHEMA_VERSION
    
    async def test_create_database(self, temp_db_path: Path) -> None:
        """Test create_database helper."""
        db = await create_database(temp_db_path)
        
        try:
            # Check WAL mode is enabled
            cursor = await db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row[0].lower() == "wal"
            
            # Check schema is initialized
            manager = SchemaManager(db)
            assert await manager.verify_schema()
        finally:
            await db.close()


# ============================================================================
# DocumentRepository Tests
# ============================================================================


class TestDocumentRepository:
    """Tests for DocumentRepository."""
    
    async def test_insert_and_get(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test inserting and retrieving a document."""
        repo = DocumentRepository(db)
        
        await repo.insert(sample_document)
        
        retrieved = await repo.get_by_id(sample_document.id)
        assert retrieved is not None
        assert retrieved.id == sample_document.id
        assert retrieved.title == sample_document.title
        assert retrieved.module == sample_document.module
    
    async def test_get_by_url(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test retrieving document by URL."""
        repo = DocumentRepository(db)
        await repo.insert(sample_document)
        
        retrieved = await repo.get_by_url(sample_document.url)
        assert retrieved is not None
        assert retrieved.id == sample_document.id
    
    async def test_get_by_s3_key(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test retrieving document by S3 key."""
        repo = DocumentRepository(db)
        await repo.insert(sample_document)
        
        retrieved = await repo.get_by_s3_key(sample_document.s3_key)
        assert retrieved is not None
        assert retrieved.id == sample_document.id
    
    async def test_get_nonexistent(self, db: aiosqlite.Connection) -> None:
        """Test retrieving nonexistent document returns None."""
        repo = DocumentRepository(db)
        
        assert await repo.get_by_id("nonexistent") is None
        assert await repo.get_by_url("https://example.com") is None
    
    async def test_upsert_insert(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test upsert inserts new document."""
        repo = DocumentRepository(db)
        
        await repo.upsert(sample_document)
        
        retrieved = await repo.get_by_id(sample_document.id)
        assert retrieved is not None
    
    async def test_upsert_update(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test upsert updates existing document."""
        repo = DocumentRepository(db)
        
        await repo.insert(sample_document)
        
        # Create updated version
        updated = DocumentRecord(
            id=sample_document.id,
            url=sample_document.url,
            title="Updated Title",
            module=sample_document.module,
            submodule=sample_document.submodule,
            major_version=sample_document.major_version,
            minor_version=sample_document.minor_version,
            patch_version=sample_document.patch_version,
            content_hash="d" * 64,  # Changed hash
            file_path=sample_document.file_path,
            s3_key=sample_document.s3_key,
            size_bytes=99999,
            scraped_at=sample_document.scraped_at,
            schema_version=sample_document.schema_version,
        )
        
        await repo.upsert(updated)
        
        retrieved = await repo.get_by_id(sample_document.id)
        assert retrieved is not None
        assert retrieved.title == "Updated Title"
        assert retrieved.content_hash == "d" * 64
        assert retrieved.is_indexed is False  # Reset due to hash change
    
    async def test_get_by_module_version(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test retrieving documents by module and version."""
        repo = DocumentRepository(db)
        await repo.insert(sample_document)
        
        docs = await repo.get_by_module_version("spring-boot", 4)
        assert len(docs) == 1
        assert docs[0].id == sample_document.id

        docs_with_submodule = await repo.get_by_module_version("spring-boot", 4, submodule="redis")
        assert len(docs_with_submodule) == 0
    
    async def test_get_unindexed(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test retrieving unindexed documents."""
        repo = DocumentRepository(db)
        
        # Insert with synced_at set
        doc_with_sync = DocumentRecord(
            **{**sample_document.model_dump(), "synced_at": datetime.now(timezone.utc)}
        )
        await repo.insert(doc_with_sync)
        
        unindexed = await repo.get_unindexed()
        assert len(unindexed) == 1
    
    async def test_mark_indexed(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test marking documents as indexed."""
        repo = DocumentRepository(db)
        await repo.insert(sample_document)
        
        count = await repo.mark_indexed([sample_document.id])
        assert count == 1
        
        retrieved = await repo.get_by_id(sample_document.id)
        assert retrieved is not None
        assert retrieved.is_indexed is True

    async def test_reset_indexed(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test resetting indexed flags back to unindexed."""
        repo = DocumentRepository(db)
        doc = DocumentRecord(
            **{
                **sample_document.model_dump(),
                "is_indexed": True,
                "synced_at": datetime.now(timezone.utc),
            }
        )
        await repo.insert(doc)

        reset = await repo.reset_indexed()
        assert reset == 1

        retrieved = await repo.get_by_id(sample_document.id)
        assert retrieved is not None
        assert retrieved.is_indexed is False
    
    async def test_mark_synced(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test marking document as synced."""
        repo = DocumentRepository(db)
        await repo.insert(sample_document)
        
        sync_time = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = await repo.mark_synced(sample_document.id, sync_time)
        assert result is True
        
        retrieved = await repo.get_by_id(sample_document.id)
        assert retrieved is not None
        assert retrieved.synced_at == sync_time
    
    async def test_delete(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test deleting a document."""
        repo = DocumentRepository(db)
        await repo.insert(sample_document)
        
        deleted = await repo.delete(sample_document.id)
        assert deleted is True
        
        assert await repo.get_by_id(sample_document.id) is None
    
    async def test_delete_nonexistent(self, db: aiosqlite.Connection) -> None:
        """Test deleting nonexistent document."""
        repo = DocumentRepository(db)
        
        deleted = await repo.delete("nonexistent")
        assert deleted is False
    
    async def test_count(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test counting documents."""
        repo = DocumentRepository(db)
        
        assert await repo.count() == 0
        
        await repo.insert(sample_document)
        assert await repo.count() == 1
        assert await repo.count("spring-boot") == 1
        assert await repo.count("spring-framework") == 0
    
    async def test_get_content_hashes(
        self,
        db: aiosqlite.Connection,
        sample_document: DocumentRecord,
    ) -> None:
        """Test getting content hashes mapping."""
        repo = DocumentRepository(db)
        await repo.insert(sample_document)
        
        hashes = await repo.get_content_hashes("spring-boot", "4.0.5")
        assert sample_document.s3_key in hashes
        assert hashes[sample_document.s3_key] == sample_document.content_hash


# ============================================================================
# SyncHistoryRepository Tests
# ============================================================================


class TestSyncHistoryRepository:
    """Tests for SyncHistoryRepository."""
    
    async def test_start_sync(self, db: aiosqlite.Connection) -> None:
        """Test starting a sync operation."""
        repo = SyncHistoryRepository(db)
        
        sync_id = await repo.start_sync("spring-boot", "4.0.5")
        assert sync_id > 0
    
    async def test_complete_sync(self, db: aiosqlite.Connection) -> None:
        """Test completing a sync operation."""
        repo = SyncHistoryRepository(db)
        
        sync_id = await repo.start_sync("spring-boot", "4.0.5")
        await repo.complete_sync(
            sync_id,
            files_added=10,
            files_modified=5,
            files_removed=2,
            bytes_downloaded=100000,
        )
        
        latest = await repo.get_latest("spring-boot", "4.0.5")
        assert latest is not None
        assert latest.status == SyncStatus.COMPLETED
        assert latest.files_added == 10
        assert latest.files_modified == 5
        assert latest.files_removed == 2
    
    async def test_fail_sync(self, db: aiosqlite.Connection) -> None:
        """Test failing a sync operation."""
        repo = SyncHistoryRepository(db)
        
        sync_id = await repo.start_sync("spring-boot", "4.0.5")
        await repo.fail_sync(sync_id, "Connection timeout")
        
        latest = await repo.get_latest("spring-boot", "4.0.5")
        assert latest is not None
        assert latest.status == SyncStatus.FAILED
        assert latest.error_message == "Connection timeout"
    
    async def test_get_in_progress(self, db: aiosqlite.Connection) -> None:
        """Test getting in-progress syncs."""
        repo = SyncHistoryRepository(db)
        
        await repo.start_sync("spring-boot", "4.0.5")
        await repo.start_sync("spring-framework", "7.0.0")
        
        in_progress = await repo.get_in_progress()
        assert len(in_progress) == 2
    
    async def test_get_history(self, db: aiosqlite.Connection) -> None:
        """Test getting sync history."""
        repo = SyncHistoryRepository(db)
        
        sync1 = await repo.start_sync("spring-boot", "4.0.5")
        await repo.complete_sync(sync1)
        
        sync2 = await repo.start_sync("spring-boot", "4.0.5")
        await repo.complete_sync(sync2)
        
        history = await repo.get_history("spring-boot", limit=10)
        assert len(history) == 2

        history_submodule = await repo.get_history("spring-boot", submodule="redis", limit=10)
        assert len(history_submodule) == 0


# ============================================================================
# LocalManifestRepository Tests
# ============================================================================


class TestLocalManifestRepository:
    """Tests for LocalManifestRepository."""
    
    async def test_save_and_get(
        self,
        db: aiosqlite.Connection,
        sample_manifest: SyncManifest,
    ) -> None:
        """Test saving and retrieving manifest."""
        repo = LocalManifestRepository(db)
        
        await repo.save(
            module="spring-boot",
            version="4.0.5",
            manifest=sample_manifest,
            manifest_hash="e" * 64,
        )
        
        record = await repo.get("spring-boot", "4.0.5")
        assert record is not None
        assert record.module == "spring-boot"
        assert record.version == "4.0.5"
        assert record.manifest_hash == "e" * 64

        record_submodule = await repo.get("spring-boot", "4.0.5", submodule="redis")
        assert record_submodule is None
    
    async def test_get_manifest_object(
        self,
        db: aiosqlite.Connection,
        sample_manifest: SyncManifest,
    ) -> None:
        """Test getting parsed manifest object."""
        repo = LocalManifestRepository(db)
        
        await repo.save(
            module="spring-boot",
            version="4.0.5",
            manifest=sample_manifest,
            manifest_hash="e" * 64,
        )
        
        record = await repo.get("spring-boot", "4.0.5")
        assert record is not None
        
        manifest = record.get_manifest()
        assert manifest.version == sample_manifest.version
        assert manifest.file_count == sample_manifest.file_count
    
    async def test_update_existing(
        self,
        db: aiosqlite.Connection,
        sample_manifest: SyncManifest,
    ) -> None:
        """Test updating existing manifest."""
        repo = LocalManifestRepository(db)
        
        await repo.save(
            module="spring-boot",
            version="4.0.5",
            manifest=sample_manifest,
            manifest_hash="e" * 64,
        )
        
        # Update with new hash
        await repo.save(
            module="spring-boot",
            version="4.0.5",
            manifest=sample_manifest,
            manifest_hash="f" * 64,
        )
        
        record = await repo.get("spring-boot", "4.0.5")
        assert record is not None
        assert record.manifest_hash == "f" * 64
    
    async def test_delete(
        self,
        db: aiosqlite.Connection,
        sample_manifest: SyncManifest,
    ) -> None:
        """Test deleting manifest."""
        repo = LocalManifestRepository(db)
        
        await repo.save(
            module="spring-boot",
            version="4.0.5",
            manifest=sample_manifest,
            manifest_hash="e" * 64,
        )
        
        deleted = await repo.delete("spring-boot", "4.0.5")
        assert deleted is True
        
        assert await repo.get("spring-boot", "4.0.5") is None


# ============================================================================
# StorageManager Tests
# ============================================================================


class TestStorageManager:
    """Tests for StorageManager."""
    
    async def test_context_manager(self, temp_db_path: Path) -> None:
        """Test StorageManager as async context manager."""
        async with StorageManager(temp_db_path) as storage:
            assert storage.documents is not None
            assert storage.sync_history is not None
            assert storage.manifests is not None
    
    async def test_repositories_accessible(
        self,
        storage_manager: StorageManager,
        sample_document: DocumentRecord,
    ) -> None:
        """Test all repositories are accessible."""
        # Insert via documents repo
        await storage_manager.documents.insert(sample_document)
        
        # Verify via get
        doc = await storage_manager.documents.get_by_id(sample_document.id)
        assert doc is not None
    
    async def test_not_connected_raises(self, temp_db_path: Path) -> None:
        """Test accessing repos before connect raises error."""
        manager = StorageManager(temp_db_path)
        
        with pytest.raises(RuntimeError, match="not connected"):
            _ = manager.documents


# ============================================================================
# DocumentRecord Tests
# ============================================================================


class TestDocumentRecord:
    """Tests for DocumentRecord model."""
    
    def test_version_string(self, sample_document: DocumentRecord) -> None:
        """Test version_string property."""
        assert sample_document.version_string == "4.0.5"
    
    def test_from_row(self) -> None:
        """Test creating from database row."""
        # Simulate a row dict
        row_data = {
            "id": "test123",
            "url": "https://example.com",
            "title": "Test",
            "module": "spring-boot",
            "submodule": None,
            "major_version": 4,
            "minor_version": 1,
            "patch_version": 2,
            "content_hash": "a" * 64,
            "file_path": "test.md",
            "s3_key": "docs/test.md",
            "size_bytes": 1000,
            "scraped_at": "2024-01-15T10:30:00+00:00",
            "synced_at": None,
            "schema_version": "1.0.0",
            "is_indexed": 0,
        }
        
        # Create mock row
        class MockRow:
            def __getitem__(self, key: str) -> Any:
                return row_data[key]
        
        record = DocumentRecord.from_row(MockRow())
        
        assert record.id == "test123"
        assert record.major_version == 4
        assert record.is_indexed is False
