"""EverSpring MCP - SQLite schema definitions.

This module defines the database schema for tracking:
- Document metadata (scraped pages)
- Sync history (audit trail)
- Local manifest cache (for incremental sync)

Supports schema migrations for backward compatibility.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

from everspring_mcp.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("storage.schema")

# Current schema version - increment when schema changes
CURRENT_SCHEMA_VERSION = 2


class TableName(str, Enum):
    """Database table names."""

    DOCUMENTS = "documents"
    SYNC_HISTORY = "sync_history"
    LOCAL_MANIFEST = "local_manifest"
    SCHEMA_VERSION = "schema_version"


# SQL for creating tables
SCHEMA_V1 = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);

-- Main documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,                    -- SHA256 hash of URL (stable ID)
    url TEXT NOT NULL UNIQUE,               -- Original source URL
    title TEXT NOT NULL,                    -- Page title
    module TEXT NOT NULL,                   -- spring-boot, spring-framework, etc.
    major_version INTEGER NOT NULL,         -- Major version number
    minor_version INTEGER NOT NULL DEFAULT 0,
    patch_version INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,             -- SHA256 of markdown content
    file_path TEXT NOT NULL,                -- Local relative path
    s3_key TEXT NOT NULL,                   -- S3 object key
    size_bytes INTEGER NOT NULL,            -- File size in bytes
    scraped_at TEXT NOT NULL,               -- ISO timestamp when scraped
    synced_at TEXT,                         -- ISO timestamp when synced locally
    schema_version TEXT NOT NULL,           -- For backward compat
    is_indexed INTEGER NOT NULL DEFAULT 0,  -- Has been vectorized? (0/1)
    UNIQUE(module, major_version, file_path)
);

-- Sync history for auditing
CREATE TABLE IF NOT EXISTS sync_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    module TEXT NOT NULL,                   -- Spring module
    submodule TEXT,                         -- Optional submodule
    version TEXT NOT NULL,                  -- Version string "4.0.5"
    status TEXT NOT NULL,                   -- pending, in_progress, completed, failed
    started_at TEXT NOT NULL,               -- ISO timestamp
    completed_at TEXT,                      -- ISO timestamp
    files_added INTEGER NOT NULL DEFAULT 0,
    files_modified INTEGER NOT NULL DEFAULT 0,
    files_removed INTEGER NOT NULL DEFAULT 0,
    bytes_downloaded INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    manifest_version TEXT                   -- S3 manifest version used
);

-- Local manifest cache
CREATE TABLE IF NOT EXISTS local_manifest (
    module TEXT NOT NULL,
    submodule TEXT,
    version TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,            -- SHA256 of manifest content
    manifest_json TEXT NOT NULL,            -- Full manifest JSON
    fetched_at TEXT NOT NULL,               -- ISO timestamp
    PRIMARY KEY (module, submodule, version)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_documents_module_version 
    ON documents(module, major_version, minor_version);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash 
    ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_is_indexed 
    ON documents(is_indexed);
CREATE INDEX IF NOT EXISTS idx_documents_s3_key 
    ON documents(s3_key);
CREATE INDEX IF NOT EXISTS idx_sync_history_module_version 
    ON sync_history(module, submodule, version);
CREATE INDEX IF NOT EXISTS idx_sync_history_status 
    ON sync_history(status);
CREATE INDEX IF NOT EXISTS idx_local_manifest_module_version
    ON local_manifest(module, submodule, version);
"""

SCHEMA_V2 = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);

-- Main documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,                    -- SHA256 hash of URL (stable ID)
    url TEXT NOT NULL UNIQUE,               -- Original source URL
    title TEXT NOT NULL,                    -- Page title
    module TEXT NOT NULL,                   -- spring-boot, spring-framework, etc.
    submodule TEXT,                         -- Optional submodule key
    major_version INTEGER NOT NULL,         -- Major version number
    minor_version INTEGER NOT NULL DEFAULT 0,
    patch_version INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,             -- SHA256 of markdown content
    file_path TEXT NOT NULL,                -- Local relative path
    s3_key TEXT NOT NULL,                   -- S3 object key
    size_bytes INTEGER NOT NULL,            -- File size in bytes
    scraped_at TEXT NOT NULL,               -- ISO timestamp when scraped
    synced_at TEXT,                         -- ISO timestamp when synced locally
    schema_version TEXT NOT NULL,           -- For backward compat
    is_indexed INTEGER NOT NULL DEFAULT 0,  -- Has been vectorized? (0/1)
    UNIQUE(module, submodule, major_version, file_path)
);

-- Sync history for auditing
CREATE TABLE IF NOT EXISTS sync_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    module TEXT NOT NULL,                   -- Spring module
    submodule TEXT,                         -- Optional submodule
    version TEXT NOT NULL,                  -- Version string "4.0.5"
    status TEXT NOT NULL,                   -- pending, in_progress, completed, failed
    started_at TEXT NOT NULL,               -- ISO timestamp
    completed_at TEXT,                      -- ISO timestamp
    files_added INTEGER NOT NULL DEFAULT 0,
    files_modified INTEGER NOT NULL DEFAULT 0,
    files_removed INTEGER NOT NULL DEFAULT 0,
    bytes_downloaded INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    manifest_version TEXT                   -- S3 manifest version used
);

-- Local manifest cache
CREATE TABLE IF NOT EXISTS local_manifest (
    module TEXT NOT NULL,
    submodule TEXT,
    version TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,            -- SHA256 of manifest content
    manifest_json TEXT NOT NULL,            -- Full manifest JSON
    fetched_at TEXT NOT NULL,               -- ISO timestamp
    PRIMARY KEY (module, submodule, version)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_documents_module_version 
    ON documents(module, submodule, major_version, minor_version);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash 
    ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_is_indexed 
    ON documents(is_indexed);
CREATE INDEX IF NOT EXISTS idx_documents_s3_key 
    ON documents(s3_key);
CREATE INDEX IF NOT EXISTS idx_sync_history_module_version 
    ON sync_history(module, submodule, version);
CREATE INDEX IF NOT EXISTS idx_sync_history_status 
    ON sync_history(status);
"""


def get_schema_sql(version: int = CURRENT_SCHEMA_VERSION) -> str:
    """Get SQL schema for a specific version.
    
    Args:
        version: Schema version number
        
    Returns:
        SQL string for creating tables
        
    Raises:
        ValueError: If version is not supported
    """
    schemas = {
        1: SCHEMA_V1,
        2: SCHEMA_V2,
    }

    if version not in schemas:
        raise ValueError(f"Unsupported schema version: {version}")

    return schemas[version]


class SchemaManager:
    """Manages database schema creation and migrations.
    
    Handles:
    - Initial schema creation
    - Version tracking
    - Future migrations
    
    Example:
        async with aiosqlite.connect(db_path) as db:
            manager = SchemaManager(db)
            await manager.initialize()
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        """Initialize schema manager.
        
        Args:
            db: SQLite connection
        """
        self.db = db

    async def get_current_version(self) -> int | None:
        """Get current schema version from database.
        
        Returns:
            Current version number, or None if not initialized
        """
        try:
            cursor = await self.db.execute(
                "SELECT MAX(version) FROM schema_version"
            )
            row = await cursor.fetchone()
            return row[0] if row and row[0] is not None else None
        except aiosqlite.OperationalError:
            # Table doesn't exist
            return None

    async def initialize(self) -> None:
        """Initialize database schema.
        
        Creates tables if they don't exist, or migrates
        if schema version is outdated.
        """
        current = await self.get_current_version()

        if current is None:
            # Fresh database - create schema
            logger.info("Creating database schema v%d", CURRENT_SCHEMA_VERSION)
            await self._create_schema()
        elif current < CURRENT_SCHEMA_VERSION:
            # Need migration
            logger.info(
                "Migrating database from v%d to v%d",
                current,
                CURRENT_SCHEMA_VERSION,
            )
            await self._migrate(current)
        else:
            logger.debug("Database schema is up to date (v%d)", current)

    async def _create_schema(self) -> None:
        """Create initial schema."""
        sql = get_schema_sql(CURRENT_SCHEMA_VERSION)
        await self.db.executescript(sql)

        # Record schema version
        await self.db.execute(
            "INSERT INTO schema_version (version, description) VALUES (?, ?)",
            (CURRENT_SCHEMA_VERSION, "Initial schema"),
        )
        await self.db.commit()
        logger.info("Schema v%d created successfully", CURRENT_SCHEMA_VERSION)

    async def _migrate(self, from_version: int) -> None:
        """Migrate database from older version.
        
        Args:
            from_version: Current database version
        """
        # Define migrations as version -> migration function
        migrations: dict[int, str] = {
            2: """
            ALTER TABLE documents ADD COLUMN submodule TEXT;
            ALTER TABLE sync_history ADD COLUMN submodule TEXT;
            ALTER TABLE local_manifest ADD COLUMN submodule TEXT;
            DROP INDEX IF EXISTS idx_documents_module_version;
            DROP INDEX IF EXISTS idx_sync_history_module_version;
            DROP INDEX IF EXISTS idx_local_manifest_module_version;
            CREATE INDEX IF NOT EXISTS idx_documents_module_version 
                ON documents(module, submodule, major_version, minor_version);
            CREATE INDEX IF NOT EXISTS idx_sync_history_module_version 
                ON sync_history(module, submodule, version);
            CREATE INDEX IF NOT EXISTS idx_local_manifest_module_version 
                ON local_manifest(module, submodule, version);
            """,
        }

        for version in range(from_version + 1, CURRENT_SCHEMA_VERSION + 1):
            if version in migrations:
                logger.info("Applying migration to v%d", version)
                await self.db.executescript(migrations[version])
                await self.db.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (version, f"Migration from v{version - 1}"),
                )

        await self.db.commit()
        logger.info("Migration to v%d complete", CURRENT_SCHEMA_VERSION)

    async def verify_schema(self) -> bool:
        """Verify all required tables exist.
        
        Returns:
            True if schema is valid
        """
        required_tables = [
            TableName.DOCUMENTS.value,
            TableName.SYNC_HISTORY.value,
            TableName.LOCAL_MANIFEST.value,
            TableName.SCHEMA_VERSION.value,
        ]

        cursor = await self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        rows = await cursor.fetchall()
        existing_tables = {row[0] for row in rows}

        for table in required_tables:
            if table not in existing_tables:
                logger.error("Missing required table: %s", table)
                return False

        return True


async def create_database(db_path: Path) -> aiosqlite.Connection:
    """Create and initialize a new database.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Initialized database connection
    """
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row

    # Enable foreign keys and WAL mode for better performance
    await db.execute("PRAGMA foreign_keys = ON")
    await db.execute("PRAGMA journal_mode = WAL")

    # Initialize schema
    manager = SchemaManager(db)
    await manager.initialize()

    return db


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "TableName",
    "get_schema_sql",
    "SchemaManager",
    "create_database",
]
