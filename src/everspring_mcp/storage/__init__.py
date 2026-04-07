"""EverSpring MCP - Storage module.

This module provides SQLite-based storage for document metadata:
- Schema definitions and migrations
- Document and sync history repositories
- Pydantic model ↔ SQLite row conversion
"""

from everspring_mcp.storage.repository import (
    DocumentRepository,
    LocalManifestRepository,
    StorageManager,
    SyncHistoryRepository,
)
from everspring_mcp.storage.schema import (
    CURRENT_SCHEMA_VERSION,
    SchemaManager,
    get_schema_sql,
)

__all__ = [
    # Schema
    "CURRENT_SCHEMA_VERSION",
    "SchemaManager",
    "get_schema_sql",
    # Repositories
    "DocumentRepository",
    "SyncHistoryRepository",
    "LocalManifestRepository",
    "StorageManager",
]
