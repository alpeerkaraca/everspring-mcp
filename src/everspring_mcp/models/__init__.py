"""EverSpring MCP - Pydantic models for Spring documentation.

This package provides all data models for the EverSpring MCP system:
- Base models with schema versioning and SHA-256 hashing
- Spring module and version definitions
- Scraped content models
- Document metadata and search models
- S3 sync and Knowledge Pack models
"""

from .base import (
    HashableContent,
    HttpUrl,
    MarkdownContent,
    SHA256Hash,
    TimestampedModel,
    VersionedModel,
    compute_hash,
    validate_sha256_hash,
)
from .content import (
    CodeExample,
    CodeLanguage,
    ContentType,
    DeprecationInfo,
    DeprecationStatus,
    DocumentSection,
    ScrapedPage,
)
from .metadata import (
    DeprecatedAPI,
    DocumentIndex,
    DocumentMetadata,
    SearchableDocument,
)
from .spring import (
    SpringModule,
    SpringVersion,
    VersionRange,
)
from .sync import (
    ChangeType,
    FileChange,
    FileEntry,
    KnowledgePack,
    S3ObjectRef,
    SyncDelta,
    SyncManifest,
    SyncStatus,
)

__all__ = [
    # Base
    "VersionedModel",
    "TimestampedModel",
    "HashableContent",
    "SHA256Hash",
    "MarkdownContent",
    "HttpUrl",
    "compute_hash",
    "validate_sha256_hash",
    # Spring
    "SpringModule",
    "SpringVersion",
    "VersionRange",
    # Content
    "ContentType",
    "CodeLanguage",
    "DeprecationStatus",
    "DeprecationInfo",
    "CodeExample",
    "DocumentSection",
    "ScrapedPage",
    # Metadata
    "DocumentMetadata",
    "DeprecatedAPI",
    "DocumentIndex",
    "SearchableDocument",
    # Sync
    "SyncStatus",
    "ChangeType",
    "S3ObjectRef",
    "FileEntry",
    "SyncManifest",
    "KnowledgePack",
    "FileChange",
    "SyncDelta",
]
