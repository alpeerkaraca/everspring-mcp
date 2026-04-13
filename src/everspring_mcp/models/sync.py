"""EverSpring MCP - S3 sync and Knowledge Pack models.

This module provides models for S3 synchronization:
- S3ObjectRef: Reference to S3 object
- SyncManifest: Manifest for version tracking
- KnowledgePack: Packaged documentation for distribution
- SyncDelta: Incremental sync changes
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Self

from pydantic import Field, computed_field, model_validator

from everspring_mcp.models.base import (
    SHA256Hash,
    TimestampedModel,
    VersionedModel,
)
from everspring_mcp.models.metadata import DocumentMetadata
from everspring_mcp.models.spring import SpringModule, SpringVersion


class SyncStatus(str, Enum):
    """Status of a sync operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ChangeType(str, Enum):
    """Type of change in sync delta."""

    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"


class S3ObjectRef(VersionedModel):
    """Reference to an S3 object."""

    bucket: str = Field(
        pattern=r"^[a-z0-9][a-z0-9\-\.]{1,61}[a-z0-9]$",
        description="S3 bucket name",
    )
    key: str = Field(
        min_length=1,
        description="S3 object key",
    )
    etag: str | None = Field(
        default=None,
        description="S3 ETag for version tracking",
    )
    last_modified: datetime | None = Field(
        default=None,
        description="Last modification timestamp",
    )
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Object size in bytes",
    )
    content_hash: SHA256Hash | None = Field(
        default=None,
        description="SHA-256 hash of object content",
    )

    @property
    def s3_uri(self) -> str:
        """Full S3 URI."""
        return f"s3://{self.bucket}/{self.key}"

    def __str__(self) -> str:
        return self.s3_uri


class FileEntry(VersionedModel):
    """Entry in a sync manifest file list."""

    path: str = Field(
        min_length=1,
        description="Relative file path within the pack",
    )
    content_hash: SHA256Hash = Field(
        description="SHA-256 hash of file content",
    )
    size_bytes: int = Field(
        ge=0,
        description="File size in bytes",
    )
    document_id: str | None = Field(
        default=None,
        description="Associated document ID if applicable",
    )


class SyncManifest(TimestampedModel):
    """Manifest for tracking Knowledge Pack versions.
    
    Used for incremental sync to minimize S3 egress costs.
    """

    version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$",
        description="Manifest version (semver format)",
    )
    pack_hash: SHA256Hash = Field(
        description="SHA-256 hash of entire pack content",
    )
    file_count: int = Field(
        ge=0,
        description="Number of files in the pack",
    )
    total_size_bytes: int = Field(
        ge=0,
        description="Total size of all files in bytes",
    )
    files: list[FileEntry] = Field(
        default_factory=list,
        description="List of files with hashes",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When manifest was generated",
    )

    @model_validator(mode="after")
    def validate_file_count(self) -> Self:
        """Ensure file_count matches actual files."""
        if self.file_count != len(self.files):
            raise ValueError(
                f"file_count ({self.file_count}) does not match "
                f"actual file count ({len(self.files)})"
            )
        return self

    @computed_field
    @property
    def computed_total_size(self) -> int:
        """Compute total size from file entries."""
        return sum(f.size_bytes for f in self.files)

    def get_file_hashes(self) -> dict[str, str]:
        """Get mapping of file paths to content hashes."""
        return {f.path: f.content_hash for f in self.files}


class KnowledgePack(TimestampedModel):
    """Packaged documentation for S3 distribution.
    
    A Knowledge Pack contains all documentation for a specific
    Spring module version, ready for distribution via S3.
    """

    id: str = Field(
        pattern=r"^[a-z0-9\-]+$",
        description="Unique pack identifier",
    )
    module: SpringModule = Field(
        description="Spring module this pack covers",
    )
    submodule: str | None = Field(
        default=None,
        description="Optional submodule this pack covers",
    )
    version: SpringVersion = Field(
        description="Spring version this pack documents",
    )
    manifest: SyncManifest = Field(
        description="Pack manifest with file hashes",
    )
    documents: list[DocumentMetadata] = Field(
        default_factory=list,
        description="Metadata for documents in this pack",
    )
    s3_location: S3ObjectRef | None = Field(
        default=None,
        description="S3 location if uploaded",
    )

    @model_validator(mode="after")
    def validate_module_consistency(self) -> Self:
        """Ensure all documents belong to the same module."""
        for doc in self.documents:
            if doc.module != self.module:
                raise ValueError(
                    f"Document {doc.id} module {doc.module} does not match "
                    f"pack module {self.module}"
                )
            if doc.submodule != self.submodule:
                raise ValueError(
                    f"Document {doc.id} submodule {doc.submodule} does not match "
                    f"pack submodule {self.submodule}"
                )
        return self

    @computed_field
    @property
    def document_count(self) -> int:
        """Number of documents in this pack."""
        return len(self.documents)

    @property
    def pack_name(self) -> str:
        """Generate pack name from module and version."""
        if self.submodule:
            return f"{self.module.value}-{self.submodule}-{self.version.version_string}"
        return f"{self.module.value}-{self.version.version_string}"

    def __str__(self) -> str:
        return f"KnowledgePack({self.pack_name})"


class FileChange(VersionedModel):
    """A single file change in a sync delta."""

    path: str = Field(
        min_length=1,
        description="File path",
    )
    change_type: ChangeType = Field(
        description="Type of change",
    )
    old_hash: SHA256Hash | None = Field(
        default=None,
        description="Previous content hash (for modified/removed)",
    )
    new_hash: SHA256Hash | None = Field(
        default=None,
        description="New content hash (for added/modified)",
    )
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="New file size (for added/modified)",
    )

    @model_validator(mode="after")
    def validate_hashes_for_change_type(self) -> Self:
        """Ensure appropriate hashes are set for change type."""
        if self.change_type == ChangeType.ADDED:
            if self.new_hash is None:
                raise ValueError("new_hash required for ADDED change")
        elif self.change_type == ChangeType.MODIFIED:
            if self.old_hash is None or self.new_hash is None:
                raise ValueError("Both old_hash and new_hash required for MODIFIED change")
        elif self.change_type == ChangeType.REMOVED:
            if self.old_hash is None:
                raise ValueError("old_hash required for REMOVED change")
        return self


class SyncDelta(TimestampedModel):
    """Incremental sync changes between two manifest versions.
    
    Used to minimize S3 egress by only transferring changed content.
    """

    from_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$",
        description="Source manifest version",
    )
    to_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$",
        description="Target manifest version",
    )
    module: SpringModule = Field(
        description="Spring module for this delta",
    )
    changes: list[FileChange] = Field(
        default_factory=list,
        description="List of file changes",
    )
    status: SyncStatus = Field(
        default=SyncStatus.PENDING,
        description="Current sync status",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if sync failed",
    )

    @computed_field
    @property
    def added_count(self) -> int:
        """Number of added files."""
        return sum(1 for c in self.changes if c.change_type == ChangeType.ADDED)

    @computed_field
    @property
    def modified_count(self) -> int:
        """Number of modified files."""
        return sum(1 for c in self.changes if c.change_type == ChangeType.MODIFIED)

    @computed_field
    @property
    def removed_count(self) -> int:
        """Number of removed files."""
        return sum(1 for c in self.changes if c.change_type == ChangeType.REMOVED)

    @computed_field
    @property
    def total_changes(self) -> int:
        """Total number of changes."""
        return len(self.changes)

    @computed_field
    @property
    def bytes_to_download(self) -> int:
        """Estimated bytes to download for added/modified files."""
        return sum(
            c.size_bytes or 0
            for c in self.changes
            if c.change_type in (ChangeType.ADDED, ChangeType.MODIFIED)
        )

    def has_changes(self) -> bool:
        """Check if there are any changes to sync."""
        return len(self.changes) > 0

    @classmethod
    def compute(
        cls,
        old_manifest: SyncManifest,
        new_manifest: SyncManifest,
        module: SpringModule,
    ) -> SyncDelta:
        """Compute delta between two manifests.
        
        Args:
            old_manifest: Previous manifest
            new_manifest: New manifest
            module: Spring module
            
        Returns:
            SyncDelta with computed changes
        """
        old_files = old_manifest.get_file_hashes()
        new_files = new_manifest.get_file_hashes()

        changes: list[FileChange] = []

        # Find added and modified files
        for path, new_hash in new_files.items():
            new_entry = next(f for f in new_manifest.files if f.path == path)
            if path not in old_files:
                changes.append(FileChange(
                    path=path,
                    change_type=ChangeType.ADDED,
                    new_hash=new_hash,
                    size_bytes=new_entry.size_bytes,
                ))
            elif old_files[path] != new_hash:
                changes.append(FileChange(
                    path=path,
                    change_type=ChangeType.MODIFIED,
                    old_hash=old_files[path],
                    new_hash=new_hash,
                    size_bytes=new_entry.size_bytes,
                ))

        # Find removed files
        for path, old_hash in old_files.items():
            if path not in new_files:
                changes.append(FileChange(
                    path=path,
                    change_type=ChangeType.REMOVED,
                    old_hash=old_hash,
                ))

        return cls(
            from_version=old_manifest.version,
            to_version=new_manifest.version,
            module=module,
            changes=changes,
        )


__all__ = [
    # Enums
    "SyncStatus",
    "ChangeType",
    # Models
    "S3ObjectRef",
    "FileEntry",
    "SyncManifest",
    "KnowledgePack",
    "FileChange",
    "SyncDelta",
]
