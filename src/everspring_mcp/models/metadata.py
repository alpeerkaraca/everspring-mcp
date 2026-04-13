"""EverSpring MCP - Document metadata models.

This module provides models for document metadata and search:
- DocumentMetadata: Core metadata for a document
- DocumentIndex: Full document with metadata for indexing
- DeprecatedAPI: Tracking deprecated APIs
- SearchableDocument: Embedding-ready document representation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Self

from pydantic import Field, computed_field, model_validator

from everspring_mcp.models.base import (
    HttpUrl,
    MarkdownContent,
    SHA256Hash,
    TimestampedModel,
    VersionedModel,
    compute_hash,
)
from everspring_mcp.models.content import (
    CodeExample,
    ContentType,
    DeprecationInfo,
    DeprecationStatus,
    DocumentSection,
)
from everspring_mcp.models.spring import SpringModule, SpringVersion, VersionRange


class DocumentMetadata(TimestampedModel):
    """Core metadata for a Spring documentation page."""

    id: str = Field(
        pattern=r"^[a-z0-9\-_]+$",
        description="Unique document identifier",
    )
    title: str = Field(
        min_length=1,
        description="Document title",
    )
    url: HttpUrl = Field(
        description="Source URL",
    )
    module: SpringModule = Field(
        description="Spring module",
    )
    submodule: str | None = Field(
        default=None,
        description="Optional submodule key",
    )
    version: SpringVersion = Field(
        description="Spring version",
    )
    content_type: ContentType = Field(
        default=ContentType.REFERENCE,
        description="Type of documentation",
    )
    content_hash: SHA256Hash = Field(
        description="SHA-256 hash of document content",
    )
    last_scraped: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When document was last scraped",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorization tags",
    )

    @model_validator(mode="after")
    def validate_version_module(self) -> Self:
        """Ensure version module matches document module."""
        if self.version.module != self.module:
            raise ValueError(
                f"Version module {self.version.module} does not match "
                f"document module {self.module}"
            )
        return self


class DeprecatedAPI(TimestampedModel):
    """Tracking model for deprecated Spring APIs.
    
    Used by Deprecation Guard to maintain a registry
    of deprecated patterns and their replacements.
    """

    id: str = Field(
        pattern=r"^[a-z0-9\-_\.]+$",
        description="Unique API identifier (e.g., 'spring-boot.web-security-config')",
    )
    name: str = Field(
        min_length=1,
        description="API name (class, method, annotation, etc.)",
    )
    module: SpringModule = Field(
        description="Spring module containing this API",
    )
    api_type: str = Field(
        description="Type of API (class, method, annotation, property, etc.)",
    )
    deprecation: DeprecationInfo = Field(
        description="Deprecation details",
    )
    affected_versions: VersionRange = Field(
        description="Version range where this API is deprecated",
    )
    documentation_url: HttpUrl | None = Field(
        default=None,
        description="URL to relevant documentation",
    )
    code_pattern: str | None = Field(
        default=None,
        description="Code pattern to detect usage (regex)",
    )
    replacement_example: CodeExample | None = Field(
        default=None,
        description="Example showing the recommended replacement",
    )

    @model_validator(mode="after")
    def validate_module_consistency(self) -> Self:
        """Ensure all module references are consistent."""
        if self.deprecation.deprecated_since.module != self.module:
            raise ValueError("deprecation.deprecated_since module must match api module")
        if self.affected_versions.module != self.module:
            raise ValueError("affected_versions module must match api module")
        return self

    @property
    def is_removed(self) -> bool:
        """Check if this API has been removed."""
        return self.deprecation.status == DeprecationStatus.REMOVED


class DocumentIndex(TimestampedModel):
    """Full document representation for indexing and search.
    
    Combines metadata, sections, and computed fields
    for efficient retrieval and display.
    """

    metadata: DocumentMetadata = Field(
        description="Document metadata",
    )
    sections: list[DocumentSection] = Field(
        default_factory=list,
        description="Document sections",
    )
    full_content: MarkdownContent = Field(
        description="Complete document content in Markdown",
    )
    deprecated_apis: list[DeprecatedAPI] = Field(
        default_factory=list,
        description="Deprecated APIs mentioned in this document",
    )

    @computed_field
    @property
    def content_hash(self) -> str:
        """Compute content hash for integrity verification."""
        return compute_hash(self.full_content)

    @computed_field
    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.full_content.split())

    @computed_field
    @property
    def section_count(self) -> int:
        """Total number of sections (including nested)."""
        def count_sections(sections: list[DocumentSection]) -> int:
            total = len(sections)
            for section in sections:
                total += count_sections(section.subsections)
            return total
        return count_sections(self.sections)

    @computed_field
    @property
    def code_example_count(self) -> int:
        """Total number of code examples."""
        count = 0
        for section in self.sections:
            count += len(section.all_code_examples)
        return count

    @property
    def all_tags(self) -> list[str]:
        """Get all tags including module and content type."""
        tags = list(self.metadata.tags)
        tags.append(self.metadata.module.value)
        tags.append(self.metadata.content_type.value)
        return list(set(tags))

    def has_deprecations(self) -> bool:
        """Check if document mentions any deprecations."""
        return len(self.deprecated_apis) > 0


class SearchableDocument(VersionedModel):
    """Document representation optimized for vector search.
    
    Contains fields needed for embedding generation and
    metadata filtering in ChromaDB.
    """

    id: str = Field(
        description="Unique document/chunk identifier",
    )
    document_id: str = Field(
        description="Parent document identifier",
    )
    chunk_index: int = Field(
        ge=0,
        description="Index of this chunk within the document",
    )
    content: str = Field(
        min_length=1,
        description="Text content for embedding",
    )
    content_hash: SHA256Hash = Field(
        description="SHA-256 hash of content",
    )

    # Metadata for filtering (as required by AGENTS.md)
    module: SpringModule = Field(
        description="Spring module for filtering",
    )
    submodule: str | None = Field(
        default=None,
        description="Optional submodule for filtering",
    )
    version_major: int = Field(
        ge=1,
        description="Major version for filtering",
    )
    version_minor: int = Field(
        ge=0,
        description="Minor version for filtering",
    )
    content_type: ContentType = Field(
        description="Content type for filtering",
    )

    # Additional metadata
    title: str = Field(
        description="Document/section title",
    )
    url: HttpUrl = Field(
        description="Source URL",
    )
    section_path: str = Field(
        default="",
        description="Path to section (e.g., 'Getting Started > Configuration')",
    )
    has_code: bool = Field(
        default=False,
        description="Whether chunk contains code examples",
    )
    has_deprecation: bool = Field(
        default=False,
        description="Whether chunk mentions deprecated APIs",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Searchable tags",
    )

    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        """Verify content hash matches content."""
        computed = compute_hash(self.content)
        if self.content_hash != computed:
            raise ValueError(
                f"Content hash mismatch. Expected {computed}, got {self.content_hash}"
            )
        return self

    @classmethod
    def from_document_index(
        cls,
        doc: DocumentIndex,
        chunk_content: str,
        chunk_index: int,
        section_path: str = "",
        has_code: bool = False,
    ) -> SearchableDocument:
        """Create SearchableDocument from DocumentIndex.
        
        Args:
            doc: Source document
            chunk_content: Content for this chunk
            chunk_index: Index of chunk
            section_path: Path to section
            has_code: Whether chunk has code
            
        Returns:
            SearchableDocument ready for embedding
        """
        return cls(
            id=f"{doc.metadata.id}-{chunk_index}",
            document_id=doc.metadata.id,
            chunk_index=chunk_index,
            content=chunk_content,
            content_hash=compute_hash(chunk_content),
            module=doc.metadata.module,
            submodule=doc.metadata.submodule,
            version_major=doc.metadata.version.major,
            version_minor=doc.metadata.version.minor,
            content_type=doc.metadata.content_type,
            title=doc.metadata.title,
            url=doc.metadata.url,
            section_path=section_path,
            has_code=has_code,
            has_deprecation=doc.has_deprecations(),
            tags=doc.all_tags,
        )

    def to_chroma_metadata(self) -> dict:
        """Convert to ChromaDB metadata dict.
        
        ChromaDB requires primitive types in metadata.
        """
        return {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "module": self.module.value,
            "submodule": self.submodule or "",
            "version_major": self.version_major,
            "version_minor": self.version_minor,
            "content_type": self.content_type.value,
            "title": self.title,
            "url": self.url,
            "section_path": self.section_path,
            "has_code": self.has_code,
            "has_deprecation": self.has_deprecation,
            "tags": ",".join(self.tags),
        }


class SearchResult(VersionedModel):
    """Result from hybrid retrieval search.
    
    Combines content with metadata and ranking scores.
    """

    id: str = Field(
        description="Chunk identifier (doc_id-chunk_index)",
    )
    content: str = Field(
        description="Retrieved text content (markdown chunk)",
    )
    title: str = Field(
        description="Document title",
    )
    url: str = Field(
        description="Source documentation URL",
    )
    module: str = Field(
        description="Spring module name",
    )
    submodule: str | None = Field(
        default=None,
        description="Spring submodule (e.g., redis for spring-data)",
    )
    version_major: int = Field(
        description="Major version number",
    )
    version_minor: int = Field(
        description="Minor version number",
    )
    score: float = Field(
        ge=0.0,
        description="Combined retrieval score (higher = better)",
    )
    dense_rank: int | None = Field(
        default=None,
        description="Rank from dense (cosine) retrieval",
    )
    sparse_rank: int | None = Field(
        default=None,
        description="Rank from sparse (BM25) retrieval",
    )
    section_path: str = Field(
        default="",
        description="Section hierarchy path",
    )
    has_code: bool = Field(
        default=False,
        description="Whether chunk contains code examples",
    )


__all__ = [
    "DocumentMetadata",
    "DeprecatedAPI",
    "DocumentIndex",
    "SearchableDocument",
    "SearchResult",
]
