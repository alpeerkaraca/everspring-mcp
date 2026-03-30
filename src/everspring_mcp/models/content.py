"""EverSpring MCP - Scraped content models.

This module provides models for scraped documentation content:
- ScrapedPage: Raw scraped page with hash verification
- DocumentSection: Hierarchical document structure
- CodeExample: Code snippets with metadata
- DeprecationInfo: Deprecation tracking
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Self

from pydantic import Field, computed_field, model_validator

from .base import (
    HashableContent,
    HttpUrl,
    MarkdownContent,
    SHA256Hash,
    TimestampedModel,
    VersionedModel,
    compute_hash,
)
from .spring import SpringModule, SpringVersion


class ContentType(str, Enum):
    """Type of documentation content."""
    
    GUIDE = "guide"
    REFERENCE = "reference"
    API_DOC = "api-doc"
    TUTORIAL = "tutorial"
    RELEASE_NOTES = "release-notes"
    MIGRATION_GUIDE = "migration-guide"


class CodeLanguage(str, Enum):
    """Supported code languages in examples."""
    
    JAVA = "java"
    KOTLIN = "kotlin"
    GROOVY = "groovy"
    XML = "xml"
    YAML = "yaml"
    PROPERTIES = "properties"
    JSON = "json"
    SHELL = "shell"
    SQL = "sql"


class DeprecationStatus(str, Enum):
    """Status of a deprecated feature."""
    
    DEPRECATED = "deprecated"
    FOR_REMOVAL = "for-removal"
    REMOVED = "removed"


class DeprecationInfo(VersionedModel):
    """Information about a deprecated API or feature.
    
    Used by the Deprecation Guard feature to warn LLMs
    about outdated patterns.
    """
    
    status: DeprecationStatus = Field(
        default=DeprecationStatus.DEPRECATED,
        description="Current deprecation status",
    )
    deprecated_since: SpringVersion = Field(
        description="Version when deprecation was introduced",
    )
    removal_version: SpringVersion | None = Field(
        default=None,
        description="Version when feature will be/was removed",
    )
    replacement: str | None = Field(
        default=None,
        description="Suggested replacement API or pattern",
    )
    reason: str | None = Field(
        default=None,
        description="Reason for deprecation",
    )
    migration_guide_url: HttpUrl | None = Field(
        default=None,
        description="URL to migration documentation",
    )
    
    @model_validator(mode="after")
    def validate_versions(self) -> Self:
        """Ensure removal version is after deprecation version."""
        if self.removal_version:
            if self.removal_version.module != self.deprecated_since.module:
                raise ValueError(
                    "removal_version must be for same module as deprecated_since"
                )
            if self.removal_version < self.deprecated_since:
                raise ValueError(
                    "removal_version cannot be before deprecated_since"
                )
        return self


class CodeExample(VersionedModel):
    """A code example from documentation."""
    
    language: CodeLanguage = Field(
        description="Programming language of the code",
    )
    code: str = Field(
        min_length=1,
        description="The code content",
    )
    title: str | None = Field(
        default=None,
        description="Optional title for the example",
    )
    description: str | None = Field(
        default=None,
        description="Optional description of what the code does",
    )
    annotations: list[str] = Field(
        default_factory=list,
        description="Spring annotations used in this example",
    )
    is_recommended: bool = Field(
        default=True,
        description="Whether this is a recommended pattern",
    )
    deprecation: DeprecationInfo | None = Field(
        default=None,
        description="Deprecation info if this is an outdated pattern",
    )
    
    @computed_field
    @property
    def code_hash(self) -> str:
        """SHA-256 hash of the code content."""
        return compute_hash(self.code)


class DocumentSection(VersionedModel):
    """A section within a documentation page.
    
    Supports hierarchical structure with nested subsections.
    """
    
    id: str = Field(
        pattern=r"^[a-z0-9\-]+$",
        description="Section anchor ID",
    )
    title: str = Field(
        min_length=1,
        description="Section heading",
    )
    level: int = Field(
        ge=1,
        le=6,
        description="Heading level (1-6)",
    )
    content: MarkdownContent = Field(
        description="Markdown content of this section",
    )
    code_examples: list[CodeExample] = Field(
        default_factory=list,
        description="Code examples in this section",
    )
    subsections: list[DocumentSection] = Field(
        default_factory=list,
        description="Nested subsections",
    )
    
    @computed_field
    @property
    def content_hash(self) -> str:
        """SHA-256 hash of section content."""
        return compute_hash(self.content)
    
    @property
    def all_code_examples(self) -> list[CodeExample]:
        """Recursively collect all code examples."""
        examples = list(self.code_examples)
        for subsection in self.subsections:
            examples.extend(subsection.all_code_examples)
        return examples


class ScrapedPage(TimestampedModel):
    """A scraped documentation page with integrity verification.
    
    Stores both raw HTML and converted Markdown, with SHA-256
    hash verification for data integrity.
    """
    
    url: HttpUrl = Field(
        description="Source URL of the page",
    )
    module: SpringModule = Field(
        description="Spring module this page belongs to",
    )
    version: SpringVersion = Field(
        description="Spring version this page documents",
    )
    content_type: ContentType = Field(
        default=ContentType.REFERENCE,
        description="Type of documentation",
    )
    title: str = Field(
        min_length=1,
        description="Page title",
    )
    raw_html: str = Field(
        description="Original HTML content",
    )
    markdown_content: MarkdownContent = Field(
        description="Converted Markdown content",
    )
    content_hash: SHA256Hash = Field(
        description="SHA-256 hash of markdown_content",
    )
    sections: list[DocumentSection] = Field(
        default_factory=list,
        description="Parsed document sections",
    )
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the page was scraped",
    )
    
    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        """Verify content hash matches markdown content."""
        computed = compute_hash(self.markdown_content)
        if self.content_hash != computed:
            raise ValueError(
                f"Content hash mismatch. Expected {computed}, got {self.content_hash}"
            )
        return self
    
    @model_validator(mode="after")
    def validate_version_module(self) -> Self:
        """Ensure version module matches page module."""
        if self.version.module != self.module:
            raise ValueError(
                f"Version module {self.version.module} does not match "
                f"page module {self.module}"
            )
        return self
    
    @computed_field
    @property
    def html_hash(self) -> str:
        """SHA-256 hash of raw HTML content."""
        return compute_hash(self.raw_html)
    
    @classmethod
    def create(
        cls,
        url: str,
        module: SpringModule,
        version: SpringVersion,
        title: str,
        raw_html: str,
        markdown_content: str,
        content_type: ContentType = ContentType.REFERENCE,
        sections: list[DocumentSection] | None = None,
    ) -> ScrapedPage:
        """Factory method that automatically computes content hash.
        
        Args:
            url: Source URL
            module: Spring module
            version: Spring version
            title: Page title
            raw_html: Original HTML
            markdown_content: Converted Markdown
            content_type: Documentation type
            sections: Optional parsed sections
            
        Returns:
            ScrapedPage with computed content_hash
        """
        return cls(
            url=url,
            module=module,
            version=version,
            title=title,
            raw_html=raw_html,
            markdown_content=markdown_content,
            content_hash=compute_hash(markdown_content),
            content_type=content_type,
            sections=sections or [],
        )


__all__ = [
    # Enums
    "ContentType",
    "CodeLanguage",
    "DeprecationStatus",
    # Models
    "DeprecationInfo",
    "CodeExample",
    "DocumentSection",
    "ScrapedPage",
]
