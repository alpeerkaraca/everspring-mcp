"""EverSpring MCP - Base Pydantic models and validators.

This module provides foundational models for the EverSpring MCP system:
- VersionedModel: Schema versioning for backward compatibility
- HashableContent: SHA-256 hash computation and verification
- TimestampedModel: Automatic timestamp tracking
- Custom field types and validators
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Annotated, ClassVar, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

# Custom field types
SHA256Hash = Annotated[
    str,
    Field(
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 hash as 64 lowercase hexadecimal characters",
    ),
]

MarkdownContent = Annotated[
    str,
    Field(min_length=1, description="Markdown-formatted content"),
]

HttpUrl = Annotated[
    str,
    Field(
        pattern=r"^https?://[^\s]+$",
        description="Valid HTTP or HTTPS URL",
    ),
]


class VersionedModel(BaseModel):
    """Base model with schema versioning for backward compatibility.
    
    All models inherit from this to ensure data can be migrated
    when schema changes occur in future versions.
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        validate_default=True,
        extra="forbid",
    )

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1
    schema_version: int = Field(
        default=1,
        ge=1,
        description="Schema version for backward compatibility",
    )

    @model_validator(mode="after")
    def check_schema_version(self) -> Self:
        """Validate schema version and warn if outdated."""
        if self.schema_version > self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version {self.schema_version} is newer than "
                f"supported version {self.CURRENT_SCHEMA_VERSION}. "
                "Please update EverSpring MCP."
            )
        return self


class TimestampedModel(VersionedModel):
    """Model with automatic timestamp tracking."""

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the record was last updated",
    )

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        """Ensure updated_at is not before created_at."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")
        return self


class HashableContent(VersionedModel):
    """Model with SHA-256 hash computation and verification.
    
    Used for content integrity validation as required by
    EverSpring's data integrity principles.
    """

    content: str = Field(
        min_length=1,
        description="The content to be hashed",
    )
    content_hash: SHA256Hash | None = Field(
        default=None,
        description="Pre-computed SHA-256 hash for verification",
    )

    @computed_field
    @property
    def computed_hash(self) -> str:
        """Compute SHA-256 hash of the content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """Verify that stored hash matches computed hash.
        
        Returns:
            True if content_hash matches computed_hash or if content_hash is None.
        """
        if self.content_hash is None:
            return True
        return self.content_hash == self.computed_hash

    def with_hash(self) -> Self:
        """Return a new instance with content_hash set to computed value.
        
        Since models are frozen, this creates a new instance.
        """
        return self.model_copy(update={"content_hash": self.computed_hash})


def validate_sha256_hash(value: str) -> str:
    """Validate that a string is a valid SHA-256 hash."""
    if not re.match(r"^[a-f0-9]{64}$", value):
        raise ValueError(
            "Invalid SHA-256 hash format. "
            "Expected 64 lowercase hexadecimal characters."
        )
    return value


def compute_hash(content: str | bytes) -> str:
    """Compute SHA-256 hash of content.
    
    Args:
        content: String or bytes to hash.
        
    Returns:
        64-character lowercase hexadecimal hash string.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


__all__ = [
    # Models
    "VersionedModel",
    "TimestampedModel",
    "HashableContent",
    # Field types
    "SHA256Hash",
    "MarkdownContent",
    "HttpUrl",
    # Utilities
    "validate_sha256_hash",
    "compute_hash",
]
