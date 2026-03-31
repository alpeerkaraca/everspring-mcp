"""EverSpring MCP - Sync configuration.

Configuration models for S3 synchronization.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _default_data_dir() -> Path:
    """Get default data directory."""
    return Path.home() / ".everspring"


class SyncConfig(BaseModel):
    """Configuration for S3 sync operations.
    
    Attributes:
        s3_bucket: S3 bucket name
        s3_region: AWS region
        s3_prefix: Key prefix for all objects
        local_data_dir: Local data directory
        docs_subdir: Subdirectory for downloaded docs
        db_filename: SQLite database filename
        download_concurrency: Max concurrent downloads
        chunk_size: Download chunk size in bytes
    """
    
    model_config = ConfigDict(frozen=True)
    
    # Environment variable names
    ENV_BUCKET: ClassVar[str] = "EVERSPRING_S3_BUCKET"
    ENV_REGION: ClassVar[str] = "AWS_REGION"
    ENV_PREFIX: ClassVar[str] = "EVERSPRING_S3_PREFIX"
    ENV_DATA_DIR: ClassVar[str] = "EVERSPRING_DATA_DIR"
    
    # S3 settings
    s3_bucket: str = Field(
        default="everspring-mcp-kb",
        pattern=r"^[a-z0-9][a-z0-9\-\.]{1,61}[a-z0-9]$",
        description="S3 bucket name",
    )
    s3_region: str = Field(
        default="eu-central-1",
        description="AWS region",
    )
    s3_prefix: str = Field(
        default="docs",
        description="S3 key prefix",
    )
    
    # Local paths
    local_data_dir: Path = Field(
        default_factory=_default_data_dir,
        description="Local data directory",
    )
    docs_subdir: str = Field(
        default="docs",
        description="Subdirectory for downloaded documents",
    )
    db_filename: str = Field(
        default="metadata.db",
        description="SQLite database filename",
    )
    
    # Sync settings
    download_concurrency: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max concurrent downloads",
    )
    chunk_size: int = Field(
        default=8192,
        ge=1024,
        description="Download chunk size in bytes",
    )
    
    @field_validator("local_data_dir", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path) -> Path:
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v
    
    @property
    def db_path(self) -> Path:
        """Full path to SQLite database."""
        return self.local_data_dir / self.db_filename
    
    @property
    def docs_dir(self) -> Path:
        """Full path to local docs directory."""
        return self.local_data_dir / self.docs_subdir
    
    def get_local_path(self, s3_key: str) -> Path:
        """Get local file path for an S3 key.
        
        Args:
            s3_key: S3 object key (e.g., "docs/spring-boot/4.0.5/abc123.md")
            
        Returns:
            Local file path
        """
        # Remove s3_prefix from key if present
        relative_key = s3_key
        if s3_key.startswith(f"{self.s3_prefix}/"):
            relative_key = s3_key[len(self.s3_prefix) + 1:]
        
        return self.docs_dir / relative_key
    
    def get_s3_key(self, module: str, version: str, filename: str) -> str:
        """Build S3 key for a document.
        
        Args:
            module: Spring module (e.g., "spring-boot")
            version: Version string (e.g., "4.0.5")
            filename: Document filename
            
        Returns:
            Full S3 key
        """
        return f"{self.s3_prefix}/{module}/{version}/{filename}"
    
    def get_manifest_key(self, module: str, version: str) -> str:
        """Get S3 key for manifest file.
        
        Args:
            module: Spring module
            version: Version string
            
        Returns:
            Manifest S3 key
        """
        return f"{self.s3_prefix}/{module}/{version}/manifest.json"
    
    @classmethod
    def from_env(cls) -> SyncConfig:
        """Create configuration from environment variables.
        
        Returns:
            SyncConfig with values from environment
        """
        import os
        
        kwargs = {}
        
        if bucket := os.environ.get(cls.ENV_BUCKET):
            kwargs["s3_bucket"] = bucket
        if region := os.environ.get(cls.ENV_REGION):
            kwargs["s3_region"] = region
        if prefix := os.environ.get(cls.ENV_PREFIX):
            kwargs["s3_prefix"] = prefix
        if data_dir := os.environ.get(cls.ENV_DATA_DIR):
            kwargs["local_data_dir"] = Path(data_dir)
        
        return cls(**kwargs)
    
    def ensure_directories(self) -> None:
        """Create local directories if they don't exist."""
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)


__all__ = ["SyncConfig"]
