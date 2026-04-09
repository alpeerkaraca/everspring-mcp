"""EverSpring MCP - Sync configuration.

Configuration models for S3 synchronization.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from everspring_mcp.vector.embeddings import (
    DEFAULT_MAIN_MODEL,
    MAIN_TIER,
    VALID_EMBEDDING_TIERS,
)


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
    ENV_MODEL_TIER: ClassVar[str] = "EVERSPRING_MODEL_TIER"
    ENV_MODEL_NAME: ClassVar[str] = "EVERSPRING_EMBED_MODEL"

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
        default="spring-docs",
        description="Root S3 key prefix",
    )
    raw_data_subprefix: str = Field(
        default="raw-data",
        description="Subprefix for raw markdown + metadata objects",
    )
    db_snapshots_subprefix: str = Field(
        default="db-snapshots",
        description="Subprefix for DB snapshot archives",
    )
    models_subprefix: str = Field(
        default="models",
        description="Subprefix for model artifacts",
    )
    model_tier: str = Field(
        default=MAIN_TIER,
        description="Model tier used for snapshot/model artifact paths",
    )
    model_name: str = Field(
        default=DEFAULT_MAIN_MODEL,
        min_length=1,
        description="Embedding model name used for snapshot namespace",
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
    chroma_subdir: str = Field(
        default="chroma",
        description="Subdirectory for local ChromaDB persistence",
    )
    packs_subdir: str = Field(
        default="packs",
        description="Subdirectory for downloaded knowledge pack archives",
    )
    db_filename: str = Field(
        default="metadata.db",
        description="SQLite database filename",
    )
    knowledge_pack_filename: str = Field(
        default="knowledge_pack.zip",
        description="Archive filename for bundled SQLite + ChromaDB data",
    )

    # Sync settings
    parallel_jobs: int = Field(
        default=5,
        ge=1,
        le=200,
        description="Parallel workers for manifest sync operations",
    )
    download_concurrency: int = Field(
        default=5,
        ge=1,
        le=200,
        description="Legacy alias for parallel_jobs (kept for backward compatibility)",
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

    @field_validator("model_tier")
    @classmethod
    def validate_model_tier(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in VALID_EMBEDDING_TIERS:
            raise ValueError(f"model_tier must be one of {sorted(VALID_EMBEDDING_TIERS)}")
        return normalized

    @model_validator(mode="before")
    @classmethod
    def normalize_parallel_settings(cls, data: Any) -> Any:
        """Keep parallel_jobs and download_concurrency in sync."""
        if not isinstance(data, dict):
            return data

        values = dict(data)
        has_parallel = values.get("parallel_jobs") is not None
        has_download = values.get("download_concurrency") is not None

        if has_parallel and not has_download:
            values["download_concurrency"] = values["parallel_jobs"]
        elif has_download and not has_parallel:
            values["parallel_jobs"] = values["download_concurrency"]
        elif has_parallel and has_download:
            values["download_concurrency"] = values["parallel_jobs"]

        return values

    @property
    def db_path(self) -> Path:
        """Full path to SQLite database."""
        return self.local_data_dir / self.db_filename

    @property
    def docs_dir(self) -> Path:
        """Full path to local docs directory."""
        return self.local_data_dir / self.docs_subdir

    @property
    def chroma_dir(self) -> Path:
        """Full path to local ChromaDB directory."""
        return self.local_data_dir / self.chroma_subdir

    @property
    def packs_dir(self) -> Path:
        """Full path to local archive cache directory."""
        return self.local_data_dir / self.packs_subdir

    def get_local_path(self, s3_key: str) -> Path:
        """Get local file path for an S3 key.

        Args:
            s3_key: S3 object key (e.g., "spring-docs/raw-data/spring-boot/4.0.5/abc123/document.md")

        Returns:
            Local file path
        """
        raw_prefix = f"{self.s3_prefix}/{self.raw_data_subprefix}/"
        relative_key = s3_key
        if s3_key.startswith(raw_prefix):
            relative_key = s3_key[len(raw_prefix):]
        elif s3_key.startswith(f"{self.s3_prefix}/"):
            relative_key = s3_key[len(self.s3_prefix) + 1:]

        return self.docs_dir / relative_key

    @staticmethod
    def _module_segment(module: str, submodule: str | None = None) -> str:
        """Build module segment with optional submodule suffix."""
        clean_module = module.strip()
        clean_submodule = (submodule or "").strip()
        if clean_submodule:
            return f"{clean_module}-{clean_submodule}"
        return clean_module

    @staticmethod
    def _format_snapshot_date(snapshot_date: date | datetime | None = None) -> str:
        """Format snapshot date as YYYY_MM_DD."""
        if snapshot_date is None:
            snapshot_date = datetime.utcnow().date()
        if isinstance(snapshot_date, datetime):
            snapshot_date = snapshot_date.date()
        return snapshot_date.strftime("%Y_%m_%d")

    def get_raw_data_prefix(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> str:
        """Get raw-data prefix for a module/submodule/version."""
        module_segment = self._module_segment(module, submodule=submodule)
        return f"{self.s3_prefix}/{self.raw_data_subprefix}/{module_segment}/{version}"

    @staticmethod
    def _model_slug(model_name: str) -> str:
        model_tail = model_name.strip().split("/")[-1]
        slug = re.sub(r"[^a-z0-9]+", "-", model_tail.lower()).strip("-")
        return slug or "model"

    def get_snapshot_namespace(
        self,
        model_name: str | None = None,
        tier: str | None = None,
    ) -> str:
        effective_model = (model_name or self.model_name).strip()
        effective_tier = (tier or self.model_tier).strip().lower()
        return f"{self._model_slug(effective_model)}-{effective_tier}"

    def get_db_snapshots_prefix(
        self,
        model_name: str | None = None,
        tier: str | None = None,
    ) -> str:
        """Get db-snapshots prefix."""
        namespace = self.get_snapshot_namespace(model_name=model_name, tier=tier)
        return f"{self.s3_prefix}/{self.db_snapshots_subprefix}/{namespace}"

    def get_s3_key(
        self,
        module: str,
        version: str,
        relative_path: str,
        submodule: str | None = None,
    ) -> str:
        """Build S3 key for a raw-data document or metadata file.

        Args:
            module: Spring module (e.g., "spring-boot")
            version: Version string (e.g., "4.0.5")
            relative_path: Path relative to module/version (supports subdirs)

        Returns:
            Full S3 key
        """
        clean_path = relative_path.lstrip("/")
        return f"{self.get_raw_data_prefix(module, version, submodule=submodule)}/{clean_path}"

    def get_manifest_key(self, module: str, version: str, submodule: str | None = None) -> str:
        """Get S3 key for manifest file.

        Args:
            module: Spring module
            version: Version string

        Returns:
            Manifest S3 key
        """
        return f"{self.get_raw_data_prefix(module, version, submodule=submodule)}/manifest.json"

    def get_chroma_snapshot_key(
        self,
        snapshot_date: date | datetime | None = None,
        model_name: str | None = None,
        tier: str | None = None,
    ) -> str:
        """Get S3 key for ChromaDB snapshot archive."""
        date_token = self._format_snapshot_date(snapshot_date)
        return (
            f"{self.get_db_snapshots_prefix(model_name=model_name, tier=tier)}"
            f"/chroma_db_{date_token}.zip"
        )

    def get_sqlite_snapshot_key(
        self,
        snapshot_date: date | datetime | None = None,
        model_name: str | None = None,
        tier: str | None = None,
    ) -> str:
        """Get S3 key for SQLite metadata snapshot archive."""
        date_token = self._format_snapshot_date(snapshot_date)
        return (
            f"{self.get_db_snapshots_prefix(model_name=model_name, tier=tier)}"
            f"/sqlite_metadata_{date_token}.zip"
        )

    def get_models_prefix(self, tier: str = "slim") -> str:
        """Get model artifacts prefix for a specific tier."""
        clean_tier = tier.strip().lower()
        return f"{self.s3_prefix}/{self.models_subprefix}/{clean_tier}"

    @staticmethod
    def get_local_model_dir(tier: str = "slim") -> Path:
        """Get local model artifact directory for a specific tier."""
        clean_tier = tier.strip().lower()
        return Path.home() / ".everspring" / "models" / clean_tier

    def get_knowledge_pack_key(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> str:
        """Get S3 key for a zipped knowledge pack."""
        if submodule:
            return (
                f"{self.s3_prefix}/{module}/{submodule}/{version}/"
                f"{self.knowledge_pack_filename}"
            )
        return f"{self.s3_prefix}/{module}/{version}/{self.knowledge_pack_filename}"

    def get_local_pack_path(
        self,
        module: str,
        version: str,
        submodule: str | None = None,
    ) -> Path:
        """Get local archive path for a module version pack."""
        base_name = f"{module}-{version}"
        if submodule:
            base_name = f"{module}-{submodule}-{version}"
        safe_name = base_name.replace("/", "-").replace("\\", "-")
        return self.packs_dir / f"{safe_name}-{self.knowledge_pack_filename}"

    def get_chroma_snapshot_local_path(
        self,
        snapshot_date: date | datetime | None = None,
    ) -> Path:
        """Get local file path for Chroma snapshot archive."""
        date_token = self._format_snapshot_date(snapshot_date)
        return self.packs_dir / f"chroma_db_{date_token}.zip"

    def get_sqlite_snapshot_local_path(
        self,
        snapshot_date: date | datetime | None = None,
    ) -> Path:
        """Get local file path for SQLite snapshot archive."""
        date_token = self._format_snapshot_date(snapshot_date)
        return self.packs_dir / f"sqlite_metadata_{date_token}.zip"

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
        if model_tier := os.environ.get(cls.ENV_MODEL_TIER):
            kwargs["model_tier"] = model_tier
        if model_name := os.environ.get(cls.ENV_MODEL_NAME):
            kwargs["model_name"] = model_name

        return cls(**kwargs)

    def ensure_directories(self) -> None:
        """Create local directories if they don't exist."""
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.packs_dir.mkdir(parents=True, exist_ok=True)


__all__ = ["SyncConfig"]
