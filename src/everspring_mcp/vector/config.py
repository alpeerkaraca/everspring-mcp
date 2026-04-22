"""EverSpring MCP - Vector configuration.

Configuration for ChromaDB and embedding pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from everspring_mcp.vector.embeddings import (
    DEFAULT_MAIN_MODEL,
    MAIN_TIER,
    VALID_EMBEDDING_TIERS,
)

TIER_CHUNKING_DEFAULTS: dict[str, dict[str, int]] = {
    "main": {"chunk_size": 2048, "overlap": 200},
    "slim": {"chunk_size": 512, "overlap": 50},
    "xslim": {"chunk_size": 384, "overlap": 40},
}


def chunk_defaults_for_tier(tier: str) -> tuple[int, int]:
    normalized = tier.strip().lower()
    if normalized not in TIER_CHUNKING_DEFAULTS:
        raise ValueError(f"Unsupported embedding tier '{tier}'")
    defaults = TIER_CHUNKING_DEFAULTS[normalized]
    return defaults["chunk_size"], defaults["overlap"]


def _default_chroma_dir() -> Path:
    """Default persistent ChromaDB directory."""
    return Path.home() / ".everspring" / "chroma"


def _default_data_dir() -> Path:
    """Default local data directory."""
    return Path.home() / ".everspring"


def _default_chunk_workers() -> int:
    """Default worker count for CPU-bound chunk preparation."""
    import os

    cpu_count = os.cpu_count() or 4
    return max(2, min(16, cpu_count - 1))


class VectorConfig(BaseModel):
    """Configuration for vectorization pipeline.

    Attributes:
        chroma_dir: Persistent ChromaDB directory
        collection_name: ChromaDB collection name
        embedding_model: Embedding model ID
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks
        batch_size: Embedding batch size
        chunk_workers: Parallel workers for chunk preparation
        chroma_upsert_batch_size: Batch size for ChromaDB upserts
        data_dir: Local data directory (for sqlite/db)
        db_filename: SQLite filename
        docs_subdir: Local docs subdirectory
    """

    model_config = ConfigDict(frozen=True)

    # Environment variable names
    ENV_CHROMA_DIR: ClassVar[str] = "EVERSPRING_CHROMA_DIR"
    ENV_COLLECTION: ClassVar[str] = "EVERSPRING_CHROMA_COLLECTION"
    ENV_EMBED_MODEL: ClassVar[str] = "EVERSPRING_EMBED_MODEL"
    ENV_DATA_DIR: ClassVar[str] = "EVERSPRING_DATA_DIR"
    ENV_CHUNK_WORKERS: ClassVar[str] = "EVERSPRING_INDEX_CHUNK_WORKERS"
    ENV_CHROMA_UPSERT_BATCH_SIZE: ClassVar[str] = "EVERSPRING_CHROMA_UPSERT_BATCH_SIZE"
    ENV_EMBED_TIER: ClassVar[str] = "EVERSPRING_EMBED_TIER"
    ENV_PREFETCH_BATCHES: ClassVar[str] = "EVERSPRING_INDEX_PREFETCH_BATCHES"

    chroma_dir: Path = Field(
        default_factory=_default_chroma_dir,
        description="ChromaDB persistent directory",
    )
    collection_name: str = Field(
        default="spring_docs",
        min_length=1,
        description="ChromaDB collection name",
    )
    embedding_model: str = Field(
        default=DEFAULT_MAIN_MODEL,
        min_length=1,
        description="Embedding model ID",
    )
    embedding_tier: str = Field(
        default=MAIN_TIER,
        description="Embedding model tier (main, slim, xslim)",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=100,
        le=2048,
        description="Max tokens per chunk (model limit is 2048)",
    )
    overlap_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Token overlap between chunks",
    )
    batch_size: int = Field(
        default=128,
        ge=1,
        description="Embedding batch size",
    )
    chunk_workers: int = Field(
        default_factory=_default_chunk_workers,
        ge=1,
        description="Parallel workers for document read + chunking",
    )
    chroma_upsert_batch_size: int = Field(
        default=512,
        ge=1,
        description="Number of vectors per Chroma upsert call",
    )
    prefetch_batches: int = Field(
        default=3,
        ge=1,
        description="Prepared embedding batches to prefetch ahead of GPU consumption",
    )
    data_dir: Path = Field(
        default_factory=_default_data_dir,
        description="Local data directory",
    )
    db_filename: str = Field(
        default="metadata.db",
        description="SQLite database filename",
    )
    docs_subdir: str = Field(
        default="docs",
        description="Local docs subdirectory",
    )

    @field_validator("chroma_dir", "data_dir", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path) -> Path:
        """Convert string to Path."""
        path_obj = Path(v) if isinstance(v, str) else v
        return path_obj.expanduser().resolve()
    @field_validator("embedding_tier")
    @classmethod
    def validate_embedding_tier(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in VALID_EMBEDDING_TIERS:
            raise ValueError(
                f"embedding_tier must be one of {sorted(VALID_EMBEDDING_TIERS)}"
            )
        return normalized

    @model_validator(mode="before")
    @classmethod
    def apply_tier_chunk_defaults(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        tier_value = data.get("embedding_tier", MAIN_TIER)
        normalized_tier = (
            tier_value.strip().lower() if isinstance(tier_value, str) else MAIN_TIER
        )
        if normalized_tier in TIER_CHUNKING_DEFAULTS:
            chunk_size, overlap_tokens = chunk_defaults_for_tier(normalized_tier)
            if data.get("max_tokens") is None:
                data["max_tokens"] = chunk_size
            if data.get("overlap_tokens") is None:
                data["overlap_tokens"] = overlap_tokens
        return data

    @property
    def db_path(self) -> Path:
        """Full path to SQLite database."""
        return self.data_dir / self.db_filename

    @property
    def docs_dir(self) -> Path:
        """Full path to local docs directory."""
        return self.data_dir / self.docs_subdir

    @classmethod
    def from_env(cls) -> VectorConfig:
        """Create config from environment variables."""
        import os

        kwargs = {}
        if value := os.environ.get(cls.ENV_CHROMA_DIR):
            kwargs["chroma_dir"] = value
        if value := os.environ.get(cls.ENV_COLLECTION):
            kwargs["collection_name"] = value
        if value := os.environ.get(cls.ENV_EMBED_MODEL):
            kwargs["embedding_model"] = value
        if value := os.environ.get(cls.ENV_EMBED_TIER):
            kwargs["embedding_tier"] = value
        if value := os.environ.get(cls.ENV_DATA_DIR):
            kwargs["data_dir"] = value
        if value := os.environ.get(cls.ENV_CHUNK_WORKERS):
            kwargs["chunk_workers"] = value
        if value := os.environ.get(cls.ENV_CHROMA_UPSERT_BATCH_SIZE):
            kwargs["chroma_upsert_batch_size"] = value
        if value := os.environ.get(cls.ENV_PREFETCH_BATCHES):
            kwargs["prefetch_batches"] = value

        return cls(**kwargs)

    def ensure_directories(self) -> None:
        """Create required directories."""
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)


__all__ = ["VectorConfig", "chunk_defaults_for_tier"]
