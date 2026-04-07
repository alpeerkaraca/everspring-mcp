"""EverSpring MCP - Vector configuration.

Configuration for ChromaDB and embedding pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
        default="google/embeddinggemma-300m",
        min_length=1,
        description="Embedding model ID",
    )
    max_tokens: int = Field(
        default=512,
        ge=100,
        le=2048,
        description="Max tokens per chunk (model limit is 2048)",
    )
    overlap_tokens: int = Field(
        default=50,
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
        return Path(v) if isinstance(v, str) else v

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
        if value := os.environ.get(cls.ENV_DATA_DIR):
            kwargs["data_dir"] = value
        if value := os.environ.get(cls.ENV_CHUNK_WORKERS):
            kwargs["chunk_workers"] = value
        if value := os.environ.get(cls.ENV_CHROMA_UPSERT_BATCH_SIZE):
            kwargs["chroma_upsert_batch_size"] = value

        return cls(**kwargs)

    def ensure_directories(self) -> None:
        """Create required directories."""
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)


__all__ = ["VectorConfig"]
