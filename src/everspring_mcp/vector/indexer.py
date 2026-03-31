"""EverSpring MCP - Vector indexer.

Reads unindexed documents from SQLite, chunks Markdown,
embeds them, and upserts into ChromaDB.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..storage.repository import StorageManager
from ..models.metadata import SearchableDocument
from ..models.content import ContentType
from ..models.spring import SpringModule
from ..vector.chunking import MarkdownChunker
from ..vector.embeddings import Embedder
from ..vector.chroma_client import ChromaClient
from ..vector.config import VectorConfig


@dataclass(frozen=True)
class IndexStats:
    """Statistics from indexing operation."""
    
    documents_indexed: int
    chunks_indexed: int


class VectorIndexer:
    """Indexes markdown documents into ChromaDB."""
    
    def __init__(
        self,
        config: VectorConfig | None = None,
        storage: StorageManager | None = None,
    ) -> None:
        self.config = config or VectorConfig.from_env()
        self.chunker = MarkdownChunker(
            max_tokens=self.config.max_tokens,
            overlap_tokens=self.config.overlap_tokens,
        )
        self.embedder = Embedder(
            model_name=self.config.embedding_model,
            batch_size=self.config.batch_size,
        )
        self.chroma = ChromaClient(self.config)
        self._storage = storage
        self._owns_storage = storage is None
    
    async def __aenter__(self) -> VectorIndexer:
        self.config.ensure_directories()
        if self._storage is None:
            self._storage = StorageManager(self.config.db_path)
            await self._storage.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._owns_storage and self._storage:
            await self._storage.close()
    
    async def index_unindexed(self, limit: int = 50) -> IndexStats:
        """Index unindexed documents.
        
        Args:
            limit: Maximum documents to process
        """
        if not self._storage:
            raise RuntimeError("VectorIndexer not initialized")
        
        docs = await self._storage.documents.get_unindexed(limit=limit)
        if not docs:
            return IndexStats(documents_indexed=0, chunks_indexed=0)
        
        total_chunks = 0
        indexed_docs = 0
        for doc in docs:
            relative_path = Path(doc.file_path)
            if self._is_metadata_path(relative_path):
                # Skip metadata JSON files
                continue

            markdown_path = self.config.docs_dir / relative_path
            metadata_path = self._metadata_path_for(relative_path)
            metadata = self._load_metadata(metadata_path)
            source_url = metadata.get("url") if metadata else None
            if not source_url or not str(source_url).startswith("http"):
                # Skip if we can't provide a valid source URL for SearchableDocument
                continue

            if not markdown_path.exists():
                continue
            content = markdown_path.read_text(encoding="utf-8")
            chunks = self.chunker.chunk(content)
            if not chunks:
                continue
            
            texts = [c.content for c in chunks]
            vectors = await self.embedder.embed_batches(texts)
            ids = [f"{doc.id}-{i}" for i in range(len(chunks))]
            
            metadatas = []
            for i, chunk in enumerate(chunks):
                tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
                searchable = SearchableDocument(
                    id=ids[i],
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                    module=SpringModule(doc.module),
                    version_major=doc.major_version,
                    version_minor=doc.minor_version,
                    content_type=ContentType.REFERENCE,
                    title=metadata.get("title") or doc.title,
                    url=source_url,
                    section_path=chunk.section_path,
                    has_code=chunk.has_code,
                    has_deprecation=False,
                    tags=tags,
                )
                metadatas.append(searchable.to_chroma_metadata())
            
            self.chroma.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
            await self._storage.documents.mark_indexed([doc.id])
            total_chunks += len(chunks)
            indexed_docs += 1
        
        return IndexStats(documents_indexed=indexed_docs, chunks_indexed=total_chunks)

    @staticmethod
    def _is_metadata_path(path: Path) -> bool:
        return str(path).replace("\\", "/").startswith("metadata/") and path.suffix == ".json"

    def _metadata_path_for(self, markdown_path: Path) -> Path:
        """Compute metadata JSON path for a markdown file."""
        url_hash = markdown_path.stem
        return self.config.docs_dir / "metadata" / f"{url_hash}.json"

    @staticmethod
    def _load_metadata(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


__all__ = ["VectorIndexer", "IndexStats"]
