"""EverSpring MCP - Vector indexer.

Reads unindexed documents from SQLite, chunks Markdown,
embeds them, and upserts into ChromaDB.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from everspring_mcp.models.content import ContentType
from everspring_mcp.models.metadata import SearchableDocument
from everspring_mcp.models.spring import SpringModule
from everspring_mcp.storage.repository import StorageManager
from everspring_mcp.utils.logging import get_logger
from everspring_mcp.vector.chroma_client import ChromaClient
from everspring_mcp.vector.chunking import MarkdownChunker
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.embeddings import Embedder

logger = get_logger("vector.indexer")
HNSW_FINALIZATION_MESSAGE = (
    "Finalizing HNSW index and committing to disk... Please do not interrupt."
)


@dataclass(frozen=True)
class IndexStats:
    """Statistics from indexing operation."""

    documents_indexed: int
    chunks_indexed: int


@dataclass(frozen=True)
class _PreparedChunk:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _PreparedDocument:
    document_id: str
    chunks: list[_PreparedChunk]


@dataclass(frozen=True)
class _DocumentTask:
    id: str
    file_path: str
    title: str
    module: str
    submodule: str | None
    major_version: int
    minor_version: int


@dataclass(frozen=True)
class _VectorPayload:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float]
    sparse_weights: dict[str, float] | None = None


_WORKER_CHUNKER_CACHE: dict[tuple[str, int, int], MarkdownChunker] = {}


def _is_metadata_path(path: Path) -> bool:
    """Check if path is a metadata JSON file, not markdown."""
    normalized = str(path).replace("\\", "/")
    if path.name == "metadata.json":
        return True
    return "/metadata/" in normalized and path.suffix == ".json"


def _metadata_path_for(markdown_path: Path) -> Path:
    if markdown_path.name == "document.md":
        return markdown_path.parent / "metadata.json"
    url_hash = markdown_path.stem
    return markdown_path.parent / "metadata" / f"{url_hash}.json"


def _resolve_markdown_path(docs_dir: Path, relative_path: Path) -> Path | None:
    if relative_path.is_absolute():
        return None
    if any(part == ".." for part in relative_path.parts):
        return None
    docs_root = docs_dir.resolve()
    markdown_path = (docs_root / relative_path).resolve()
    if not markdown_path.is_relative_to(docs_root):
        return None
    return markdown_path


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _worker_chunker(
    model_name: str, max_tokens: int, overlap_tokens: int
) -> MarkdownChunker:
    key = (model_name, max_tokens, overlap_tokens)
    chunker = _WORKER_CHUNKER_CACHE.get(key)
    if chunker is None:
        chunker = MarkdownChunker(
            model_name=model_name,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        _WORKER_CHUNKER_CACHE[key] = chunker
    return chunker


def _prepare_document_worker(
    task: _DocumentTask,
    docs_dir: str,
    model_name: str,
    max_tokens: int,
    overlap_tokens: int,
) -> _PreparedDocument | None:
    relative_path = Path(task.file_path)
    if _is_metadata_path(relative_path):
        return None

    markdown_path = _resolve_markdown_path(Path(docs_dir), relative_path)
    if markdown_path is None or not markdown_path.exists():
        return None

    metadata_path = _metadata_path_for(markdown_path)
    metadata = _load_metadata(metadata_path)
    source_url = metadata.get("url") if metadata else None
    if not source_url or not str(source_url).startswith("http"):
        return None

    content = markdown_path.read_text(encoding="utf-8")
    chunks = _worker_chunker(model_name, max_tokens, overlap_tokens).chunk(content)
    if not chunks:
        return None

    raw_tags = metadata.get("tags")
    tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
    title = str(metadata.get("title") or task.title)
    prepared_chunks: list[_PreparedChunk] = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{task.id}-{i}"
        searchable = SearchableDocument(
            id=chunk_id,
            document_id=task.id,
            chunk_index=i,
            content=chunk.content,
            content_hash=chunk.content_hash,
            module=SpringModule(task.module),
            submodule=task.submodule,
            version_major=task.major_version,
            version_minor=task.minor_version,
            content_type=ContentType.REFERENCE,
            title=title,
            url=str(source_url),
            section_path=chunk.section_path,
            has_code=chunk.has_code,
            has_deprecation=False,
            tags=tags,
        )
        prepared_chunks.append(
            _PreparedChunk(
                chunk_id=chunk_id,
                document_id=task.id,
                content=chunk.content,
                metadata=searchable.to_chroma_metadata(),
            ),
        )
    return _PreparedDocument(document_id=task.id, chunks=prepared_chunks)


class VectorIndexer:
    """Indexes markdown documents into ChromaDB."""

    SQLITE_MARK_BATCH_SIZE = 500

    def __init__(
        self,
        config: VectorConfig | None = None,
        storage: StorageManager | None = None,
    ) -> None:
        start = time.perf_counter()
        self.config = config or VectorConfig.from_env()
        self.chunker = MarkdownChunker(
            model_name=self.config.embedding_model,
            max_tokens=self.config.max_tokens,
            overlap_tokens=self.config.overlap_tokens,
        )
        self.embedder = Embedder(
            model_name=self.config.embedding_model,
            batch_size=self.config.batch_size,
            tier=self.config.embedding_tier,
        )
        self.chroma = ChromaClient(self.config)
        self._storage = storage
        self._owns_storage = storage is None
        logger.info(f"VectorIndexer initialized in {time.perf_counter() - start:.2f}s")

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
            logger.debug("No unindexed documents found")
            return IndexStats(documents_indexed=0, chunks_indexed=0)

        logger.info("Indexing %d documents", len(docs))
        chunk_workers = min(len(docs), max(1, self.config.chunk_workers))
        embed_batch_size = max(1, self.config.batch_size)
        upsert_batch_size = max(1, self.config.chroma_upsert_batch_size)
        chunk_max_tokens = int(self.config.max_tokens)
        chunk_overlap_tokens = int(self.config.overlap_tokens)
        prefetch_chunks_limit = max(
            embed_batch_size, self.config.prefetch_batches * embed_batch_size
        )

        total_chunks = 0
        indexed_docs = 0
        pending_chunks: list[_PreparedChunk] = []
        pending_vectors: list[_VectorPayload] = []
        doc_chunk_totals: dict[str, int] = {}
        doc_chunk_indexed: dict[str, int] = {}
        marked_docs: set[str] = set()

        progress_enabled = sys.stderr.isatty()
        log_context = (
            logging_redirect_tqdm(loggers=[logging.getLogger("everspring_mcp")])
            if progress_enabled
            else nullcontext()
        )
        heartbeat_emitted = False

        with log_context:
            with (
                tqdm(
                    total=len(docs),
                    desc="Indexing documents",
                    unit="doc",
                    dynamic_ncols=True,
                    disable=not progress_enabled,
                ) as docs_bar,
                tqdm(
                    total=0,
                    desc="Embedding chunks",
                    unit="chunk",
                    dynamic_ncols=True,
                    disable=not progress_enabled,
                ) as chunks_bar,
            ):
                prepared_queue: asyncio.Queue[_PreparedDocument | None] = asyncio.Queue(
                    maxsize=max(1, prefetch_chunks_limit // embed_batch_size),
                )
                loop = asyncio.get_running_loop()
                pool = concurrent.futures.ProcessPoolExecutor(max_workers=chunk_workers)

                async def _flush_pending_vectors() -> tuple[int, int]:
                    flushed_total_chunks = 0
                    flushed_total_docs = 0
                    while len(pending_vectors) >= upsert_batch_size:
                        vector_batch = pending_vectors[:upsert_batch_size]
                        del pending_vectors[:upsert_batch_size]
                        (
                            flushed_chunks,
                            flushed_docs,
                        ) = await self._flush_vector_payloads(
                            payloads=vector_batch,
                            doc_chunk_totals=doc_chunk_totals,
                            doc_chunk_indexed=doc_chunk_indexed,
                            marked_docs=marked_docs,
                        )
                        flushed_total_chunks += flushed_chunks
                        flushed_total_docs += flushed_docs
                    return flushed_total_chunks, flushed_total_docs

                async def _consume_prepared(
                    prepared: _PreparedDocument,
                ) -> tuple[int, int]:
                    nonlocal heartbeat_emitted
                    consumed_chunks = 0
                    consumed_docs = 0
                    doc_chunk_totals[prepared.document_id] = len(prepared.chunks)
                    doc_chunk_indexed.setdefault(prepared.document_id, 0)
                    pending_chunks.extend(prepared.chunks)
                    if progress_enabled:
                        chunks_bar.total = (chunks_bar.total or 0) + len(
                            prepared.chunks
                        )
                        chunks_bar.refresh()

                    while len(pending_chunks) >= embed_batch_size:
                        chunk_batch = pending_chunks[:embed_batch_size]
                        del pending_chunks[:embed_batch_size]
                        vectors = await self._embed_with_progress(
                            texts=[item.content for item in chunk_batch],
                            progress_bar=chunks_bar if progress_enabled else None,
                        )
                        pending_vectors.extend(
                            self._to_vector_payloads(chunk_batch, vectors)
                        )
                        flushed_chunks, flushed_docs = await _flush_pending_vectors()
                        consumed_chunks += flushed_chunks
                        consumed_docs += flushed_docs

                    if pending_vectors and not heartbeat_emitted:
                        self._emit_finalization_heartbeat()
                        heartbeat_emitted = True
                    return consumed_chunks, consumed_docs

                async def _producer() -> None:
                    docs_iter = iter(docs)
                    active_tasks: set[asyncio.Future[_PreparedDocument | None]] = set()
                    excluded_files = 0

                    def _schedule_next() -> bool:
                        nonlocal excluded_files
                        try:
                            while True:
                                next_doc = next(docs_iter)
                                if self.config.exclude_patterns:
                                    import fnmatch
                                    # Match against the original URL which retains the recognizable filename structure
                                    match_target = str(next_doc.url) if next_doc.url else str(next_doc.file_path)
                                    if any(fnmatch.fnmatch(match_target, pat) for pat in self.config.exclude_patterns):
                                        excluded_files += 1
                                        continue
                                break
                        except StopIteration:
                            return False
                        task_data = _DocumentTask(
                            id=str(next_doc.id),
                            file_path=str(next_doc.file_path),
                            title=str(next_doc.title),
                            module=str(next_doc.module),
                            submodule=next_doc.submodule,
                            major_version=int(next_doc.major_version),
                            minor_version=int(next_doc.minor_version),
                        )
                        task = loop.run_in_executor(
                            pool,
                            _prepare_document_worker,
                            task_data,
                            str(self.config.docs_dir),
                            self.config.embedding_model,
                            chunk_max_tokens,
                            chunk_overlap_tokens,
                        )
                        active_tasks.add(task)
                        return True

                    for _ in range(chunk_workers):
                        if not _schedule_next():
                            break

                    while active_tasks:
                        done, _ = await asyncio.wait(
                            active_tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in done:
                            active_tasks.remove(task)
                            prepared = task.result()
                            docs_bar.update(1)
                            if prepared is not None and prepared.chunks:
                                await prepared_queue.put(prepared)
                            _schedule_next()

                    await prepared_queue.put(None)

                producer_task = asyncio.create_task(_producer())
                try:
                    while True:
                        prepared = await prepared_queue.get()
                        if prepared is None:
                            break
                        flushed_chunks, flushed_docs = await _consume_prepared(prepared)
                        total_chunks += flushed_chunks
                        indexed_docs += flushed_docs

                    while pending_chunks:
                        chunk_batch = pending_chunks[:embed_batch_size]
                        del pending_chunks[:embed_batch_size]
                        vectors = await self._embed_with_progress(
                            texts=[item.content for item in chunk_batch],
                            progress_bar=chunks_bar if progress_enabled else None,
                        )
                        pending_vectors.extend(
                            self._to_vector_payloads(chunk_batch, vectors)
                        )
                        flushed_chunks, flushed_docs = await _flush_pending_vectors()
                        total_chunks += flushed_chunks
                        indexed_docs += flushed_docs
                finally:
                    await producer_task
                    pool.shutdown(wait=True, cancel_futures=True)

                if pending_vectors and not heartbeat_emitted:
                    self._emit_finalization_heartbeat()
                    heartbeat_emitted = True

                while pending_vectors:
                    vector_batch = pending_vectors[:upsert_batch_size]
                    del pending_vectors[:upsert_batch_size]
                    flushed_chunks, flushed_docs = await self._flush_vector_payloads(
                        payloads=vector_batch,
                        doc_chunk_totals=doc_chunk_totals,
                        doc_chunk_indexed=doc_chunk_indexed,
                        marked_docs=marked_docs,
                    )
                    total_chunks += flushed_chunks
                    indexed_docs += flushed_docs

        if indexed_docs > 0 and not heartbeat_emitted:
            self._emit_finalization_heartbeat()

        logger.info("Indexing complete: %d docs, %d chunks", indexed_docs, total_chunks)
        return IndexStats(documents_indexed=indexed_docs, chunks_indexed=total_chunks)

    def _prepare_document(self, doc: Any) -> _PreparedDocument | None:
        relative_path = Path(doc.file_path)
        if self._is_metadata_path(relative_path):
            return None

        markdown_path = _resolve_markdown_path(self.config.docs_dir, relative_path)
        if markdown_path is None or not markdown_path.exists():
            return None

        metadata_path = self._metadata_path_for(relative_path)
        metadata = self._load_metadata(metadata_path)
        source_url = metadata.get("url") if metadata else None
        if not source_url or not str(source_url).startswith("http"):
            return None

        content = markdown_path.read_text(encoding="utf-8")
        chunks = self.chunker.chunk(content)
        if not chunks:
            return None

        raw_tags = metadata.get("tags")
        tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
        title = str(metadata.get("title") or doc.title)

        prepared_chunks: list[_PreparedChunk] = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc.id}-{i}"
            searchable = SearchableDocument(
                id=chunk_id,
                document_id=doc.id,
                chunk_index=i,
                content=chunk.content,
                content_hash=chunk.content_hash,
                module=SpringModule(doc.module),
                submodule=doc.submodule,
                version_major=doc.major_version,
                version_minor=doc.minor_version,
                content_type=ContentType.REFERENCE,
                title=title,
                url=str(source_url),
                section_path=chunk.section_path,
                has_code=chunk.has_code,
                has_deprecation=False,
                tags=tags,
            )
            prepared_chunks.append(
                _PreparedChunk(
                    chunk_id=chunk_id,
                    document_id=doc.id,
                    content=chunk.content,
                    metadata=searchable.to_chroma_metadata(),
                ),
            )
        return _PreparedDocument(document_id=doc.id, chunks=prepared_chunks)

    @staticmethod
    def _to_vector_payloads(
        chunks: list[_PreparedChunk],
        vectors: list[dict[str, Any]],
    ) -> list[_VectorPayload]:
        if len(chunks) != len(vectors):
            raise RuntimeError("Chunk/vector count mismatch during embedding")

        payloads = []
        for i, chunk in enumerate(chunks):
            dense = vectors[i]["dense"]
            sparse = vectors[i]["sparse"]

            # Metadata Size Protection: Top 150 tokens
            top_sparse = None
            if sparse:
                top_sparse = dict(
                    sorted(sparse.items(), key=lambda x: x[1], reverse=True)[:150]
                )

            payloads.append(
                _VectorPayload(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    embedding=dense,
                    sparse_weights=top_sparse,
                )
            )
        return payloads

    async def _flush_vector_payloads(
        self,
        payloads: list[_VectorPayload],
        doc_chunk_totals: dict[str, int],
        doc_chunk_indexed: dict[str, int],
        marked_docs: set[str],
    ) -> tuple[int, int]:
        if not payloads:
            return 0, 0

        # Safe Metadata Storage: Serialize sparse weights
        metadatas = []
        for item in payloads:
            meta = item.metadata.copy()
            if item.sparse_weights:
                meta["sparse_weights"] = json.dumps(
                    item.sparse_weights, separators=(",", ":")
                )
            metadatas.append(meta)

        await asyncio.to_thread(
            self.chroma.upsert,
            ids=[item.chunk_id for item in payloads],
            embeddings=[item.embedding for item in payloads],
            documents=[item.content for item in payloads],
            metadatas=metadatas,
        )

        touched_doc_ids: set[str] = set()
        for item in payloads:
            touched_doc_ids.add(item.document_id)
            doc_chunk_indexed[item.document_id] = (
                doc_chunk_indexed.get(item.document_id, 0) + 1
            )

        completed_docs = [
            doc_id
            for doc_id in touched_doc_ids
            if doc_chunk_totals.get(doc_id, 0) > 0
            and doc_chunk_indexed.get(doc_id, 0) >= doc_chunk_totals[doc_id]
            and doc_id not in marked_docs
        ]
        if completed_docs:
            await self._mark_docs_indexed_batched(completed_docs)
            marked_docs.update(completed_docs)

        return len(payloads), len(completed_docs)

    async def _mark_docs_indexed_batched(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        if not self._storage:
            raise RuntimeError("VectorIndexer not initialized")

        for start in range(0, len(doc_ids), self.SQLITE_MARK_BATCH_SIZE):
            batch_ids = doc_ids[start : start + self.SQLITE_MARK_BATCH_SIZE]
            await self._storage.documents.mark_indexed(batch_ids)

    async def _embed_with_progress(
        self,
        texts: list[str],
        progress_bar: Any | None = None,
    ) -> list[dict[str, Any]]:
        logger.info("Embedding batch of %d texts", len(texts))
        vectors = await self.embedder.embed_texts(texts)
        if progress_bar is not None:
            progress_bar.update(len(texts))
        return vectors

    @staticmethod
    def _emit_finalization_heartbeat() -> None:
        logger.info(HNSW_FINALIZATION_MESSAGE)

    @staticmethod
    def _is_metadata_path(path: Path) -> bool:
        """Check if path is a metadata JSON file, not markdown."""
        normalized = str(path).replace("\\", "/")
        if path.name == "metadata.json":
            return True
        # Match legacy paths like 'spring-boot/4.0.5/metadata/xxx.json' or 'metadata/xxx.json'
        return "/metadata/" in normalized and path.suffix == ".json"

    def _metadata_path_for(self, markdown_path: Path) -> Path:
        """Compute metadata JSON path for a markdown file.

        Supports:
        - New layout: '{module}/{version}/{url_hash}/document.md' -> sibling 'metadata.json'
        - Legacy layout: '{module}/{version}/{url_hash}.md' -> 'metadata/{url_hash}.json'
        """
        if markdown_path.name == "document.md":
            return self.config.docs_dir / markdown_path.parent / "metadata.json"

        url_hash = markdown_path.stem
        # Legacy metadata sits in a metadata/ subfolder keyed by URL hash.
        return (
            self.config.docs_dir
            / markdown_path.parent
            / "metadata"
            / f"{url_hash}.json"
        )

    @staticmethod
    def _load_metadata(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


__all__ = ["VectorIndexer", "IndexStats"]
