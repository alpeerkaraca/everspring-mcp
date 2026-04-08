"""EverSpring MCP - BM25 sparse index for keyword matching.

Provides BM25Okapi-based sparse retrieval to complement
dense vector search in the hybrid retrieval pipeline.
"""

from __future__ import annotations

import hashlib
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from everspring_mcp.utils.logging import get_logger
from everspring_mcp.vector.config import VectorConfig

logger = get_logger("vector.bm25")


@dataclass(frozen=True)
class BM25SearchResult:
    """Single BM25 search result."""
    
    doc_id: str
    score: float
    rank: int


class BM25Index:
    """BM25 sparse index for keyword-based retrieval.
    
    Builds and persists a BM25Okapi index from document corpus.
    Supports filtering by metadata before search.
    """
    
    def __init__(self, config: VectorConfig) -> None:
        start = time.perf_counter()
        self.config = config
        self._index: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        logger.info(f"BM25Index initialized in {time.perf_counter() - start:.3f}s")
    
    @property
    def index_path(self) -> Path:
        """Path to persisted BM25 index."""
        return self.config.data_dir / "bm25_index.pkl"

    @property
    def hash_path(self) -> Path:
        """Path to SHA-256 integrity file for the persisted BM25 index."""
        return self.index_path.with_suffix(".sha256")
    
    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self._index is not None
    
    def build(
        self,
        doc_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Build BM25 index from documents.
        
        Args:
            doc_ids: Document identifiers
            documents: Document text content
            metadatas: Document metadata dicts
        """
        start = time.perf_counter()
        logger.info(f"Building BM25 index from {len(documents)} documents")
        self._doc_ids = doc_ids
        self._documents = documents
        self._metadatas = metadatas
        
        # Tokenize documents for BM25
        tokenized = [self._tokenize(doc) for doc in documents]
        self._index = BM25Okapi(tokenized)
        logger.info(f"BM25 index built in {time.perf_counter() - start:.2f}s")
    
    def save(self) -> None:
        """Persist index to disk with SHA-256 integrity file."""
        if not self._index:
            raise RuntimeError("No index to save")

        data = {
            "doc_ids": self._doc_ids,
            "documents": self._documents,
            "metadatas": self._metadatas,
            "tokenized": [self._tokenize(doc) for doc in self._documents],
        }

        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

        # Write companion SHA-256 file for integrity verification on load.
        digest = hashlib.sha256(self.index_path.read_bytes()).hexdigest()
        self.hash_path.write_text(digest, encoding="utf-8")
        logger.info(f"BM25 index saved to {self.index_path}")
    
    def load(self) -> bool:
        """Load index from disk with optional SHA-256 integrity check.

        Returns:
            True if loaded successfully, False if no index file exists

        Raises:
            ValueError: If a companion hash file exists and the digest does not match,
                        indicating the index file may have been tampered with.
        """
        if not self.index_path.exists():
            logger.debug("No BM25 index file found")
            return False

        # Verify integrity when a companion hash file is present.
        if self.hash_path.exists():
            expected_digest = self.hash_path.read_text(encoding="utf-8").strip()
            actual_digest = hashlib.sha256(self.index_path.read_bytes()).hexdigest()
            if actual_digest != expected_digest:
                raise ValueError(
                    f"BM25 index integrity check failed for {self.index_path}. "
                    "The file may have been tampered with. Remove it and rebuild."
                )
        else:
            logger.warning(
                "BM25 index loaded without integrity verification (no .sha256 file found). "
                "Rebuild the index to generate the verification file."
            )

        start = time.perf_counter()
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self._doc_ids = data["doc_ids"]
        self._documents = data["documents"]
        self._metadatas = data["metadatas"]
        self._index = BM25Okapi(data["tokenized"])
        logger.info(f"BM25 index loaded ({len(self._doc_ids)} docs) in {time.perf_counter() - start:.2f}s")
        return True
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        module: str | None = None,
        version_major: int | None = None,
    ) -> list[BM25SearchResult]:
        """Search index with BM25 scoring.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            module: Optional module filter
            version_major: Optional major version filter
            
        Returns:
            List of BM25SearchResult sorted by score (descending)
        """
        if not self._index:
            if not self.load():
                return []
        
        assert self._index is not None
        
        # Get scores for all documents
        query_tokens = self._tokenize(query)
        scores = self._index.get_scores(query_tokens)
        
        # Create (doc_id, score, metadata) tuples
        scored = list(zip(self._doc_ids, scores, self._metadatas))
        
        # Apply filters
        if module:
            scored = [(d, s, m) for d, s, m in scored if m.get("module") == module]
        if version_major is not None:
            scored = [(d, s, m) for d, s, m in scored if m.get("version_major") == version_major]
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results with rank
        results = []
        for rank, (doc_id, score, _) in enumerate(scored[:top_k], start=1):
            results.append(BM25SearchResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank,
            ))
        
        return results
    
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25.
        
        Simple whitespace + punctuation tokenization with lowercasing.
        """
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens
    
    def get_document(self, doc_id: str) -> tuple[str, dict[str, Any]] | None:
        """Get document content and metadata by ID.
        
        Returns:
            Tuple of (content, metadata) or None if not found
        """
        try:
            idx = self._doc_ids.index(doc_id)
            return self._documents[idx], self._metadatas[idx]
        except ValueError:
            return None


__all__ = ["BM25Index", "BM25SearchResult"]
