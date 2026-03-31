"""Tests for vector indexing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from everspring_mcp.vector.chunking import MarkdownChunker
from everspring_mcp.vector.indexer import VectorIndexer
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.embeddings import Embedder


@pytest.fixture
def sample_markdown() -> str:
    return """
# Title
Some intro text.

## Section One
Content in section one.

```java
System.out.println("Hello");
```

## Section Two
More content here.
""".strip()


def test_chunking_hybrid(sample_markdown: str) -> None:
    chunker = MarkdownChunker(max_tokens=50, overlap_tokens=5)
    chunks = chunker.chunk(sample_markdown)
    assert len(chunks) > 1
    assert any(c.has_code for c in chunks)
    assert all(c.content_hash for c in chunks)


@pytest.mark.asyncio
async def test_indexer_no_docs(tmp_path: Path) -> None:
    config = VectorConfig(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")
    async with VectorIndexer(config=config) as indexer:
        stats = await indexer.index_unindexed(limit=5)
    assert stats.documents_indexed == 0
    assert stats.chunks_indexed == 0


@pytest.mark.asyncio
async def test_embedder_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = Embedder(model_name="google/embedding-gemma-300m", batch_size=2)
    
    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 3 for _ in texts]
    
    monkeypatch.setattr(embedder, "embed_texts", fake_embed_texts)
    vectors = await embedder.embed_batches(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 3 for v in vectors)
