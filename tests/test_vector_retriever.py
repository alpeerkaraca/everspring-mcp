"""Tests for hybrid retriever behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.retriever import HybridRetriever


def _mock_dense_results() -> list[dict[str, Any]]:
    return [
        {
            "id": "doc1-0",
            "content": "alpha content",
            "metadata": {
                "title": "Alpha",
                "url": "https://docs.spring.io/a",
                "module": "spring-boot",
                "submodule": "",
                "version_major": 4,
                "version_minor": 0,
                "section_path": "A",
                "has_code": False,
            },
            "distance": 0.2,
        },
        {
            "id": "doc2-0",
            "content": "beta content",
            "metadata": {
                "title": "Beta",
                "url": "https://docs.spring.io/a",
                "module": "spring-boot",
                "submodule": "",
                "version_major": 4,
                "version_minor": 0,
                "section_path": "B",
                "has_code": True,
            },
            "distance": 0.3,
        },
        {
            "id": "doc3-0",
            "content": "gamma content",
            "metadata": {
                "title": "Gamma",
                "url": "https://docs.spring.io/c",
                "module": "spring-boot",
                "submodule": "",
                "version_major": 4,
                "version_minor": 0,
                "section_path": "C",
                "has_code": False,
            },
            "distance": 0.4,
        },
    ]


@pytest.mark.asyncio
async def test_search_deduplicates_urls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = VectorConfig(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")
    retriever = HybridRetriever(config=config)

    async def fake_dense_search(
        query: str,
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        return _mock_dense_results()

    class SparseResult:
        def __init__(self, doc_id: str) -> None:
            self.doc_id = doc_id

    monkeypatch.setattr(retriever, "_dense_search", fake_dense_search)
    monkeypatch.setattr(
        retriever._bm25,
        "search",
        lambda *args, **kwargs: [SparseResult("doc2-0"), SparseResult("doc1-0"), SparseResult("doc3-0")],
    )

    results = await retriever.search("datasource", top_k=3, deduplicate_urls=True)

    urls = [r.url for r in results]
    assert len(urls) == len(set(urls))
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_no_dedup_returns_multiple_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = VectorConfig(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")
    retriever = HybridRetriever(config=config)

    async def fake_dense_search(
        query: str,
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        return _mock_dense_results()

    class SparseResult:
        def __init__(self, doc_id: str) -> None:
            self.doc_id = doc_id

    monkeypatch.setattr(retriever, "_dense_search", fake_dense_search)
    monkeypatch.setattr(
        retriever._bm25,
        "search",
        lambda *args, **kwargs: [SparseResult("doc2-0"), SparseResult("doc1-0"), SparseResult("doc3-0")],
    )

    results = await retriever.search("datasource", top_k=3, deduplicate_urls=False)

    assert len(results) == 3
    assert results[-1].id == "doc3-0"
    assert {results[0].id, results[1].id} == {"doc1-0", "doc2-0"}
