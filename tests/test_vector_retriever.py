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


def test_build_bm25_index_fetches_chroma_in_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = VectorConfig(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")
    retriever = HybridRetriever(config=config)

    class FakeCollection:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def count(self) -> int:
            return 2501

        def get(
            self,
            include: list[str],
            limit: int,
            offset: int,
        ) -> dict[str, list[Any]]:
            assert include == ["documents", "metadatas"]
            self.calls.append((limit, offset))
            end = min(offset + limit, 2501)
            ids = [f"doc-{i}" for i in range(offset, end)]
            return {
                "ids": ids,
                "documents": [f"content-{i}" for i in range(offset, end)],
                "metadatas": [{"module": "spring-boot", "version_major": 4} for _ in ids],
            }

    fake_collection = FakeCollection()
    monkeypatch.setattr(retriever._chroma, "get_collection", lambda: fake_collection)

    captured: dict[str, int] = {}

    def fake_build(
        doc_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        captured["doc_ids"] = len(doc_ids)
        captured["documents"] = len(documents)
        captured["metadatas"] = len(metadatas)

    monkeypatch.setattr(retriever._bm25, "build", fake_build)
    monkeypatch.setattr(retriever._bm25, "save", lambda: captured.setdefault("saved", 1))

    retriever.build_bm25_index()

    assert fake_collection.calls == [
        (retriever.BM25_BUILD_BATCH_SIZE, 0),
        (retriever.BM25_BUILD_BATCH_SIZE, retriever.BM25_BUILD_BATCH_SIZE),
        (retriever.BM25_BUILD_BATCH_SIZE, retriever.BM25_BUILD_BATCH_SIZE * 2),
    ]
    assert captured == {
        "doc_ids": 2501,
        "documents": 2501,
        "metadatas": 2501,
        "saved": 1,
    }
