"""Tests for index CLI BM25 integration."""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import pytest

from everspring_mcp import main as cli_main
from everspring_mcp.vector.config import VectorConfig


def test_index_parser_accepts_build_bm25_flag() -> None:
    """index parser should expose --build-bm25 flag."""
    parser = cli_main._build_parser()
    args = parser.parse_args(["index", "--build-bm25"])
    assert args.build_bm25 is True


def test_index_parser_accepts_performance_flags() -> None:
    """index parser should expose throughput tuning flags."""
    parser = cli_main._build_parser()
    args = parser.parse_args(
        ["index", "--chunk-workers", "12", "--upsert-batch-size", "1024", "--batch-size", "256"],
    )
    assert args.chunk_workers == 12
    assert args.upsert_batch_size == 1024
    assert args.batch_size == 256


@pytest.mark.asyncio
async def test_run_index_builds_bm25_when_flag_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_run_index should build BM25 when --build-bm25 is enabled."""
    captured: dict[str, object] = {}

    class FakeIndexer:
        def __init__(self, config: VectorConfig) -> None:
            captured["indexer_config"] = config

        async def __aenter__(self) -> FakeIndexer:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        async def index_unindexed(self, limit: int = 50) -> object:
            captured["index_limit"] = limit
            return types.SimpleNamespace(documents_indexed=3, chunks_indexed=9)

    class FakeRetriever:
        def __init__(self, config: VectorConfig) -> None:
            captured["retriever_config"] = config

        def build_bm25_index(self) -> None:
            captured["bm25_built"] = True

    def fake_from_env(cls: type[VectorConfig]) -> VectorConfig:
        return cls(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")

    monkeypatch.setattr(cli_main, "VectorIndexer", FakeIndexer)
    monkeypatch.setattr(cli_main, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(VectorConfig, "from_env", classmethod(fake_from_env))

    args = argparse.Namespace(
        submodule=None,
        module=None,
        data_dir=None,
        db_filename=None,
        docs_subdir=None,
        chroma_dir=None,
        collection=None,
        embed_model=None,
        max_tokens=None,
        overlap_tokens=None,
        batch_size=None,
        chunk_workers=None,
        upsert_batch_size=None,
        reindex=False,
        version=None,
        limit=25,
        build_bm25=True,
        json=True,
    )

    exit_code = await cli_main._run_index(args)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured.get("index_limit") == 25
    assert captured.get("bm25_built") is True
    assert payload["documents_indexed"] == 3
    assert payload["chunks_indexed"] == 9
    assert payload["bm25_index_built"] is True
    assert payload["bm25_index_path"].endswith("bm25_index.pkl")
