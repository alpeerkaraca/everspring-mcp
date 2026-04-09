"""Tests for index CLI BM25 integration."""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import pytest

from everspring_mcp import main as cli_main
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.embeddings import (
    DEFAULT_MAIN_MODEL,
    DEFAULT_SLIM_MODEL,
    DEFAULT_XSLIM_MODEL,
)


@pytest.fixture(autouse=True)
def disable_auto_snapshot_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_refresh(*, model_name: str, tier: str, data_dir: Path) -> None:
        del model_name, tier, data_dir

    monkeypatch.setattr(cli_main, "_auto_refresh_runtime_snapshots", _noop_refresh)


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


def test_index_parser_accepts_tier_flag() -> None:
    parser = cli_main._build_parser()
    args = parser.parse_args(["index", "--tier", "xslim"])
    assert args.tier == "xslim"


def test_search_parser_accepts_tier_flag() -> None:
    parser = cli_main._build_parser()
    args = parser.parse_args(["search", "--query", "bean lifecycle", "--tier", "slim"])
    assert args.tier == "slim"


def test_serve_parser_accepts_tier_flag() -> None:
    parser = cli_main._build_parser()
    args = parser.parse_args(["serve", "--tier", "main"])
    assert args.tier == "main"


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
        tier="slim",
        json=True,
    )

    exit_code = await cli_main._run_index(args)
    capsys.readouterr()

    assert exit_code == 0
    assert captured.get("index_limit") == 25
    assert captured.get("bm25_built") is True


@pytest.mark.asyncio
async def test_run_index_skips_bm25_for_main_tier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeIndexer:
        def __init__(self, config: VectorConfig) -> None:
            del config

        async def __aenter__(self) -> FakeIndexer:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        async def index_unindexed(self, limit: int = 50) -> object:
            del limit
            return types.SimpleNamespace(documents_indexed=1, chunks_indexed=2)

    class FakeRetriever:
        def __init__(self, config: VectorConfig) -> None:
            del config

        def build_bm25_index(self) -> None:
            captured["bm25_built"] = True

    monkeypatch.setattr(cli_main, "VectorIndexer", FakeIndexer)
    monkeypatch.setattr(cli_main, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        VectorConfig,
        "from_env",
        classmethod(lambda cls: cls(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")),
    )

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
        limit=1,
        build_bm25=True,
        tier="main",
        json=True,
    )
    exit_code = await cli_main._run_index(args)
    assert exit_code == 0
    assert "bm25_built" not in captured


@pytest.mark.asyncio
async def test_run_index_applies_bge_model_for_selected_tier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeIndexer:
        def __init__(self, config: VectorConfig) -> None:
            captured["tier"] = config.embedding_tier
            captured["model"] = config.embedding_model
            captured["max_tokens"] = config.max_tokens
            captured["overlap_tokens"] = config.overlap_tokens
            captured["chroma_dir"] = str(config.chroma_dir)

        async def __aenter__(self) -> FakeIndexer:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        async def index_unindexed(self, limit: int = 50) -> object:
            del limit
            return types.SimpleNamespace(documents_indexed=0, chunks_indexed=0)

    monkeypatch.setattr(cli_main, "VectorIndexer", FakeIndexer)
    monkeypatch.setattr(
        VectorConfig,
        "from_env",
        classmethod(lambda cls: cls(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")),
    )

    for selected_tier, model_name, expected_max_tokens, expected_overlap_tokens in (
        ("main", DEFAULT_MAIN_MODEL, 2048, 200),
        ("slim", DEFAULT_SLIM_MODEL, 512, 50),
        ("xslim", DEFAULT_XSLIM_MODEL, 384, 40),
    ):
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
            limit=1,
            build_bm25=False,
            tier=selected_tier,
            json=True,
        )
        exit_code = await cli_main._run_index(args)
        assert exit_code == 0
        assert captured["tier"] == selected_tier
        assert captured["model"] == model_name
        assert captured["max_tokens"] == expected_max_tokens
        assert captured["overlap_tokens"] == expected_overlap_tokens
        assert captured["chroma_dir"].endswith(
            f"/.everspring/chroma-{selected_tier}-{cli_main._model_slug(model_name)}"
        )


@pytest.mark.asyncio
async def test_run_index_coerces_path_overrides_to_path_objects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeIndexer:
        def __init__(self, config: VectorConfig) -> None:
            captured["data_dir_type"] = type(config.data_dir)
            captured["chroma_dir_type"] = type(config.chroma_dir)

        async def __aenter__(self) -> FakeIndexer:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        async def index_unindexed(self, limit: int = 50) -> object:
            del limit
            return types.SimpleNamespace(documents_indexed=0, chunks_indexed=0)

    class FakeRetriever:
        def __init__(self, config: VectorConfig) -> None:
            del config

        def build_bm25_index(self) -> None:
            return

    monkeypatch.setattr(cli_main, "VectorIndexer", FakeIndexer)
    monkeypatch.setattr(cli_main, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        VectorConfig,
        "from_env",
        classmethod(lambda cls: cls(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")),
    )

    args = argparse.Namespace(
        submodule=None,
        module=None,
        data_dir=str(tmp_path / "data-override"),
        db_filename=None,
        docs_subdir=None,
        chroma_dir=str(tmp_path / "chroma-override"),
        collection=None,
        embed_model=None,
        max_tokens=None,
        overlap_tokens=None,
        batch_size=None,
        chunk_workers=None,
        upsert_batch_size=None,
        reindex=False,
        version=None,
        limit=1,
        build_bm25=False,
        tier="main",
        json=True,
    )

    exit_code = await cli_main._run_index(args)
    assert exit_code == 0
    assert issubclass(captured["data_dir_type"], Path)
    assert issubclass(captured["chroma_dir_type"], Path)


@pytest.mark.asyncio
async def test_run_search_uses_tier_model_chroma_dir_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeRetriever:
        def __init__(self, config: VectorConfig) -> None:
            captured["tier"] = config.embedding_tier
            captured["model"] = config.embedding_model
            captured["chroma_dir"] = str(config.chroma_dir)

        def ensure_bm25_index(self) -> bool:
            return True

        def build_bm25_index(self) -> None:
            return

        async def search(
            self,
            *,
            query: str,
            top_k: int,
            module: str | None,
            version_major: int | None,
            deduplicate_urls: bool,
        ) -> list[object]:
            del query, top_k, module, version_major, deduplicate_urls
            return []

    monkeypatch.setattr(cli_main, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        VectorConfig,
        "from_env",
        classmethod(lambda cls: cls(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")),
    )

    args = argparse.Namespace(
        query="bean lifecycle",
        top_k=3,
        module=None,
        version=None,
        no_dedup=False,
        build_index=False,
        tier="slim",
        json=False,
    )

    exit_code = await cli_main._run_search(args)
    assert exit_code == 0
    assert captured["tier"] == "slim"
    assert captured["model"] == DEFAULT_SLIM_MODEL
    assert captured["chroma_dir"].endswith("/.everspring/chroma-slim-bge-base-en-v1-5")
