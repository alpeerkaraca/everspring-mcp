"""Tests for model-cache CLI command."""

from __future__ import annotations

import argparse

import pytest

from everspring_mcp import main as cli_main
from everspring_mcp.cli import index as index_cli
from everspring_mcp.cli import sync as sync_cli
from everspring_mcp.cli import mcp as mcp_cli
from everspring_mcp.cli import model_cache as cache_cli
from everspring_mcp.cli import utils as cli_utils
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.embeddings import DEFAULT_MAIN_MODEL


@pytest.mark.asyncio
async def test_run_model_cache_prefetches_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    async def fake_prefetch_model(self) -> None:  # type: ignore[no-untyped-def]
        calls["prefetched"] = True

    monkeypatch.setattr(
        cache_cli.Embedder,
        "prefetch_model",
        fake_prefetch_model,
    )

    args = argparse.Namespace(
        embed_model="custom/model",
        batch_size=8,
        json=True,
    )

    exit_code = await cache_cli._run_model_cache(args)

    assert exit_code == 0
    assert calls["prefetched"] is True


@pytest.mark.asyncio
async def test_run_model_cache_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeEmbedder:
        def __init__(self, model_name: str, batch_size: int, tier: str) -> None:
            captured["model_name"] = model_name
            captured["batch_size"] = batch_size
            captured["tier"] = tier

        async def prefetch_model(self) -> None:
            captured["prefetched"] = True

    monkeypatch.setattr(cache_cli, "Embedder", FakeEmbedder)
    monkeypatch.setattr(VectorConfig, "from_env", classmethod(lambda cls: cls()))

    args = argparse.Namespace(
        embed_model=None,
        batch_size=None,
        json=True,
    )

    exit_code = await cache_cli._run_model_cache(args)

    assert exit_code == 0
    assert captured["prefetched"] is True
    assert captured["model_name"] == DEFAULT_MAIN_MODEL
    assert captured["batch_size"] == 128
    assert captured["tier"] == "main"
