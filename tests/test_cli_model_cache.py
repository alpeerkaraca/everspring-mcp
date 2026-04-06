"""Tests for model-cache CLI command."""

from __future__ import annotations

import argparse

import pytest

from everspring_mcp import main as cli_main
from everspring_mcp.vector.config import VectorConfig


@pytest.mark.asyncio
async def test_run_model_cache_prefetches_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    async def fake_prefetch_model(self) -> None:  # type: ignore[no-untyped-def]
        calls["prefetched"] = True

    monkeypatch.setattr(
        cli_main.Embedder,
        "prefetch_model",
        fake_prefetch_model,
    )

    args = argparse.Namespace(
        embed_model="custom/model",
        batch_size=8,
        json=True,
    )

    exit_code = await cli_main._run_model_cache(args)

    assert exit_code == 0
    assert calls["prefetched"] is True


@pytest.mark.asyncio
async def test_run_model_cache_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeEmbedder:
        def __init__(self, model_name: str, batch_size: int) -> None:
            captured["model_name"] = model_name
            captured["batch_size"] = batch_size

        async def prefetch_model(self) -> None:
            captured["prefetched"] = True

    monkeypatch.setattr(cli_main, "Embedder", FakeEmbedder)
    monkeypatch.setattr(VectorConfig, "from_env", classmethod(lambda cls: cls()))

    args = argparse.Namespace(
        embed_model=None,
        batch_size=None,
        json=True,
    )

    exit_code = await cli_main._run_model_cache(args)

    assert exit_code == 0
    assert captured["prefetched"] is True
    assert captured["model_name"] == "google/embeddinggemma-300m"
    assert captured["batch_size"] == 32

