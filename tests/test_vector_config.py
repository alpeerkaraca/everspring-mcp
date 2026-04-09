"""Tests for vector config tier defaults."""

from __future__ import annotations

from everspring_mcp.vector.config import VectorConfig


def test_vector_config_applies_tier_chunk_defaults() -> None:
    main = VectorConfig(embedding_tier="main")
    slim = VectorConfig(embedding_tier="slim")
    xslim = VectorConfig(embedding_tier="xslim")

    assert (main.max_tokens, main.overlap_tokens) == (2048, 200)
    assert (slim.max_tokens, slim.overlap_tokens) == (512, 50)
    assert (xslim.max_tokens, xslim.overlap_tokens) == (384, 40)


def test_vector_config_keeps_explicit_chunk_overrides() -> None:
    config = VectorConfig(embedding_tier="xslim", max_tokens=600, overlap_tokens=60)
    assert (config.max_tokens, config.overlap_tokens) == (600, 60)


def test_vector_config_prefetch_batches_default() -> None:
    config = VectorConfig()
    assert config.prefetch_batches == 3
