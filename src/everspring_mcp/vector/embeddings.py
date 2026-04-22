"""EverSpring MCP - Embedding generation utilities using Strategy and Factory patterns."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from everspring_mcp.utils.logging import get_logger

logger = get_logger("vector.embeddings")

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


MAIN_TIER = "main"
SLIM_TIER = "slim"
XSLIM_TIER = "xslim"
VALID_EMBEDDING_TIERS = {MAIN_TIER, SLIM_TIER, XSLIM_TIER}

DEFAULT_MAIN_MODEL = "BAAI/bge-m3"
DEFAULT_SLIM_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_XSLIM_MODEL = "BAAI/bge-small-en-v1.5"

TIER_DEFAULT_MODELS: dict[str, str] = {
    MAIN_TIER: DEFAULT_MAIN_MODEL,
    SLIM_TIER: DEFAULT_SLIM_MODEL,
    XSLIM_TIER: DEFAULT_XSLIM_MODEL,
}


def default_model_for_tier(tier: str) -> str:
    """Resolve the default model name for a given tier."""
    normalized = tier.strip().lower()
    if normalized not in TIER_DEFAULT_MODELS:
        raise ValueError(f"Unsupported embedding tier '{tier}'")
    return TIER_DEFAULT_MODELS[normalized]


Vector: TypeAlias = list[float]


class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @abstractmethod
    async def embed(self, texts: list[str], batch_size: int) -> list[Vector]:
        """Embed a list of texts."""
        pass

    @property
    @abstractmethod
    def tier_name(self) -> str:
        """The name of the tier this strategy represents."""
        pass

    def _resolve_device_and_dtype(self) -> tuple[str, Any]:
        import torch

        if torch.backends.mps.is_available():
            return "mps", torch.float32
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return "cuda", torch.bfloat16
            return "cuda", torch.float16
        return "cpu", torch.float32

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device, dtype = self._resolve_device_and_dtype()
            logger.info("Loading %s model: %s", self.tier_name, self.model_name)
            self._model = SentenceTransformer(
                self.model_name,
                device=device,
                model_kwargs={"dtype": dtype},
                processor_kwargs={"use_fast": True},
            )
        return self._model


class BGEM3Strategy(EmbeddingStrategy):
    """Concrete strategy for the BGE-M3 (Main) model."""

    @property
    def tier_name(self) -> str:
        return MAIN_TIER

    async def embed(self, texts: list[str], batch_size: int) -> list[Vector]:
        model = await asyncio.to_thread(self._ensure_loaded)
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in np.asarray(embeddings)]


class BGESlimStrategy(EmbeddingStrategy):
    """Concrete strategy for the BGE-Base (Slim) model."""

    @property
    def tier_name(self) -> str:
        return SLIM_TIER

    async def embed(self, texts: list[str], batch_size: int) -> list[Vector]:
        model = await asyncio.to_thread(self._ensure_loaded)
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in np.asarray(embeddings)]


class BGEXSlimStrategy(EmbeddingStrategy):
    """Concrete strategy for the BGE-Small (X-Slim) model."""

    @property
    def tier_name(self) -> str:
        return XSLIM_TIER

    async def embed(self, texts: list[str], batch_size: int) -> list[Vector]:
        model = await asyncio.to_thread(self._ensure_loaded)
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in np.asarray(embeddings)]


class StrategyFactory:
    """Factory for creating embedding strategies based on tier."""

    @staticmethod
    def create(tier: str, model_name: str | None = None) -> EmbeddingStrategy:
        normalized_tier = tier.strip().lower()
        resolved_model = model_name or TIER_DEFAULT_MODELS.get(normalized_tier)

        if not resolved_model:
            raise ValueError(f"Unsupported embedding tier: {tier}")

        if normalized_tier == MAIN_TIER:
            return BGEM3Strategy(resolved_model)
        elif normalized_tier == SLIM_TIER:
            return BGESlimStrategy(resolved_model)
        elif normalized_tier == XSLIM_TIER:
            return BGEXSlimStrategy(resolved_model)
        else:
            raise ValueError(f"Unknown tier: {tier}")


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding result for a chunk."""

    chunk_id: str
    vector: Vector


class Embedder:
    """Embedding generator delegating to specific strategies."""

    def __init__(
        self, model_name: str, batch_size: int = 128, tier: str = MAIN_TIER
    ) -> None:
        self.batch_size = batch_size
        self._strategy = StrategyFactory.create(tier, model_name)
        logger.info(
            "Embedder initialized with %s strategy (model=%s)",
            self._strategy.__class__.__name__,
            self._strategy.model_name,
        )

    async def embed_texts(self, texts: list[str]) -> list[Vector]:
        """Embed a list of texts asynchronously."""
        if not texts:
            return []
        return await self._strategy.embed(texts, self.batch_size)

    async def embed_batches(self, texts: list[str]) -> list[Vector]:
        """Embed texts in batches."""
        vectors: list[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = await self.embed_texts(batch)
            vectors.extend(batch_vectors)
        return vectors

    async def prefetch_model(self) -> None:
        """Preload the strategy's model."""
        await asyncio.to_thread(self._strategy._ensure_loaded)


__all__ = [
    "EmbeddingStrategy",
    "BGEM3Strategy",
    "BGESlimStrategy",
    "BGEXSlimStrategy",
    "StrategyFactory",
    "EmbeddingResult",
    "Embedder",
    "MAIN_TIER",
    "SLIM_TIER",
    "XSLIM_TIER",
    "VALID_EMBEDDING_TIERS",
    "default_model_for_tier",
]
