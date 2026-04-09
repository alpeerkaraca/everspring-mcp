"""EverSpring MCP - Embedding generation utilities."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

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
    normalized = tier.strip().lower()
    if normalized not in TIER_DEFAULT_MODELS:
        raise ValueError(f"Unsupported embedding tier '{tier}'")
    return TIER_DEFAULT_MODELS[normalized]


class _EmbeddingBackend(Protocol):
    tokenizer: Any | None
    max_seq_length: int | None
    raw_model: SentenceTransformer

    def embed(self, texts: list[str], batch_size: int) -> np.ndarray: ...


class _SentenceTransformerBackend:
    def __init__(self, model: SentenceTransformer) -> None:
        self.raw_model = model
        self.tokenizer: Any | None = getattr(model, "tokenizer", None)
        max_seq = getattr(model, "max_seq_length", None)
        self.max_seq_length: int | None = max_seq if isinstance(max_seq, int) else None

    def embed(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = self.raw_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors)


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding result for a chunk."""

    chunk_id: str
    vector: list[float]


class Embedder:
    """Embedding generator for markdown chunks with tiered BGE backends."""

    def __init__(self, model_name: str, batch_size: int = 128, tier: str = MAIN_TIER) -> None:
        start = time.perf_counter()
        normalized_tier = tier.strip().lower()
        if normalized_tier not in VALID_EMBEDDING_TIERS:
            raise ValueError(
                f"Unsupported embedding tier '{tier}'. Valid values: {sorted(VALID_EMBEDDING_TIERS)}"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.tier = normalized_tier
        self._model: _EmbeddingBackend | None = None
        logger.info(
            "Embedder initialized (tier=%s, model=%s) in %.3fs",
            self.tier,
            self._resolved_model_name(),
            time.perf_counter() - start,
        )

    @staticmethod
    def _resolve_device_and_dtype() -> tuple[str, Any]:
        import torch

        # 1. Apple Silicon (M1/M2/M3)
        if torch.backends.mps.is_available():
            logger.info("Apple Silicon (MPS) detected, using float32 (MPS best practice)")
            return "mps", torch.float32

        # 2. CUDA (NVIDIA & AMD ROCm)
        if torch.cuda.is_available():
            # check bfloat16
            if torch.cuda.is_bf16_supported():
                logger.info("CUDA bfloat16 support detected, using bfloat16")
                return "cuda", torch.bfloat16

            # if not standart fp16
            logger.info("CUDA detected but bfloat16 not supported, falling back to float16")
            return "cuda", torch.float16

        # 3. CPU
        logger.info("No GPU acceleration found, falling back to CPU (float32)")
        return "cpu", torch.float32

    def _resolved_model_name(self) -> str:
        if self.model_name == DEFAULT_MAIN_MODEL:
            return default_model_for_tier(self.tier)
        return self.model_name

    def _load_sentence_transformer_backend(self) -> _SentenceTransformerBackend:
        import_start = time.perf_counter()
        from sentence_transformers import SentenceTransformer

        logger.info(
            "sentence_transformers imported in %.2fs",
            time.perf_counter() - import_start,
        )

        device, dtype = self._resolve_device_and_dtype()
        model_name = self._resolved_model_name()
        start = time.perf_counter()
        logger.info("Loading embedding model (%s): %s", self.tier, model_name)
        model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"dtype": dtype},
            tokenizer_kwargs={"use_fast": True},
        )
        logger.info("Embedding model loaded in %.2fs", time.perf_counter() - start)
        return _SentenceTransformerBackend(model)

    def _get_model(self) -> _EmbeddingBackend:
        if self._model is None:
            self._model = self._load_sentence_transformer_backend()
        return self._model

    def ensure_model_loaded(self) -> _EmbeddingBackend:
        """Ensure embedding model is loaded and return backend wrapper."""
        return self._get_model()

    async def prefetch_model(self) -> None:
        """Preload embedding model so artifacts are cached before first query."""
        await asyncio.to_thread(self.ensure_model_loaded)

    @staticmethod
    def _resolve_max_seq_length(model: _EmbeddingBackend) -> int | None:
        max_seq = model.max_seq_length
        if isinstance(max_seq, int) and max_seq > 0:
            return max_seq

        tokenizer: Any = model.tokenizer
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            return tokenizer_limit

        return None

    def _prepare_texts(self, model: _EmbeddingBackend, texts: list[str]) -> list[str]:
        max_seq_length = self._resolve_max_seq_length(model)
        tokenizer: Any = model.tokenizer
        if max_seq_length is None or tokenizer is None:
            return texts

        prepared: list[str] = []
        for idx, text in enumerate(texts):
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                verbose=False,
            )
            token_count = len(token_ids)
            if token_count <= max_seq_length:
                prepared.append(text)
                continue

            truncated_tokens = token_ids[:max_seq_length]
            truncated = tokenizer.decode(
                truncated_tokens,
                skip_special_tokens=True,
            )
            if not truncated:
                truncated = tokenizer.decode(
                    truncated_tokens,
                    skip_special_tokens=False,
                )
            logger.debug(
                "Truncated embedding input at index %d from %d to %d tokens",
                idx,
                token_count,
                max_seq_length,
            )
            prepared.append(truncated)

        return prepared

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts asynchronously."""
        if not texts:
            return []

        model = self.ensure_model_loaded()
        prepared_texts = self._prepare_texts(model, texts)
        embeddings_np = await asyncio.to_thread(
            model.embed,
            prepared_texts,
            self.batch_size,
        )
        return [emb.tolist() for emb in embeddings_np]

    async def embed_batches(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = await self.embed_texts(batch)
            vectors.extend(batch_vectors)
        return vectors


__all__ = [
    "EmbeddingResult",
    "Embedder",
    "DEFAULT_MAIN_MODEL",
    "DEFAULT_SLIM_MODEL",
    "DEFAULT_XSLIM_MODEL",
    "MAIN_TIER",
    "SLIM_TIER",
    "XSLIM_TIER",
    "VALID_EMBEDDING_TIERS",
    "default_model_for_tier",
]
