"""EverSpring MCP - Embedding generation utilities."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from everspring_mcp.utils.logging import get_logger

logger = get_logger("vector.embeddings")

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding result for a chunk."""

    chunk_id: str
    vector: list[float]


class Embedder:
    """Embedding generator for markdown chunks.

    Uses sentence-transformers for embedding generation.
    """

    def __init__(self, model_name: str, batch_size: int = 128) -> None:
        start = time.perf_counter()
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
        logger.info(
            f"Embedder initialized (model={model_name}) in {time.perf_counter() - start:.3f}s"
        )

    @staticmethod
    def _resolve_cuda_bfloat16() -> tuple[str, Any]:
        import torch

        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                logger.info(
                    "CUDA bfloat16 support detected, using bfloat16 for embeddings"
                )
                return "cuda", torch.bfloat16
            else:
                logger.warning(
                    "CUDA is available but bfloat16 support is not detected. Falling back to float32 for embeddings."
                )
                return "cuda", torch.float16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info(
                "Apple Silicon GPU support detected, using float32 for embeddings"
            )
            return "mps", torch.float32

        logger.warning(
            "CUDA is not available. Falling back to CPU with float32 for embeddings, which may be slow."
        )
        return "cpu", torch.float32

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            import_start = time.perf_counter()
            logger.info("Importing sentence_transformers...")
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"sentence_transformers imported in {time.perf_counter() - import_start:.2f}s"
            )

            start = time.perf_counter()
            device, torch_dtype = self._resolve_cuda_bfloat16()
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=device,
                model_kwargs={"torch_dtype": torch_dtype},
            )
            logger.info(f"Embedding model loaded in {time.perf_counter() - start:.2f}s")
        return self._model

    def ensure_model_loaded(self) -> SentenceTransformer:
        """Ensure embedding model is loaded and return it."""
        return self._get_model()

    async def prefetch_model(self) -> None:
        """Preload embedding model so artifacts are cached before first query."""
        await asyncio.to_thread(self.ensure_model_loaded)

    @staticmethod
    def _resolve_max_seq_length(model: SentenceTransformer) -> int | None:
        max_seq = getattr(model, "max_seq_length", None)
        if isinstance(max_seq, int) and max_seq > 0:
            return max_seq

        tokenizer: Any = getattr(model, "tokenizer", None)
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            return tokenizer_limit

        return None

    def _prepare_texts(self, model: SentenceTransformer, texts: list[str]) -> list[str]:
        max_seq_length = self._resolve_max_seq_length(model)
        tokenizer: Any = getattr(model, "tokenizer", None)
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
        model = self.ensure_model_loaded()
        prepared_texts = self._prepare_texts(model, texts)
        embeddings = await asyncio.to_thread(
            model.encode,
            prepared_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        # Convert numpy arrays to lists of floats
        return [emb.tolist() for emb in embeddings]

    async def embed_batches(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = await self.embed_texts(batch)
            vectors.extend(batch_vectors)
        return vectors


__all__ = ["EmbeddingResult", "Embedder"]
