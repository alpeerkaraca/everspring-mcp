"""Tests for vector indexing pipeline."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from everspring_mcp.vector.chunking import MarkdownChunker
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.embeddings import Embedder
from everspring_mcp.vector.indexer import HNSW_FINALIZATION_MESSAGE, VectorIndexer


@pytest.fixture
def sample_markdown() -> str:
    return """
# Title
Some intro text.

## Section One
Content in section one.

```java
System.out.println("Hello");
```

## Section Two
More content here.
""".strip()


def test_chunking_hybrid(sample_markdown: str) -> None:
    chunker = MarkdownChunker(
        model_name="google/embeddinggemma-300m",
        max_tokens=50,
        overlap_tokens=5,
    )
    chunks = chunker.chunk(sample_markdown)
    assert len(chunks) > 1
    assert any(c.has_code for c in chunks)
    assert all(c.content_hash for c in chunks)


def test_chunking_removes_copied_artifacts() -> None:
    chunker = MarkdownChunker(
        model_name="google/embeddinggemma-300m",
        max_tokens=120,
        overlap_tokens=10,
    )
    markdown = """
## Configure
Use this snippet:
```java
class Demo {}
```
Copied!
""".strip()

    chunks = chunker.chunk(markdown)
    assert chunks
    assert all("Copied!" not in c.content for c in chunks)


@pytest.mark.asyncio
async def test_indexer_no_docs(tmp_path: Path) -> None:
    config = VectorConfig(data_dir=tmp_path, chroma_dir=tmp_path / "chroma")
    async with VectorIndexer(config=config) as indexer:
        stats = await indexer.index_unindexed(limit=5)
    assert stats.documents_indexed == 0
    assert stats.chunks_indexed == 0


@pytest.mark.asyncio
async def test_embedder_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = Embedder(model_name="google/embedding-gemma-300m", batch_size=2)

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 3 for _ in texts]

    monkeypatch.setattr(embedder, "embed_texts", fake_embed_texts)
    vectors = await embedder.embed_batches(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 3 for v in vectors)


@pytest.mark.asyncio
async def test_prefetch_model_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    load_calls = 0

    class FakeSentenceTransformer:
        def __init__(
            self,
            model_name: str,
            device: str | None = None,
            model_kwargs: dict[str, object] | None = None,
        ) -> None:
            del model_name
            assert device == "cuda"
            assert model_kwargs is not None
            nonlocal load_calls
            load_calls += 1

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    fake_torch = types.SimpleNamespace(
        cuda=FakeCuda(),
        bfloat16="bf16",
    )
    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embedder = Embedder(model_name="fake/model")
    await embedder.prefetch_model()
    await embedder.prefetch_model()

    assert load_calls == 1


def test_embedder_resolve_cuda_bfloat16_fails_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_bf16_supported() -> bool:
            return False

    fake_torch = types.SimpleNamespace(
        cuda=FakeCuda(),
        bfloat16="bf16",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        Embedder._resolve_cuda_bfloat16()


@pytest.mark.asyncio
async def test_flush_vector_payloads_batches_and_marks_document_once() -> None:
    class FakeChroma:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def upsert(
            self,
            ids: list[str],
            embeddings: list[list[float]],
            documents: list[str],
            metadatas: list[dict[str, object]],
        ) -> None:
            self.calls.append(
                {
                    "ids": ids,
                    "embeddings": embeddings,
                    "documents": documents,
                    "metadatas": metadatas,
                },
            )

    class FakeDocumentRepo:
        def __init__(self) -> None:
            self.mark_calls: list[list[str]] = []

        async def mark_indexed(self, doc_ids: list[str]) -> int:
            self.mark_calls.append(doc_ids)
            return len(doc_ids)

    fake_docs_repo = FakeDocumentRepo()
    fake_storage = types.SimpleNamespace(documents=fake_docs_repo)
    fake_chroma = FakeChroma()

    indexer = object.__new__(VectorIndexer)
    indexer.chroma = fake_chroma
    indexer._storage = fake_storage

    payloads = [
        types.SimpleNamespace(
            chunk_id="doc1-0",
            document_id="doc1",
            content="c1",
            metadata={"module": "spring-boot"},
            embedding=[0.1, 0.2],
        ),
        types.SimpleNamespace(
            chunk_id="doc1-1",
            document_id="doc1",
            content="c2",
            metadata={"module": "spring-boot"},
            embedding=[0.3, 0.4],
        ),
    ]

    flushed_chunks, flushed_docs = await VectorIndexer._flush_vector_payloads(
        indexer,
        payloads=payloads,
        doc_chunk_totals={"doc1": 2},
        doc_chunk_indexed={"doc1": 0},
        marked_docs=set(),
    )

    assert flushed_chunks == 2
    assert flushed_docs == 1
    assert len(fake_chroma.calls) == 1
    assert fake_chroma.calls[0]["ids"] == ["doc1-0", "doc1-1"]
    assert fake_docs_repo.mark_calls == [["doc1"]]


def test_chunker_tokenizer_encode_is_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    verbose_calls: list[bool] = []

    class FakeTokenizer:
        def encode(
            self,
            text: str,
            add_special_tokens: bool = False,
            verbose: bool = True,
        ) -> list[int]:
            del add_special_tokens
            verbose_calls.append(verbose)
            return list(range(max(1, len(text.split()))))

        def decode(
            self,
            token_ids: list[int],
            skip_special_tokens: bool = True,
        ) -> str:
            del skip_special_tokens
            return " ".join("tok" for _ in token_ids)

    monkeypatch.setattr(
        "everspring_mcp.vector.chunking.AutoTokenizer.from_pretrained",
        lambda _model_name: FakeTokenizer(),
    )

    chunker = MarkdownChunker(
        model_name="fake/model",
        max_tokens=40,
        overlap_tokens=10,
    )
    chunks = chunker.chunk("## Title\n" + ("word " * 500))

    assert chunks
    assert verbose_calls
    assert all(flag is False for flag in verbose_calls)


@pytest.mark.asyncio
async def test_embedder_truncates_over_limit_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTokenizer:
        def __init__(self) -> None:
            self.verbose_calls: list[bool] = []

        def encode(
            self,
            text: str,
            add_special_tokens: bool = False,
            verbose: bool = True,
        ) -> list[int]:
            del add_special_tokens
            self.verbose_calls.append(verbose)
            return [ord(ch) for ch in text]

        def decode(
            self,
            token_ids: list[int],
            skip_special_tokens: bool = True,
        ) -> str:
            del skip_special_tokens
            return "".join(chr(token) for token in token_ids)

    class FakeVector:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return self._values

    class FakeModel:
        def __init__(self) -> None:
            self.max_seq_length = 5
            self.tokenizer = FakeTokenizer()
            self.last_texts: list[str] = []

        def encode(
            self,
            texts: list[str],
            convert_to_numpy: bool = True,
            batch_size: int | None = None,
            show_progress_bar: bool | None = None,
            normalize_embeddings: bool | None = None,
        ) -> list[FakeVector]:
            del convert_to_numpy
            del batch_size
            del show_progress_bar
            del normalize_embeddings
            self.last_texts = list(texts)
            return [FakeVector([0.1, 0.2, 0.3]) for _ in texts]

    fake_model = FakeModel()
    embedder = Embedder(model_name="fake/model")
    monkeypatch.setattr(embedder, "ensure_model_loaded", lambda: fake_model)

    vectors = await embedder.embed_texts(["abcdefghi", "ok"])

    assert len(vectors) == 2
    assert fake_model.last_texts == ["abcde", "ok"]
    assert fake_model.tokenizer.verbose_calls
    assert all(flag is False for flag in fake_model.tokenizer.verbose_calls)


def test_chunking_enforces_token_limit() -> None:
    """Test that chunks never exceed max_tokens."""
    chunker = MarkdownChunker(
        model_name="google/embeddinggemma-300m",
        max_tokens=100,
        overlap_tokens=10,
    )

    # Create a long markdown doc
    long_text = "\n\n".join([f"## Section {i}\n" + "word " * 200 for i in range(5)])
    chunks = chunker.chunk(long_text)

    # Verify no chunk exceeds limit
    for chunk in chunks:
        token_count = chunker._count_tokens(chunk.content)
        assert token_count <= 100, f"Chunk exceeded limit: {token_count} tokens"

    # Verify we got multiple chunks (doc was too long for one)
    assert len(chunks) > 1


@pytest.mark.asyncio
async def test_embed_with_progress_updates_progress_bar() -> None:
    """Embedding helper should advance progress by processed batch size."""

    class FakeEmbedder:
        batch_size = 2

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    class FakeProgressBar:
        def __init__(self) -> None:
            self.updated = 0

        def update(self, amount: int) -> None:
            self.updated += amount

    indexer = object.__new__(VectorIndexer)
    indexer.embedder = FakeEmbedder()
    progress = FakeProgressBar()

    vectors = await VectorIndexer._embed_with_progress(
        indexer,
        texts=["a", "b", "c", "d", "e"],
        progress_bar=progress,
    )

    assert len(vectors) == 5
    assert progress.updated == 5


def test_emit_finalization_heartbeat_logs_expected_message(caplog: pytest.LogCaptureFixture) -> None:
    """Heartbeat helper should emit the expected finalization log text."""
    caplog.set_level("INFO", logger="everspring_mcp.vector.indexer")

    VectorIndexer._emit_finalization_heartbeat()

    assert HNSW_FINALIZATION_MESSAGE in caplog.text

