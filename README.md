# EverSpring MCP

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-6E56CF)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-4F46E5)
![MCP](https://img.shields.io/badge/MCP-SDK%20Server-111827)
![CUDA](https://img.shields.io/badge/CUDA-BF16%20Path-0891B2)
![AMD ROCm](https://img.shields.io/badge/AMD%20ROCm-RX%209000%20Ready-ED1C24?logo=amd&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS%20Fallback-A2AAAD?logo=apple&logoColor=white)

**Enterprise-grade Hybrid RAG for Spring documentation.**
Engineered around one operating model: **Build on Power, Search on Light**.

## Introduction

EverSpring MCP separates heavy offline computation from online query serving:

| Profile | Responsibility | Typical Hardware |
| --- | --- | --- |
| **Build Node** | Chunking, embedding, Chroma upserts, BM25 build, snapshot publish | RTX/ROCm workstation |
| **Search Node** | Snapshot download, hybrid retrieval, MCP serving | Laptop / thin client |

This split lets you centralize expensive indexing once, publish artifacts to S3, then run fast retrieval anywhere by restoring snapshots (`SQLite + Chroma + BM25`).

## Architecture Overview

```text
Scrape -> Sync -> Index -> Hybrid Retrieve -> MCP Tool
```

1. **Scrape** (`ScraperPipeline`): crawls and normalizes Spring docs to S3 raw-data layout.
2. **Sync** (`SyncOrchestrator` + `S3SyncService`): manifest-based deltas and snapshot upload/download.
3. **Index** (`VectorIndexer`): concurrent chunk production + batched embedding/upsert.
4. **Hybrid Retrieve** (`HybridRetriever`): Chroma dense + BM25 sparse + RRF fusion.
5. **Serve** (`MCPServer`): MCP stdio tool `search_spring_docs`.

### Engineering Highlights

#### 1) Producer-Consumer Indexing (GIL-Aware Throughput Design)
- `VectorIndexer.index_unindexed(...)` pools chunk work into a shared `pending_chunks` buffer.
- Chunk preparation is parallelized, while GPU embedding and Chroma writes happen in large batches.
- Batched upserts (`chroma_upsert_batch_size`) minimize write overhead and keep memory bounded.
- Observed on reference build machine: **RAM ~18GB -> ~3.8GB**, throughput to **~123 chunks/sec (~60k tokens/sec)**.

#### 2) Auto-Detect Hardware Fallback
- `Embedder._resolve_cuda_bfloat16()` selects runtime path dynamically.
- Prioritizes CUDA bf16 path, supports ROCm-backed torch setups, falls back to MPS float32 on Apple Silicon, then CPU float32.
- Scales from high-end build GPUs (RTX 4090 / RX 9070 class) to standard developer laptops.

#### 3) Zero-Cold-Start MCP Design
- `MCPServer` binds stdio immediately (late heavy work).
- Startup lifecycle asynchronously preheats:
  - `await retriever._embedder.prefetch_model()`
  - `retriever.ensure_bm25_index()`
- Result: no model-load surprise during normal first interactive query path.

#### 4) Hybrid RRF Retrieval
- Dense retrieval: **ChromaDB** cosine search.
- Sparse retrieval: **BM25Okapi** lexical search.
- Fusion: **Reciprocal Rank Fusion** (`RRF_K = 60`) in `HybridRetriever`.
- Module/version filters are applied before ranking paths are merged.

## Hardware Requirements & Auto-Fallback

| Runtime | Device Path | Numeric Type | Notes |
| --- | --- | --- | --- |
| NVIDIA CUDA (preferred) | `cuda` | `bfloat16` when available | Best throughput profile |
| CUDA without bf16 support | `cuda` | `float16` | Automatic CUDA downgrade |
| Apple Silicon | `mps` | `float32` | M1/M2/M3 compatible fallback |
| CPU-only | `cpu` | `float32` | Functional, slower path |

**Minimum baseline:** Python 3.11+, local writable `~/.everspring`, and indexed artifacts (or S3 snapshot restore).

## Installation

### Standard install

```bash
uv sync
```

### AMD ROCm install (Linux x86_64)

> Current `amd` extra is scoped to Linux x86_64 ROCm wheels (Python < 3.12).

```bash
uv sync --extra amd
export HSA_OVERRIDE_GFX_VERSION=12.0.1
```
> Check is your ROCm setup is detected correctly by PyTorch:
```bash
python -c "
import torch;print('PyTorch version:', torch.__version__);print('Your GPU device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device detected');print('CUDA availability:', torch.cuda.is_available());print('BF16 support:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A');print('MPS availability:', torch.backends.mps.is_available());
"
```
> From now on you should use 'python' command instead of 'uv run python' to ensure AMD ROCm environment variables are respected.

## CLI Usage
### Before running application:
- Ensure you have access to [Google Embedding Gemma-300m on Hugging Face](https://huggingface.co/google/embeddinggemma-300m)
- Ensure you have logged in to Hugging Face CLI with `huggingface-cli login` and have access to the model.
- Or set your 'HF_TOKEN' environment variable with a valid Hugging Face token that has access to the model.
- If you are planning to host your own files, ensure you have access to an S3 bucket and have configured your AWS credentials properly.


### Build node: high-throughput indexing

```bash
uv run python -m everspring_mcp.main index \
  --limit 1000000 \
  --batch-size 512 \
  --chunk-workers 16 \
  --upsert-batch-size 1024 \
  --build-bm25 \
  --json
```

### Build node: publish snapshots

```bash
uv run python -m everspring_mcp.main sync --mode snapshot-upload --json
```

### Search node: restore snapshots

```bash
uv run python -m everspring_mcp.main sync --mode snapshot-download --json
```

### Optional: prime manifests for bulk targets

```bash
uv run python -m everspring_mcp.main sync --all --mode manifest-prime --parallel-jobs 5 --json
```

### Local smoke search

```bash
uv run python -m everspring_mcp.main search \
  --query "How to configure multiple SecurityFilterChain beans?" \
  --top-k 3 \
  --json
```

## MCP Server Integration

### Download or restore snapshots on search node before starting MCP server.
```bash
uv run python -m everspring_mcp.main sync --mode snapshot-download --parallel-jobs 5 --json
```

### Start MCP server (check is everything is working end-to-end)

```bash
uv run python -m everspring_mcp.main serve
```

### Verify integration with the SDK test client

```bash
uv run python -m everspring_mcp.main client
```

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "everspring-mcp": {
      "command": "uv",
      "args": ["run", "python", "-u", "-m", "everspring_mcp.main", "serve"],
      "env": { "PYTHONUNBUFFERED": "1" },
      "cwd": "C:\\\\path\\\\to\\\\everspring_mcp"
    }
  }
}
```

Once connected, call `search_spring_docs` with:
- `query` (required)
- `top_k` (default: `3`)
- `module` (optional)
- `version_major` (optional)
