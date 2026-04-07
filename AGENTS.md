# EverSpring MCP Assistant Contract

This document defines non-negotiable engineering constraints for AI assistants working in this repository.

## Mission

Maintain and improve an enterprise-grade Hybrid RAG pipeline for Spring documentation while preserving performance-critical architecture.

## Core Principle

**Build on Power, Search on Light**

- Build nodes (high-end GPU) handle indexing, BM25 generation, and snapshot publishing.
- Search nodes (thin clients/laptops) restore snapshots and serve queries.

Do not collapse these roles in architecture changes unless explicitly requested.

## Non-Negotiable Performance Rules

### 1) Never Break the Buffer

`vector/indexer.py` uses a producer-consumer design with a `pending_chunks` buffer.
This buffering model is **sacred for performance**.

- Do **not** revert to document-by-document synchronous indexing loops.
- Do **not** remove producer-consumer buffering, even if a simpler loop looks cleaner.
- Do **not** undo the GIL-avoidance design by forcing sequential chunking/embedding writes.
- Keep concurrent chunk preparation.
- Keep threshold-based embedding and Chroma write batching.
- Keep mark-indexed behavior tied to successful vector writes.

### 2) Keep Chroma Writes Batched

- Do not replace batched upserts with per-document or per-chunk upserts.
- Preserve configurable upsert batch sizing (`chroma_upsert_batch_size` / CLI override).

### 3) Preserve Model-Aware Chunking

- Chunk sizing must remain tokenizer-aware and respect token limits.
- Do not replace model-token counting with naive character/word counting.

## Hardware and Precision Constraints

- GPU path is mandatory for optimized indexing.
- `bfloat16` is mandatory by design in embedder loading.
- Do not introduce `float32` or CPU fallback paths unless the user explicitly asks for them.

### AMD ROCm Guidance

- Package manager is `uv`.
- ROCm PyTorch must be managed via `[project.optional-dependencies]` `amd` extra.
- Required runtime environment for RX 9000 class GPUs:
  - `HSA_OVERRIDE_GFX_VERSION=12.0.1`

## Dependency and Tooling Policy

- Use `uv` for dependency and run workflows.
- Prefer existing project commands:
  - `uv sync`
  - `uv sync --extra amd`
  - `uv run pytest -q`
  - `uv run ruff check src tests`
  - `uv run mypy src`

Do not introduce ad-hoc package managers or runtime toolchains.

## Retrieval Contract (Hybrid RRF)

Any change to dense retrieval must account for the full hybrid stack:

1. Dense retrieval in Chroma
2. Sparse retrieval in BM25
3. Fusion in `HybridRetriever` using RRF (`RRF_K`)

If you modify Chroma query logic, you must evaluate and adjust BM25 and fusion behavior as needed. Never optimize one path while silently regressing hybrid relevance quality.
Any adjustments to ranking behavior should explicitly consider `RRF_K`.

## Data and Snapshot Contract

- Snapshot upload/download flows are first-class operational paths.
- BM25 persistence and restore behavior must remain aligned with snapshot workflows.
- Preserve atomicity and rollback safety when altering snapshot apply logic.

## Change Safety Checklist

Before finalizing vector/search/sync changes:

1. Verify indexing throughput assumptions still hold (concurrency + batching intact).
2. Verify hardware constraints are still enforced (`CUDA` + `bfloat16`).
3. Verify hybrid retrieval remains dense + sparse + RRF.
4. Verify snapshot flows still keep build-node/search-node separation intact.

## Communication Channel Integrity
- Never use print() for debugging or status updates in the MCP serve path.
- All non-JSON output MUST go through the centralized logger to sys.stderr. Any data on sys.stdout that is not a valid MCP JSON-RPC message will break the connection.
