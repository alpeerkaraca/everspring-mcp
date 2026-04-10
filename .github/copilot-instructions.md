# EverSpring MCP - Copilot Instructions

## Build, test, and lint commands

Use `uv` for local workflows.

| Task | Command |
| --- | --- |
| Install dependencies | `uv sync` |
| Install dependencies (including dev tools) | `uv sync --dev` |
| Build package artifact | `uv build` |
| Lint | `uv run ruff check src tests` |
| Lint one file | `uv run ruff check src\everspring_mcp\mcp\server.py` |
| Format code | `uv run ruff format src tests` |
| Type-check | `uv run mypy src` |
| Run full test suite | `uv run pytest -q` |
| Run one test file | `uv run pytest tests\test_vector_indexer.py -q` |
| Run one test case | `uv run pytest tests\test_vector_indexer.py::test_chunking_enforces_token_limit -q` |

## High-level architecture

### 1. Scrape pipeline (cloud/ingestion side)
- CLI entry: `python -m everspring_mcp.main scrape` in `src\everspring_mcp\main.py`.
- `ScraperPipeline` (`src\everspring_mcp\scraper\pipeline.py`) orchestrates:
  1. `SpringBrowser.navigate_with_retry(...)` for resilient page fetches.
  2. `SpringDocParser.parse(...)` for version extraction + markdown conversion.
  3. Hash-aware S3 upload (`content-hash` metadata) for incremental updates.
- Raw content is written under:
  - `spring-docs/raw-data/{module[-submodule]}/{version}/{url_hash}/document.md`
  - `spring-docs/raw-data/{module[-submodule]}/{version}/{url_hash}/metadata.json`

### 2. Sync pipeline (S3 -> local docs + SQLite tracking)
- CLI entry: `python -m everspring_mcp.main sync`.
- `SyncOrchestrator` (`src\everspring_mcp\sync\orchestrator.py`) coordinates:
  1. Fetch manifest from S3.
  2. Fallback build from S3 listing if missing.
  3. Delta computation + concurrent download through `S3SyncService`.
  4. SQLite updates via `StorageManager`.
- `sync --mode manifest-prime` pre-generates/uploads missing manifests.
- `sync --mode snapshot-upload` uploads local SQLite/Chroma snapshots to `spring-docs/db-snapshots/`.
- `sync --mode snapshot-download` restores the latest matching SQLite/Chroma snapshot pair for search-node startup.

### 3. Indexing pipeline (local docs -> Chroma vectors + BM25 corpus)
- CLI entry: `python -m everspring_mcp.main index`.
- `VectorIndexer` (`src\everspring_mcp\vector\indexer.py`) reads unindexed docs from SQLite, chunks markdown, embeds chunks, upserts into Chroma, then marks docs indexed.
- Indexing is a producer-consumer pipeline: chunk preparation runs in a `ProcessPoolExecutor`, staged in `pending_chunks`, then embedded/upserted in configured batches.
- Chunking and embedding are model-aware:
  - `MarkdownChunker` uses tokenizer-based limits.
  - `Embedder` enforces `max_seq_length` safety before `SentenceTransformer.encode(...)`.

### 4. Retrieval + MCP serving
- `HybridRetriever` (`src\everspring_mcp\vector\retriever.py`) handles retrieval:
  - `tier=main`: dense-only path.
  - `tier=slim`/`tier=xslim`: dense Chroma + sparse BM25 fused by Reciprocal Rank Fusion (`RRF_K=60`).
- `MCPServer` (`src\everspring_mcp\mcp\server.py`) is SDK-native (`mcp.server.Server`) and publishes one MCP tool: `search_spring_docs`.
- The server uses late initialization and startup preheating in lifespan hooks (`prefetch_model()` + `ensure_bm25_index()`) to avoid first-query cold starts.

## Key conventions for this repository

1. **Pydantic-first domain/config models**
   - Core records/configs are Pydantic models (`SyncConfig`, `VectorConfig`, `DocumentRecord`, `SyncManifest`, etc.).
   - Prefer model validation/coercion over ad-hoc dict parsing.

2. **Version handling rules are strict and content-type aware**
   - API docs require explicit release version context.
   - Reference docs may fallback to provided target version when `span.version` is missing.
   - Version mismatch between parsed page and target is treated as an error.

3. **Never remove version selector nodes during parser cleanup**
   - `_clean_html` in `scraper\parser.py` explicitly protects configured version selectors (for example `span.version`) before pruning noisy nodes.

4. **Submodule layout uses `module-submodule` path segments**
   - S3 raw-data and sync prefixes flatten submodule into the module segment (for example `spring-data-jpa`).
   - Sync code must preserve compatibility with current and legacy prefixes.

5. **SQLite `documents.file_path` is docs-root relative**
   - Sync stores file paths relative to local `docs` root.
   - Indexer resolves markdown with `config.docs_dir / file_path`.
   - New-layout metadata is expected as sibling `metadata.json` next to `document.md`.

6. **Discovery safety defaults**
   - External-link following is forcibly disabled.
   - HTTP 404 links are suppressed during traversal by default (`suppress_http_404=True`).

7. **Incremental sync and integrity checks are mandatory behavior**
   - Manifest/delta flow is the default sync path.
   - `content-hash` metadata is used to avoid unnecessary upload/download work.
   - Download/upload retry policy uses tenacity with exponential backoff.

8. **Producer-consumer batching is performance-critical**
   - Keep `pending_chunks` buffering, chunk workers, embedding batching, and Chroma upsert batching intact.
   - Do not regress to document-by-document synchronous indexing.
   - Keep mark-indexed behavior coupled to successful vector writes.

9. **Hybrid search filtering and tier behavior**
   - Module/version filters are applied before ranking/scoring in dense and sparse paths.
   - URL deduplication is enabled by default in retrieval.
   - BM25/RRF apply to `slim`/`xslim`; `main` tier uses dense-only retrieval by design.

10. **MCP stdio output discipline**
    - Do not use `print()` in the MCP serve path.
    - Emit logs through the project logger (`everspring_mcp.utils.logging`) so stdout stays valid JSON-RPC traffic.
