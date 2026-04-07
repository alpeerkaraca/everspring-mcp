# EverSpring MCP - Copilot Instructions

## Build, test, and lint commands

Use `uv` for local workflows.

| Task | Command |
| --- | --- |
| Install dependencies (including dev tools) | `uv sync --dev` |
| Build package artifact | `uv build` |
| Lint | `uv run ruff check src tests` |
| Lint one file | `uv run ruff check src\everspring_mcp\sync\s3_sync.py` |
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

### 3. Indexing pipeline (local docs -> Chroma vectors + BM25 corpus)
- CLI entry: `python -m everspring_mcp.main index`.
- `VectorIndexer` (`src\everspring_mcp\vector\indexer.py`) reads unindexed docs from SQLite, chunks markdown, embeds chunks, upserts into Chroma, then marks docs indexed.
- Chunking and embedding are model-aware:
  - `MarkdownChunker` uses tokenizer-based limits.
  - `Embedder` enforces `max_seq_length` safety before `SentenceTransformer.encode(...)`.
- Hybrid retrieval uses:
  - Dense Chroma search + sparse BM25 (`BM25Index`)
  - Reciprocal Rank Fusion in `HybridRetriever` (`src\everspring_mcp\vector\retriever.py`).

### 4. MCP serving layer
- `MCPServer` (`src\everspring_mcp\mcp\server.py`) exposes `search_spring_docs`, `get_spring_docs_status`, and `list_spring_modules`.
- `SpringDocsTool` (`src\everspring_mcp\mcp\tools.py`) adds progress events and score-threshold filtering on top of `HybridRetriever`.

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

8. **Hybrid search filtering expectations**
   - Module/version filters are applied before ranking/scoring in dense and sparse paths.
   - URL deduplication is enabled by default in retrieval.
