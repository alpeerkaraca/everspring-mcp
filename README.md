# EverSpring MCP
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-6E56CF)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-4F46E5)
![MCP](https://img.shields.io/badge/MCP-SDK%20Server-111827)
![CUDA](https://img.shields.io/badge/CUDA-BF16%20Path-0891B2)
![AMD ROCm](https://img.shields.io/badge/AMD%20ROCm-RX%209000%20Ready-ED1C24?logo=amd&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS%20Fallback-A2AAAD?logo=apple&logoColor=white)

## 🚀 Quickstart (3 Steps to Run)

1. **Install dependencies**
   ```bash
   uv sync
   ```
2. **Download a pre-built slim snapshot**
   ```bash
   uv run python -m everspring_mcp.main sync --mode snapshot-download --snapshot-tier slim --snapshot-model BAAI/bge-base-en-v1.5
   ```
3. **Start the MCP server**
   ```bash
   uv run python -m everspring_mcp.main serve --tier slim
   ```

EverSpring MCP is a Spring documentation ingestion, indexing, and retrieval system with an MCP server interface.  
Primary CLI entrypoint: `python -m everspring_mcp.main`.

## What `main.py` does

`main.py` exposes operational commands for:

- **scrape**: discover/crawl Spring docs and upload raw artifacts to S3
- **sync**: manifest sync and snapshot upload/download
- **status**: local sync status reporting
- **index**: chunk + embed + upsert into Chroma
- **search**: query local index
- **serve**: start MCP stdio server
- **client**: interactive terminal client
- **model-cache**: prefetch embedding model artifacts

It also includes:

- tier-aware model defaults (`main`, `slim`, `xslim`)
- tier/model-aware default Chroma directory naming
- startup snapshot auto-refresh for `index/search/serve`

---

## High-level architecture flow

```text
Scrape -> Sync -> Index -> Retrieve -> MCP Serve
```

1. **Scrape**: Spring documentation is discovered, parsed, and uploaded to S3 (`raw-data` layout).
2. **Sync**: Local `docs` + `metadata.db` are synchronized (manifest mode) or restored from snapshots.
3. **Index**: Local docs are chunked, embedded, and written to Chroma with batching.
4. **Retrieve**: Query-time retrieval runs tier-aware search logic.
5. **Serve**: MCP server exposes documentation search tools over stdio.

---

## Hybrid RRF retrieval

For non-main tiers, retrieval uses a hybrid pipeline:

- **Dense retrieval** from Chroma
- **Sparse retrieval** from BM25
- **Fusion** with Reciprocal Rank Fusion (RRF)

Main details:

- RRF constant is defined in retriever (`RRF_K = 60`).
- `tier=main` intentionally bypasses BM25/RRF and uses model-native dense path.
- `tier=slim/xslim` uses hybrid dense+sparse retrieval.

---

## Installation

```bash
uv sync --dev
```

For AMD ROCm workflows (if you use the project extra):

```bash
uv sync --extra amd
export HSA_OVERRIDE_GFX_VERSION=12.0.1
```

---

## Tier defaults

| Tier | Default embedding model | Default chunk size (`max_tokens`) | Default overlap (`overlap_tokens`) |
| --- | --- | ---: | ---: |
| `main` | `BAAI/bge-m3` | 2048 | 200 |
| `slim` | `BAAI/bge-base-en-v1.5` | 512 | 50 |
| `xslim` | `BAAI/bge-small-en-v1.5` | 384 | 40 |

Notes:

- Default Chroma dir pattern: `~/.everspring/chroma-{tier}-{model_slug}`
- Current retrieval behavior:
  - `tier=main`: dense/model-native path (BM25 build/check skipped)
  - `tier!=main`: hybrid path (dense + BM25)

---

## MCP server build/run instructions

1. **Install dependencies**
   ```bash
   uv sync --dev
   ```
2. **Prepare local data** (either sync snapshots or run full scrape/sync/index flow)
3. **Start MCP server**
   ```bash
   uv run python -m everspring_mcp.main serve --tier main
   ```
4. **Optional local terminal client**
   ```bash
   uv run python -m everspring_mcp.main client
   ```

---

## Global CLI arguments (`main.py`)

| Argument | Required | Values | Default | Description |
| --- | --- | --- | --- | --- |
| `--log-level` | No | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` | Console/file log level |
| `--log-file` | No | path | `None` | Optional extra log file |

---

## Command reference (arguments, values, defaults)

## `scrape`

| Argument | Required | Values | Default | Description |
| --- | --- | --- | --- | --- |
| `--entry-url` | Conditional* | URL | `None` | Discovery entry URL |
| `--module` | Conditional* | Spring module enum values | `None` | Target Spring module |
| `--version` | Conditional* | version string (`4.0.5`) | `None` | Target version |
| `--submodule` | No | string | `None` | Optional submodule |
| `--content-type` | No | content type enum values | `reference` | Doc content type |
| `--registry-path` | No | file path | `None` | Submodule registry JSON |
| `--concurrency` | No | int | `5` | Parallel scrape concurrency |
| `--s3-bucket` | No | string | `None` | S3 bucket override |
| `--s3-region` | No | string | `None` | S3 region override |
| `--s3-prefix` | No | string | `None` | S3 prefix override |
| `--no-hash-check` | No | flag | `false` | Disable content hash checks |
| `--json` | No | flag | `false` | JSON output |

\* `entry-url`, `module`, `version` must be provided together when used.

## `sync`

| Argument | Required | Values | Default | Description |
| --- | --- | --- | --- | --- |
| `--mode` | No | `manifest`, `manifest-prime`, `snapshot-upload`, `snapshot-download` | `manifest` | Sync operation mode |
| `--module` | Conditional | string | `None` | Module for non-`--all` manifest sync |
| `--version` | Conditional | string | `None` | Version for non-`--all` manifest sync |
| `--submodule` | No | string | `None` | Optional submodule |
| `--all` | No | flag | `false` | Sync all targets from matrix |
| `--force` | No | flag | `false` | Force manifest/manifest-prime operations |
| `--parallel-jobs` | No | int (`>=1`) | `5` | Parallel workers |
| `--snapshot-model` | No | model name | `None` | Snapshot namespace model override |
| `--snapshot-tier` | No | `main`, `slim`, `xslim` | `None` | Snapshot namespace tier override |
| `--s3-bucket` | No | string | `None` | S3 bucket override |
| `--s3-region` | No | string | `None` | S3 region override |
| `--s3-prefix` | No | string | `None` | S3 prefix override |
| `--data-dir` | No | path | `None` | Local data directory override |
| `--json` | No | flag | `false` | JSON output |

Snapshot namespace path format:

`s3://<bucket>/<prefix>/db-snapshots/{model_slug}-{tier}/...`

## `status`

| Argument | Required | Values | Default | Description |
| --- | --- | --- | --- | --- |
| `--module` | No | string | `None` | Module filter |
| `--version` | No | string | `None` | Version filter |
| `--submodule` | No | string | `None` | Submodule filter |
| `--all` | No | flag | `false` | Show status for all local manifest targets |
| `--data-dir` | No | path | `None` | Local data directory override |
| `--json` | No | flag | `false` | JSON output |

## `index`

| Argument | Required | Values | Default | Description |
| --- | --- | --- | --- | --- |
| `--limit` | No | int | `50` | Max unindexed docs to process |
| `--tier` | No | `main`, `slim`, `xslim` | `main` | Embedding tier |
| `--embed-model` | No | model name | tier default | Embedding model override |
| `--batch-size` | No | int | config default | Embedding batch size |
| `--chunk-workers` | No | int | config default | CPU chunk preparation workers |
| `--upsert-batch-size` | No | int | config default | Chroma upsert batch size |
| `--max-tokens` | No | int | tier default | Chunk token limit override |
| `--overlap-tokens` | No | int | tier default | Chunk overlap override |
| `--build-bm25` | No | flag | `false` | Build BM25 (skipped for `tier=main`) |
| `--reindex` | No | flag | `false` | Reset indexed flags and rebuild vectors |
| `--module` | No | string | `None` | Filter for `--reindex` |
| `--version` | No | int | `None` | Major version filter for `--reindex` |
| `--submodule` | No | string | `None` | Submodule filter for `--reindex` |
| `--data-dir`, `--db-filename`, `--docs-subdir`, `--chroma-dir`, `--collection` | No | paths/strings | `None` | Local/index storage overrides |
| `--json` | No | flag | `false` | JSON output |

## `search`

| Argument | Required | Values | Default | Description |
| --- | --- | --- | --- | --- |
| `--query`, `-q` | **Yes** | string | — | Search query |
| `--top-k`, `-k` | No | int | `3` | Result count |
| `--module` | No | string | `None` | Module filter |
| `--version` | No | int | `None` | Major version filter |
| `--build-index` | No | flag | `false` | Rebuild BM25 before search (`tier!=main`) |
| `--no-dedup` | No | flag | `false` | Disable URL dedup |
| `--tier` | No | `main`, `slim`, `xslim` | `main` | Search tier/model |
| `--json` | No | flag | `false` | JSON output |

## `serve`

| Argument | Required | Values | Default | Description |
| --- | --- | --- | --- | --- |
| `--transport` | No | `stdio` | `stdio` | MCP transport |
| `--tier` | No | `main`, `slim`, `xslim` | `main` | Serving tier/model |
| `--json` | No | flag | `false` | Emit startup status JSON |

## `client`

| Argument | Required | Values | Default | Description |
| --- | --- | --- | --- | --- |
| `--no-progress` | No | flag | `false` | Disable progress display |

## `model-cache`

| Argument | Required | Values | Default | Description |
| --- | --- | --- | --- | --- |
| `--embed-model` | No | model name | config default | Model override |
| `--batch-size` | No | int | config default | Batch size override |
| `--json` | No | flag | `false` | JSON output |

---

## Environment variables

## Required (by workflow)

| Variable | Required when | Description |
| --- | --- | --- |
| `EVERSPRING_S3_BUCKET` | `scrape` (if no explicit `--s3-bucket` override) | S3 bucket for scrape uploads |
| AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional session token) | Any S3 operation (`scrape`, `sync` snapshot/manifest) | AWS auth for S3 |

## Optional (configuration overrides)

| Variable | Default | Used by |
| --- | --- | --- |
| `EVERSPRING_S3_BUCKET` | `everspring-mcp-kb` | sync/scrape |
| `AWS_REGION` | `eu-central-1` (sync), `us-east-1` fallback in scraper config | sync/scrape |
| `EVERSPRING_S3_PREFIX` | `spring-docs` (sync), `spring-docs/raw-data` (scrape env config) | sync/scrape |
| `EVERSPRING_DATA_DIR` | `~/.everspring` | sync/vector |
| `EVERSPRING_MODEL_TIER` | `main` | sync snapshot namespace |
| `EVERSPRING_EMBED_MODEL` | tier default | vector + sync snapshot namespace model |
| `EVERSPRING_CHROMA_DIR` | `~/.everspring/chroma` (overridden dynamically by CLI tier/model) | vector |
| `EVERSPRING_CHROMA_COLLECTION` | `spring_docs` | vector |
| `EVERSPRING_EMBED_TIER` | `main` | vector |
| `EVERSPRING_INDEX_CHUNK_WORKERS` | auto (`cpu_count` based) | vector index |
| `EVERSPRING_CHROMA_UPSERT_BATCH_SIZE` | `512` | vector index |
| `EVERSPRING_INDEX_PREFETCH_BATCHES` | `3` | vector index prefetch |

---

## Operational examples

## Snapshot upload/download for `bge-m3-main`

```bash
python -m everspring_mcp.main sync \
  --mode snapshot-upload \
  --snapshot-model BAAI/bge-m3 \
  --snapshot-tier main \
  --json

python -m everspring_mcp.main sync \
  --mode snapshot-download \
  --snapshot-model BAAI/bge-m3 \
  --snapshot-tier main \
  --json
```

## Index/Search/Serve (main tier)

```bash
python -m everspring_mcp.main index --tier main --build-bm25 --json
python -m everspring_mcp.main search --tier main --query "security filter chain" --json
python -m everspring_mcp.main serve --tier main
```

---

## Notes

- `index/search/serve` perform startup snapshot freshness checks for active model+tier; if a newer remote snapshot is applied, process restarts automatically once.
- Keep `data_dir` consistent between `sync`, `index`, and `search/serve` to avoid “docs exist but DB empty” mismatches.

---

## Claude Desktop (`claude_desktop_config.json`)

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
