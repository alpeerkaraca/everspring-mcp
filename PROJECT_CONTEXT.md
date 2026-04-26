# EverSpring MCP - Project Context

This file contains the architectural context of the `everspring-mcp` project. It is intended to be read by AI agents to quickly understand the system.

## 1. Primary Purpose
EverSpring MCP is a sophisticated Retrieval-Augmented Generation (RAG) system specifically designed for Spring framework documentation. It exposes this search capability to LLMs via the Model Context Protocol (MCP).

## 2. Architectural Overview & Tiers
The system features a tiered architecture supporting different embedding models and retrieval strategies:
- **main**: Full feature set, utilizing the BGE-M3 model for native Hybrid Search (Dense + Sparse embeddings).
- **slim** / **xslim**: Resource-constrained tiers, utilizing smaller models and supporting Dense-only embeddings natively.

## 3. Core Modules & Data Flow

The core data flow follows a pipeline: **Scrape -> S3 -> Sync -> Index -> Serve**

1.  **Scraper (`src/everspring_mcp/scraper/`)**:
    - Orchestrates scraping (Playwright), parsing, and uploading documentation to S3.
    - Key component: `ScraperPipeline` (`scraper/pipeline.py`).
2.  **Sync & Storage (`src/everspring_mcp/sync/`, `src/everspring_mcp/storage/`)**:
    - Manages the synchronization of data from S3 to local storage. Supports snapshot-based deployment for quick setup and incremental syncs via content hashing.
    - Key component: `SyncOrchestrator` (`sync/orchestrator.py`).
3.  **Vector & Indexing (`src/everspring_mcp/vector/`)**:
    - Handles file pre-filtering (hard-excluding noisy glob patterns like Javadocs), chunking, embedding, and indexing of documentation into ChromaDB. Implements hybrid retrieval logic combining dense and sparse vectors (native in 'main', fallback to BM25 in others).
    - Key components: `VectorIndexer` (`vector/indexer.py`), `HybridRetriever` (`vector/retriever.py`).
4.  **MCP Server (`src/everspring_mcp/mcp/`)**:
    - The core MCP server implementation that exposes the `search_spring_docs` tool to LLMs.
    - Key component: `MCPServer` (`mcp/server.py`).

## 4. Key Technologies
- **Model Context Protocol (MCP)**: For exposing tools to LLMs.
- **ChromaDB**: The primary vector database.
- **BGE-M3**: Embedding model for native dense/sparse hybrid search.
- **BM25**: Sparse retrieval algorithm used for fallback/hybrid search.
- **Playwright**: Used for scraping dynamic documentation.
- **Pydantic**: For robust data modeling and validation.
- **AWS S3**: Central storage for scraped documentation snapshots.
- **FastAPI / HTTP**: For local serving/testing capabilities.

## 5. Maintenance Instruction
**Agent Mandate:** If you make significant architectural changes, add new core modules, alter the primary data flow, or change key technologies, you MUST update this `PROJECT_CONTEXT.md` file to reflect those changes to ensure future context remains accurate.
