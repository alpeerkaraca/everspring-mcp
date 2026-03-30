# Agent Context: EverSpring MCP

## Project Intent
EverSpring MCP is a specialized RAG (Retrieval-Augmented Generation) system designed to provide LLMs with real-time, verified access to the Spring Ecosystem documentation (Spring Boot 4+, Spring Framework 7+, etc.).

## Architecture Principles
- **Hybrid Cloud:** AWS Lambda for scraping, S3 for distribution, and Local MCP for execution.
- **Data Integrity:** Content must be validated via SHA-256 hashing before ingestion.
- **Type Safety:** Strict use of Pydantic models for all data structures (No raw dicts).
- **Asynchronous First:** Use `asyncio` and `httpx` for I/O bound operations (Scraping/API).

## Tech Stack Requirements
- **Language:** Python 3.11+
- **MCP Framework:** FastMCP (Python SDK)
- **Database:** SQLite (Metadata) & ChromaDB (Vector, Persistent Storage)
- **Cloud:** AWS Boto3 (S3 interaction)
- **Scraping:** Playwright (for JS-rendered Spring docs)

## Embedding Strategy
- **Model:** google/embedding-gemma-300m (Chosen for high-density semantic representation at low memory footprint).
- **Vector DB:** ChromaDB (Persistent storage).
- **Search:** Cosine Similarity with metadata filtering (Spring Version, Module).

## Retrieval Strategy: Hybrid Search
- **Dense Retrieval:** Cosine Similarity using `google/embedding-gemma-300m`.
- **Sparse Retrieval:** BM25 (Best Matching 25) for exact keyword/annotation matching.
- **Reranking/Fusion:** Implement Reciprocal Rank Fusion (RRF) to combine results from both scorers.
- **Filtering:** Always apply metadata filters (version, module) BEFORE ranking.

## Role Definitions for Copilot
- While performing every operation pay attention to security best practices, especially when handling data and interacting with AWS services.
- When writing **Scrapers**: Focus on resilience, rate-limiting, and clean Markdown conversion.
- When writing **MCP Tools**: Ensure clear docstrings as they are used by the LLM as tool definitions.
- When writing **Sync Logic**: Prioritize "Incremental Sync" to minimize S3 egress costs.
- When writing **Vectorization Logic**: Ensure that the embedding process is consistent and that metadata is correctly associated with each vector.
- When writing **Tests**: Use `pytest` and focus on edge cases (e.g., handling doc structure changes, network failures).
- When writing **LLM Prompts**: Ensure clarity and specificity to guide the LLM effectively, and include examples where possible.
- When writing **Pydantic Models**: Define clear and concise models with appropriate validation to ensure data integrity throughout the system.
- When writing **AWS Lambda Functions**: Focus on efficient resource usage, proper error handling, and secure access to AWS services (e.g., using IAM roles).
- When writing **S3 Interaction Logic**: Ensure that all interactions with S3 are secure, efficient, and include proper error handling to manage potential issues with network or permissions.
- When writing **ChromaDB Logic**: Ensure that vector storage and retrieval are optimized for performance, and that metadata is correctly associated with each vector for effective filtering.
- When writing **Terraform Scripts**: Follow best practices for infrastructure as code, including modularization, use of variables, and proper state management to ensure reproducibility and maintainability.