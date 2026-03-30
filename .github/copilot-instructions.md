# EverSpring MCP - Copilot Instructions

## Project Overview

EverSpring MCP is a Hybrid RAG system providing LLMs with real-time, verified access to Spring Ecosystem documentation (Spring Boot 4+, Spring Framework 7+). It uses AWS Lambda for scraping, S3 for distribution, and a local MCP server for execution.

## Tech Stack

- **Language:** Python 3.11+
- **MCP Framework:** FastMCP (Python SDK)
- **Database:** SQLite (Metadata) & ChromaDB (Vector, Persistent Storage)
- **Scraping:** Playwright (for JS-rendered Spring docs)
- **Cloud:** AWS Boto3 (S3 interaction)
- **Data Models:** Pydantic for all data structures (no raw dicts)

## Architecture Principles

- **Hybrid Cloud:** AWS Lambda for scraping, S3 for distribution, Local MCP for execution
- **Data Integrity:** Content must be validated via SHA-256 hashing before ingestion
- **Type Safety:** Strict use of Pydantic models for all data structures
- **Async First:** Use `asyncio` and `httpx` for all I/O bound operations
- **Incremental Sync:** Minimize S3 egress costs by syncing only changed content

## Embedding Strategy

- **Model:** google/embedding-gemma-300m (high-density semantic representation, low memory footprint)
- **Vector DB:** ChromaDB (Persistent storage)
- **Search:** Cosine Similarity with metadata filtering (Spring Version, Module)

## Retrieval Strategy: Hybrid Search

- **Dense Retrieval:** Cosine Similarity using `google/embedding-gemma-300m`
- **Sparse Retrieval:** BM25 for exact keyword/annotation matching
- **Reranking/Fusion:** Reciprocal Rank Fusion (RRF) to combine results from both scorers
- **Filtering:** Always apply metadata filters (version, module) BEFORE ranking

## Component Guidelines

### Scrapers
- Focus on resilience and rate-limiting
- Convert scraped content to clean Markdown
- Handle network failures gracefully

### MCP Tools
- Write clear docstrings—they serve as tool definitions for the LLM
- Expose semantic search and deprecation checking capabilities

### Vectorization
- Ensure embedding consistency across runs
- Associate correct metadata with each vector for effective filtering

### Sync Logic
- Implement incremental sync using hash comparison
- Pull only changed "Knowledge Packs" from S3

### Pydantic Models
- Define clear and concise models with appropriate validation
- Ensure data integrity throughout the system

### AWS Lambda Functions
- Focus on efficient resource usage and proper error handling
- Use IAM roles for secure access to AWS services

### S3 Interaction Logic
- Ensure secure, efficient interactions with proper error handling
- Manage potential issues with network or permissions

### ChromaDB Logic
- Optimize vector storage and retrieval for performance
- Associate correct metadata with each vector for effective filtering

### Terraform Scripts
- Follow IaC best practices: modularization, variables, proper state management
- Ensure reproducibility and maintainability

## Testing

- Use `pytest` for all tests
- Focus on edge cases: doc structure changes, network failures, hash mismatches

## Security

- Pay attention to security best practices when handling data and interacting with AWS services
- Use IAM roles and policies to restrict access to AWS resources
- Ensure that sensitive information (like AWS credentials) is not hardcoded and is managed securely (e.g., using environment variables or AWS Secrets Manager)
- Validate and sanitize all inputs to prevent potential security vulnerabilities
- Regularly review and update dependencies to mitigate security risks
- Implement proper error handling to avoid exposing sensitive information in error messages
- Consider implementing logging and monitoring for security-related events, especially for AWS interactions

