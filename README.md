# 🌿 EverSpring MCP

An autonomous Model Context Protocol (MCP) server that keeps your LLM's Spring knowledge "Evergreen". 

## 🚀 The Problem
LLM knowledge cut-offs result in outdated advice for rapidly evolving frameworks like **Spring Boot 4+** and **Spring Framework 7+**. Developers often receive deprecated code patterns because the LLM hasn't seen the latest documentation.

## 🛠️ The Solution (Hybrid RAG)
EverSpring solves this by creating a verified bridge between official Spring docs and your LLM.
- **Cloud Scraper (AWS Lambda):** Periodically tracks changes using SHA-256 hash verification.
- **Central Registry (AWS S3):** Distributes verified, pre-processed Markdown and Vector indices.
- **Local MCP Server:** Syncs with the registry and serves the LLM with the latest context, ensuring **code privacy**.

## 🏗️ Architecture Principles
- **Hybrid Cloud:** AWS Lambda for scraping, S3 for distribution, Local MCP for execution.
- **Data Integrity:** Content validated via SHA-256 hashing before ingestion.
- **Type Safety:** Strict use of Pydantic models for all data structures (no raw dicts).
- **Asynchronous First:** Use `asyncio` and `httpx` for all I/O bound operations.

## 🔧 Key Features
- **Hybrid Search:** Combines dense retrieval (Cosine Similarity) with sparse retrieval (BM25) using Reciprocal Rank Fusion.
- **Semantic Search:** Find the right Spring patterns using `google/embedding-gemma-300m` embeddings.
- **Deprecation Guard:** Automatically flags deprecated APIs based on current release notes.
- **Zero-Config Sync:** Automatically pulls the latest "Knowledge Pack" from S3 via incremental sync.

## 💻 Tech Stack
- **Language:** Python 3.11+
- **MCP Framework:** FastMCP (Python SDK)
- **Database:** SQLite (Metadata) & ChromaDB (Vector, Persistent Storage)
- **Embeddings:** google/embedding-gemma-300m
- **Scraping:** Playwright (for JS-rendered Spring docs)
- **Cloud:** AWS Lambda, S3, EventBridge, Boto3
- **IaC:** Terraform

---
*Created by Alper Karaca - Aiming to bridge the gap between Java Excellence and AI Efficiency.*