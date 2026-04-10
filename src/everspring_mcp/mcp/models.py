"""EverSpring MCP - MCP Models for tool parameters and responses.

Pydantic models for MCP tool inputs/outputs with validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SearchStatus(str, Enum):
    """Status of a search operation."""

    SUCCESS = "success"
    NO_RESULTS = "no_results"
    BELOW_THRESHOLD = "below_threshold"
    ERROR = "error"


class SearchParameters(BaseModel):
    """Parameters for Spring documentation search."""

    query: str = Field(
        min_length=3,
        max_length=500,
        description="Natural language query to search for in Spring documentation",
    )
    module: str | None = Field(
        default=None,
        description="Spring module to search within (e.g., 'spring-boot', 'spring-framework')",
    )
    version: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Major version number to filter results (e.g., 4 for Spring Boot 4.x)",
    )
    submodule: str | None = Field(
        default=None,
        description="Submodule for multi-module projects like 'redis' for spring-data-redis",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of results to return",
    )
    score_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold; results below this are rejected",
    )


class SearchResultItem(BaseModel):
    """A single search result item."""

    rank: int = Field(
        ge=1,
        description="Ranking position (1 is best)",
    )
    title: str = Field(
        description="Document title",
    )
    url: str = Field(
        description="Documentation URL",
    )
    module: str = Field(
        description="Spring module name",
    )
    version: str = Field(
        description="Version string (e.g., 'v4.0')",
    )
    submodule: str | None = Field(
        default=None,
        description="Submodule if applicable",
    )
    score: float = Field(
        ge=0.0,
        description="Relevance score",
    )
    score_breakdown: str = Field(
        description="Explanation of score components",
    )
    content: str = Field(
        description="Relevant content snippet",
    )
    has_code: bool = Field(
        default=False,
        description="Whether result contains code examples",
    )


class SearchResponse(BaseModel):
    """Response from a search operation."""

    status: SearchStatus = Field(
        description="Status of the search operation",
    )
    message: str = Field(
        description="Human-readable status message",
    )
    query: str = Field(
        description="Original query",
    )
    filters_applied: dict[str, Any] = Field(
        default_factory=dict,
        description="Filters that were applied",
    )
    results_found: int = Field(
        ge=0,
        description="Total number of results found",
    )
    results_returned: int = Field(
        ge=0,
        description="Number of results returned after filtering",
    )
    score_threshold: float = Field(
        description="Score threshold that was applied",
    )
    results: list[SearchResultItem] = Field(
        default_factory=list,
        description="Search results",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings generated during search",
    )


class ProgressNotification(BaseModel):
    """Progress notification during search operations."""

    stage: str = Field(
        description="Current processing stage",
    )
    message: str = Field(
        description="Progress message",
    )
    percentage: float = Field(
        ge=0.0,
        le=100.0,
        description="Completion percentage",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details",
    )


class ModuleInfo(BaseModel):
    """Information about available modules."""

    name: str = Field(
        description="Module name",
    )
    versions: list[int] = Field(
        description="Available major versions",
    )
    submodules: list[str] = Field(
        default_factory=list,
        description="Available submodules",
    )
    doc_count: int = Field(
        ge=0,
        description="Number of indexed documents",
    )


class StatusResponse(BaseModel):
    """Server status response."""

    healthy: bool = Field(
        description="Whether server is healthy",
    )
    index_ready: bool = Field(
        description="Whether vector index is ready",
    )
    modules: list[ModuleInfo] = Field(
        default_factory=list,
        description="Available modules and versions",
    )
    total_documents: int = Field(
        ge=0,
        description="Total indexed documents",
    )
    bm25_index_loaded: bool = Field(
        description="Whether BM25 index is loaded",
    )


class StructuredErrorResponse(BaseModel):
    """Structured MCP error payload for LLM auto-correction flows."""

    model_config = ConfigDict(extra="forbid")

    error_type: str = Field(
        min_length=1,
        description="Stable machine-readable error category",
    )
    message: str = Field(
        min_length=1,
        description="Human-readable summary of the error",
    )
    resolution_hints: list[str] = Field(
        default_factory=list,
        description="Actionable next steps to resolve the error",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured context for correction hints/options",
    )


__all__ = [
    "SearchStatus",
    "SearchParameters",
    "SearchResultItem",
    "SearchResponse",
    "ProgressNotification",
    "ModuleInfo",
    "StatusResponse",
    "StructuredErrorResponse",
]
