"""EverSpring MCP - Builder pattern for LLM prompt construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from everspring_mcp.models.metadata import SearchResult


class PromptBuilder:
    """Builder for constructing strictly formatted LLM contexts."""

    def __init__(self) -> None:
        self._system_prompt: str | None = None
        self._query: str | None = None
        self._results: list[SearchResult] = []
        self._filters: dict[str, Any] = {}

    def add_system_prompt(self, text: str) -> PromptBuilder:
        """Set the initial system instructions."""
        self._system_prompt = text.strip()
        return self

    def add_user_query(self, query: str) -> PromptBuilder:
        """Set the user's search query."""
        self._query = query.strip()
        return self

    def add_filters(self, **filters: Any) -> PromptBuilder:
        """Add metadata filters used in the search."""
        self._filters.update(filters)
        return self

    def add_retrieved_context(self, results: list[Any]) -> PromptBuilder:
        """Add retrieved search results to the context."""
        self._results.extend(results)
        return self

    def build(self) -> str:
        """Assemble the final prompt context as a Markdown string."""
        lines: list[str] = []

        if self._system_prompt:
            lines.extend([self._system_prompt, ""])

        lines.append("## Spring Docs Search Results")
        lines.append("")

        if self._query:
            lines.append(f"**Query:** {self._query}")

        filter_strs: list[str] = []
        for key, value in self._filters.items():
            if value is not None:
                filter_strs.append(f"{key}={value}")

        if filter_strs:
            lines.append(f"**Filters:** {', '.join(filter_strs)}")

        lines.append("")

        if not self._results:
            lines.append("No relevant documentation chunks found.")
        else:
            for index, result in enumerate(self._results, start=1):
                # Handle both SearchResult (raw) and SearchResultItem (SDK model)
                if hasattr(result, "version"):
                    version = result.version
                elif hasattr(result, "version_major") and hasattr(
                    result, "version_minor"
                ):
                    version = f"v{result.version_major}.{result.version_minor}"
                else:
                    version = "unknown"

                module = getattr(result, "module", "unknown")
                submodule = getattr(result, "submodule", None)
                if submodule:
                    module = f"{module}/{submodule}"

                title = getattr(result, "title", "Untitled")
                url = getattr(result, "url", "#")
                score = getattr(result, "score", 0.0)
                section = getattr(
                    result, "section_path", getattr(result, "title", "Unknown Section")
                )
                content = getattr(result, "content", "").strip()

                lines.extend(
                    [
                        f"### {index}. {title}",
                        f"- **URL:** {url}",
                        f"- **Module:** {module}",
                        f"- **Version:** {version}",
                        f"- **Score:** {score:.4f}",
                        f"- **Section:** {section}",
                        "",
                        content,
                        "",
                    ]
                )

        return "\n".join(lines).strip()
