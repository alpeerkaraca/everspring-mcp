"""EverSpring MCP - Markdown chunking utilities.

Hybrid chunking strategy:
1) Clean known markdown artifacts
2) Split by headings
3) Enforce max token size with overlap and natural breakpoints
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass

from transformers import AutoTokenizer


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")

# Patterns to clean from Spring docs
CLEANUP_PATTERNS = [
    re.compile(r"Copied!$", re.MULTILINE),  # Copy button artifact
    re.compile(r"Copied!\s*```", re.MULTILINE),  # Copied! before closing fence
    re.compile(r"```\s*Copied!", re.MULTILINE),  # Copied! after code block
    re.compile(r"Copied!(?=\s|$)"),  # Inline copy button artifact
]


@dataclass(frozen=True)
class MarkdownChunk:
    """Represents a chunk of markdown content."""
    
    content: str
    section_path: str
    has_code: bool
    content_hash: str


class MarkdownChunker:
    """Chunk markdown by headings then enforce token size.
    
    Uses the actual tokenizer from the embedding model to ensure accurate
    token counting and proper chunk sizing.
    
    Args:
        model_name: HuggingFace model name (for tokenizer)
        max_tokens: Max tokens per chunk
        overlap_tokens: Overlap between chunks
    """
    
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))
    
    def _clean_content(self, text: str) -> str:
        """Remove Spring docs artifacts from content."""
        result = text
        for pattern in CLEANUP_PATTERNS:
            result = pattern.sub("", result)
        # Clean up extra whitespace
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()
    
    def _split_by_headings(self, markdown: str) -> list[tuple[str, str]]:
        """Split markdown into sections by headings.
        
        Returns list of (section_path, content).
        """
        lines = markdown.splitlines()
        sections: list[tuple[str, list[str]]] = []
        current_path: list[str] = []
        current_content: list[str] = []
        
        def flush() -> None:
            if current_content:
                sections.append((" > ".join(current_path), current_content.copy()))
                current_content.clear()
        
        for line in lines:
            match = HEADING_PATTERN.match(line.strip())
            if match:
                # New heading starts a new section
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                
                # Maintain heading hierarchy
                current_path = current_path[: level - 1]
                current_path.append(title)
                current_content.append(line)
            else:
                current_content.append(line)
        
        flush()
        return [(path, "\n".join(content)) for path, content in sections]
    
    def _split_by_tokens(self, text: str) -> list[str]:
        """Split text into token-limited chunks with natural boundaries."""
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_tokens:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_text = self._tokenizer.decode(tokens[start:end], skip_special_tokens=True).strip()

            if end < len(tokens):
                chunk_text = self._find_natural_break(chunk_text)
            if not chunk_text:
                chunk_text = self._tokenizer.decode(tokens[start:end], skip_special_tokens=True).strip()
            if not chunk_text:
                break

            chunks.append(chunk_text)
            if end == len(tokens):
                break

            used_tokens = len(self._tokenizer.encode(chunk_text, add_special_tokens=False))
            step = max(1, used_tokens - self.overlap_tokens)
            start += step

        return chunks

    def _find_natural_break(self, text: str) -> str:
        """Find a natural break point (paragraph, sentence, or code fence)."""
        if not text:
            return text

        # Avoid returning incomplete fenced code at chunk tail.
        if text.count("```") % 2 == 1:
            last_fence = text.rfind("```")
            if last_fence > len(text) * 0.4:
                trimmed = text[:last_fence].strip()
                if trimmed:
                    return trimmed

        search_start = len(text) // 2
        for marker in ("\n\n", "\n- ", "\n* ", ". ", "! ", "? ", "; "):
            pos = text.rfind(marker, search_start)
            if pos != -1:
                return text[: pos + len(marker)].strip()

        return text.strip()
    
    def chunk(self, markdown: str) -> list[MarkdownChunk]:
        """Chunk markdown using hybrid strategy.
        
        Returns list of MarkdownChunk objects.
        """
        # Clean content first
        cleaned = self._clean_content(markdown)
        heading_sections = self._split_by_headings(cleaned)
        chunks: list[MarkdownChunk] = []
        
        for section_path, content in heading_sections:
            for piece in self._split_by_tokens(content):
                stripped = piece.strip()
                if not stripped:
                    continue
                has_code = "```" in stripped or "`" in stripped
                content_hash = self._hash_content(stripped)
                chunks.append(MarkdownChunk(
                    content=stripped,
                    section_path=section_path,
                    has_code=has_code,
                    content_hash=content_hash,
                ))
        
        return chunks

    @staticmethod
    def _hash_content(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["MarkdownChunk", "MarkdownChunker"]
