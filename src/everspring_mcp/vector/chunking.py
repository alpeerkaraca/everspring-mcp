"""EverSpring MCP - Markdown chunking utilities.

Hybrid chunking strategy:
1) Split by headings
2) Enforce max token size with overlap
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import tiktoken


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass(frozen=True)
class MarkdownChunk:
    """Represents a chunk of markdown content."""
    
    content: str
    section_path: str
    has_code: bool
    content_hash: str


class MarkdownChunker:
    """Chunk markdown by headings then enforce token size.
    
    Args:
        max_tokens: Max tokens per chunk
        overlap_tokens: Overlap between chunks
    """
    
    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._encoder = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))
    
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
        """Split text into token-limited chunks with overlap."""
        tokens = self._encoder.encode(text)
        if len(tokens) <= self.max_tokens:
            return [text]
        
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            if end == len(tokens):
                break
            start = max(0, end - self.overlap_tokens)
        
        return chunks
    
    def chunk(self, markdown: str) -> list[MarkdownChunk]:
        """Chunk markdown using hybrid strategy.
        
        Returns list of MarkdownChunk objects.
        """
        heading_sections = self._split_by_headings(markdown)
        chunks: list[MarkdownChunk] = []
        
        for section_path, content in heading_sections:
            for piece in self._split_by_tokens(content):
                has_code = "```" in piece or "`" in piece
                content_hash = self._hash_content(piece)
                chunks.append(MarkdownChunk(
                    content=piece.strip(),
                    section_path=section_path,
                    has_code=has_code,
                    content_hash=content_hash,
                ))
        
        return [c for c in chunks if c.content]

    @staticmethod
    def _hash_content(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["MarkdownChunk", "MarkdownChunker"]
