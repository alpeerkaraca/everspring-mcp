"""Text utilities for EverSpring MCP."""

from __future__ import annotations

import re


def sanitize_title(title: str, replacement: str = "-") -> str:
    """Replace non-ASCII characters with a safe replacement.

    Args:
        title: Raw title string.
        replacement: Character to replace non-ASCII bytes with.

    Returns:
        Sanitized title.
    """
    if not title:
        return title

    sanitized = "".join(ch if ord(ch) < 128 else replacement for ch in title)
    sanitized = re.sub(rf"{re.escape(replacement)}{{2,}}", replacement, sanitized)
    return sanitized.strip()
