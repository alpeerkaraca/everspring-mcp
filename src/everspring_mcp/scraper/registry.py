"""Submodule registry for multi-module documentation targets."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from everspring_mcp.models.spring import SpringModule


class SubmoduleTarget(BaseModel):
    """Registry entry for a module or submodule documentation target."""

    model_config = ConfigDict(frozen=True)

    module_key: SpringModule = Field(description="Spring module key")
    base_url: str = Field(
        pattern=r"^https?://[^\s]+$",
        description="Base URL for the documentation entry point",
    )
    submodule_key: str | None = Field(
        default=None,
        description="Optional submodule key (e.g., redis)",
    )
    version_selector: str = Field(
        default="span.version",
        description="CSS selector for version extraction",
    )


class SubmoduleRegistry(BaseModel):
    """Registry of module/submodule targets."""

    model_config = ConfigDict(frozen=True)

    targets: list[SubmoduleTarget] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> SubmoduleRegistry:
        """Load registry from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)


__all__ = ["SubmoduleRegistry", "SubmoduleTarget"]
