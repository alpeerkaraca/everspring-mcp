"""EverSpring MCP - Spring version and module definitions.

This module provides Spring ecosystem version management:
- SpringModule: Enum for all supported Spring modules
- SpringVersion: Version model with minimum version validation
- VersionRange: Version constraints for compatibility checks
"""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import Field, model_validator

from everspring_mcp.models.base import VersionedModel


class SpringModule(str, Enum):
    """Supported Spring ecosystem modules."""
    
    BOOT = "spring-boot"
    FRAMEWORK = "spring-framework"
    SECURITY = "spring-security"
    DATA = "spring-data"
    CLOUD = "spring-cloud"
    AI = "spring-ai"
    
    @property
    def display_name(self) -> str:
        """Human-readable module name."""
        return self.value.replace("-", " ").title()
    
    @property
    def minimum_supported_version(self) -> int:
        """Minimum major version supported by EverSpring."""
        minimums = {
            SpringModule.BOOT: 4,
            SpringModule.FRAMEWORK: 7,
            SpringModule.SECURITY: 6,
            SpringModule.DATA: 4,
            SpringModule.CLOUD: 4,
            SpringModule.AI: 1
        }
        return minimums.get(self, 1)


class SpringVersion(VersionedModel):
    """Spring module version with validation.
    
    Enforces minimum version requirements:
    - Spring Boot: 4+
    - Spring Framework: 7+
    - Spring Security: 6+
    - Spring Data: 4+
    - Spring Cloud: 4+
    - Spring AI: 1+
    """
    
    module: SpringModule = Field(
        description="The Spring module this version applies to",
    )
    major: int = Field(
        ge=1,
        description="Major version number",
    )
    minor: int = Field(
        default=0,
        ge=0,
        description="Minor version number",
    )
    patch: int = Field(
        default=0,
        ge=0,
        description="Patch version number",
    )
    qualifier: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9\-\.]+$",
        description="Version qualifier (e.g., 'RELEASE', 'M1', 'RC1')",
    )
    
    @model_validator(mode="after")
    def validate_minimum_version(self) -> Self:
        """Ensure version meets minimum requirements for the module."""
        min_version = self.module.minimum_supported_version
        if self.major < min_version:
            raise ValueError(
                f"{self.module.display_name} requires version {min_version}+, "
                f"got {self.major}.{self.minor}.{self.patch}"
            )
        return self
    
    @property
    def version_tuple(self) -> tuple[int, int, int]:
        """Version as comparable tuple."""
        return (self.major, self.minor, self.patch)
    
    @property
    def version_string(self) -> str:
        """Version as string (e.g., '4.0.0' or '4.0.0-RELEASE')."""
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.qualifier:
            return f"{base}-{self.qualifier}"
        return base
    
    def __str__(self) -> str:
        return f"{self.module.value}:{self.version_string}"
    
    def __lt__(self, other: SpringVersion) -> bool:
        if not isinstance(other, SpringVersion):
            return NotImplemented
        if self.module != other.module:
            raise ValueError("Cannot compare versions of different modules")
        return self.version_tuple < other.version_tuple
    
    def __le__(self, other: SpringVersion) -> bool:
        return self == other or self < other
    
    def __gt__(self, other: SpringVersion) -> bool:
        if not isinstance(other, SpringVersion):
            return NotImplemented
        if self.module != other.module:
            raise ValueError("Cannot compare versions of different modules")
        return self.version_tuple > other.version_tuple
    
    def __ge__(self, other: SpringVersion) -> bool:
        return self == other or self > other
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpringVersion):
            return NotImplemented
        return (
            self.module == other.module
            and self.version_tuple == other.version_tuple
        )
    
    def __hash__(self) -> int:
        return hash((self.module, self.version_tuple))
    
    @classmethod
    def parse(cls, module: SpringModule, version_str: str) -> SpringVersion:
        """Parse version string into SpringVersion.
        
        Args:
            module: The Spring module
            version_str: Version string (e.g., '4.0.0', '4.0.0-RELEASE')
            
        Returns:
            Parsed SpringVersion instance
        """
        parts = version_str.split("-", 1)
        version_parts = parts[0].split(".")
        
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0
        qualifier = parts[1] if len(parts) > 1 else None
        
        return cls(
            module=module,
            major=major,
            minor=minor,
            patch=patch,
            qualifier=qualifier,
        )


class VersionRange(VersionedModel):
    """Version range constraint for compatibility checks."""
    
    module: SpringModule = Field(
        description="The Spring module this range applies to",
    )
    min_version: SpringVersion | None = Field(
        default=None,
        description="Minimum version (inclusive)",
    )
    max_version: SpringVersion | None = Field(
        default=None,
        description="Maximum version (inclusive)",
    )
    
    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """Ensure range is valid and modules match."""
        if self.min_version and self.min_version.module != self.module:
            raise ValueError("min_version module must match range module")
        if self.max_version and self.max_version.module != self.module:
            raise ValueError("max_version module must match range module")
        if self.min_version and self.max_version:
            if self.min_version > self.max_version:
                raise ValueError("min_version cannot be greater than max_version")
        return self
    
    def contains(self, version: SpringVersion) -> bool:
        """Check if version falls within this range.
        
        Args:
            version: The version to check
            
        Returns:
            True if version is within range (inclusive)
        """
        if version.module != self.module:
            return False
        if self.min_version and version < self.min_version:
            return False
        if self.max_version and version > self.max_version:
            return False
        return True
    
    def __str__(self) -> str:
        min_str = self.min_version.version_string if self.min_version else "*"
        max_str = self.max_version.version_string if self.max_version else "*"
        return f"{self.module.value}:[{min_str}, {max_str}]"


__all__ = [
    "SpringModule",
    "SpringVersion",
    "VersionRange",
]
