"""EverSpring MCP - Spring documentation HTML parser.

This module provides SpringDocParser for parsing Spring documentation:
- HTML extraction (sidebar, main content, code blocks)
- Markdown conversion using markdownify
- Section hierarchy building
- ScrapedPage model creation
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag
from markdownify import MarkdownConverter
from pydantic import BaseModel, ConfigDict, Field

from everspring_mcp.models.content import (
    ApiClassSignature,
    CodeExample,
    CodeLanguage,
    ContentType,
    DocumentSection,
    MethodParameter,
    MethodSignature,
    ScrapedPage,
)
from everspring_mcp.models.spring import SpringModule, SpringVersion
from everspring_mcp.scraper.exceptions import ContentExtractionError
from everspring_mcp.utils import sanitize_title
from everspring_mcp.utils.logging import get_logger

logger = get_logger("scraper.parser")


# Default CSS selectors for Spring documentation structure
SPRING_DOC_SELECTORS: dict[str, list[str]] = {
    "title": [
        "h1.title",
        "article h1",
        "main h1",
        "#content h1",
        "meta[property='og:title']",
        "title",
    ],
    "sidebar": [
        "nav.toc",
        "aside.toc",
        ".book-toc",
        "#toc",
        ".nav-toc",
        "nav[role='navigation']",
        ".nav-list"
    ],
    "main_content": [
        "article.doc",
        "main.content",
        "#content",
        "article",
        "main",
        ".doc-content",
    ],
    "code_blocks": [
        "pre code",
        ".listingblock pre",
        ".highlight pre",
        "pre.programlisting",
        "pre.code",
    ],
    "version": [
        "span.version",
    ],
}

# Language detection patterns for code blocks
LANGUAGE_PATTERNS: dict[str, CodeLanguage] = {
    r"java": CodeLanguage.JAVA,
    r"kotlin|kt": CodeLanguage.KOTLIN,
    r"groovy": CodeLanguage.GROOVY,
    r"xml": CodeLanguage.XML,
    r"ya?ml": CodeLanguage.YAML,
    r"properties|props": CodeLanguage.PROPERTIES,
    r"json": CodeLanguage.JSON,
    r"sh|bash|shell|zsh": CodeLanguage.SHELL,
    r"sql": CodeLanguage.SQL,
}

# Spring annotation pattern for extraction
ANNOTATION_PATTERN = re.compile(r"@(\w+)(?:\([^)]*\))?")

# Java modifiers used in class/method signature extraction
JAVA_MODIFIERS = {
    "public",
    "protected",
    "private",
    "static",
    "final",
    "abstract",
    "default",
    "synchronized",
    "native",
    "strictfp",
    "sealed",
    "non-sealed",
    "transient",
    "volatile",
}


class ParserConfig(BaseModel):
    """Configuration for SpringDocParser.
    
    Attributes:
        title_selectors: CSS selectors for page title
        sidebar_selectors: CSS selectors for navigation sidebar
        content_selectors: CSS selectors for main content
        code_selectors: CSS selectors for code blocks
        strip_tags: HTML tags to remove entirely
        unwrap_tags: HTML tags to unwrap (keep content, remove tag)
    """

    model_config = ConfigDict(frozen=True)

    title_selectors: list[str] = Field(
        default_factory=lambda: SPRING_DOC_SELECTORS["title"].copy(),
        description="CSS selectors for page title",
    )
    sidebar_selectors: list[str] = Field(
        default_factory=lambda: SPRING_DOC_SELECTORS["sidebar"].copy(),
        description="CSS selectors for sidebar navigation",
    )
    content_selectors: list[str] = Field(
        default_factory=lambda: SPRING_DOC_SELECTORS["main_content"].copy(),
        description="CSS selectors for main content",
    )
    code_selectors: list[str] = Field(
        default_factory=lambda: SPRING_DOC_SELECTORS["code_blocks"].copy(),
        description="CSS selectors for code blocks",
    )
    version_selectors: list[str] = Field(
        default_factory=lambda: SPRING_DOC_SELECTORS["version"].copy(),
        description="CSS selectors for version extraction",
    )
    strip_tags: list[str] = Field(
        default_factory=lambda: ["script", "style", "noscript", "iframe"],
        description="Tags to remove entirely",
    )
    unwrap_tags: list[str] = Field(
        default_factory=lambda: ["span", "div"],
        description="Tags to unwrap (keep content)",
    )
    noise_selectors: list[str] = Field(
        default_factory=lambda: [
            "nav",
            "footer",
            "aside",
            ".toc",
            ".book-toc",
            "#toc",
            ".nav-toc",
            ".edit-link",
            ".edit-page-link",
            ".pagination",
            ".pager",
            ".pagination-controls",
            "a[rel='next']",
            "a[rel='prev']",
            ".next-page",
            ".prev-page",
            ".pagination-next",
            ".pagination-prev",
            "button.next",
            "button.prev",
            "button[aria-label='Next']",
            "button[aria-label='Previous']",
        ],
        description="CSS selectors for noisy navigation/editor/pagination elements",
    )


class SpringMarkdownConverter(MarkdownConverter):
    """Custom Markdown converter for Spring documentation.
    
    Handles Spring-specific elements and preserves code formatting.
    """

    def convert_pre(self, el: Tag, text: str, convert_as_inline: bool = False, parent_tags: list | None = None) -> str:
        """Convert pre element preserving code language."""
        # Try to detect language from various attributes
        lang = ""
        code_el = el.find("code")

        if code_el and isinstance(code_el, Tag):
            # Check class for language hint
            classes = code_el.get("class", [])
            if isinstance(classes, list):
                for cls in classes:
                    if cls.startswith("language-"):
                        lang = cls.replace("language-", "")
                        break
                    elif cls.startswith("highlight-"):
                        lang = cls.replace("highlight-", "")
                        break

            # Check data-lang attribute
            if not lang:
                lang = code_el.get("data-lang", "") or ""

        # Also check pre element itself
        if not lang:
            pre_classes = el.get("class", [])
            if isinstance(pre_classes, list):
                for cls in pre_classes:
                    for pattern, code_lang in LANGUAGE_PATTERNS.items():
                        if re.search(pattern, cls, re.IGNORECASE):
                            lang = code_lang.value
                            break

        # Get code content
        code = el.get_text()

        return f"\n```{lang}\n{code.strip()}\n```\n"

    def convert_a(self, el: Tag, text: str, convert_as_inline: bool = False, parent_tags: list | None = None) -> str:
        """Convert anchor with proper link handling."""
        href = el.get("href", "")
        title = el.get("title", "")

        if not href:
            return text

        # Handle anchor-only links
        if href.startswith("#"):
            return f"[{text}]({href})"

        if title:
            return f'[{text}]({href} "{title}")'
        return f"[{text}]({href})"

    def convert_table(self, el: Tag, text: str, convert_as_inline: bool = False, parent_tags: list | None = None) -> str:
        """Convert table to Markdown table format."""
        rows = el.find_all("tr")
        if not rows:
            return text

        result = []
        for i, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            row_text = " | ".join(cell.get_text(strip=True) for cell in cells)
            result.append(f"| {row_text} |")

            # Add header separator after first row
            if i == 0:
                separator = " | ".join("---" for _ in cells)
                result.append(f"| {separator} |")

        return "\n" + "\n".join(result) + "\n"


class SpringDocParser:
    """Parser for Spring documentation HTML.
    
    Extracts content from Spring docs and converts to structured
    Pydantic models with clean Markdown.
    
    Usage:
        parser = SpringDocParser()
        scraped_page = parser.parse(
            html=html_content,
            url="https://docs.spring.io/...",
            module=SpringModule.BOOT,
            version=SpringVersion(module=SpringModule.BOOT, major=4),
        )
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        """Initialize parser.
        
        Args:
            config: Parser configuration. Uses defaults if not provided.
        """
        self.config = config or ParserConfig()
        self._converter = SpringMarkdownConverter(
            heading_style="atx",
            bullets="-",
            strong_em_symbol="*",
            code_language_callback=self._detect_language,
        )

    def _detect_language(self, el: Tag) -> str:
        """Detect programming language from element attributes."""
        classes = el.get("class", [])
        if isinstance(classes, list):
            for cls in classes:
                for pattern, lang in LANGUAGE_PATTERNS.items():
                    if re.search(pattern, cls, re.IGNORECASE):
                        return lang.value
        return ""

    def _create_soup(self, html: str) -> BeautifulSoup:
        """Create BeautifulSoup object from HTML."""
        return BeautifulSoup(html, "lxml")

    def _find_first(
        self,
        soup: BeautifulSoup,
        selectors: list[str],
    ) -> Tag | None:
        """Find first matching element from list of selectors."""
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element
        return None

    def _clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove unwanted elements from HTML."""
        # Remove script, style, etc.
        for tag_name in self.config.strip_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        protected_version_nodes: list[Tag] = []
        seen_version_nodes: set[int] = set()
        for selector in self.config.version_selectors:
            for node in soup.select(selector):
                if not isinstance(node, Tag):
                    continue
                node_id = id(node)
                if node_id in seen_version_nodes:
                    continue
                seen_version_nodes.add(node_id)
                protected_version_nodes.append(node)

        removed_noise = 0
        for selector in self.config.noise_selectors:
            for node in soup.select(selector):
                if not isinstance(node, Tag):
                    continue
                # Preserve code examples even when class names overlap.
                if node.find_parent(["pre", "code"]):
                    continue
                # Keep version marker nodes and move them out of noisy containers.
                if node in protected_version_nodes:
                    continue
                protected_descendants = [
                    protected for protected in protected_version_nodes if node in protected.parents
                ]
                for protected in protected_descendants:
                    node.insert_before(protected.extract())
                node.decompose()
                removed_noise += 1

        if removed_noise:
            logger.debug("Pruned %d noisy nodes before markdown conversion", removed_noise)

        return soup

    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Page title string
            
        Raises:
            ContentExtractionError: If title cannot be found
        """
        # Try meta tag first for og:title
        meta = soup.find("meta", property="og:title")
        if meta and isinstance(meta, Tag):
            content = meta.get("content")
            if content:
                return str(content).strip()

        # Try configured selectors
        for selector in self.config.title_selectors:
            if selector.startswith("meta"):
                continue
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)

        raise ContentExtractionError("Could not extract page title")

    def extract_version(self, soup: BeautifulSoup, selectors: list[str] | None = None) -> str:
        """Extract documentation version from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            Version string (e.g., "4.0.5")

        Raises:
            ContentExtractionError: If version cannot be found
        """
        element = self._find_first(soup, selectors or self.config.version_selectors)
        if element:
            text = element.get_text(strip=True)
            if text:
                return text
        raise ContentExtractionError("Could not extract version from page")

    def extract_sidebar(self, soup: BeautifulSoup) -> list[dict]:
        """Extract sidebar navigation hierarchy.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of navigation items with title, url, and children
        """
        sidebar = self._find_first(soup, self.config.sidebar_selectors)
        if not sidebar:
            logger.warning("No sidebar found")
            return []

        def parse_nav_items(element: Tag) -> list[dict]:
            items = []
            for li in element.find_all("li", recursive=False):
                item: dict = {}

                # Find link
                link = li.find("a")
                if link and isinstance(link, Tag):
                    item["title"] = link.get_text(strip=True)
                    item["url"] = link.get("href", "")

                # Find nested list
                nested = li.find(["ul", "ol"])
                if nested and isinstance(nested, Tag):
                    item["children"] = parse_nav_items(nested)
                else:
                    item["children"] = []

                if item.get("title"):
                    items.append(item)

            return items

        # Find the main list
        nav_list = sidebar.find(["ul", "ol"])
        if nav_list and isinstance(nav_list, Tag):
            return parse_nav_items(nav_list)

        return []

    def extract_main_content(self, soup: BeautifulSoup) -> Tag:
        """Extract main content element.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Main content Tag element
            
        Raises:
            ContentExtractionError: If content cannot be found
        """
        content = self._find_first(soup, self.config.content_selectors)
        if not content:
            raise ContentExtractionError("Could not extract main content")
        return content

    def extract_code_blocks(self, soup: BeautifulSoup) -> list[CodeExample]:
        """Extract code blocks from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of CodeExample models
        """
        examples = []

        for selector in self.config.code_selectors:
            for element in soup.select(selector):
                # Get the pre element
                pre = element if element.name == "pre" else element.find_parent("pre")
                if not pre:
                    continue

                code = element.get_text()
                if not code.strip():
                    continue

                # Detect language
                lang = self._detect_code_language(element)

                # Extract annotations from Java/Kotlin code
                annotations = []
                if lang in (CodeLanguage.JAVA, CodeLanguage.KOTLIN):
                    annotations = ANNOTATION_PATTERN.findall(code)

                # Get title from preceding element
                title = None
                prev_sibling = pre.find_previous_sibling()
                if prev_sibling and isinstance(prev_sibling, Tag):
                    if prev_sibling.name in ("p", "div"):
                        title_text = prev_sibling.get_text(strip=True)
                        if len(title_text) < 100:  # Reasonable title length
                            title = title_text

                try:
                    example = CodeExample(
                        language=lang,
                        code=code.strip(),
                        title=title,
                        annotations=list(set(annotations)),
                    )
                    examples.append(example)
                except Exception as e:
                    logger.warning(f"Failed to create CodeExample: {e}")

        return examples

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse repeated whitespace in extracted text."""
        return re.sub(r"\s+", " ", text).strip()

    def _split_top_level(self, text: str, delimiter: str = ",") -> list[str]:
        """Split text on delimiter while preserving nested generic/type expressions."""
        parts: list[str] = []
        current: list[str] = []
        angle = 0
        paren = 0
        bracket = 0

        for char in text:
            if char == "<":
                angle += 1
            elif char == ">" and angle > 0:
                angle -= 1
            elif char == "(":
                paren += 1
            elif char == ")" and paren > 0:
                paren -= 1
            elif char == "[":
                bracket += 1
            elif char == "]" and bracket > 0:
                bracket -= 1

            if (
                char == delimiter
                and angle == 0
                and paren == 0
                and bracket == 0
            ):
                segment = "".join(current).strip()
                if segment:
                    parts.append(segment)
                current = []
                continue

            current.append(char)

        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    def _extract_javadoc_package(self, soup: BeautifulSoup) -> str:
        """Extract package name from a Javadoc page header."""
        selectors = [
            "div.header div.sub-title",
            "div.header div.subTitle",
            "div.sub-title",
            "div.subTitle",
            "header div.sub-title",
        ]
        package_pattern = re.compile(r"\bpackage\s+([A-Za-z_][\w.]*)")

        for selector in selectors:
            for element in soup.select(selector):
                text = self._normalize_whitespace(element.get_text(" ", strip=True))
                match = package_pattern.search(text)
                if match:
                    return match.group(1)

                if re.fullmatch(r"[A-Za-z_][\w.]*(?:\.[A-Za-z_][\w.]*)*", text):
                    return text

        package_link = soup.select_one("a[href$='package-summary.html']")
        if package_link and isinstance(package_link, Tag):
            text = self._normalize_whitespace(package_link.get_text(" ", strip=True))
            if re.fullmatch(r"[A-Za-z_][\w.]*(?:\.[A-Za-z_][\w.]*)*", text):
                return text

        raise ContentExtractionError("Could not extract Javadoc package name")

    def _extract_type_signature_text(self, soup: BeautifulSoup) -> str:
        """Extract class/interface declaration text from a Javadoc page."""
        selectors = [
            ".type-signature",
            ".typeSignature",
            "section.class-description pre",
            "div.class-description pre",
            "div.description > pre",
            "li.blockList > pre",
            "li.block-list > pre",
        ]

        for selector in selectors:
            for element in soup.select(selector):
                text = self._normalize_whitespace(element.get_text(" ", strip=True))
                if not text:
                    continue
                if re.search(r"\b(@interface|class|interface|enum)\b", text):
                    return text

        # Fallback for pages where declaration is rendered in header text.
        title = soup.select_one("h1.title, h1")
        if title and isinstance(title, Tag):
            title_text = self._normalize_whitespace(title.get_text(" ", strip=True))
            match = re.search(r"\b(Class|Interface|Enum|Annotation)\s+([A-Za-z_]\w*)", title_text)
            if match:
                mapped = {
                    "Class": "class",
                    "Interface": "interface",
                    "Enum": "enum",
                    "Annotation": "@interface",
                }[match.group(1)]
                return f"public {mapped} {match.group(2)}"

        raise ContentExtractionError("Could not extract Javadoc class/interface declaration")

    def _parse_class_signature(
        self,
        signature_text: str,
    ) -> tuple[str, str, list[str], str | None, list[str], list[str]]:
        """Parse class/interface declaration into structured parts."""
        signature = self._normalize_whitespace(signature_text)
        kind_match = re.search(r"\b(@interface|class|interface|enum)\b", signature)
        if not kind_match:
            raise ContentExtractionError("Javadoc declaration missing class/interface keyword")

        kind_token = kind_match.group(1)
        signature_type = "annotation" if kind_token == "@interface" else kind_token

        prefix = signature[:kind_match.start()].strip()
        suffix = signature[kind_match.end():].strip()

        annotations = re.findall(r"@\w+(?:\([^)]*\))?", prefix)
        prefix_no_annotations = re.sub(r"@\w+(?:\([^)]*\))?", "", prefix).strip()
        modifiers = [token for token in prefix_no_annotations.split() if token in JAVA_MODIFIERS]

        name_match = re.match(r"([A-Za-z_]\w*)", suffix)
        if not name_match:
            raise ContentExtractionError("Javadoc declaration missing class/interface name")

        class_name = name_match.group(1)
        inheritance = suffix[name_match.end():].strip()

        extends: str | None = None
        implements: list[str] = []

        extends_match = re.search(r"\bextends\b\s+(.+?)(?=\bimplements\b|$)", inheritance)
        if extends_match:
            extends = self._normalize_whitespace(extends_match.group(1).strip(" ,"))

        implements_match = re.search(r"\bimplements\b\s+(.+)$", inheritance)
        if implements_match:
            implements = [
                self._normalize_whitespace(item)
                for item in self._split_top_level(implements_match.group(1))
                if item.strip()
            ]
        elif signature_type == "interface" and extends_match:
            # Interfaces use "extends" for superinterfaces; keep them in the same
            # field so retrieval can still reason over interface contracts.
            implements = [
                self._normalize_whitespace(item)
                for item in self._split_top_level(extends_match.group(1))
                if item.strip()
            ]

        return class_name, signature_type, modifiers, extends, implements, annotations

    def _extract_method_tag_docs(self, container: Tag) -> tuple[dict[str, str], str | None, list[str]]:
        """Extract @param, @return, and throws docs from a Javadoc method block."""
        param_docs: dict[str, str] = {}
        return_doc: str | None = None
        throws: list[str] = []

        for dl in container.select("dl"):
            current_label: str | None = None
            for node in dl.children:
                if not isinstance(node, Tag):
                    continue

                if node.name == "dt":
                    current_label = self._normalize_whitespace(node.get_text(" ", strip=True)).lower().rstrip(":")
                    continue

                if node.name != "dd" or not current_label:
                    continue

                text = self._normalize_whitespace(node.get_text(" ", strip=True))
                if not text:
                    continue

                if current_label.startswith("parameters"):
                    code = node.find("code")
                    param_name = ""
                    if code and isinstance(code, Tag):
                        param_name = self._normalize_whitespace(code.get_text(" ", strip=True))
                    if not param_name:
                        match = re.match(r"([A-Za-z_]\w*)\s*(?:-|:)\s*(.*)", text)
                        if match:
                            param_name = match.group(1)
                            text = match.group(2)
                    if param_name:
                        cleaned = text
                        if cleaned.startswith(param_name):
                            cleaned = cleaned[len(param_name):].lstrip(" -:")
                        param_docs[param_name] = cleaned.strip()

                elif current_label.startswith("returns"):
                    return_doc = text

                elif current_label.startswith("throws") or current_label.startswith("exception"):
                    code = node.find("code")
                    throws_name = ""
                    if code and isinstance(code, Tag):
                        throws_name = self._normalize_whitespace(code.get_text(" ", strip=True))
                    if not throws_name:
                        throws_match = re.match(r"([A-Za-z_][\w.]*)", text)
                        if throws_match:
                            throws_name = throws_match.group(1)
                    if throws_name and throws_name not in throws:
                        throws.append(throws_name)

        return param_docs, return_doc, throws

    def _parse_method_parameters(
        self,
        parameter_text: str,
        param_docs: dict[str, str],
    ) -> list[MethodParameter]:
        """Parse a Java method parameter list."""
        if not parameter_text or parameter_text == "()":
            return []

        parameters: list[MethodParameter] = []
        for raw_param in self._split_top_level(parameter_text):
            cleaned = self._normalize_whitespace(raw_param)
            if not cleaned:
                continue

            cleaned = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", cleaned)
            cleaned = re.sub(r"\b(final|volatile|transient)\b\s+", "", cleaned)

            name_match = re.search(r"([A-Za-z_]\w*)\s*$", cleaned)
            if not name_match:
                continue

            param_name = name_match.group(1)
            param_type = cleaned[:name_match.start()].strip()
            if not param_type:
                continue

            parameters.append(
                MethodParameter(
                    name=param_name,
                    type=param_type,
                    description=param_docs.get(param_name) or None,
                )
            )

        return parameters

    def _parse_method_signature(
        self,
        signature_text: str,
        param_docs: dict[str, str],
        return_doc: str | None,
        throws_docs: list[str],
        *,
        is_deprecated: bool,
    ) -> MethodSignature | None:
        """Parse a Java method declaration line into a MethodSignature model."""
        signature = self._normalize_whitespace(signature_text).rstrip(";")
        if "(" not in signature or ")" not in signature:
            return None

        open_idx = signature.find("(")
        close_idx = signature.rfind(")")
        if close_idx <= open_idx:
            return None

        method_head = signature[:open_idx].strip()
        params_text = signature[open_idx + 1 : close_idx].strip()
        trailing = signature[close_idx + 1 :].strip()

        name_match = re.search(r"([A-Za-z_]\w*)\s*$", method_head)
        if not name_match:
            return None

        method_name = name_match.group(1)
        prefix = method_head[:name_match.start()].strip()
        if not prefix:
            return None

        annotations = re.findall(r"@\w+(?:\([^)]*\))?", prefix)
        prefix = re.sub(r"@\w+(?:\([^)]*\))?", "", prefix).strip()

        tokens = prefix.split()
        modifiers = [token for token in tokens if token in JAVA_MODIFIERS]
        non_modifier = [token for token in tokens if token not in JAVA_MODIFIERS]
        return_type = " ".join(non_modifier).strip()

        if return_type.startswith("<"):
            generic_depth = 0
            split_index = 0
            for idx, char in enumerate(return_type):
                if char == "<":
                    generic_depth += 1
                elif char == ">" and generic_depth > 0:
                    generic_depth -= 1
                    if generic_depth == 0:
                        split_index = idx + 1
                        break
            return_type = return_type[split_index:].strip()

        if not return_type:
            return None

        throws = list(throws_docs)
        throws_match = re.search(r"\bthrows\b\s+(.+)$", trailing)
        if throws_match:
            for item in self._split_top_level(throws_match.group(1)):
                throw_item = self._normalize_whitespace(item)
                if throw_item and throw_item not in throws:
                    throws.append(throw_item)

        return MethodSignature(
            name=method_name,
            return_type=return_type,
            return_description=return_doc,
            parameters=self._parse_method_parameters(params_text, param_docs),
            modifiers=modifiers,
            annotations=annotations,
            is_deprecated=is_deprecated,
            throws=throws,
        )

    def extract_api_signature(self, javadoc_html: str | BeautifulSoup | Tag) -> ApiClassSignature:
        """Extract high-signal class/interface + method signatures from Javadoc HTML.

        This intentionally ignores descriptive prose and keeps only:
        - Class/interface definition
        - Implemented interfaces
        - Method signatures with @param/@return tags
        """
        if isinstance(javadoc_html, BeautifulSoup):
            soup = javadoc_html
        elif isinstance(javadoc_html, Tag):
            soup = self._create_soup(str(javadoc_html))
        else:
            soup = self._create_soup(javadoc_html)

        soup = self._clean_html(soup)

        package_name = self._extract_javadoc_package(soup)
        declaration_text = self._extract_type_signature_text(soup)
        class_name, signature_type, modifiers, extends, implements, annotations = self._parse_class_signature(
            declaration_text
        )

        method_sections = soup.select(
            "section.method-details section.detail, "
            "section.method-details li.block-list, "
            "section.method-details li.blockList, "
            "section.methodDetail, "
            "section.methodDetails li.blockList"
        )

        methods: list[MethodSignature] = []
        seen_method_keys: set[tuple[str, str, str]] = set()
        for section in method_sections:
            if not isinstance(section, Tag):
                continue

            signature_el = section.select_one(".member-signature, .memberSignature, pre")
            if not signature_el or not isinstance(signature_el, Tag):
                continue

            signature_text = self._normalize_whitespace(signature_el.get_text(" ", strip=True))
            if not signature_text:
                continue

            param_docs, return_doc, throws_docs = self._extract_method_tag_docs(section)
            deprecated_text = self._normalize_whitespace(section.get_text(" ", strip=True)).lower()
            method = self._parse_method_signature(
                signature_text,
                param_docs,
                return_doc,
                throws_docs,
                is_deprecated="deprecated" in deprecated_text,
            )
            if not method:
                continue

            key = (
                method.name,
                ",".join(param.type for param in method.parameters),
                method.return_type,
            )
            if key in seen_method_keys:
                continue
            seen_method_keys.add(key)
            methods.append(method)

        class_is_deprecated = (
            any(annotation == "@Deprecated" for annotation in annotations)
            or bool(soup.select_one(".deprecated-label, .deprecatedLabel, .deprecation-block, .deprecated"))
        )

        return ApiClassSignature(
            name=class_name,
            package=package_name,
            type=signature_type,
            modifiers=modifiers,
            extends=extends,
            implements=implements,
            annotations=annotations,
            methods=methods,
            is_deprecated=class_is_deprecated,
        )

    def _api_signature_to_markdown(self, signature: ApiClassSignature) -> str:
        """Render API signature model into compact, high-signal markdown."""
        definition_parts = [*signature.annotations, *signature.modifiers]
        keyword = "@interface" if signature.type == "annotation" else signature.type
        definition_parts.extend([keyword, signature.name])

        if signature.extends:
            definition_parts.extend(["extends", signature.extends])
        if signature.implements:
            definition_parts.extend(["implements", ", ".join(signature.implements)])

        lines = [
            f"# {signature.fully_qualified_name}",
            "",
            "## Type Definition",
            f"`{' '.join(part for part in definition_parts if part).strip()}`",
            "",
        ]

        if signature.implements:
            lines.append("## Implemented Interfaces")
            for interface_name in signature.implements:
                lines.append(f"- `{interface_name}`")
            lines.append("")

        lines.append("## Method Signatures")
        for method in signature.methods:
            method_parts = [*method.annotations, *method.modifiers, method.return_type]
            params = ", ".join(f"{param.type} {param.name}" for param in method.parameters)
            method_line = f"`{' '.join(part for part in method_parts if part).strip()} {method.name}({params})`"
            lines.append(f"- {method_line}")

            for param in method.parameters:
                if param.description:
                    lines.append(f"  - `@param {param.name}`: {param.description}")

            if method.return_description:
                lines.append(f"  - `@return`: {method.return_description}")

            for throw_name in method.throws:
                lines.append(f"  - `@throws {throw_name}`")

        return "\n".join(lines).strip()

    def _detect_code_language(self, element: Tag) -> CodeLanguage:
        """Detect programming language from code element."""
        # Check class attributes
        classes = element.get("class", [])
        if isinstance(classes, str):
            classes = classes.split()

        for cls in classes:
            # Common patterns: language-java, highlight-java, java
            clean_cls = cls.lower()
            for prefix in ("language-", "highlight-", "lang-"):
                if clean_cls.startswith(prefix):
                    clean_cls = clean_cls[len(prefix):]
                    break

            for pattern, lang in LANGUAGE_PATTERNS.items():
                if re.search(pattern, clean_cls, re.IGNORECASE):
                    return lang

        # Check data-lang attribute
        data_lang = element.get("data-lang", "")
        if data_lang:
            for pattern, lang in LANGUAGE_PATTERNS.items():
                if re.search(pattern, str(data_lang), re.IGNORECASE):
                    return lang

        # Check parent pre element
        pre = element.find_parent("pre")
        if pre and isinstance(pre, Tag):
            pre_classes = pre.get("class", [])
            if isinstance(pre_classes, str):
                pre_classes = pre_classes.split()
            for cls in pre_classes:
                for pattern, lang in LANGUAGE_PATTERNS.items():
                    if re.search(pattern, cls, re.IGNORECASE):
                        return lang

        # Default to Java for Spring docs
        return CodeLanguage.JAVA

    def to_markdown(self, html: str | Tag) -> str:
        """Convert HTML to clean Markdown.
        
        Args:
            html: HTML string or BeautifulSoup Tag
            
        Returns:
            Clean Markdown string
        """
        clean_soup = self._create_soup(str(html))
        clean_soup = self._clean_html(clean_soup)

        # Use custom converter
        md = self._converter.convert(str(clean_soup))

        # Clean up Spring docs copy-button artifacts
        md = re.sub(r"Copied!(?=\s|$)", "", md)

        # Clean up excessive whitespace
        md = re.sub(r"\n{3,}", "\n\n", md)
        md = re.sub(r" +", " ", md)
        md = md.strip()

        return md

    def parse_sections(
        self,
        content: Tag,
        code_examples: list[CodeExample],
    ) -> list[DocumentSection]:
        """Parse content into hierarchical sections.
        
        Args:
            content: Main content element
            code_examples: List of extracted code examples
            
        Returns:
            List of DocumentSection models
        """
        sections: list[DocumentSection] = []
        current_stack: list[tuple[int, DocumentSection | None, list[DocumentSection]]] = [
            (0, None, sections)
        ]

        # Find all headings
        headings = content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

        for heading in headings:
            if not isinstance(heading, Tag):
                continue

            level = int(heading.name[1])
            title = heading.get_text(strip=True)

            # Generate ID from heading
            raw_heading_id = str(heading.get("id", "") or "")
            if raw_heading_id:
                heading_id = re.sub(r"[^a-z0-9\-]+", "-", raw_heading_id.lower())
            else:
                heading_id = re.sub(r"[^a-z0-9]+", "-", title.lower())
            heading_id = re.sub(r"-{2,}", "-", heading_id).strip("-")
            if not heading_id:
                heading_id = f"section-{len(sections)}"

            # Get content until next heading
            section_content = []
            section_examples = []

            for sibling in heading.next_siblings:
                if isinstance(sibling, Tag):
                    if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        break
                    section_content.append(str(sibling))

                    # Check for code blocks
                    for code in sibling.find_all("pre"):
                        code_text = code.get_text()
                        for example in code_examples:
                            if example.code.strip() == code_text.strip():
                                section_examples.append(example)
                                break

            section_html = "".join(section_content)
            section_md = self.to_markdown(section_html) if section_html else title

            try:
                section = DocumentSection(
                    id=heading_id[:50],
                    title=title,
                    level=level,
                    content=section_md if section_md.strip() else title,
                    code_examples=section_examples,
                    subsections=[],
                )
            except Exception as e:
                logger.warning(f"Failed to create section: {e}")
                continue

            # Find correct parent level
            while current_stack[-1][0] >= level:
                current_stack.pop()

            # Add to parent's subsections
            current_stack[-1][2].append(section)

            # Push this section for potential children
            current_stack.append((level, section, section.subsections))

        return sections

    def parse(
        self,
        html: str,
        url: str,
        module: SpringModule,
        version: SpringVersion | None,
        submodule: str | None = None,
        content_type: ContentType = ContentType.REFERENCE,
    ) -> ScrapedPage:
        """Parse HTML into ScrapedPage model.
        
        Args:
            html: Raw HTML content
            url: Source URL
            module: Spring module
            version: Spring version
            content_type: Type of documentation
            
        Returns:
            ScrapedPage model with extracted content
            
        Raises:
            ContentExtractionError: If parsing fails
        """
        logger.info(f"Parsing page: {url}")

        # Parse HTML
        soup = self._create_soup(html)
        soup = self._clean_html(soup)

        # Extract title
        try:
            title = self.extract_title(soup)
        except ContentExtractionError:
            title = url.split("/")[-1].replace("-", " ").title()
            logger.warning(f"Using fallback title: {title}")

        title = sanitize_title(title)

        parsed_version: SpringVersion
        if content_type == ContentType.API_DOC and version is not None:
            try:
                version_text = self.extract_version(soup)
            except ContentExtractionError:
                parsed_version = version
                logger.debug(
                    "Using provided release version %s for API doc page: %s",
                    version.version_string,
                    url,
                )
            else:
                try:
                    parsed_version = SpringVersion.parse(module, version_text)
                except ValueError as exc:
                    raise ContentExtractionError(str(exc), url=url) from exc
                if parsed_version != version:
                    raise ContentExtractionError(
                        f"Version mismatch: expected {version.version_string}, got {parsed_version.version_string}",
                        url=url,
                    )
        else:
            if content_type == ContentType.API_DOC and version is None:
                raise ContentExtractionError(
                    "API documentation scraping requires an explicit release version",
                    url=url,
                )
            try:
                version_text = self.extract_version(soup)
            except ContentExtractionError:
                if content_type == ContentType.REFERENCE and version is not None:
                    parsed_version = version
                    logger.debug(
                        "Using provided release version %s for reference page: %s",
                        version.version_string,
                        url,
                    )
                else:
                    raise
            else:
                try:
                    parsed_version = SpringVersion.parse(module, version_text)
                except ValueError as exc:
                    raise ContentExtractionError(str(exc), url=url) from exc
                if version and parsed_version != version:
                    raise ContentExtractionError(
                        f"Version mismatch: expected {version.version_string}, got {parsed_version.version_string}",
                        url=url,
                    )

        if content_type == ContentType.API_DOC:
            try:
                api_signature = self.extract_api_signature(soup)
            except ContentExtractionError:
                logger.debug(
                    "No class-level API signature extracted for %s; using generic markdown parser",
                    url,
                )
            else:
                return ScrapedPage.create(
                    url=url,
                    module=module,
                    version=parsed_version,
                    submodule=submodule,
                    title=title,
                    raw_html=html,
                    markdown_content=self._api_signature_to_markdown(api_signature),
                    content_type=content_type,
                    sections=[],
                )

        # Extract main content
        try:
            content_element = self.extract_main_content(soup)
        except ContentExtractionError:
            # Fall back to body
            content_element = soup.find("body")
            if not content_element or not isinstance(content_element, Tag):
                raise ContentExtractionError("No content found in page", url=url)

        # Extract code examples
        code_examples = self.extract_code_blocks(content_element)
        logger.debug(f"Extracted {len(code_examples)} code examples")

        # Convert to Markdown
        markdown_content = self.to_markdown(content_element)
        if not markdown_content.strip():
            raise ContentExtractionError("Empty content after conversion", url=url)

        # Parse sections
        sections = self.parse_sections(content_element, code_examples)
        logger.debug(f"Parsed {len(sections)} top-level sections")

        # Create ScrapedPage using factory method
        return ScrapedPage.create(
            url=url,
            module=module,
            version=parsed_version,
            submodule=submodule,
            title=title,
            raw_html=html,
            markdown_content=markdown_content,
            content_type=content_type,
            sections=sections,
        )


__all__ = [
    "ParserConfig",
    "SpringDocParser",
    "SpringMarkdownConverter",
    "SPRING_DOC_SELECTORS",
    "LANGUAGE_PATTERNS",
]
