"""EverSpring MCP - Spring documentation HTML parser.

This module provides SpringDocParser for parsing Spring documentation:
- HTML extraction (sidebar, main content, code blocks)
- Markdown conversion using markdownify
- Section hierarchy building
- ScrapedPage model creation
"""

from __future__ import annotations

import logging
import re
from typing import ClassVar

from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import MarkdownConverter, markdownify
from pydantic import BaseModel, ConfigDict, Field

from ..models.base import compute_hash
from ..models.content import (
    CodeExample,
    CodeLanguage,
    ContentType,
    DocumentSection,
    ScrapedPage,
)
from ..models.spring import SpringModule, SpringVersion
from .exceptions import ContentExtractionError

logger = logging.getLogger(__name__)


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
    strip_tags: list[str] = Field(
        default_factory=lambda: ["script", "style", "noscript", "iframe"],
        description="Tags to remove entirely",
    )
    unwrap_tags: list[str] = Field(
        default_factory=lambda: ["span", "div"],
        description="Tags to unwrap (keep content)",
    )


class SpringMarkdownConverter(MarkdownConverter):
    """Custom Markdown converter for Spring documentation.
    
    Handles Spring-specific elements and preserves code formatting.
    """
    
    def convert_pre(self, el: Tag, text: str, convert_as_inline: bool) -> str:
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
    
    def convert_a(self, el: Tag, text: str, convert_as_inline: bool) -> str:
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
    
    def convert_table(self, el: Tag, text: str, convert_as_inline: bool) -> str:
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
        if isinstance(html, Tag):
            html = str(html)
        
        # Use custom converter
        md = self._converter.convert(html)
        
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
            heading_id = heading.get("id", "")
            if not heading_id:
                heading_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            
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
                    id=heading_id[:50] if heading_id else f"section-{len(sections)}",
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
        version: SpringVersion,
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
            version=version,
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
