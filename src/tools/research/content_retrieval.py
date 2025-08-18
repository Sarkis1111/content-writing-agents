"""
Content Retrieval Tool for extracting and processing web content.

Provides web scraping capabilities with content extraction, cleaning,
and support for multiple content types (HTML, PDF, etc.).
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Set
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin, urlparse
import re
import hashlib

import aiohttp
from bs4 import BeautifulSoup, NavigableString
import requests
from pydantic import BaseModel, Field, HttpUrl

from ...core.config.loader import get_settings
from ...core.errors.exceptions import ToolExecutionError, APIError
from ...utils.retry import with_retry


logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Container for extracted content."""
    url: str
    title: str
    content: str
    meta_description: str
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    word_count: int = 0
    language: Optional[str] = None
    content_type: str = "html"
    images: List[str] = None
    links: List[str] = None
    headings: List[str] = None


class ContentExtractionRequest(BaseModel):
    """Content extraction request configuration."""
    url: HttpUrl = Field(..., description="URL to extract content from")
    include_images: bool = Field(default=False, description="Extract image URLs")
    include_links: bool = Field(default=False, description="Extract internal/external links")
    include_headings: bool = Field(default=True, description="Extract heading structure")
    clean_content: bool = Field(default=True, description="Clean and normalize content")
    max_content_length: int = Field(default=50000, description="Maximum content length")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class ContentExtractionResponse(BaseModel):
    """Content extraction response."""
    success: bool
    url: str
    content: Optional[ExtractedContent] = None
    error: Optional[str] = None
    extraction_time: float
    timestamp: datetime


class ContentRetrievalTool:
    """
    Content Retrieval Tool for comprehensive web content extraction.
    
    Supports HTML content extraction, PDF processing, content cleaning,
    and metadata extraction with rate limiting and caching.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache: Dict[str, ExtractedContent] = {}
        self.cache_ttl = 7200  # 2 hour cache TTL
        
        # User agent for web scraping
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
        # Content type handlers
        self.content_handlers = {
            'text/html': self._extract_html_content,
            'application/pdf': self._extract_pdf_content,
            'text/plain': self._extract_text_content
        }
        
        # Selectors for content extraction
        self.content_selectors = [
            'article', 'main', '[role="main"]', '.content', '.post-content',
            '.entry-content', '.article-content', '.story-body', '.post-body'
        ]
        
        # Selectors to remove (noise)
        self.noise_selectors = [
            'nav', 'footer', 'header', '.advertisement', '.ads', '.sidebar',
            '.social-share', '.comments', '.related-posts', '.popup',
            'script', 'style', 'noscript', '.cookie-notice'
        ]
    
    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached result is still valid."""
        age = (datetime.now() - timestamp).total_seconds()
        return age < self.cache_ttl
    
    @with_retry(max_attempts=3, delay=1.0)
    async def _fetch_content(self, url: str, timeout: int = 30) -> tuple[bytes, str, Dict[str, str]]:
        """Fetch raw content from URL."""
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        raise APIError(f"HTTP {response.status}: {response.reason}")
                    
                    content = await response.read()
                    content_type = response.headers.get('content-type', '').lower().split(';')[0]
                    response_headers = dict(response.headers)
                    
                    return content, content_type, response_headers
                    
        except asyncio.TimeoutError:
            raise ToolExecutionError(f"Timeout while fetching content from {url}")
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            raise ToolExecutionError(f"Failed to fetch content: {e}")
    
    def _extract_html_content(self, content: bytes, url: str, **kwargs) -> ExtractedContent:
        """Extract content from HTML."""
        try:
            soup = BeautifulSoup(content, 'lxml')
            
            # Remove noise elements
            for selector in self.noise_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Extract metadata
            title = self._extract_title(soup)
            meta_description = self._extract_meta_description(soup)
            author = self._extract_author(soup)
            publish_date = self._extract_publish_date(soup)
            language = self._extract_language(soup)
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Extract additional elements if requested
            images = self._extract_images(soup, url) if kwargs.get('include_images') else []
            links = self._extract_links(soup, url) if kwargs.get('include_links') else []
            headings = self._extract_headings(soup) if kwargs.get('include_headings') else []
            
            # Clean content if requested
            if kwargs.get('clean_content', True):
                main_content = self._clean_content(main_content)
            
            # Limit content length
            max_length = kwargs.get('max_content_length', 50000)
            if len(main_content) > max_length:
                main_content = main_content[:max_length] + "..."
            
            return ExtractedContent(
                url=url,
                title=title,
                content=main_content,
                meta_description=meta_description,
                author=author,
                publish_date=publish_date,
                word_count=len(main_content.split()),
                language=language,
                content_type='html',
                images=images,
                links=links,
                headings=headings
            )
            
        except Exception as e:
            logger.error(f"HTML content extraction failed for {url}: {e}")
            raise ToolExecutionError(f"HTML extraction failed: {e}")
    
    def _extract_pdf_content(self, content: bytes, url: str, **kwargs) -> ExtractedContent:
        """Extract content from PDF (placeholder for future implementation)."""
        # TODO: Implement PDF extraction using PyPDF2 or similar
        return ExtractedContent(
            url=url,
            title="PDF Document",
            content="PDF content extraction not yet implemented",
            meta_description="",
            content_type='pdf'
        )
    
    def _extract_text_content(self, content: bytes, url: str, **kwargs) -> ExtractedContent:
        """Extract content from plain text."""
        try:
            text_content = content.decode('utf-8')
            
            # Clean content if requested
            if kwargs.get('clean_content', True):
                text_content = self._clean_content(text_content)
            
            return ExtractedContent(
                url=url,
                title=f"Text Document - {urlparse(url).path.split('/')[-1]}",
                content=text_content,
                meta_description="",
                word_count=len(text_content.split()),
                content_type='text'
            )
            
        except Exception as e:
            logger.error(f"Text content extraction failed for {url}: {e}")
            raise ToolExecutionError(f"Text extraction failed: {e}")
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try multiple title sources
        title_selectors = [
            'h1',
            'title', 
            '[property="og:title"]',
            '[name="twitter:title"]',
            '.post-title',
            '.article-title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True) if hasattr(element, 'get_text') else element.get('content', '')
                if title:
                    return title
        
        return "Untitled"
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description."""
        meta_selectors = [
            '[name="description"]',
            '[property="og:description"]',
            '[name="twitter:description"]'
        ]
        
        for selector in meta_selectors:
            element = soup.select_one(selector)
            if element:
                description = element.get('content', '').strip()
                if description:
                    return description
        
        return ""
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author information."""
        author_selectors = [
            '[rel="author"]',
            '[name="author"]',
            '[property="article:author"]',
            '.author',
            '.byline',
            '.post-author'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get_text(strip=True) if hasattr(element, 'get_text') else element.get('content', '')
                if author:
                    return author
        
        return None
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publish date."""
        date_selectors = [
            '[property="article:published_time"]',
            '[name="publication-date"]',
            'time[datetime]',
            '.publish-date',
            '.post-date'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('datetime') or element.get('content') or element.get_text(strip=True)
                if date_str:
                    return self._parse_date(date_str)
        
        return None
    
    def _extract_language(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract content language."""
        html_tag = soup.find('html')
        if html_tag:
            return html_tag.get('lang')
        return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content using various strategies."""
        # Strategy 1: Try semantic selectors
        for selector in self.content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                content = content_element.get_text(separator=' ', strip=True)
                if len(content) > 100:  # Reasonable content length threshold
                    return content
        
        # Strategy 2: Find the element with the most text content
        all_elements = soup.find_all(['div', 'section', 'article', 'main'])
        best_element = None
        max_text_length = 0
        
        for element in all_elements:
            text_length = len(element.get_text(strip=True))
            if text_length > max_text_length:
                max_text_length = text_length
                best_element = element
        
        if best_element:
            return best_element.get_text(separator=' ', strip=True)
        
        # Strategy 3: Fallback to body content
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ', strip=True)
        
        # Final fallback
        return soup.get_text(separator=' ', strip=True)
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs."""
        images = []
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src')
            if src:
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, src)
                images.append(absolute_url)
        return images
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract link URLs."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)
        return links
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract heading structure."""
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = heading.get_text(strip=True)
            if text:
                headings.append(f"{heading.name}: {text}")
        return headings
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        
        # Common date formats
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        if not content:
            return ""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might interfere
        content = re.sub(r'[\r\n\t]', ' ', content)
        
        # Remove multiple spaces
        content = re.sub(r' +', ' ', content)
        
        # Trim
        content = content.strip()
        
        return content
    
    async def extract_content(
        self,
        request: Union[str, ContentExtractionRequest],
        use_cache: bool = True
    ) -> ContentExtractionResponse:
        """
        Extract content from a URL.
        
        Args:
            request: URL string or ContentExtractionRequest object
            use_cache: Whether to use cached results
            
        Returns:
            ContentExtractionResponse with extracted content
        """
        start_time = datetime.now()
        
        # Convert string URL to request object
        if isinstance(request, str):
            request = ContentExtractionRequest(url=request)
        
        url = str(request.url)
        
        try:
            # Check cache
            if use_cache:
                cache_key = self._generate_cache_key(url)
                cached_content = self.cache.get(cache_key)
                if cached_content:
                    logger.info(f"Returning cached content for: {url}")
                    extraction_time = (datetime.now() - start_time).total_seconds()
                    return ContentExtractionResponse(
                        success=True,
                        url=url,
                        content=cached_content,
                        extraction_time=extraction_time,
                        timestamp=datetime.now()
                    )
            
            # Fetch content
            logger.info(f"Extracting content from: {url}")
            raw_content, content_type, headers = await self._fetch_content(url, request.timeout)
            
            # Extract content based on type
            handler = self.content_handlers.get(content_type, self._extract_html_content)
            extracted_content = handler(
                raw_content, 
                url,
                include_images=request.include_images,
                include_links=request.include_links,
                include_headings=request.include_headings,
                clean_content=request.clean_content,
                max_content_length=request.max_content_length
            )
            
            # Cache the result
            if use_cache:
                self.cache[cache_key] = extracted_content
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            return ContentExtractionResponse(
                success=True,
                url=url,
                content=extracted_content,
                extraction_time=extraction_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            return ContentExtractionResponse(
                success=False,
                url=url,
                error=str(e),
                extraction_time=extraction_time,
                timestamp=datetime.now()
            )
    
    async def bulk_extract(self, urls: List[str]) -> List[ContentExtractionResponse]:
        """
        Extract content from multiple URLs concurrently.
        
        Args:
            urls: List of URLs to extract content from
            
        Returns:
            List of ContentExtractionResponse objects
        """
        tasks = [self.extract_content(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Bulk extraction failed for URL {i}: {result}")
                responses.append(ContentExtractionResponse(
                    success=False,
                    url=urls[i],
                    error=str(result),
                    extraction_time=0.0,
                    timestamp=datetime.now()
                ))
            else:
                responses.append(result)
        
        return responses
    
    def clear_cache(self):
        """Clear the content cache."""
        self.cache.clear()
        logger.info("Content cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache),
            "cache_size_mb": sum(len(str(content)) for content in self.cache.values()) / 1024 / 1024
        }


# Tool instance for MCP integration
content_retrieval_tool = ContentRetrievalTool()


# MCP tool function
async def mcp_extract_content(
    url: str,
    include_images: bool = False,
    include_links: bool = False,
    include_headings: bool = True,
    clean_content: bool = True,
    max_content_length: int = 50000,
    timeout: int = 30
) -> Dict:
    """
    MCP-compatible content extraction function.
    
    Args:
        url: URL to extract content from
        include_images: Extract image URLs
        include_links: Extract internal/external links
        include_headings: Extract heading structure
        clean_content: Clean and normalize content
        max_content_length: Maximum content length
        timeout: Request timeout in seconds
    
    Returns:
        Dictionary with extracted content and metadata
    """
    try:
        request = ContentExtractionRequest(
            url=url,
            include_images=include_images,
            include_links=include_links,
            include_headings=include_headings,
            clean_content=clean_content,
            max_content_length=max_content_length,
            timeout=timeout
        )
        
        response = await content_retrieval_tool.extract_content(request)
        
        if response.success and response.content:
            return {
                "success": True,
                "url": response.url,
                "title": response.content.title,
                "content": response.content.content,
                "meta_description": response.content.meta_description,
                "author": response.content.author,
                "publish_date": response.content.publish_date.isoformat() if response.content.publish_date else None,
                "word_count": response.content.word_count,
                "language": response.content.language,
                "content_type": response.content.content_type,
                "images": response.content.images or [],
                "links": response.content.links or [],
                "headings": response.content.headings or [],
                "extraction_time": response.extraction_time,
                "timestamp": response.timestamp.isoformat()
            }
        else:
            return {
                "success": False,
                "error": response.error,
                "url": response.url,
                "extraction_time": response.extraction_time
            }
            
    except Exception as e:
        logger.error(f"Content extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "url": url
        }