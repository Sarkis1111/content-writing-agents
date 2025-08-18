"""
Web Search Tool for content research.

Provides web search capabilities using multiple search engines and APIs.
Supports query optimization, result filtering, and rate limiting.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib

import aiohttp
import requests
from serpapi import GoogleSearch
from pydantic import BaseModel, Field

from ...core.config.loader import get_settings
from ...core.errors.exceptions import ToolExecutionError, APIError
from ...utils.retry import with_retry


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Individual search result."""
    title: str
    url: str
    snippet: str
    source: str
    position: int
    date_published: Optional[datetime] = None
    relevance_score: Optional[float] = None


class SearchQuery(BaseModel):
    """Search query configuration."""
    query: str = Field(..., description="Search query string")
    num_results: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    language: str = Field(default="en", description="Language for search results")
    country: str = Field(default="us", description="Country code for localized results")
    safe_search: bool = Field(default=True, description="Enable safe search filtering")
    date_range: Optional[str] = Field(default=None, description="Date range filter (day, week, month, year)")


class SearchResponse(BaseModel):
    """Search response container."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    timestamp: datetime
    source_engine: str


class WebSearchTool:
    """
    Web Search Tool providing comprehensive search capabilities.
    
    Supports multiple search engines with fallback mechanisms,
    query optimization, result filtering, and caching.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache: Dict[str, SearchResponse] = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # Initialize API clients
        self._init_search_apis()
    
    def _init_search_apis(self):
        """Initialize search API clients."""
        # SerpAPI configuration
        self.serpapi_key = self.settings.get("SERPAPI_KEY")
        if not self.serpapi_key:
            logger.warning("SERPAPI_KEY not found. SerpAPI search will be unavailable.")
        
        # Google Custom Search configuration
        self.google_api_key = self.settings.get("GOOGLE_API_KEY")
        self.google_cse_id = self.settings.get("GOOGLE_CSE_ID")
        if not self.google_api_key or not self.google_cse_id:
            logger.warning("Google Custom Search credentials not found. Google search will be unavailable.")
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query."""
        query_str = f"{query.query}_{query.num_results}_{query.language}_{query.country}_{query.date_range}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached result is still valid."""
        age = (datetime.now() - timestamp).total_seconds()
        return age < self.cache_ttl
    
    @with_retry(max_attempts=3, delay=1.0)
    async def _search_serpapi(self, query: SearchQuery) -> SearchResponse:
        """Search using SerpAPI."""
        if not self.serpapi_key:
            raise APIError("SerpAPI key not configured")
        
        start_time = datetime.now()
        
        try:
            search_params = {
                "q": query.query,
                "api_key": self.serpapi_key,
                "engine": "google",
                "num": query.num_results,
                "gl": query.country,
                "hl": query.language,
                "safe": "active" if query.safe_search else "off"
            }
            
            if query.date_range:
                search_params["tbs"] = f"qdr:{query.date_range[0]}"  # d, w, m, y
            
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            search_results = []
            organic_results = results.get("organic_results", [])
            
            for i, result in enumerate(organic_results):
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    source="serpapi",
                    position=i + 1,
                    date_published=self._parse_date(result.get("date")),
                    relevance_score=self._calculate_relevance(result, query.query)
                )
                search_results.append(search_result)
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return SearchResponse(
                query=query.query,
                results=search_results,
                total_results=len(search_results),
                search_time=search_time,
                timestamp=datetime.now(),
                source_engine="serpapi"
            )
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            raise APIError(f"SerpAPI search failed: {e}")
    
    @with_retry(max_attempts=3, delay=1.0)
    async def _search_google_custom(self, query: SearchQuery) -> SearchResponse:
        """Search using Google Custom Search API."""
        if not self.google_api_key or not self.google_cse_id:
            raise APIError("Google Custom Search not configured")
        
        start_time = datetime.now()
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query.query,
                "num": min(query.num_results, 10),  # Google CSE max is 10
                "gl": query.country,
                "hl": query.language,
                "safe": "active" if query.safe_search else "off"
            }
            
            if query.date_range:
                # Convert date_range to Google's date restriction format
                date_restrict = self._convert_date_range(query.date_range)
                if date_restrict:
                    params["dateRestrict"] = date_restrict
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise APIError(f"Google Custom Search API returned {response.status}")
                    
                    data = await response.json()
            
            search_results = []
            items = data.get("items", [])
            
            for i, item in enumerate(items):
                search_result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google_custom",
                    position=i + 1,
                    relevance_score=self._calculate_relevance(item, query.query)
                )
                search_results.append(search_result)
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return SearchResponse(
                query=query.query,
                results=search_results,
                total_results=int(data.get("searchInformation", {}).get("totalResults", len(search_results))),
                search_time=search_time,
                timestamp=datetime.now(),
                source_engine="google_custom"
            )
            
        except Exception as e:
            logger.error(f"Google Custom Search failed: {e}")
            raise APIError(f"Google Custom Search failed: {e}")
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        
        try:
            # Try common date formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None
    
    def _calculate_relevance(self, result: Dict, query: str) -> float:
        """Calculate relevance score for a search result."""
        try:
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            query_lower = query.lower()
            
            # Simple relevance scoring based on keyword matches
            title_matches = sum(1 for word in query_lower.split() if word in title)
            snippet_matches = sum(1 for word in query_lower.split() if word in snippet)
            
            # Weight title matches higher
            score = (title_matches * 2 + snippet_matches) / len(query_lower.split())
            return min(score, 1.0)
            
        except Exception:
            return 0.0
    
    def _convert_date_range(self, date_range: str) -> Optional[str]:
        """Convert date range to Google's format."""
        mapping = {
            "day": "d1",
            "week": "w1", 
            "month": "m1",
            "year": "y1"
        }
        return mapping.get(date_range.lower())
    
    def _optimize_query(self, query: str) -> str:
        """Optimize search query for better results."""
        # Remove extra spaces and special characters
        optimized = " ".join(query.split())
        
        # Add quotes for exact phrases if not already present
        if '"' not in optimized and len(optimized.split()) > 3:
            # Extract potential key phrases (2-3 words)
            words = optimized.split()
            if len(words) >= 2:
                # Quote the first two words if they seem like a phrase
                optimized = f'"{" ".join(words[:2])}" {" ".join(words[2:])}'
        
        return optimized
    
    async def search(
        self, 
        query: Union[str, SearchQuery],
        use_cache: bool = True,
        fallback: bool = True
    ) -> SearchResponse:
        """
        Execute web search with multiple engine support and fallback.
        
        Args:
            query: Search query string or SearchQuery object
            use_cache: Whether to use cached results
            fallback: Whether to use fallback search engines on failure
            
        Returns:
            SearchResponse with results and metadata
        """
        # Convert string query to SearchQuery object
        if isinstance(query, str):
            query = SearchQuery(query=query)
        
        # Optimize the query
        query.query = self._optimize_query(query.query)
        
        # Check cache
        if use_cache:
            cache_key = self._generate_cache_key(query)
            cached_result = self.cache.get(cache_key)
            if cached_result and self._is_cache_valid(cached_result.timestamp):
                logger.info(f"Returning cached search results for: {query.query}")
                return cached_result
        
        # Try primary search engine (SerpAPI)
        try:
            if self.serpapi_key:
                logger.info(f"Searching with SerpAPI: {query.query}")
                response = await self._search_serpapi(query)
                
                # Cache successful results
                if use_cache:
                    self.cache[cache_key] = response
                
                return response
                
        except Exception as e:
            logger.warning(f"SerpAPI search failed: {e}")
            if not fallback:
                raise
        
        # Fallback to Google Custom Search
        try:
            if self.google_api_key and self.google_cse_id:
                logger.info(f"Searching with Google Custom Search: {query.query}")
                response = await self._search_google_custom(query)
                
                # Cache successful results
                if use_cache:
                    self.cache[cache_key] = response
                
                return response
                
        except Exception as e:
            logger.error(f"Google Custom Search also failed: {e}")
            if not fallback:
                raise
        
        # If all searches failed
        raise ToolExecutionError("All search engines failed. Please check API configurations.")
    
    async def bulk_search(self, queries: List[Union[str, SearchQuery]]) -> List[SearchResponse]:
        """
        Execute multiple searches concurrently.
        
        Args:
            queries: List of search queries
            
        Returns:
            List of SearchResponse objects
        """
        tasks = [self.search(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Bulk search failed for query {i}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def clear_cache(self):
        """Clear the search cache."""
        self.cache.clear()
        logger.info("Search cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        valid_entries = sum(1 for response in self.cache.values() 
                          if self._is_cache_valid(response.timestamp))
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries
        }


# Tool instance for MCP integration
web_search_tool = WebSearchTool()


# MCP tool function
async def mcp_web_search(
    query: str,
    num_results: int = 10,
    language: str = "en",
    country: str = "us",
    safe_search: bool = True,
    date_range: Optional[str] = None
) -> Dict:
    """
    MCP-compatible web search function.
    
    Args:
        query: Search query string
        num_results: Number of results to return (1-50)
        language: Language code for results
        country: Country code for localized results  
        safe_search: Enable safe search filtering
        date_range: Date range filter (day, week, month, year)
    
    Returns:
        Dictionary with search results and metadata
    """
    try:
        search_query = SearchQuery(
            query=query,
            num_results=num_results,
            language=language,
            country=country,
            safe_search=safe_search,
            date_range=date_range
        )
        
        response = await web_search_tool.search(search_query)
        
        return {
            "success": True,
            "query": response.query,
            "results": [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "position": result.position,
                    "source": result.source,
                    "relevance_score": result.relevance_score,
                    "date_published": result.date_published.isoformat() if result.date_published else None
                }
                for result in response.results
            ],
            "total_results": response.total_results,
            "search_time": response.search_time,
            "source_engine": response.source_engine,
            "timestamp": response.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }