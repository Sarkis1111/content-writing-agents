"""
News Search Tool for real-time news monitoring and analysis.

Provides News API integration, real-time news monitoring,
source credibility scoring, and news trend analysis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib
from urllib.parse import urlparse

import aiohttp
from newsapi import NewsApiClient
from pydantic import BaseModel, Field, HttpUrl

from ...core.config.loader import get_settings
from ...core.errors.exceptions import ToolExecutionError, APIError
from ...utils.retry import with_retry


logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Individual news article."""
    title: str
    description: str
    url: str
    source_name: str
    author: Optional[str]
    published_at: datetime
    url_to_image: Optional[str]
    content: Optional[str]
    credibility_score: float
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None


@dataclass
class NewsSource:
    """News source information."""
    id: str
    name: str
    description: str
    url: str
    category: str
    language: str
    country: str
    credibility_score: float


class NewsQuery(BaseModel):
    """News search query configuration."""
    query: Optional[str] = Field(default=None, description="Search query")
    sources: Optional[List[str]] = Field(default=None, description="Comma-separated source IDs")
    domains: Optional[List[str]] = Field(default=None, description="Comma-separated domains")
    exclude_domains: Optional[List[str]] = Field(default=None, description="Domains to exclude")
    from_date: Optional[datetime] = Field(default=None, description="Start date for articles")
    to_date: Optional[datetime] = Field(default=None, description="End date for articles")
    language: str = Field(default="en", description="Language code")
    sort_by: str = Field(default="publishedAt", description="Sort by: relevancy, popularity, publishedAt")
    page_size: int = Field(default=20, ge=1, le=100, description="Number of articles per page")
    page: int = Field(default=1, ge=1, description="Page number")


class HeadlinesQuery(BaseModel):
    """Top headlines query configuration."""
    country: Optional[str] = Field(default=None, description="Country code")
    category: Optional[str] = Field(default=None, description="Category")
    sources: Optional[List[str]] = Field(default=None, description="Source IDs")
    query: Optional[str] = Field(default=None, description="Search query")
    page_size: int = Field(default=20, ge=1, le=100, description="Number of articles")
    page: int = Field(default=1, ge=1, description="Page number")


class NewsResponse(BaseModel):
    """News search response."""
    query: str
    articles: List[NewsArticle]
    total_results: int
    search_time: float
    timestamp: datetime
    source_api: str


class NewsSearchTool:
    """
    News Search Tool for comprehensive news monitoring and analysis.
    
    Provides News API integration, real-time monitoring, source credibility
    scoring, and news trend analysis capabilities.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache: Dict[str, NewsResponse] = {}
        self.cache_ttl = 1800  # 30 minute cache TTL for news
        
        # Initialize News API client
        self._init_news_api()
        
        # Source credibility mapping (simplified scoring system)
        self.source_credibility = {
            # High credibility sources
            "bbc-news": 0.95,
            "reuters": 0.95,
            "associated-press": 0.95,
            "npr": 0.9,
            "the-guardian": 0.85,
            "cnn": 0.8,
            "the-new-york-times": 0.85,
            "the-washington-post": 0.85,
            "bloomberg": 0.85,
            "financial-times": 0.85,
            
            # Medium credibility sources
            "fox-news": 0.7,
            "cbc-news": 0.8,
            "abc-news": 0.75,
            "nbc-news": 0.75,
            "usa-today": 0.7,
            
            # Default credibility for unknown sources
            "default": 0.6
        }
        
        # News categories
        self.categories = [
            "business", "entertainment", "general", "health",
            "science", "sports", "technology"
        ]
        
        # Country codes
        self.countries = [
            "us", "gb", "ca", "au", "de", "fr", "it", "jp", "kr", "in"
        ]
    
    def _init_news_api(self):
        """Initialize News API client."""
        api_key = self.settings.get("NEWS_API_KEY")
        if not api_key:
            logger.warning("NEWS_API_KEY not found. News API will be unavailable.")
            self.news_api = None
        else:
            try:
                self.news_api = NewsApiClient(api_key=api_key)
                logger.info("News API client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize News API client: {e}")
                self.news_api = None
    
    def _generate_cache_key(self, query: Union[NewsQuery, HeadlinesQuery]) -> str:
        """Generate cache key for news query."""
        if isinstance(query, NewsQuery):
            query_str = f"search_{query.query}_{query.sources}_{query.from_date}_{query.to_date}_{query.sort_by}"
        else:  # HeadlinesQuery
            query_str = f"headlines_{query.country}_{query.category}_{query.sources}_{query.query}"
        
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached result is still valid."""
        age = (datetime.now() - timestamp).total_seconds()
        return age < self.cache_ttl
    
    def _calculate_source_credibility(self, source_id: str, source_name: str) -> float:
        """Calculate credibility score for a news source."""
        # Check if we have a specific score for this source
        source_key = source_id.lower() if source_id else source_name.lower().replace(" ", "-")
        
        if source_key in self.source_credibility:
            return self.source_credibility[source_key]
        
        # Use heuristics for unknown sources
        credibility = self.source_credibility["default"]
        
        # Boost credibility for certain indicators
        if any(indicator in source_name.lower() for indicator in ["news", "times", "post", "journal"]):
            credibility += 0.1
        
        if any(indicator in source_name.lower() for indicator in ["blog", "gossip", "rumor"]):
            credibility -= 0.2
        
        # Ensure credibility is within bounds
        return max(0.0, min(1.0, credibility))
    
    def _calculate_relevance_score(self, article: Dict, query: str) -> float:
        """Calculate relevance score for an article."""
        if not query:
            return 0.5  # Default relevance when no query
        
        query_lower = query.lower()
        score = 0.0
        
        # Title relevance (weighted higher)
        title = article.get("title", "").lower()
        title_matches = sum(1 for word in query_lower.split() if word in title)
        score += (title_matches / len(query_lower.split())) * 0.6
        
        # Description relevance
        description = article.get("description", "").lower()
        desc_matches = sum(1 for word in query_lower.split() if word in description)
        score += (desc_matches / len(query_lower.split())) * 0.4
        
        return min(score, 1.0)
    
    def _parse_article(self, article_data: Dict, query: str = "") -> NewsArticle:
        """Parse article data from News API response."""
        # Parse published date
        published_at = datetime.now()
        if article_data.get("publishedAt"):
            try:
                published_at = datetime.fromisoformat(
                    article_data["publishedAt"].replace("Z", "+00:00")
                )
            except ValueError:
                pass
        
        # Calculate scores
        source_id = article_data.get("source", {}).get("id", "")
        source_name = article_data.get("source", {}).get("name", "Unknown")
        credibility_score = self._calculate_source_credibility(source_id, source_name)
        relevance_score = self._calculate_relevance_score(article_data, query)
        
        return NewsArticle(
            title=article_data.get("title", ""),
            description=article_data.get("description", ""),
            url=article_data.get("url", ""),
            source_name=source_name,
            author=article_data.get("author"),
            published_at=published_at,
            url_to_image=article_data.get("urlToImage"),
            content=article_data.get("content"),
            credibility_score=credibility_score,
            relevance_score=relevance_score
        )
    
    @with_retry(max_attempts=3, delay=1.0)
    async def _search_news_api(self, query: NewsQuery) -> NewsResponse:
        """Search news using News API."""
        if not self.news_api:
            raise APIError("News API not configured")
        
        start_time = datetime.now()
        
        try:
            # Prepare parameters
            params = {
                "q": query.query,
                "language": query.language,
                "sort_by": query.sort_by,
                "page_size": query.page_size,
                "page": query.page
            }
            
            # Add optional parameters
            if query.sources:
                params["sources"] = ",".join(query.sources)
            if query.domains:
                params["domains"] = ",".join(query.domains)
            if query.exclude_domains:
                params["exclude_domains"] = ",".join(query.exclude_domains)
            if query.from_date:
                params["from_param"] = query.from_date.strftime("%Y-%m-%d")
            if query.to_date:
                params["to"] = query.to_date.strftime("%Y-%m-%d")
            
            # Make API call
            response = self.news_api.get_everything(**params)
            
            if response.get("status") != "ok":
                raise APIError(f"News API error: {response.get('message', 'Unknown error')}")
            
            # Parse articles
            articles = []
            for article_data in response.get("articles", []):
                article = self._parse_article(article_data, query.query or "")
                articles.append(article)
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return NewsResponse(
                query=query.query or "all",
                articles=articles,
                total_results=response.get("totalResults", len(articles)),
                search_time=search_time,
                timestamp=datetime.now(),
                source_api="newsapi"
            )
            
        except Exception as e:
            logger.error(f"News API search failed: {e}")
            raise APIError(f"News API search failed: {e}")
    
    @with_retry(max_attempts=3, delay=1.0)
    async def _get_headlines_api(self, query: HeadlinesQuery) -> NewsResponse:
        """Get top headlines using News API."""
        if not self.news_api:
            raise APIError("News API not configured")
        
        start_time = datetime.now()
        
        try:
            # Prepare parameters
            params = {
                "page_size": query.page_size,
                "page": query.page
            }
            
            # Add optional parameters
            if query.country:
                params["country"] = query.country
            if query.category:
                params["category"] = query.category
            if query.sources:
                params["sources"] = ",".join(query.sources)
            if query.query:
                params["q"] = query.query
            
            # Make API call
            response = self.news_api.get_top_headlines(**params)
            
            if response.get("status") != "ok":
                raise APIError(f"News API error: {response.get('message', 'Unknown error')}")
            
            # Parse articles
            articles = []
            for article_data in response.get("articles", []):
                article = self._parse_article(article_data, query.query or "")
                articles.append(article)
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return NewsResponse(
                query=query.query or f"headlines_{query.country}_{query.category}",
                articles=articles,
                total_results=response.get("totalResults", len(articles)),
                search_time=search_time,
                timestamp=datetime.now(),
                source_api="newsapi_headlines"
            )
            
        except Exception as e:
            logger.error(f"News API headlines failed: {e}")
            raise APIError(f"News API headlines failed: {e}")
    
    async def search_news(
        self,
        query: Union[str, NewsQuery],
        use_cache: bool = True
    ) -> NewsResponse:
        """
        Search for news articles.
        
        Args:
            query: Search query string or NewsQuery object
            use_cache: Whether to use cached results
            
        Returns:
            NewsResponse with articles and metadata
        """
        # Convert string query to NewsQuery object
        if isinstance(query, str):
            query = NewsQuery(query=query)
        
        try:
            # Check cache
            if use_cache:
                cache_key = self._generate_cache_key(query)
                cached_result = self.cache.get(cache_key)
                if cached_result and self._is_cache_valid(cached_result.timestamp):
                    logger.info(f"Returning cached news results for: {query.query}")
                    return cached_result
            
            logger.info(f"Searching news for: {query.query}")
            
            # Search using News API
            response = await self._search_news_api(query)
            
            # Sort articles by credibility and relevance
            response.articles.sort(
                key=lambda x: (x.credibility_score * 0.6 + (x.relevance_score or 0) * 0.4),
                reverse=True
            )
            
            # Cache the result
            if use_cache:
                self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"News search failed: {e}")
            raise ToolExecutionError(f"News search failed: {e}")
    
    async def get_headlines(
        self,
        query: Union[str, HeadlinesQuery] = None,
        use_cache: bool = True
    ) -> NewsResponse:
        """
        Get top headlines.
        
        Args:
            query: Headlines query string or HeadlinesQuery object
            use_cache: Whether to use cached results
            
        Returns:
            NewsResponse with top headlines
        """
        # Convert string query to HeadlinesQuery object
        if isinstance(query, str):
            query = HeadlinesQuery(query=query)
        elif query is None:
            query = HeadlinesQuery()
        
        try:
            # Check cache
            if use_cache:
                cache_key = self._generate_cache_key(query)
                cached_result = self.cache.get(cache_key)
                if cached_result and self._is_cache_valid(cached_result.timestamp):
                    logger.info(f"Returning cached headlines")
                    return cached_result
            
            logger.info(f"Getting headlines for: {query.country} {query.category}")
            
            # Get headlines using News API
            response = await self._get_headlines_api(query)
            
            # Sort articles by credibility and recency
            response.articles.sort(
                key=lambda x: (x.credibility_score * 0.7 + 
                             (1.0 - (datetime.now() - x.published_at).total_seconds() / 86400) * 0.3),
                reverse=True
            )
            
            # Cache the result
            if use_cache:
                self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Headlines retrieval failed: {e}")
            raise ToolExecutionError(f"Headlines retrieval failed: {e}")
    
    async def monitor_keywords(
        self,
        keywords: List[str],
        interval_minutes: int = 60,
        max_articles: int = 50
    ) -> Dict[str, List[NewsArticle]]:
        """
        Monitor keywords for new articles.
        
        Args:
            keywords: List of keywords to monitor
            interval_minutes: Monitoring interval in minutes
            max_articles: Maximum articles per keyword
            
        Returns:
            Dictionary mapping keywords to recent articles
        """
        results = {}
        
        # Search for each keyword
        for keyword in keywords:
            try:
                # Search for recent articles
                from_date = datetime.now() - timedelta(minutes=interval_minutes)
                query = NewsQuery(
                    query=keyword,
                    from_date=from_date,
                    sort_by="publishedAt",
                    page_size=min(max_articles, 100)
                )
                
                response = await self.search_news(query, use_cache=False)
                results[keyword] = response.articles
                
            except Exception as e:
                logger.error(f"Keyword monitoring failed for '{keyword}': {e}")
                results[keyword] = []
        
        return results
    
    async def get_sources(self, category: Optional[str] = None, language: str = "en") -> List[NewsSource]:
        """
        Get available news sources.
        
        Args:
            category: Filter by category
            language: Filter by language
            
        Returns:
            List of available news sources
        """
        if not self.news_api:
            raise ToolExecutionError("News API not configured")
        
        try:
            params = {"language": language}
            if category:
                params["category"] = category
            
            response = self.news_api.get_sources(**params)
            
            if response.get("status") != "ok":
                raise APIError(f"News API error: {response.get('message', 'Unknown error')}")
            
            sources = []
            for source_data in response.get("sources", []):
                credibility = self._calculate_source_credibility(
                    source_data.get("id", ""),
                    source_data.get("name", "")
                )
                
                source = NewsSource(
                    id=source_data.get("id", ""),
                    name=source_data.get("name", ""),
                    description=source_data.get("description", ""),
                    url=source_data.get("url", ""),
                    category=source_data.get("category", ""),
                    language=source_data.get("language", ""),
                    country=source_data.get("country", ""),
                    credibility_score=credibility
                )
                sources.append(source)
            
            # Sort by credibility
            sources.sort(key=lambda x: x.credibility_score, reverse=True)
            
            return sources
            
        except Exception as e:
            logger.error(f"Failed to get sources: {e}")
            raise ToolExecutionError(f"Failed to get sources: {e}")
    
    def clear_cache(self):
        """Clear the news cache."""
        self.cache.clear()
        logger.info("News cache cleared")
    
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
news_search_tool = NewsSearchTool()


# MCP tool function
async def mcp_search_news(
    query: str,
    sources: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 20,
    page: int = 1
) -> Dict:
    """
    MCP-compatible news search function.
    
    Args:
        query: Search query
        sources: Source IDs to search
        domains: Domains to search
        exclude_domains: Domains to exclude
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        language: Language code
        sort_by: Sort by (relevancy, popularity, publishedAt)
        page_size: Number of articles per page
        page: Page number
    
    Returns:
        Dictionary with news search results
    """
    try:
        # Parse dates
        parsed_from_date = None
        parsed_to_date = None
        
        if from_date:
            parsed_from_date = datetime.fromisoformat(from_date)
        if to_date:
            parsed_to_date = datetime.fromisoformat(to_date)
        
        news_query = NewsQuery(
            query=query,
            sources=sources,
            domains=domains,
            exclude_domains=exclude_domains,
            from_date=parsed_from_date,
            to_date=parsed_to_date,
            language=language,
            sort_by=sort_by,
            page_size=page_size,
            page=page
        )
        
        response = await news_search_tool.search_news(news_query)
        
        return {
            "success": True,
            "query": response.query,
            "articles": [
                {
                    "title": article.title,
                    "description": article.description,
                    "url": article.url,
                    "source_name": article.source_name,
                    "author": article.author,
                    "published_at": article.published_at.isoformat(),
                    "url_to_image": article.url_to_image,
                    "credibility_score": article.credibility_score,
                    "relevance_score": article.relevance_score
                }
                for article in response.articles
            ],
            "total_results": response.total_results,
            "search_time": response.search_time,
            "timestamp": response.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"News search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


# MCP function for headlines
async def mcp_get_headlines(
    country: Optional[str] = None,
    category: Optional[str] = None,
    sources: Optional[List[str]] = None,
    query: Optional[str] = None,
    page_size: int = 20,
    page: int = 1
) -> Dict:
    """
    MCP-compatible function to get top headlines.
    
    Args:
        country: Country code
        category: News category
        sources: Source IDs
        query: Search query
        page_size: Number of articles
        page: Page number
    
    Returns:
        Dictionary with top headlines
    """
    try:
        headlines_query = HeadlinesQuery(
            country=country,
            category=category,
            sources=sources,
            query=query,
            page_size=page_size,
            page=page
        )
        
        response = await news_search_tool.get_headlines(headlines_query)
        
        return {
            "success": True,
            "query": response.query,
            "articles": [
                {
                    "title": article.title,
                    "description": article.description,
                    "url": article.url,
                    "source_name": article.source_name,
                    "author": article.author,
                    "published_at": article.published_at.isoformat(),
                    "url_to_image": article.url_to_image,
                    "credibility_score": article.credibility_score
                }
                for article in response.articles
            ],
            "total_results": response.total_results,
            "search_time": response.search_time,
            "timestamp": response.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Headlines retrieval failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# MCP function for sources
async def mcp_get_news_sources(
    category: Optional[str] = None,
    language: str = "en"
) -> Dict:
    """
    MCP-compatible function to get news sources.
    
    Args:
        category: News category filter
        language: Language code
    
    Returns:
        Dictionary with available news sources
    """
    try:
        sources = await news_search_tool.get_sources(category, language)
        
        return {
            "success": True,
            "sources": [
                {
                    "id": source.id,
                    "name": source.name,
                    "description": source.description,
                    "url": source.url,
                    "category": source.category,
                    "language": source.language,
                    "country": source.country,
                    "credibility_score": source.credibility_score
                }
                for source in sources
            ],
            "count": len(sources),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get news sources: {e}")
        return {
            "success": False,
            "error": str(e)
        }