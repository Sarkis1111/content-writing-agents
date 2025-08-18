"""
Trend Finder Tool for discovering content trends and keyword popularity.

Provides Google Trends integration, social media trend analysis,
and keyword popularity tracking over time.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib

from pytrends.request import TrendReq
import pandas as pd
from pydantic import BaseModel, Field

from ...core.config.loader import get_settings
from ...core.errors.exceptions import ToolExecutionError, APIError
from ...utils.retry import with_retry


logger = logging.getLogger(__name__)


@dataclass
class TrendData:
    """Individual trend data point."""
    keyword: str
    interest_score: int
    timestamp: datetime
    region: str


@dataclass
class TrendInsight:
    """Trend analysis insight."""
    keyword: str
    avg_interest: float
    peak_interest: int
    trend_direction: str  # rising, falling, stable
    growth_rate: float
    related_queries: List[str]
    related_topics: List[str]
    regional_interest: Dict[str, int]


class TrendQuery(BaseModel):
    """Trend analysis query configuration."""
    keywords: List[str] = Field(..., min_items=1, max_items=5, description="Keywords to analyze (max 5)")
    timeframe: str = Field(default="today 3-m", description="Time frame for analysis")
    geo: str = Field(default="", description="Geographic location (empty for worldwide)")
    category: int = Field(default=0, description="Google Trends category (0 for all)")
    property_filter: str = Field(default="", description="Property filter (web, news, youtube, etc.)")


class TrendResponse(BaseModel):
    """Trend analysis response."""
    query_keywords: List[str]
    timeframe: str
    geo: str
    trends: List[TrendInsight]
    related_queries: Dict[str, List[str]]
    rising_queries: Dict[str, List[str]]
    top_queries: Dict[str, List[str]]
    regional_data: Dict[str, Dict[str, int]]
    analysis_time: float
    timestamp: datetime


class TrendFinderTool:
    """
    Trend Finder Tool for comprehensive trend analysis and keyword research.
    
    Provides Google Trends integration, keyword popularity tracking,
    regional analysis, and trend prediction capabilities.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache: Dict[str, TrendResponse] = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # Initialize Google Trends client
        self._init_trends_client()
        
        # Trend direction thresholds
        self.rising_threshold = 0.1  # 10% growth
        self.falling_threshold = -0.1  # 10% decline
    
    def _init_trends_client(self):
        """Initialize Google Trends client."""
        try:
            self.trends_client = TrendReq(
                hl='en-US',
                tz=360,  # GMT offset in minutes
                timeout=(10, 25),  # Connection and read timeout
                retries=3,
                backoff_factor=0.1
            )
            logger.info("Google Trends client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Google Trends client: {e}")
            self.trends_client = None
    
    def _generate_cache_key(self, query: TrendQuery) -> str:
        """Generate cache key for trend query."""
        query_str = f"{'-'.join(query.keywords)}_{query.timeframe}_{query.geo}_{query.category}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached result is still valid."""
        age = (datetime.now() - timestamp).total_seconds()
        return age < self.cache_ttl
    
    @with_retry(max_attempts=3, delay=2.0)
    async def _get_interest_over_time(self, query: TrendQuery) -> pd.DataFrame:
        """Get interest over time data from Google Trends."""
        if not self.trends_client:
            raise ToolExecutionError("Google Trends client not initialized")
        
        try:
            # Build payload for Google Trends
            self.trends_client.build_payload(
                kw_list=query.keywords,
                timeframe=query.timeframe,
                geo=query.geo,
                cat=query.category,
                gprop=query.property_filter
            )
            
            # Get interest over time data
            interest_df = self.trends_client.interest_over_time()
            
            if interest_df.empty:
                logger.warning(f"No trend data found for keywords: {query.keywords}")
                return pd.DataFrame()
            
            return interest_df
            
        except Exception as e:
            logger.error(f"Failed to get interest over time: {e}")
            raise APIError(f"Google Trends API error: {e}")
    
    @with_retry(max_attempts=3, delay=2.0)
    async def _get_related_queries(self, query: TrendQuery) -> Dict[str, Dict]:
        """Get related queries from Google Trends."""
        if not self.trends_client:
            return {}
        
        try:
            # Build payload
            self.trends_client.build_payload(
                kw_list=query.keywords,
                timeframe=query.timeframe,
                geo=query.geo,
                cat=query.category,
                gprop=query.property_filter
            )
            
            # Get related queries
            related_queries = self.trends_client.related_queries()
            return related_queries
            
        except Exception as e:
            logger.error(f"Failed to get related queries: {e}")
            return {}
    
    @with_retry(max_attempts=3, delay=2.0)
    async def _get_related_topics(self, query: TrendQuery) -> Dict[str, Dict]:
        """Get related topics from Google Trends."""
        if not self.trends_client:
            return {}
        
        try:
            # Build payload
            self.trends_client.build_payload(
                kw_list=query.keywords,
                timeframe=query.timeframe,
                geo=query.geo,
                cat=query.category,
                gprop=query.property_filter
            )
            
            # Get related topics
            related_topics = self.trends_client.related_topics()
            return related_topics
            
        except Exception as e:
            logger.error(f"Failed to get related topics: {e}")
            return {}
    
    @with_retry(max_attempts=3, delay=2.0)
    async def _get_regional_interest(self, query: TrendQuery) -> pd.DataFrame:
        """Get regional interest data from Google Trends."""
        if not self.trends_client:
            return pd.DataFrame()
        
        try:
            # Build payload
            self.trends_client.build_payload(
                kw_list=query.keywords,
                timeframe=query.timeframe,
                geo=query.geo,
                cat=query.category,
                gprop=query.property_filter
            )
            
            # Get regional interest
            regional_df = self.trends_client.interest_by_region()
            return regional_df
            
        except Exception as e:
            logger.error(f"Failed to get regional interest: {e}")
            return pd.DataFrame()
    
    def _analyze_trend_direction(self, interest_data: pd.Series) -> tuple[str, float]:
        """Analyze trend direction and growth rate."""
        if len(interest_data) < 2:
            return "stable", 0.0
        
        # Calculate growth rate between first and last values
        first_val = interest_data.iloc[0]
        last_val = interest_data.iloc[-1]
        
        if first_val == 0:
            growth_rate = 0.0
        else:
            growth_rate = (last_val - first_val) / first_val
        
        # Determine trend direction
        if growth_rate > self.rising_threshold:
            direction = "rising"
        elif growth_rate < self.falling_threshold:
            direction = "falling"
        else:
            direction = "stable"
        
        return direction, growth_rate
    
    def _process_related_data(self, related_data: Dict, data_type: str = "queries") -> Dict[str, List[str]]:
        """Process related queries or topics data."""
        processed = {}
        
        for keyword, data in related_data.items():
            if not data or data_type not in data:
                processed[keyword] = []
                continue
            
            # Get top and rising items
            top_items = []
            rising_items = []
            
            if 'top' in data[data_type] and data[data_type]['top'] is not None:
                top_df = data[data_type]['top']
                top_items = top_df['query'].tolist() if 'query' in top_df.columns else []
            
            if 'rising' in data[data_type] and data[data_type]['rising'] is not None:
                rising_df = data[data_type]['rising']
                rising_items = rising_df['query'].tolist() if 'query' in rising_df.columns else []
            
            # Combine and deduplicate
            all_items = list(set(top_items + rising_items))
            processed[keyword] = all_items[:10]  # Limit to top 10
        
        return processed
    
    def _extract_top_queries(self, related_queries: Dict) -> Dict[str, List[str]]:
        """Extract top queries for each keyword."""
        top_queries = {}
        
        for keyword, data in related_queries.items():
            if not data or 'query' not in data:
                top_queries[keyword] = []
                continue
            
            if 'top' in data['query'] and data['query']['top'] is not None:
                top_df = data['query']['top']
                queries = top_df['query'].tolist() if 'query' in top_df.columns else []
                top_queries[keyword] = queries[:10]
            else:
                top_queries[keyword] = []
        
        return top_queries
    
    def _extract_rising_queries(self, related_queries: Dict) -> Dict[str, List[str]]:
        """Extract rising queries for each keyword."""
        rising_queries = {}
        
        for keyword, data in related_queries.items():
            if not data or 'query' not in data:
                rising_queries[keyword] = []
                continue
            
            if 'rising' in data['query'] and data['query']['rising'] is not None:
                rising_df = data['query']['rising']
                queries = rising_df['query'].tolist() if 'query' in rising_df.columns else []
                rising_queries[keyword] = queries[:10]
            else:
                rising_queries[keyword] = []
        
        return rising_queries
    
    def _process_regional_data(self, regional_df: pd.DataFrame, keywords: List[str]) -> Dict[str, Dict[str, int]]:
        """Process regional interest data."""
        regional_data = {}
        
        if regional_df.empty:
            return {keyword: {} for keyword in keywords}
        
        for keyword in keywords:
            if keyword in regional_df.columns:
                # Get top regions for this keyword
                keyword_data = regional_df[keyword].dropna().sort_values(ascending=False)
                regional_data[keyword] = keyword_data.head(10).to_dict()
            else:
                regional_data[keyword] = {}
        
        return regional_data
    
    async def analyze_trends(
        self,
        query: Union[List[str], TrendQuery],
        use_cache: bool = True
    ) -> TrendResponse:
        """
        Analyze trends for given keywords.
        
        Args:
            query: List of keywords or TrendQuery object
            use_cache: Whether to use cached results
            
        Returns:
            TrendResponse with comprehensive trend analysis
        """
        start_time = datetime.now()
        
        # Convert list of keywords to TrendQuery
        if isinstance(query, list):
            query = TrendQuery(keywords=query)
        
        # Validate keyword count
        if len(query.keywords) > 5:
            raise ToolExecutionError("Maximum 5 keywords allowed for trend analysis")
        
        try:
            # Check cache
            if use_cache:
                cache_key = self._generate_cache_key(query)
                cached_result = self.cache.get(cache_key)
                if cached_result and self._is_cache_valid(cached_result.timestamp):
                    logger.info(f"Returning cached trend analysis for: {query.keywords}")
                    return cached_result
            
            logger.info(f"Analyzing trends for keywords: {query.keywords}")
            
            # Gather all trend data concurrently
            tasks = [
                self._get_interest_over_time(query),
                self._get_related_queries(query),
                self._get_related_topics(query),
                self._get_regional_interest(query)
            ]
            
            interest_df, related_queries, related_topics, regional_df = await asyncio.gather(*tasks)
            
            # Process interest over time data
            trends = []
            for keyword in query.keywords:
                if keyword in interest_df.columns:
                    keyword_data = interest_df[keyword]
                    
                    # Calculate trend metrics
                    avg_interest = keyword_data.mean()
                    peak_interest = keyword_data.max()
                    trend_direction, growth_rate = self._analyze_trend_direction(keyword_data)
                    
                    # Extract related data
                    keyword_related_queries = []
                    keyword_related_topics = []
                    
                    if keyword in related_queries and related_queries[keyword]:
                        if 'query' in related_queries[keyword]:
                            top_queries = related_queries[keyword]['query'].get('top')
                            if top_queries is not None and 'query' in top_queries.columns:
                                keyword_related_queries = top_queries['query'].tolist()[:5]
                    
                    if keyword in related_topics and related_topics[keyword]:
                        if 'topic' in related_topics[keyword]:
                            top_topics = related_topics[keyword]['topic'].get('top')
                            if top_topics is not None and 'topic_title' in top_topics.columns:
                                keyword_related_topics = top_topics['topic_title'].tolist()[:5]
                    
                    # Regional interest for this keyword
                    keyword_regional = {}
                    if not regional_df.empty and keyword in regional_df.columns:
                        regional_series = regional_df[keyword].dropna().sort_values(ascending=False)
                        keyword_regional = regional_series.head(5).to_dict()
                    
                    trend_insight = TrendInsight(
                        keyword=keyword,
                        avg_interest=float(avg_interest),
                        peak_interest=int(peak_interest),
                        trend_direction=trend_direction,
                        growth_rate=float(growth_rate),
                        related_queries=keyword_related_queries,
                        related_topics=keyword_related_topics,
                        regional_interest=keyword_regional
                    )
                    
                    trends.append(trend_insight)
            
            # Process all related data
            all_related_queries = self._process_related_data(related_queries, "query")
            top_queries = self._extract_top_queries(related_queries)
            rising_queries = self._extract_rising_queries(related_queries)
            regional_data = self._process_regional_data(regional_df, query.keywords)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            response = TrendResponse(
                query_keywords=query.keywords,
                timeframe=query.timeframe,
                geo=query.geo or "Worldwide",
                trends=trends,
                related_queries=all_related_queries,
                rising_queries=rising_queries,
                top_queries=top_queries,
                regional_data=regional_data,
                analysis_time=analysis_time,
                timestamp=datetime.now()
            )
            
            # Cache the result
            if use_cache:
                self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise ToolExecutionError(f"Trend analysis failed: {e}")
    
    async def get_trending_searches(self, country: str = "united_states") -> List[str]:
        """Get current trending searches for a country."""
        if not self.trends_client:
            raise ToolExecutionError("Google Trends client not initialized")
        
        try:
            # Get trending searches
            trending_searches = self.trends_client.trending_searches(pn=country)
            
            if trending_searches.empty:
                return []
            
            # Return top 20 trending searches
            return trending_searches[0].tolist()[:20]
            
        except Exception as e:
            logger.error(f"Failed to get trending searches: {e}")
            raise ToolExecutionError(f"Failed to get trending searches: {e}")
    
    async def compare_keywords(self, keywords: List[str], timeframe: str = "today 3-m") -> Dict[str, float]:
        """
        Compare relative popularity of keywords.
        
        Args:
            keywords: List of keywords to compare (max 5)
            timeframe: Time frame for comparison
            
        Returns:
            Dictionary mapping keywords to average interest scores
        """
        if len(keywords) > 5:
            raise ToolExecutionError("Maximum 5 keywords allowed for comparison")
        
        query = TrendQuery(keywords=keywords, timeframe=timeframe)
        interest_df = await self._get_interest_over_time(query)
        
        if interest_df.empty:
            return {keyword: 0.0 for keyword in keywords}
        
        # Calculate average interest for each keyword
        comparison = {}
        for keyword in keywords:
            if keyword in interest_df.columns:
                comparison[keyword] = float(interest_df[keyword].mean())
            else:
                comparison[keyword] = 0.0
        
        return comparison
    
    def clear_cache(self):
        """Clear the trends cache."""
        self.cache.clear()
        logger.info("Trends cache cleared")
    
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
trend_finder_tool = TrendFinderTool()


# MCP tool function
async def mcp_analyze_trends(
    keywords: List[str],
    timeframe: str = "today 3-m",
    geo: str = "",
    category: int = 0,
    property_filter: str = ""
) -> Dict:
    """
    MCP-compatible trend analysis function.
    
    Args:
        keywords: Keywords to analyze (max 5)
        timeframe: Time frame for analysis
        geo: Geographic location (empty for worldwide)
        category: Google Trends category (0 for all)
        property_filter: Property filter (web, news, youtube, etc.)
    
    Returns:
        Dictionary with trend analysis results
    """
    try:
        query = TrendQuery(
            keywords=keywords,
            timeframe=timeframe,
            geo=geo,
            category=category,
            property_filter=property_filter
        )
        
        response = await trend_finder_tool.analyze_trends(query)
        
        return {
            "success": True,
            "query_keywords": response.query_keywords,
            "timeframe": response.timeframe,
            "geo": response.geo,
            "trends": [
                {
                    "keyword": trend.keyword,
                    "avg_interest": trend.avg_interest,
                    "peak_interest": trend.peak_interest,
                    "trend_direction": trend.trend_direction,
                    "growth_rate": trend.growth_rate,
                    "related_queries": trend.related_queries,
                    "related_topics": trend.related_topics,
                    "regional_interest": trend.regional_interest
                }
                for trend in response.trends
            ],
            "related_queries": response.related_queries,
            "rising_queries": response.rising_queries,
            "top_queries": response.top_queries,
            "regional_data": response.regional_data,
            "analysis_time": response.analysis_time,
            "timestamp": response.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "keywords": keywords
        }


# MCP function for trending searches
async def mcp_get_trending_searches(country: str = "united_states") -> Dict:
    """
    MCP-compatible function to get trending searches.
    
    Args:
        country: Country for trending searches
    
    Returns:
        Dictionary with trending searches
    """
    try:
        trending = await trend_finder_tool.get_trending_searches(country)
        
        return {
            "success": True,
            "country": country,
            "trending_searches": trending,
            "count": len(trending),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get trending searches: {e}")
        return {
            "success": False,
            "error": str(e),
            "country": country
        }


# MCP function for keyword comparison
async def mcp_compare_keywords(
    keywords: List[str],
    timeframe: str = "today 3-m"
) -> Dict:
    """
    MCP-compatible function to compare keyword popularity.
    
    Args:
        keywords: Keywords to compare (max 5)
        timeframe: Time frame for comparison
    
    Returns:
        Dictionary with keyword comparison results
    """
    try:
        comparison = await trend_finder_tool.compare_keywords(keywords, timeframe)
        
        # Sort by interest score
        sorted_keywords = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "success": True,
            "keywords": keywords,
            "timeframe": timeframe,
            "comparison": comparison,
            "ranking": sorted_keywords,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Keyword comparison failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "keywords": keywords
        }