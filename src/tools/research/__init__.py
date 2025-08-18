"""
Research Tools Module

This module provides comprehensive research capabilities for content creation,
including web search, content retrieval, trend analysis, and news monitoring.
"""

from .web_search import (
    WebSearchTool,
    web_search_tool,
    mcp_web_search,
    SearchQuery,
    SearchResult,
    SearchResponse
)

from .content_retrieval import (
    ContentRetrievalTool,
    content_retrieval_tool,
    mcp_extract_content,
    ContentExtractionRequest,
    ExtractedContent,
    ContentExtractionResponse
)

from .trend_finder import (
    TrendFinderTool,
    trend_finder_tool,
    mcp_analyze_trends,
    mcp_get_trending_searches,
    mcp_compare_keywords,
    TrendQuery,
    TrendInsight,
    TrendResponse
)

from .news_search import (
    NewsSearchTool,
    news_search_tool,
    mcp_search_news,
    mcp_get_headlines,
    mcp_get_news_sources,
    NewsQuery,
    HeadlinesQuery,
    NewsArticle,
    NewsSource,
    NewsResponse
)


__all__ = [
    # Web Search
    'WebSearchTool',
    'web_search_tool',
    'mcp_web_search',
    'SearchQuery',
    'SearchResult', 
    'SearchResponse',
    
    # Content Retrieval
    'ContentRetrievalTool',
    'content_retrieval_tool',
    'mcp_extract_content',
    'ContentExtractionRequest',
    'ExtractedContent',
    'ContentExtractionResponse',
    
    # Trend Finder
    'TrendFinderTool',
    'trend_finder_tool',
    'mcp_analyze_trends',
    'mcp_get_trending_searches', 
    'mcp_compare_keywords',
    'TrendQuery',
    'TrendInsight',
    'TrendResponse',
    
    # News Search
    'NewsSearchTool',
    'news_search_tool',
    'mcp_search_news',
    'mcp_get_headlines',
    'mcp_get_news_sources',
    'NewsQuery',
    'HeadlinesQuery',
    'NewsArticle',
    'NewsSource',
    'NewsResponse'
]


# Tool instances for easy access
RESEARCH_TOOLS = {
    'web_search': web_search_tool,
    'content_retrieval': content_retrieval_tool,
    'trend_finder': trend_finder_tool,
    'news_search': news_search_tool
}


# MCP functions for easy registration
MCP_RESEARCH_FUNCTIONS = {
    'web_search': mcp_web_search,
    'extract_content': mcp_extract_content,
    'analyze_trends': mcp_analyze_trends,
    'get_trending_searches': mcp_get_trending_searches,
    'compare_keywords': mcp_compare_keywords,
    'search_news': mcp_search_news,
    'get_headlines': mcp_get_headlines,
    'get_news_sources': mcp_get_news_sources
}