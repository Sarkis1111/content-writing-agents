"""
CrewAI-compatible tool wrappers for research tools.
This module wraps our custom tools to work with CrewAI's BaseTool interface.
"""

import asyncio
from typing import Any, Dict, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .web_search import web_search_tool, SearchQuery
from .content_retrieval import content_retrieval_tool, ContentExtractionRequest
from .trend_finder import trend_finder_tool, TrendQuery
from .news_search import news_search_tool, NewsQuery


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Maximum number of results")
    search_type: str = Field(default="general", description="Type of search: general, news, images")
    region: str = Field(default="US", description="Search region")
    language: str = Field(default="en", description="Search language")


class WebSearchTool(BaseTool):
    """CrewAI-compatible web search tool."""
    
    name: str = "web_search"
    description: str = "Search the web for information on any topic. Useful for finding current information, news, and general knowledge."
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str, max_results: int = 10, search_type: str = "general", 
             region: str = "US", language: str = "en") -> str:
        """Execute web search synchronously."""
        try:
            # Create search query
            search_query = SearchQuery(
                query=query,
                max_results=max_results,
                search_type=search_type,
                region=region,
                language=language
            )
            
            # Run async search in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(web_search_tool.search(search_query))
                
                # Format results for CrewAI
                if response.results:
                    formatted_results = []
                    for result in response.results[:max_results]:
                        formatted_results.append(
                            f"Title: {result.title}\n"
                            f"URL: {result.url}\n"
                            f"Content: {result.snippet[:200]}...\n"
                            f"---"
                        )
                    return "\n\n".join(formatted_results)
                else:
                    return f"No results found for query: {query}"
                    
            finally:
                loop.close()
                
        except Exception as e:
            return f"Error performing web search: {str(e)}"


class ContentRetrievalInput(BaseModel):
    """Input schema for content retrieval tool."""
    url: str = Field(description="URL to extract content from")
    extract_main_content: bool = Field(default=True, description="Extract main content")
    extract_metadata: bool = Field(default=True, description="Extract metadata")
    max_length: int = Field(default=5000, description="Maximum content length")


class ContentRetrievalTool(BaseTool):
    """CrewAI-compatible content retrieval tool."""
    
    name: str = "content_retrieval"
    description: str = "Extract and retrieve content from web pages. Useful for getting full text content from URLs."
    args_schema: Type[BaseModel] = ContentRetrievalInput
    
    def _run(self, url: str, extract_main_content: bool = True, 
             extract_metadata: bool = True, max_length: int = 5000) -> str:
        """Extract content from URL synchronously."""
        try:
            # Create extraction request
            request = ContentExtractionRequest(
                url=url,
                extract_main_content=extract_main_content,
                extract_metadata=extract_metadata,
                max_length=max_length
            )
            
            # Run async extraction in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(content_retrieval_tool.extract_content(request))
                
                if response.success and response.content:
                    content = response.content
                    result = f"Title: {content.title}\n"
                    result += f"Content: {content.main_content[:max_length]}\n"
                    
                    if content.metadata:
                        result += f"Description: {content.metadata.get('description', 'N/A')}\n"
                        result += f"Author: {content.metadata.get('author', 'N/A')}\n"
                    
                    return result
                else:
                    return f"Failed to extract content from {url}: {response.error}"
                    
            finally:
                loop.close()
                
        except Exception as e:
            return f"Error extracting content: {str(e)}"


class TrendAnalysisInput(BaseModel):
    """Input schema for trend analysis tool."""
    keywords: str = Field(description="Keywords to analyze trends for (comma-separated)")
    timeframe: str = Field(default="7d", description="Timeframe for analysis: 1d, 7d, 30d, 90d")
    region: str = Field(default="US", description="Region for trend analysis")


class TrendAnalysisTool(BaseTool):
    """CrewAI-compatible trend analysis tool."""
    
    name: str = "trend_analysis"
    description: str = "Analyze trends for keywords and topics. Useful for understanding popularity and search volume changes."
    args_schema: Type[BaseModel] = TrendAnalysisInput
    
    def _run(self, keywords: str, timeframe: str = "7d", region: str = "US") -> str:
        """Analyze trends synchronously."""
        try:
            # Parse keywords
            keyword_list = [k.strip() for k in keywords.split(",")]
            
            # Create trend query
            query = TrendQuery(
                keywords=keyword_list,
                timeframe=timeframe,
                region=region
            )
            
            # Run async trend analysis in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(trend_finder_tool.analyze_trends(query))
                
                if response.success and response.insights:
                    results = []
                    for insight in response.insights:
                        results.append(
                            f"Keyword: {insight.keyword}\n"
                            f"Interest Score: {insight.interest_score:.2f}\n"
                            f"Trend Direction: {insight.trend_direction}\n"
                            f"Related Queries: {', '.join(insight.related_queries[:3])}\n"
                            f"---"
                        )
                    return "\n\n".join(results)
                else:
                    return f"No trend data found for keywords: {keywords}"
                    
            finally:
                loop.close()
                
        except Exception as e:
            return f"Error analyzing trends: {str(e)}"


class NewsSearchInput(BaseModel):
    """Input schema for news search tool."""
    query: str = Field(description="News search query")
    max_results: int = Field(default=10, description="Maximum number of articles")
    category: str = Field(default="general", description="News category")
    language: str = Field(default="en", description="News language")
    timeframe: str = Field(default="7d", description="Time range: 1d, 7d, 30d")


class NewsSearchTool(BaseTool):
    """CrewAI-compatible news search tool."""
    
    name: str = "news_search"
    description: str = "Search for recent news articles on any topic. Useful for finding current events and news coverage."
    args_schema: Type[BaseModel] = NewsSearchInput
    
    def _run(self, query: str, max_results: int = 10, category: str = "general",
             language: str = "en", timeframe: str = "7d") -> str:
        """Search news synchronously."""
        try:
            # Create news query
            news_query = NewsQuery(
                query=query,
                max_results=max_results,
                category=category,
                language=language,
                time_range=timeframe
            )
            
            # Run async news search in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(news_search_tool.search_news(news_query))
                
                if response.success and response.articles:
                    results = []
                    for article in response.articles[:max_results]:
                        results.append(
                            f"Title: {article.title}\n"
                            f"Source: {article.source}\n"
                            f"Published: {article.published_date}\n"
                            f"Summary: {article.summary[:200]}...\n"
                            f"URL: {article.url}\n"
                            f"---"
                        )
                    return "\n\n".join(results)
                else:
                    return f"No news articles found for query: {query}"
                    
            finally:
                loop.close()
                
        except Exception as e:
            return f"Error searching news: {str(e)}"


# CrewAI-compatible tool instances
crewai_web_search = WebSearchTool()
crewai_content_retrieval = ContentRetrievalTool()
crewai_trend_analysis = TrendAnalysisTool()
crewai_news_search = NewsSearchTool()

# Tool collection for easy access
CREWAI_RESEARCH_TOOLS = [
    crewai_web_search,
    crewai_content_retrieval,
    crewai_trend_analysis,
    crewai_news_search
]