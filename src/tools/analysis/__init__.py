"""
Analysis Tools Module

This module provides comprehensive analysis capabilities for content creation,
including content processing, topic extraction, content analysis, and Reddit insights.
"""

from .content_processing import (
    ContentProcessingTool,
    content_processing_tool,
    mcp_process_content,
    mcp_detect_duplicates,
    ContentProcessingRequest,
    ProcessedContent,
    ContentProcessingResponse,
    LanguageInfo
)

from .topic_extraction import (
    TopicExtractionTool,
    topic_extraction_tool,
    mcp_extract_topics,
    TopicExtractionRequest,
    ExtractedKeyword,
    ExtractedTopic,
    NamedEntity,
    TopicExtractionResult,
    TopicExtractionResponse
)

from .content_analysis import (
    ContentAnalysisTool,
    content_analysis_tool,
    mcp_analyze_content,
    ContentAnalysisRequest,
    SentimentAnalysis,
    ReadabilityMetrics,
    StyleAnalysis,
    ContentQuality,
    ContentAnalysisResult,
    ContentAnalysisResponse
)

from .reddit_search import (
    RedditSearchTool,
    reddit_search_tool,
    mcp_search_reddit,
    mcp_analyze_subreddit,
    mcp_get_trending_discussions,
    RedditSearchRequest,
    RedditPost,
    RedditComment,
    SubredditInfo,
    SubredditAnalysis,
    RedditSearchResponse
)


__all__ = [
    # Content Processing
    'ContentProcessingTool',
    'content_processing_tool',
    'mcp_process_content',
    'mcp_detect_duplicates',
    'ContentProcessingRequest',
    'ProcessedContent',
    'ContentProcessingResponse',
    'LanguageInfo',
    
    # Topic Extraction
    'TopicExtractionTool',
    'topic_extraction_tool',
    'mcp_extract_topics',
    'TopicExtractionRequest',
    'ExtractedKeyword',
    'ExtractedTopic',
    'NamedEntity',
    'TopicExtractionResult',
    'TopicExtractionResponse',
    
    # Content Analysis
    'ContentAnalysisTool',
    'content_analysis_tool',
    'mcp_analyze_content',
    'ContentAnalysisRequest',
    'SentimentAnalysis',
    'ReadabilityMetrics',
    'StyleAnalysis',
    'ContentQuality',
    'ContentAnalysisResult',
    'ContentAnalysisResponse',
    
    # Reddit Search
    'RedditSearchTool',
    'reddit_search_tool',
    'mcp_search_reddit',
    'mcp_analyze_subreddit',
    'mcp_get_trending_discussions',
    'RedditSearchRequest',
    'RedditPost',
    'RedditComment',
    'SubredditInfo',
    'SubredditAnalysis',
    'RedditSearchResponse'
]


# Tool instances for easy access
ANALYSIS_TOOLS = {
    'content_processing': content_processing_tool,
    'topic_extraction': topic_extraction_tool,
    'content_analysis': content_analysis_tool,
    'reddit_search': reddit_search_tool
}


# MCP functions for easy registration
MCP_ANALYSIS_FUNCTIONS = {
    'process_content': mcp_process_content,
    'detect_duplicates': mcp_detect_duplicates,
    'extract_topics': mcp_extract_topics,
    'analyze_content': mcp_analyze_content,
    'search_reddit': mcp_search_reddit,
    'analyze_subreddit': mcp_analyze_subreddit,
    'get_trending_discussions': mcp_get_trending_discussions
}