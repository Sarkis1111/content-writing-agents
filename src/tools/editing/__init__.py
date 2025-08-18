"""
Editing Tools Module - Quality Assurance and Content Optimization

This module contains tools for content quality assurance, optimization, and refinement.
Includes grammar checking, SEO analysis, readability scoring, and sentiment analysis.

Available Tools:
- GrammarChecker: Grammar and style checking with multiple detection methods
- SEOAnalyzer: Search engine optimization analysis and recommendations
- ReadabilityScorer: Multi-metric readability analysis and audience alignment
- SentimentAnalyzer: Emotional tone analysis and brand voice consistency

Dependencies:
- language_tool_python: Grammar checking
- textstat: Readability metrics
- textblob: Sentiment and grammar analysis
- vaderSentiment: Social media optimized sentiment analysis
- beautifulsoup4: HTML parsing for SEO analysis
- nltk: Natural language processing
"""

# Tool imports
from .grammar_checker import (
    GrammarChecker,
    GrammarCheckRequest,
    GrammarCheckResults,
    GrammarError,
    ErrorType,
    ErrorSeverity,
    WritingStyle,
    Language,
    grammar_checker_tool,
    mcp_check_grammar
)

from .seo_analyzer import (
    SEOAnalyzer,
    SEOAnalysisRequest,
    SEOAnalysisResults,
    SEOIssue,
    SEOIssueType,
    SEOPriority,
    KeywordAnalysis,
    ContentStructure,
    MetaTagsAnalysis,
    LinkAnalysis,
    ContentType as SEOContentType,
    seo_analyzer_tool,
    mcp_analyze_seo
)

from .readability_scorer import (
    ReadabilityScorer,
    ReadabilityRequest,
    ReadabilityResults,
    ReadabilityScore,
    ReadabilityMetric,
    VocabularyAnalysis,
    SentenceAnalysis,
    AudienceAlignment,
    TargetAudience,
    ContentPurpose,
    readability_scorer_tool,
    mcp_score_readability
)

from .sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentAnalysisRequest,
    SentimentAnalysisResults,
    SentimentScore,
    EmotionAnalysis,
    BrandVoiceAnalysis,
    AudienceReaction,
    EmotionType,
    SentimentPolarity,
    BrandVoice,
    ContentMood,
    sentiment_analyzer_tool,
    mcp_analyze_sentiment
)

# Export tool instances for easy access
EDITING_TOOLS = {
    'grammar_checker': grammar_checker_tool,
    'seo_analyzer': seo_analyzer_tool,
    'readability_scorer': readability_scorer_tool,
    'sentiment_analyzer': sentiment_analyzer_tool
}

# Export all MCP functions for external integration
MCP_EDITING_FUNCTIONS = {
    # Grammar Checker Functions
    'check_grammar': mcp_check_grammar,
    
    # SEO Analyzer Functions
    'analyze_seo': mcp_analyze_seo,
    
    # Readability Scorer Functions
    'score_readability': mcp_score_readability,
    
    # Sentiment Analyzer Functions
    'analyze_sentiment': mcp_analyze_sentiment
}

# Export main classes for direct usage
__all__ = [
    # Grammar Checker
    'GrammarChecker',
    'GrammarCheckRequest',
    'GrammarCheckResults', 
    'GrammarError',
    'ErrorType',
    'ErrorSeverity',
    'WritingStyle',
    'Language',
    'grammar_checker_tool',
    'mcp_check_grammar',
    
    # SEO Analyzer
    'SEOAnalyzer',
    'SEOAnalysisRequest',
    'SEOAnalysisResults',
    'SEOIssue',
    'SEOIssueType',
    'SEOPriority',
    'KeywordAnalysis',
    'ContentStructure',
    'MetaTagsAnalysis',
    'LinkAnalysis',
    'SEOContentType',
    'seo_analyzer_tool',
    'mcp_analyze_seo',
    
    # Readability Scorer
    'ReadabilityScorer',
    'ReadabilityRequest',
    'ReadabilityResults',
    'ReadabilityScore',
    'ReadabilityMetric',
    'VocabularyAnalysis',
    'SentenceAnalysis',
    'AudienceAlignment',
    'TargetAudience',
    'ContentPurpose',
    'readability_scorer_tool',
    'mcp_score_readability',
    
    # Sentiment Analyzer
    'SentimentAnalyzer',
    'SentimentAnalysisRequest',
    'SentimentAnalysisResults',
    'SentimentScore',
    'EmotionAnalysis',
    'BrandVoiceAnalysis',
    'AudienceReaction',
    'EmotionType',
    'SentimentPolarity',
    'BrandVoice',
    'ContentMood',
    'sentiment_analyzer_tool',
    'mcp_analyze_sentiment',
    
    # Collections
    'EDITING_TOOLS',
    'MCP_EDITING_FUNCTIONS'
]


# Utility functions
def get_tool(tool_name: str):
    """
    Get an editing tool instance by name
    
    Args:
        tool_name: Name of the tool ('grammar_checker', 'seo_analyzer', etc.)
        
    Returns:
        Tool instance or None if not found
    """
    return EDITING_TOOLS.get(tool_name)


def list_available_tools():
    """
    List all available editing tools with descriptions
    
    Returns:
        Dictionary of tool names and descriptions
    """
    return {
        'grammar_checker': {
            'description': 'Grammar and style checking with multiple detection methods',
            'capabilities': [
                'Grammar and spelling correction',
                'Style consistency checking',
                'Language-specific rules',
                'Multi-engine detection (LanguageTool, TextBlob, patterns)',
                'Writing style analysis',
                'Auto-correction suggestions'
            ],
            'languages': ['en-US', 'en-GB', 'en-CA', 'en-AU'],
            'writing_styles': ['academic', 'business', 'casual', 'creative', 'technical']
        },
        'seo_analyzer': {
            'description': 'Search engine optimization analysis and recommendations',
            'capabilities': [
                'Keyword density analysis',
                'Meta tag optimization',
                'Content structure recommendations',
                'Internal/external link analysis',
                'Image SEO optimization',
                'Technical SEO checks'
            ],
            'content_types': ['blog_post', 'product_page', 'landing_page', 'news_article'],
            'analysis_areas': ['keywords', 'meta_tags', 'headings', 'links', 'readability']
        },
        'readability_scorer': {
            'description': 'Multi-metric readability analysis and audience alignment',
            'capabilities': [
                'Multiple readability formulas (8 metrics)',
                'Target audience alignment',
                'Vocabulary complexity analysis',
                'Sentence structure analysis',
                'Grade level recommendations',
                'Improvement suggestions'
            ],
            'metrics': ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog_index', 'smog_index'],
            'audiences': ['elementary_school', 'general_adult', 'professional', 'academic']
        },
        'sentiment_analyzer': {
            'description': 'Emotional tone analysis and brand voice consistency',
            'capabilities': [
                'Multi-dimensional sentiment analysis',
                'Emotion detection (8 primary emotions)',
                'Brand voice consistency checking',
                'Audience reaction prediction',
                'Content mood analysis',
                'Sentence-level sentiment tracking'
            ],
            'emotions': ['joy', 'trust', 'fear', 'surprise', 'sadness', 'anger'],
            'brand_voices': ['professional', 'friendly', 'authoritative', 'playful', 'empathetic']
        }
    }


def get_mcp_function(function_name: str):
    """
    Get an MCP function by name
    
    Args:
        function_name: Name of the MCP function
        
    Returns:
        MCP function or None if not found
    """
    return MCP_EDITING_FUNCTIONS.get(function_name)


def list_mcp_functions():
    """
    List all available MCP functions with descriptions
    
    Returns:
        Dictionary of MCP function names and descriptions
    """
    return {
        'check_grammar': 'Check grammar, spelling, and style consistency with comprehensive analysis',
        'analyze_seo': 'Analyze content for SEO optimization opportunities and recommendations',
        'score_readability': 'Score readability using multiple metrics and provide audience alignment',
        'analyze_sentiment': 'Analyze sentiment, emotions, and brand voice consistency'
    }


# Tool configuration and validation
def validate_editing_tools_config():
    """
    Validate configuration for editing tools
    
    Returns:
        Dictionary with validation results
    """
    import os
    
    results = {
        'grammar_checker': {
            'available': True,  # Core functionality doesn't require API keys
            'requirements': ['language_tool_python', 'textblob', 'nltk'],
            'status': 'ready',
            'optional_features': ['LanguageTool server for better performance']
        },
        'seo_analyzer': {
            'available': True,  # No API keys required for basic functionality
            'requirements': ['beautifulsoup4', 'textstat', 'nltk'],
            'status': 'ready',
            'optional_features': ['External SEO APIs for enhanced analysis']
        },
        'readability_scorer': {
            'available': True,  # All functionality available offline
            'requirements': ['textstat', 'nltk'],
            'status': 'ready',
            'optional_features': ['Custom readability formulas']
        },
        'sentiment_analyzer': {
            'available': True,  # TextBlob and VADER work offline
            'requirements': ['textblob', 'vaderSentiment', 'nltk'],
            'status': 'ready',
            'optional_features': ['Advanced emotion detection APIs']
        }
    }
    
    return results