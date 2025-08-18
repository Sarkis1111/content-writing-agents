"""
Tools Module - Comprehensive Content Creation and Analysis Tools

This module provides a complete suite of tools for content creation, research, analysis,
and editing, organized into specialized categories.

Tool Categories:
- Research: Web search, content retrieval, trend analysis, news monitoring
- Analysis: Content processing, topic extraction, sentiment analysis, Reddit insights
- Writing: Content generation, headline creation, image generation
- Editing: Grammar checking, SEO analysis, readability scoring (Phase 2.4)

All tools are integrated with the Model Context Protocol (MCP) for external access.
"""

# Import all tool modules
from . import research
from . import analysis  
from . import writing
from . import editing

# Import tool registries
from .research import RESEARCH_TOOLS, MCP_RESEARCH_FUNCTIONS
from .analysis import ANALYSIS_TOOLS, MCP_ANALYSIS_FUNCTIONS
from .writing import WRITING_TOOLS, MCP_WRITING_FUNCTIONS
from .editing import EDITING_TOOLS, MCP_EDITING_FUNCTIONS

# Combined tool registry for easy access
ALL_TOOLS = {
    **RESEARCH_TOOLS,
    **ANALYSIS_TOOLS,
    **WRITING_TOOLS,
    **EDITING_TOOLS
}

# Combined MCP function registry
ALL_MCP_FUNCTIONS = {
    **MCP_RESEARCH_FUNCTIONS,
    **MCP_ANALYSIS_FUNCTIONS,
    **MCP_WRITING_FUNCTIONS,
    **MCP_EDITING_FUNCTIONS
}

# Tool categories for organization
TOOL_CATEGORIES = {
    'research': {
        'tools': RESEARCH_TOOLS,
        'description': 'Data gathering and information retrieval tools',
        'phase': '2.1'
    },
    'analysis': {
        'tools': ANALYSIS_TOOLS,
        'description': 'Content processing and insight generation tools',
        'phase': '2.2'
    },
    'writing': {
        'tools': WRITING_TOOLS,
        'description': 'Content creation and visual generation tools',
        'phase': '2.3'
    },
    'editing': {
        'tools': EDITING_TOOLS,
        'description': 'Quality assurance and content optimization tools',
        'phase': '2.4'
    }
}

__all__ = [
    # Modules
    'research',
    'analysis', 
    'writing',
    'editing',
    
    # Tool registries
    'ALL_TOOLS',
    'ALL_MCP_FUNCTIONS',
    'TOOL_CATEGORIES',
    'RESEARCH_TOOLS',
    'ANALYSIS_TOOLS', 
    'WRITING_TOOLS',
    'EDITING_TOOLS',
    'MCP_RESEARCH_FUNCTIONS',
    'MCP_ANALYSIS_FUNCTIONS',
    'MCP_WRITING_FUNCTIONS',
    'MCP_EDITING_FUNCTIONS'
]


# Utility functions
def get_tool(tool_name: str):
    """
    Get a tool instance by name from any category
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool instance or None if not found
    """
    return ALL_TOOLS.get(tool_name)


def get_tools_by_category(category: str):
    """
    Get all tools in a specific category
    
    Args:
        category: Category name ('research', 'analysis', 'writing', 'editing')
        
    Returns:
        Dictionary of tools in the category
    """
    category_info = TOOL_CATEGORIES.get(category)
    return category_info['tools'] if category_info else {}


def list_all_tools():
    """
    List all available tools with their categories and descriptions
    
    Returns:
        Dictionary organized by categories
    """
    tools_info = {}
    
    for category, info in TOOL_CATEGORIES.items():
        tools_info[category] = {
            'description': info['description'],
            'phase': info['phase'],
            'tools': list(info['tools'].keys())
        }
    
    return tools_info


def get_mcp_function(function_name: str):
    """
    Get an MCP function by name from any category
    
    Args:
        function_name: Name of the MCP function
        
    Returns:
        MCP function or None if not found
    """
    return ALL_MCP_FUNCTIONS.get(function_name)


def list_mcp_functions():
    """
    List all MCP functions organized by category
    
    Returns:
        Dictionary of MCP functions by category
    """
    return {
        'research': list(MCP_RESEARCH_FUNCTIONS.keys()),
        'analysis': list(MCP_ANALYSIS_FUNCTIONS.keys()),
        'writing': list(MCP_WRITING_FUNCTIONS.keys()),
        'editing': list(MCP_EDITING_FUNCTIONS.keys())
    }


def validate_all_tools():
    """
    Validate configuration and availability of all tools
    
    Returns:
        Dictionary with validation results for all tools
    """
    import os
    from datetime import datetime
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'unknown',
        'categories': {}
    }
    
    # Check research tools
    validation_results['categories']['research'] = {
        'status': 'ready' if any([
            os.getenv('SERPAPI_KEY'),
            os.getenv('GOOGLE_API_KEY'),
            os.getenv('NEWS_API_KEY')
        ]) else 'partial',
        'tools_count': len(RESEARCH_TOOLS),
        'api_dependencies': ['SERPAPI_KEY', 'GOOGLE_API_KEY', 'NEWS_API_KEY']
    }
    
    # Check analysis tools
    validation_results['categories']['analysis'] = {
        'status': 'ready',  # Most analysis tools don't require API keys
        'tools_count': len(ANALYSIS_TOOLS),
        'api_dependencies': ['REDDIT_CLIENT_ID (optional)']
    }
    
    # Check writing tools
    validation_results['categories']['writing'] = {
        'status': 'ready' if os.getenv('OPENAI_API_KEY') else 'missing_api_key',
        'tools_count': len(WRITING_TOOLS),
        'api_dependencies': ['OPENAI_API_KEY']
    }
    
    # Check editing tools
    validation_results['categories']['editing'] = {
        'status': 'ready',  # Most editing tools work offline
        'tools_count': len(EDITING_TOOLS),
        'api_dependencies': ['None - all tools work offline']
    }
    
    # Determine overall status
    statuses = [cat['status'] for cat in validation_results['categories'].values()]
    if all(status == 'ready' for status in statuses):
        validation_results['overall_status'] = 'ready'
    elif any(status == 'ready' for status in statuses):
        validation_results['overall_status'] = 'partial'
    else:
        validation_results['overall_status'] = 'not_ready'
    
    return validation_results


def get_tool_dependencies():
    """
    Get all external dependencies required by tools
    
    Returns:
        Dictionary of dependencies organized by type
    """
    return {
        'api_keys': {
            'required': ['OPENAI_API_KEY'],  # Required for writing tools
            'optional': ['SERPAPI_KEY', 'GOOGLE_API_KEY', 'NEWS_API_KEY', 'REDDIT_CLIENT_ID']
        },
        'python_packages': {
            'research': [
                'google-search-results>=2.4.2',
                'beautifulsoup4>=4.12.0', 
                'lxml>=4.9.0',
                'pytrends>=4.9.2',
                'newsapi-python>=0.2.6'
            ],
            'analysis': [
                'nltk>=3.8.1',
                'spacy>=3.7.0',
                'scikit-learn>=1.3.0',
                'textblob>=0.17.1',
                'langdetect>=1.0.9',
                'praw>=7.7.1',
                'textstat>=0.7.3',
                'vaderSentiment>=3.3.2'
            ],
            'writing': [
                'openai>=1.0.0',
                'pillow>=10.0.0',
                'aiohttp>=3.8.0',
                'aiofiles>=23.0.0'
            ]
        },
        'external_services': [
            'OpenAI API (GPT & DALL-E)',
            'SerpAPI (optional)',
            'Google Custom Search (optional)',
            'News API (optional)',
            'Reddit API (optional)'
        ]
    }


def create_complete_workflow(topic: str, content_type: str = "blog_post"):
    """
    Create a complete content workflow using tools from all categories
    
    Args:
        topic: Main topic for content creation
        content_type: Type of content to create
        
    Returns:
        Dictionary with workflow steps and tool assignments
    """
    workflow = {
        'topic': topic,
        'content_type': content_type,
        'phases': [
            {
                'phase': '1_research',
                'description': 'Gather information and insights',
                'tools': ['web_search', 'trend_finder', 'news_search'],
                'expected_outputs': ['research_data', 'trending_topics', 'news_insights']
            },
            {
                'phase': '2_analysis', 
                'description': 'Process and analyze collected data',
                'tools': ['content_processing', 'topic_extraction', 'content_analysis'],
                'expected_outputs': ['processed_content', 'key_topics', 'sentiment_analysis']
            },
            {
                'phase': '3_creation',
                'description': 'Generate content, headlines, and visuals',
                'tools': ['content_writer', 'headline_generator', 'image_generator'],
                'expected_outputs': ['generated_content', 'headline_variants', 'visual_assets']
            }
        ],
        'estimated_time': '15-30 minutes',
        'dependencies': ['OPENAI_API_KEY required', 'other API keys optional for enhanced results']
    }
    
    return workflow