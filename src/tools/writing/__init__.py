"""
Writing Tools Module - AI-Powered Content Generation

This module contains tools for creating written and visual content using advanced AI models.
Includes content writing, headline generation, and image creation capabilities.

Available Tools:
- ContentWriter: GPT-powered content generation with multiple formats and styles
- HeadlineGenerator: AI-powered headline creation with A/B testing and optimization
- ImageGenerator: DALL-E integration for content-relevant image generation

Dependencies:
- openai: OpenAI API for GPT and DALL-E models
- Pillow: Image processing capabilities
- aiohttp: Async HTTP client for image downloads
- aiofiles: Async file operations
"""

# Tool imports
from .content_writer import (
    ContentWriter,
    ContentRequest,
    GeneratedContent,
    ContentType as WriterContentType,
    Tone,
    Style,
    GPTModel,
    content_writer_tool,
    mcp_generate_content,
    mcp_generate_variants
)

from .headline_generator import (
    HeadlineGenerator,
    HeadlineRequest,
    GeneratedHeadline,
    HeadlineResults,
    HeadlineStyle,
    HeadlineTone,
    Platform,
    EmotionalTrigger,
    headline_generator_tool,
    mcp_generate_headlines,
    mcp_optimize_headline_for_platform
)

from .image_generator import (
    ImageGenerator,
    ImageRequest,
    GeneratedImage,
    ImageResults,
    ImageSize,
    ImageStyle,
    ContentType as ImageContentType,
    ColorScheme,
    image_generator_tool,
    mcp_generate_images
)

# Export tool instances for easy access
WRITING_TOOLS = {
    'content_writer': content_writer_tool,
    'headline_generator': headline_generator_tool,
    'image_generator': image_generator_tool
}

# Export all MCP functions for external integration
MCP_WRITING_FUNCTIONS = {
    # Content Writer Functions
    'generate_content': mcp_generate_content,
    'generate_content_variants': mcp_generate_variants,
    
    # Headline Generator Functions
    'generate_headlines': mcp_generate_headlines,
    'optimize_headline_for_platform': mcp_optimize_headline_for_platform,
    
    # Image Generator Functions
    'generate_images': mcp_generate_images
}

# Export main classes for direct usage
__all__ = [
    # Content Writer
    'ContentWriter',
    'ContentRequest', 
    'GeneratedContent',
    'WriterContentType',
    'Tone',
    'Style',
    'GPTModel',
    'content_writer_tool',
    'mcp_generate_content',
    'mcp_generate_variants',
    
    # Headline Generator
    'HeadlineGenerator',
    'HeadlineRequest',
    'GeneratedHeadline',
    'HeadlineResults',
    'HeadlineStyle',
    'HeadlineTone',
    'Platform',
    'EmotionalTrigger',
    'headline_generator_tool',
    'mcp_generate_headlines',
    'mcp_optimize_headline_for_platform',
    
    # Image Generator
    'ImageGenerator',
    'ImageRequest',
    'GeneratedImage',
    'ImageResults',
    'ImageSize',
    'ImageStyle',
    'ImageContentType',
    'ColorScheme',
    'image_generator_tool',
    'mcp_generate_images',
    
    # Collections
    'WRITING_TOOLS',
    'MCP_WRITING_FUNCTIONS'
]


# Utility functions
def get_tool(tool_name: str):
    """
    Get a writing tool instance by name
    
    Args:
        tool_name: Name of the tool ('content_writer', 'headline_generator', 'image_generator')
        
    Returns:
        Tool instance or None if not found
    """
    return WRITING_TOOLS.get(tool_name)


def list_available_tools():
    """
    List all available writing tools with descriptions
    
    Returns:
        Dictionary of tool names and descriptions
    """
    return {
        'content_writer': {
            'description': 'GPT-powered content generation with multiple formats and styles',
            'capabilities': [
                'Blog posts, articles, social media content',
                'Multiple tones and writing styles',
                'SEO optimization and keyword integration',
                'Quality scoring and analysis',
                'A/B testing variants'
            ],
            'models': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o']
        },
        'headline_generator': {
            'description': 'AI-powered headline creation with optimization and A/B testing',
            'capabilities': [
                'Multiple headline styles (question, how-to, listicle, etc.)',
                'Platform-specific optimization',
                'Emotional impact scoring',
                'Click-through rate prediction',
                'A/B testing recommendations'
            ],
            'styles': ['question', 'how_to', 'listicle', 'news', 'benefit', 'curiosity', 'urgent']
        },
        'image_generator': {
            'description': 'DALL-E integration for content-relevant image generation',
            'capabilities': [
                'Content-relevant image generation',
                'Multiple styles and formats',
                'Brand color integration',
                'Mood and atmosphere control',
                'Batch generation for variants'
            ],
            'formats': ['1024x1024', '1024x1792', '1792x1024']
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
    return MCP_WRITING_FUNCTIONS.get(function_name)


def list_mcp_functions():
    """
    List all available MCP functions with descriptions
    
    Returns:
        Dictionary of MCP function names and descriptions
    """
    return {
        'generate_content': 'Generate content using GPT models with customization options',
        'generate_content_variants': 'Generate multiple content variants for A/B testing',
        'generate_headlines': 'Generate headlines with analysis and optimization',
        'optimize_headline_for_platform': 'Optimize existing headlines for specific platforms',
        'generate_images': 'Generate images using DALL-E with content relevance'
    }


# Tool configuration and validation
def validate_writing_tools_config():
    """
    Validate configuration for writing tools
    
    Returns:
        Dictionary with validation results
    """
    import os
    
    results = {
        'content_writer': {
            'available': bool(os.getenv('OPENAI_API_KEY')),
            'requirements': ['OPENAI_API_KEY environment variable'],
            'status': 'ready' if os.getenv('OPENAI_API_KEY') else 'missing_api_key'
        },
        'headline_generator': {
            'available': bool(os.getenv('OPENAI_API_KEY')),
            'requirements': ['OPENAI_API_KEY environment variable'],
            'status': 'ready' if os.getenv('OPENAI_API_KEY') else 'missing_api_key'
        },
        'image_generator': {
            'available': bool(os.getenv('OPENAI_API_KEY')),
            'requirements': ['OPENAI_API_KEY environment variable', 'PIL (Pillow)', 'aiohttp', 'aiofiles'],
            'status': 'ready' if os.getenv('OPENAI_API_KEY') else 'missing_api_key'
        }
    }
    
    return results


# Example usage demonstrations
async def demo_content_writer():
    """Demonstrate content writer capabilities"""
    from .content_writer import ContentRequest, ContentType, Tone, Style
    
    request = ContentRequest(
        topic="The Benefits of Remote Work for Productivity",
        content_type=ContentType.BLOG_POST,
        tone=Tone.PROFESSIONAL,
        style=Style.JOURNALISTIC,
        target_length=800,
        keywords=["remote work", "productivity", "work from home"]
    )
    
    result = await content_writer_tool.generate_content(request)
    return result


async def demo_headline_generator():
    """Demonstrate headline generator capabilities"""
    from .headline_generator import HeadlineRequest, HeadlineStyle, HeadlineTone, Platform
    
    request = HeadlineRequest(
        topic="Boost Your Website Conversion Rate",
        style=HeadlineStyle.HOW_TO,
        tone=HeadlineTone.PROFESSIONAL,
        platform=Platform.BLOG,
        num_variants=3,
        include_numbers=True
    )
    
    result = await headline_generator_tool.generate_headlines(request)
    return result


async def demo_image_generator():
    """Demonstrate image generator capabilities"""
    from .image_generator import ImageRequest, ImageContentType, ImageStyle, ColorScheme
    
    request = ImageRequest(
        content_topic="Sustainable Technology Innovation",
        content_type=ImageContentType.BLOG_HEADER,
        style=ImageStyle.MODERN,
        color_scheme=ColorScheme.COOL,
        mood="inspiring"
    )
    
    result = await image_generator_tool.generate_images(request)
    return result


# Integration helpers
def create_content_package(
    topic: str,
    content_type: str = "blog_post",
    include_headlines: bool = True,
    include_images: bool = True,
    num_headline_variants: int = 5,
    num_image_variants: int = 2
):
    """
    Create a complete content package with text, headlines, and images
    
    Args:
        topic: Main topic for content
        content_type: Type of content to create
        include_headlines: Generate headlines
        include_images: Generate images
        num_headline_variants: Number of headline variants
        num_image_variants: Number of image variants
        
    Returns:
        Dictionary with content generation parameters
    """
    package = {
        'content_request': {
            'topic': topic,
            'content_type': content_type,
            'include_outline': True,
            'include_meta': True
        }
    }
    
    if include_headlines:
        package['headline_request'] = {
            'topic': topic,
            'num_variants': num_headline_variants,
            'platform': 'blog' if content_type == 'blog_post' else 'social_media'
        }
    
    if include_images:
        package['image_request'] = {
            'content_topic': topic,
            'content_type': 'blog_header' if content_type == 'blog_post' else 'article_illustration',
            'num_variants': num_image_variants
        }
    
    return package


def get_tool_capabilities():
    """
    Get comprehensive capabilities of all writing tools
    
    Returns:
        Dictionary with detailed capabilities
    """
    return {
        'content_generation': {
            'formats': ['blog_post', 'article', 'social_media', 'email', 'press_release'],
            'tones': ['professional', 'casual', 'conversational', 'formal', 'friendly'],
            'styles': ['academic', 'journalistic', 'creative', 'technical', 'marketing'],
            'features': ['seo_optimization', 'quality_scoring', 'variant_generation']
        },
        'headline_generation': {
            'styles': ['question', 'how_to', 'listicle', 'news', 'benefit', 'curiosity'],
            'platforms': ['blog', 'email', 'social_media', 'ads'],
            'features': ['ab_testing', 'emotional_scoring', 'ctr_prediction']
        },
        'image_generation': {
            'formats': ['1024x1024', '1024x1792', '1792x1024'],
            'styles': ['natural', 'vivid', 'photographic', 'digital_art', 'painting'],
            'features': ['content_relevance', 'brand_colors', 'mood_control']
        }
    }