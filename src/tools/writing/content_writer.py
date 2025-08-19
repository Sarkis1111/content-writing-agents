"""
Content Writer Tool - GPT Integration for Content Generation

This module provides AI-powered content generation capabilities using OpenAI's GPT models.
Supports multiple content types, tone/style customization, and advanced generation parameters.

Key Features:
- Multiple GPT model support (GPT-3.5-turbo, GPT-4, GPT-4-turbo)
- Content type specialization (blog, article, social, email, etc.)
- Tone and style customization
- Template-based generation
- Quality scoring and validation
- Concurrent generation for A/B testing
- Advanced prompt engineering
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal, Any
from enum import Enum
import re
import json

import openai
from pydantic import BaseModel, Field, validator
from openai import AsyncOpenAI

try:
    from ...core.errors import ToolError
    from ...core.logging.logger import get_logger
    from ...utils.simple_retry import with_retry
except ImportError:
    # Handle direct import case
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from core.errors import ToolError
    from core.logging.logger import get_logger
    from utils.simple_retry import with_retry

logger = get_logger(__name__)


class ContentType(str, Enum):
    """Supported content types for generation"""
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    PRESS_RELEASE = "press_release"
    PRODUCT_DESCRIPTION = "product_description"
    LANDING_PAGE = "landing_page"
    NEWSLETTER = "newsletter"
    TECHNICAL_DOCS = "technical_docs"
    CREATIVE_WRITING = "creative_writing"


class Tone(str, Enum):
    """Content tone options"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    HUMOROUS = "humorous"
    PERSUASIVE = "persuasive"
    INFORMATIVE = "informative"
    INSPIRATIONAL = "inspirational"


class Style(str, Enum):
    """Writing style options"""
    ACADEMIC = "academic"
    JOURNALISTIC = "journalistic"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    MARKETING = "marketing"
    STORYTELLING = "storytelling"
    LISTICLE = "listicle"
    HOW_TO = "how_to"
    INTERVIEW = "interview"
    REVIEW = "review"


class GPTModel(str, Enum):
    """Supported GPT models"""
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4O = "gpt-4o"


class ContentRequest(BaseModel):
    """Content generation request parameters"""
    
    topic: str = Field(..., description="Main topic or title for the content")
    content_type: ContentType = Field(..., description="Type of content to generate")
    tone: Tone = Field(default=Tone.PROFESSIONAL, description="Tone of the content")
    style: Style = Field(default=Style.JOURNALISTIC, description="Writing style")
    
    target_length: Optional[int] = Field(
        default=None, 
        description="Target word count (optional)",
        ge=50,
        le=10000
    )
    
    target_audience: Optional[str] = Field(
        default=None,
        description="Target audience description"
    )
    
    key_points: Optional[List[str]] = Field(
        default=None,
        description="Key points to include in the content"
    )
    
    keywords: Optional[List[str]] = Field(
        default=None,
        description="SEO keywords to naturally incorporate"
    )
    
    context_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context data (research, facts, etc.)"
    )
    
    model: GPTModel = Field(default=GPTModel.GPT_4O, description="GPT model to use")
    temperature: float = Field(default=0.7, description="Creativity level", ge=0.0, le=2.0)
    
    include_outline: bool = Field(default=False, description="Include content outline")
    include_meta: bool = Field(default=True, description="Include meta description and tags")
    
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Additional custom instructions"
    )

    @validator('keywords')
    def validate_keywords(cls, v):
        if v and len(v) > 20:
            raise ValueError("Maximum 20 keywords allowed")
        return v

    @validator('key_points')
    def validate_key_points(cls, v):
        if v and len(v) > 15:
            raise ValueError("Maximum 15 key points allowed")
        return v


class GeneratedContent(BaseModel):
    """Generated content response"""
    
    content: str = Field(..., description="Generated content")
    title: str = Field(..., description="Generated title")
    
    outline: Optional[List[str]] = Field(default=None, description="Content outline")
    meta_description: Optional[str] = Field(default=None, description="Meta description")
    tags: Optional[List[str]] = Field(default=None, description="Content tags")
    
    word_count: int = Field(..., description="Actual word count")
    estimated_reading_time: int = Field(..., description="Estimated reading time in minutes")
    
    quality_score: float = Field(..., description="Content quality score (0-100)")
    quality_factors: Dict[str, float] = Field(..., description="Quality breakdown")
    
    model_used: str = Field(..., description="GPT model used for generation")
    generation_time: float = Field(..., description="Generation time in seconds")
    
    prompt_tokens: int = Field(..., description="Input tokens used")
    completion_tokens: int = Field(..., description="Output tokens generated")
    total_cost: Optional[float] = Field(default=None, description="Estimated cost in USD")


class ContentTemplate:
    """Content templates for different types"""
    
    TEMPLATES = {
        ContentType.BLOG_POST: {
            "structure": [
                "Engaging introduction with hook",
                "Main content with subheadings",
                "Key insights and examples",
                "Conclusion with call-to-action"
            ],
            "prompt_prefix": "Write a comprehensive blog post about",
            "length_guide": "800-2000 words"
        },
        
        ContentType.ARTICLE: {
            "structure": [
                "Compelling headline and lead",
                "Background and context",
                "Main arguments with evidence",
                "Expert quotes and data",
                "Balanced conclusion"
            ],
            "prompt_prefix": "Write a well-researched article on",
            "length_guide": "1000-3000 words"
        },
        
        ContentType.SOCIAL_MEDIA: {
            "structure": [
                "Attention-grabbing opening",
                "Key message or value",
                "Engagement hook",
                "Hashtags and call-to-action"
            ],
            "prompt_prefix": "Create engaging social media content about",
            "length_guide": "50-280 characters"
        },
        
        ContentType.EMAIL: {
            "structure": [
                "Compelling subject line",
                "Personal greeting",
                "Clear value proposition",
                "Strong call-to-action"
            ],
            "prompt_prefix": "Write a professional email about",
            "length_guide": "150-500 words"
        },
        
        ContentType.PRODUCT_DESCRIPTION: {
            "structure": [
                "Product name and key benefit",
                "Feature highlights",
                "Use cases and benefits",
                "Technical specifications",
                "Call-to-action"
            ],
            "prompt_prefix": "Write a compelling product description for",
            "length_guide": "100-300 words"
        }
    }
    
    @classmethod
    def get_template(cls, content_type: ContentType) -> Dict[str, Any]:
        """Get template for content type"""
        return cls.TEMPLATES.get(
            content_type, 
            cls.TEMPLATES[ContentType.ARTICLE]  # Default fallback
        )


class ContentWriter:
    """AI-powered content generation tool using OpenAI GPT models"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Content Writer
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Model pricing (per 1K tokens) - approximate as of 2024
        self.model_pricing = {
            GPTModel.GPT_3_5_TURBO: {"input": 0.0015, "output": 0.002},
            GPTModel.GPT_4: {"input": 0.03, "output": 0.06},
            GPTModel.GPT_4_TURBO: {"input": 0.01, "output": 0.03},
            GPTModel.GPT_4O: {"input": 0.005, "output": 0.015}
        }
        
        logger.info("ContentWriter initialized with OpenAI integration")

    def _build_system_prompt(self, request: ContentRequest) -> str:
        """Build system prompt based on request parameters"""
        
        template = ContentTemplate.get_template(request.content_type)
        
        system_prompt = f"""You are an expert content writer specializing in {request.content_type.value} content.

Writing Guidelines:
- Tone: {request.tone.value}
- Style: {request.style.value}
- Content Type: {request.content_type.value}

Content Structure: {', '.join(template['structure'])}
Recommended Length: {template.get('length_guide', 'As appropriate')}
"""
        
        if request.target_length:
            system_prompt += f"\nTarget Word Count: Approximately {request.target_length} words"
        
        if request.target_audience:
            system_prompt += f"\nTarget Audience: {request.target_audience}"
        
        system_prompt += """

Quality Requirements:
- Clear, engaging, and well-structured content
- Appropriate for the specified tone and style
- Include relevant examples and insights
- Ensure smooth flow and readability
- Use active voice where appropriate
- Avoid jargon unless necessary for the audience

Always provide high-quality, original content that meets the specific requirements."""
        
        return system_prompt

    def _build_user_prompt(self, request: ContentRequest) -> str:
        """Build user prompt with all request details"""
        
        template = ContentTemplate.get_template(request.content_type)
        prompt = f"{template['prompt_prefix']} '{request.topic}'"
        
        if request.key_points:
            prompt += f"\n\nKey Points to Cover:\n" + "\n".join([f"- {point}" for point in request.key_points])
        
        if request.keywords:
            prompt += f"\n\nSEO Keywords to Include Naturally: {', '.join(request.keywords)}"
        
        if request.context_data:
            prompt += f"\n\nAdditional Context:\n{json.dumps(request.context_data, indent=2)}"
        
        if request.custom_instructions:
            prompt += f"\n\nAdditional Instructions: {request.custom_instructions}"
        
        formatting_instructions = f"""

Formatting Requirements:
- Provide a compelling title
- Structure with clear headings and subheadings
- Use bullet points or numbered lists where appropriate
- Include a brief outline if requested
"""
        
        if request.include_meta:
            formatting_instructions += "- Include a meta description (150-160 characters)\n- Suggest 3-5 relevant tags\n"
        
        prompt += formatting_instructions
        
        return prompt

    def _calculate_reading_time(self, word_count: int) -> int:
        """Calculate estimated reading time (average 200 words per minute)"""
        return max(1, round(word_count / 200))

    def _assess_content_quality(self, content: str, request: ContentRequest) -> tuple[float, Dict[str, float]]:
        """Assess content quality based on multiple factors"""
        
        factors = {}
        
        # Length appropriateness (0-100)
        word_count = len(content.split())
        if request.target_length:
            length_ratio = word_count / request.target_length
            if 0.8 <= length_ratio <= 1.2:
                factors["length"] = 100.0
            elif 0.6 <= length_ratio <= 1.5:
                factors["length"] = 75.0
            else:
                factors["length"] = 50.0
        else:
            # Generic length assessment
            if word_count < 50:
                factors["length"] = 30.0
            elif word_count < 200:
                factors["length"] = 70.0
            else:
                factors["length"] = 90.0
        
        # Structure quality (0-100)
        headings = len(re.findall(r'^#+\s', content, re.MULTILINE))
        paragraphs = len(content.split('\n\n'))
        if headings >= 2 and paragraphs >= 3:
            factors["structure"] = 95.0
        elif headings >= 1 and paragraphs >= 2:
            factors["structure"] = 80.0
        else:
            factors["structure"] = 60.0
        
        # Keyword inclusion (0-100)
        if request.keywords:
            content_lower = content.lower()
            keyword_count = sum(1 for keyword in request.keywords if keyword.lower() in content_lower)
            factors["keywords"] = (keyword_count / len(request.keywords)) * 100
        else:
            factors["keywords"] = 100.0  # N/A
        
        # Readability (0-100) - basic assessment
        avg_sentence_length = word_count / max(1, content.count('.') + content.count('!') + content.count('?'))
        if avg_sentence_length <= 20:
            factors["readability"] = 90.0
        elif avg_sentence_length <= 30:
            factors["readability"] = 75.0
        else:
            factors["readability"] = 60.0
        
        # Overall quality score
        overall_score = sum(factors.values()) / len(factors)
        
        return overall_score, factors

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: GPTModel) -> float:
        """Calculate estimated cost for API call"""
        pricing = self.model_pricing.get(model, self.model_pricing[GPTModel.GPT_4O])
        
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost

    @with_retry(max_attempts=3, delay=1.0)
    async def _generate_with_gpt(self, request: ContentRequest) -> tuple[str, Dict[str, Any]]:
        """Generate content using OpenAI GPT"""
        
        system_prompt = self._build_system_prompt(request)
        user_prompt = self._build_user_prompt(request)
        
        try:
            response = await self.client.chat.completions.create(
                model=request.model.value,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=request.temperature,
                max_tokens=4000,  # Generous limit for content generation
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            usage_stats = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "model": request.model.value
            }
            
            return content, usage_stats
            
        except Exception as e:
            logger.error(f"GPT generation failed: {str(e)}")
            raise ToolError(f"Content generation failed: {str(e)}")

    def _extract_title_and_content(self, raw_content: str) -> tuple[str, str]:
        """Extract title and main content from generated text"""
        
        lines = raw_content.strip().split('\n')
        
        # Look for title patterns
        title = None
        content_start = 0
        
        for i, line in enumerate(lines[:5]):  # Check first 5 lines
            line = line.strip()
            if not line:
                continue
                
            # Check for markdown title
            if line.startswith('# '):
                title = line[2:].strip()
                content_start = i + 1
                break
            # Check for title indicators
            elif any(indicator in line.lower() for indicator in ['title:', 'headline:']):
                title = re.sub(r'^(title:|headline:)\s*', '', line, flags=re.IGNORECASE).strip()
                content_start = i + 1
                break
            # First substantial line as title
            elif len(line) > 10 and not line.startswith(('•', '-', '1.', 'a.')):
                title = line
                content_start = i + 1
                break
        
        # Fallback: first non-empty line as title
        if not title:
            for i, line in enumerate(lines):
                if line.strip():
                    title = line.strip()
                    content_start = i + 1
                    break
        
        # Extract main content
        content_lines = lines[content_start:] if content_start < len(lines) else lines
        content = '\n'.join(content_lines).strip()
        
        return title or "Generated Content", content

    def _extract_metadata(self, content: str, request: ContentRequest) -> Dict[str, Any]:
        """Extract metadata from generated content"""
        
        metadata = {
            "outline": None,
            "meta_description": None,
            "tags": None
        }
        
        if request.include_outline:
            # Extract outline from headings
            headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            if headings:
                metadata["outline"] = headings[:10]  # Limit to 10 items
        
        if request.include_meta:
            # Look for meta description in content
            meta_match = re.search(r'meta.description[:\s]+(.{100,160})', content, re.IGNORECASE)
            if meta_match:
                metadata["meta_description"] = meta_match.group(1).strip()
            else:
                # Generate from first paragraph
                first_paragraph = content.split('\n\n')[0]
                if len(first_paragraph) > 150:
                    metadata["meta_description"] = first_paragraph[:157] + "..."
                else:
                    metadata["meta_description"] = first_paragraph
            
            # Extract or generate tags
            if request.keywords:
                metadata["tags"] = request.keywords[:5]
            else:
                # Simple tag extraction from content
                words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
                word_freq = {}
                for word in words:
                    if word not in {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said'}:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                metadata["tags"] = list(sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:5])
        
        return metadata

    async def generate_content(self, request: ContentRequest) -> GeneratedContent:
        """
        Generate content based on request parameters
        
        Args:
            request: Content generation request
            
        Returns:
            GeneratedContent with all generated content and metadata
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Generating {request.content_type.value} content for topic: {request.topic}")
            
            # Generate content with GPT
            raw_content, usage_stats = await self._generate_with_gpt(request)
            
            # Extract title and content
            title, content = self._extract_title_and_content(raw_content)
            
            # Extract metadata
            metadata = self._extract_metadata(content, request)
            
            # Calculate metrics
            word_count = len(content.split())
            reading_time = self._calculate_reading_time(word_count)
            quality_score, quality_factors = self._assess_content_quality(content, request)
            
            # Calculate cost
            cost = self._calculate_cost(
                usage_stats["prompt_tokens"],
                usage_stats["completion_tokens"],
                request.model
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = GeneratedContent(
                content=content,
                title=title,
                outline=metadata["outline"],
                meta_description=metadata["meta_description"],
                tags=metadata["tags"],
                word_count=word_count,
                estimated_reading_time=reading_time,
                quality_score=quality_score,
                quality_factors=quality_factors,
                model_used=usage_stats["model"],
                generation_time=generation_time,
                prompt_tokens=usage_stats["prompt_tokens"],
                completion_tokens=usage_stats["completion_tokens"],
                total_cost=cost
            )
            
            logger.info(f"Content generated successfully: {word_count} words, quality: {quality_score:.1f}/100")
            return result
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise ToolError(f"Content generation failed: {str(e)}")

    async def generate_multiple_variants(
        self, 
        request: ContentRequest, 
        num_variants: int = 2,
        temperature_range: Optional[tuple[float, float]] = None
    ) -> List[GeneratedContent]:
        """
        Generate multiple content variants for A/B testing
        
        Args:
            request: Base content request
            num_variants: Number of variants to generate
            temperature_range: Temperature range for variation (min, max)
            
        Returns:
            List of generated content variants
        """
        if num_variants < 1 or num_variants > 5:
            raise ValueError("Number of variants must be between 1 and 5")
        
        # Set up temperature variations
        if temperature_range:
            min_temp, max_temp = temperature_range
        else:
            base_temp = request.temperature
            min_temp = max(0.0, base_temp - 0.3)
            max_temp = min(2.0, base_temp + 0.3)
        
        # Create variant requests
        variant_requests = []
        for i in range(num_variants):
            variant_request = request.model_copy()
            if num_variants > 1:
                # Distribute temperatures across range
                variant_request.temperature = min_temp + (i / (num_variants - 1)) * (max_temp - min_temp)
            variant_requests.append(variant_request)
        
        # Generate variants concurrently
        try:
            logger.info(f"Generating {num_variants} content variants")
            tasks = [self.generate_content(req) for req in variant_requests]
            variants = await asyncio.gather(*tasks)
            
            logger.info(f"Generated {len(variants)} content variants successfully")
            return variants
            
        except Exception as e:
            logger.error(f"Multiple variant generation failed: {str(e)}")
            raise ToolError(f"Variant generation failed: {str(e)}")

    async def optimize_for_seo(self, content: str, keywords: List[str]) -> str:
        """
        Optimize existing content for SEO keywords
        
        Args:
            content: Original content
            keywords: List of keywords to optimize for
            
        Returns:
            SEO-optimized content
        """
        if not keywords:
            return content
        
        optimization_prompt = f"""Optimize the following content for SEO by naturally incorporating these keywords: {', '.join(keywords)}

Requirements:
- Maintain the original tone and style
- Ensure keyword placement feels natural
- Don't over-optimize (avoid keyword stuffing)
- Maintain readability and flow
- Focus on semantic relevance

Original Content:
{content}

Provide the optimized version:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=GPTModel.GPT_4O.value,
                messages=[
                    {"role": "system", "content": "You are an SEO expert and content optimizer."},
                    {"role": "user", "content": optimization_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            optimized_content = response.choices[0].message.content.strip()
            logger.info(f"Content optimized for {len(keywords)} SEO keywords")
            return optimized_content
            
        except Exception as e:
            logger.error(f"SEO optimization failed: {str(e)}")
            raise ToolError(f"SEO optimization failed: {str(e)}")


# Initialize tool instance
content_writer_tool = ContentWriter()


# MCP Functions for external integration
async def mcp_generate_content(
    topic: str,
    content_type: str = "article",
    tone: str = "professional",
    style: str = "journalistic",
    target_length: Optional[int] = None,
    target_audience: Optional[str] = None,
    key_points: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    MCP function to generate content
    
    Args:
        topic: Main topic or title for the content
        content_type: Type of content (blog_post, article, social_media, etc.)
        tone: Content tone (professional, casual, etc.)
        style: Writing style (journalistic, creative, etc.)
        target_length: Target word count
        target_audience: Target audience description
        key_points: Key points to include
        keywords: SEO keywords to incorporate
        model: GPT model to use
        temperature: Creativity level (0.0-2.0)
        custom_instructions: Additional instructions
        
    Returns:
        Generated content with metadata
    """
    try:
        request = ContentRequest(
            topic=topic,
            content_type=ContentType(content_type),
            tone=Tone(tone),
            style=Style(style),
            target_length=target_length,
            target_audience=target_audience,
            key_points=key_points,
            keywords=keywords,
            model=GPTModel(model),
            temperature=temperature,
            custom_instructions=custom_instructions,
            include_outline=True,
            include_meta=True
        )
        
        result = await content_writer_tool.generate_content(request)
        
        return {
            "success": True,
            "content": result.content,
            "title": result.title,
            "outline": result.outline,
            "meta_description": result.meta_description,
            "tags": result.tags,
            "word_count": result.word_count,
            "reading_time": result.estimated_reading_time,
            "quality_score": result.quality_score,
            "quality_breakdown": result.quality_factors,
            "model_used": result.model_used,
            "generation_time": result.generation_time,
            "cost": result.total_cost,
            "tokens": {
                "prompt": result.prompt_tokens,
                "completion": result.completion_tokens
            }
        }
        
    except Exception as e:
        logger.error(f"MCP content generation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


async def mcp_generate_variants(
    topic: str,
    content_type: str = "article",
    num_variants: int = 2,
    tone: str = "professional",
    style: str = "journalistic",
    target_length: Optional[int] = None,
    keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    MCP function to generate multiple content variants for A/B testing
    
    Args:
        topic: Main topic for content
        content_type: Type of content
        num_variants: Number of variants (1-5)
        tone: Content tone
        style: Writing style
        target_length: Target word count
        keywords: SEO keywords
        
    Returns:
        List of content variants with comparison metrics
    """
    try:
        request = ContentRequest(
            topic=topic,
            content_type=ContentType(content_type),
            tone=Tone(tone),
            style=Style(style),
            target_length=target_length,
            keywords=keywords
        )
        
        variants = await content_writer_tool.generate_multiple_variants(request, num_variants)
        
        # Calculate variant comparison
        variant_data = []
        for i, variant in enumerate(variants):
            variant_data.append({
                "variant_id": i + 1,
                "content": variant.content,
                "title": variant.title,
                "word_count": variant.word_count,
                "quality_score": variant.quality_score,
                "reading_time": variant.estimated_reading_time,
                "temperature_used": request.temperature,  # Note: This is approximate
                "cost": variant.total_cost
            })
        
        # Find best variant by quality score
        best_variant = max(variants, key=lambda x: x.quality_score)
        best_idx = variants.index(best_variant)
        
        return {
            "success": True,
            "variants": variant_data,
            "best_variant_id": best_idx + 1,
            "comparison": {
                "quality_scores": [v.quality_score for v in variants],
                "word_counts": [v.word_count for v in variants],
                "total_cost": sum(v.total_cost for v in variants),
                "avg_generation_time": sum(v.generation_time for v in variants) / len(variants)
            }
        }
        
    except Exception as e:
        logger.error(f"MCP variant generation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Example usage and testing
    async def test_content_writer():
        """Test the content writer functionality"""
        
        # Test basic content generation
        request = ContentRequest(
            topic="The Future of Artificial Intelligence in Healthcare",
            content_type=ContentType.BLOG_POST,
            tone=Tone.PROFESSIONAL,
            style=Style.JOURNALISTIC,
            target_length=1000,
            target_audience="Healthcare professionals and technology enthusiasts",
            key_points=[
                "AI diagnostic tools improving accuracy",
                "Challenges in AI implementation",
                "Future trends and predictions"
            ],
            keywords=["AI healthcare", "medical diagnosis", "artificial intelligence", "healthcare technology"]
        )
        
        try:
            writer = ContentWriter()
            result = await writer.generate_content(request)
            
            print(f"Generated Content:")
            print(f"Title: {result.title}")
            print(f"Word Count: {result.word_count}")
            print(f"Quality Score: {result.quality_score}/100")
            print(f"Reading Time: {result.estimated_reading_time} minutes")
            print(f"Cost: ${result.total_cost:.4f}")
            print("\n" + "="*50 + "\n")
            print(result.content[:500] + "..." if len(result.content) > 500 else result.content)
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_content_writer())