"""
Image Generator Tool - DALL-E Integration for Content-Relevant Visual Creation

This module provides AI-powered image generation capabilities using OpenAI's DALL-E models.
Features include content-relevant generation, multiple formats, style variations, and optimization.

Key Features:
- DALL-E 3 integration for high-quality image generation
- Content-relevant prompt generation
- Multiple format and size support
- Style and aesthetic customization
- Batch generation for variations
- Image optimization and processing
- Brand guideline compliance
"""

import asyncio
import os
import base64
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal, Any, Tuple
from enum import Enum
from pathlib import Path
import re

import aiohttp
import aiofiles
from PIL import Image, ImageEnhance, ImageFilter
from pydantic import BaseModel, Field, validator
from openai import AsyncOpenAI

from ...core.errors import ToolError
from ...core.logging.logger import get_logger
from ...utils.simple_retry import with_retry

logger = get_logger(__name__)


class ImageSize(str, Enum):
    """Supported image sizes for DALL-E"""
    SQUARE = "1024x1024"
    PORTRAIT = "1024x1792"
    LANDSCAPE = "1792x1024"


class ImageStyle(str, Enum):
    """Image style presets"""
    NATURAL = "natural"
    VIVID = "vivid"
    PHOTOGRAPHIC = "photographic"
    DIGITAL_ART = "digital_art"
    PAINTING = "painting"
    SKETCH = "sketch"
    CARTOON = "cartoon"
    MINIMALIST = "minimalist"
    ABSTRACT = "abstract"
    VINTAGE = "vintage"
    MODERN = "modern"
    PROFESSIONAL = "professional"


class ContentType(str, Enum):
    """Content types for context-aware generation"""
    BLOG_HEADER = "blog_header"
    SOCIAL_MEDIA = "social_media"
    ARTICLE_ILLUSTRATION = "article_illustration"
    PRODUCT_SHOWCASE = "product_showcase"
    CONCEPT_VISUALIZATION = "concept_visualization"
    INFOGRAPHIC_ELEMENT = "infographic_element"
    BACKGROUND = "background"
    ICON = "icon"
    BANNER = "banner"
    THUMBNAIL = "thumbnail"


class ColorScheme(str, Enum):
    """Color scheme options"""
    VIBRANT = "vibrant"
    MUTED = "muted"
    MONOCHROME = "monochrome"
    WARM = "warm"
    COOL = "cool"
    PASTEL = "pastel"
    HIGH_CONTRAST = "high_contrast"
    EARTH_TONES = "earth_tones"
    NEON = "neon"
    VINTAGE = "vintage"


class ImageRequest(BaseModel):
    """Image generation request parameters"""
    
    content_topic: str = Field(..., description="Main topic or subject for the image")
    content_type: ContentType = Field(..., description="Type of content image is for")
    
    prompt: Optional[str] = Field(
        default=None,
        description="Custom image prompt (auto-generated if not provided)"
    )
    
    style: ImageStyle = Field(default=ImageStyle.NATURAL, description="Visual style")
    size: ImageSize = Field(default=ImageSize.LANDSCAPE, description="Image dimensions")
    quality: Literal["standard", "hd"] = Field(default="hd", description="Image quality")
    
    color_scheme: Optional[ColorScheme] = Field(
        default=None,
        description="Preferred color scheme"
    )
    
    mood: Optional[str] = Field(
        default=None,
        description="Desired mood/atmosphere (e.g., 'inspiring', 'calm', 'energetic')"
    )
    
    target_audience: Optional[str] = Field(
        default=None,
        description="Target audience for visual appeal"
    )
    
    brand_colors: Optional[List[str]] = Field(
        default=None,
        description="Brand colors to incorporate (hex codes)"
    )
    
    avoid_elements: Optional[List[str]] = Field(
        default=None,
        description="Elements to avoid in the image"
    )
    
    include_text_space: bool = Field(
        default=False,
        description="Include space for text overlay"
    )
    
    context_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context from content"
    )
    
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Additional generation instructions"
    )
    
    num_variants: int = Field(default=1, description="Number of image variants", ge=1, le=4)

    @validator('brand_colors')
    def validate_brand_colors(cls, v):
        if v:
            for color in v:
                if not re.match(r'^#[0-9A-Fa-f]{6}$', color):
                    raise ValueError(f"Invalid hex color: {color}")
        return v

    @validator('avoid_elements')
    def validate_avoid_elements(cls, v):
        if v and len(v) > 10:
            raise ValueError("Maximum 10 elements to avoid")
        return v


class GeneratedImage(BaseModel):
    """Generated image with metadata"""
    
    image_url: str = Field(..., description="URL of the generated image")
    image_path: Optional[str] = Field(default=None, description="Local file path if saved")
    
    prompt_used: str = Field(..., description="Final prompt used for generation")
    revised_prompt: Optional[str] = Field(default=None, description="DALL-E revised prompt")
    
    size: str = Field(..., description="Image dimensions")
    quality: str = Field(..., description="Image quality level")
    style: str = Field(..., description="Visual style applied")
    
    generation_time: float = Field(..., description="Generation time in seconds")
    estimated_cost: float = Field(..., description="Estimated cost in USD")
    
    content_relevance_score: float = Field(..., description="Relevance to content (0-100)")
    visual_appeal_score: float = Field(..., description="Predicted visual appeal (0-100)")
    
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")


class ImageResults(BaseModel):
    """Complete image generation results"""
    
    images: List[GeneratedImage] = Field(..., description="Generated images")
    best_image: GeneratedImage = Field(..., description="Top-rated image")
    
    generation_stats: Dict[str, Any] = Field(..., description="Generation statistics")
    optimization_suggestions: List[str] = Field(..., description="Improvement suggestions")
    
    total_variants: int = Field(..., description="Total variants generated")
    total_cost: float = Field(..., description="Total estimated cost")
    total_time: float = Field(..., description="Total generation time")


class PromptTemplates:
    """Image prompt templates for different content types and styles"""
    
    CONTENT_TYPE_TEMPLATES = {
        ContentType.BLOG_HEADER: {
            "template": "Create a compelling header image for a blog post about {topic}. {style_desc} The image should be visually striking and relevant to the content theme.",
            "requirements": ["professional", "eye-catching", "relevant"]
        },
        
        ContentType.SOCIAL_MEDIA: {
            "template": "Design a social media image about {topic}. {style_desc} Make it engaging and shareable with strong visual impact.",
            "requirements": ["engaging", "shareable", "mobile-friendly"]
        },
        
        ContentType.ARTICLE_ILLUSTRATION: {
            "template": "Create an illustration for an article about {topic}. {style_desc} The image should support and enhance the written content.",
            "requirements": ["supportive", "clear", "informative"]
        },
        
        ContentType.PRODUCT_SHOWCASE: {
            "template": "Design a product showcase image featuring {topic}. {style_desc} Focus on highlighting key features and benefits.",
            "requirements": ["focused", "attractive", "commercial"]
        },
        
        ContentType.CONCEPT_VISUALIZATION: {
            "template": "Visualize the concept of {topic}. {style_desc} Make abstract ideas concrete and understandable through imagery.",
            "requirements": ["conceptual", "clear", "metaphorical"]
        },
        
        ContentType.BACKGROUND: {
            "template": "Create a background image related to {topic}. {style_desc} Design for use behind text or other content elements.",
            "requirements": ["subtle", "non-distracting", "texture-rich"]
        }
    }
    
    STYLE_DESCRIPTIONS = {
        ImageStyle.NATURAL: "Use natural, realistic photography style",
        ImageStyle.VIVID: "Apply vivid, saturated colors and high contrast",
        ImageStyle.PHOTOGRAPHIC: "Create photorealistic imagery",
        ImageStyle.DIGITAL_ART: "Use digital art techniques with clean lines",
        ImageStyle.PAINTING: "Apply traditional painting aesthetics",
        ImageStyle.SKETCH: "Use hand-drawn sketch style",
        ImageStyle.CARTOON: "Create cartoon or animated style imagery",
        ImageStyle.MINIMALIST: "Keep design minimal and clean",
        ImageStyle.ABSTRACT: "Use abstract forms and compositions",
        ImageStyle.VINTAGE: "Apply vintage or retro styling",
        ImageStyle.MODERN: "Use contemporary, modern design elements",
        ImageStyle.PROFESSIONAL: "Maintain professional, business-appropriate style"
    }
    
    COLOR_SCHEME_DESCRIPTIONS = {
        ColorScheme.VIBRANT: "bright, vibrant colors",
        ColorScheme.MUTED: "muted, subdued color palette",
        ColorScheme.MONOCHROME: "monochrome or single-color scheme",
        ColorScheme.WARM: "warm colors like reds, oranges, and yellows",
        ColorScheme.COOL: "cool colors like blues, greens, and purples",
        ColorScheme.PASTEL: "soft, pastel color tones",
        ColorScheme.HIGH_CONTRAST: "high contrast color combinations",
        ColorScheme.EARTH_TONES: "natural earth tones and browns",
        ColorScheme.NEON: "bright neon and electric colors",
        ColorScheme.VINTAGE: "vintage color palette"
    }
    
    MOOD_KEYWORDS = {
        "inspiring": ["uplifting", "motivational", "bright", "hopeful"],
        "calm": ["peaceful", "serene", "tranquil", "relaxing"],
        "energetic": ["dynamic", "vibrant", "active", "powerful"],
        "professional": ["clean", "polished", "sophisticated", "corporate"],
        "creative": ["artistic", "imaginative", "innovative", "expressive"],
        "modern": ["contemporary", "sleek", "cutting-edge", "minimalist"],
        "warm": ["welcoming", "cozy", "friendly", "inviting"],
        "serious": ["formal", "authoritative", "important", "focused"]
    }


class ImageGenerator:
    """AI-powered image generation tool using OpenAI DALL-E"""
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "generated_images"):
        """
        Initialize Image Generator
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            output_dir: Directory for saving generated images
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # DALL-E pricing (per image)
        self.pricing = {
            ("1024x1024", "standard"): 0.040,
            ("1024x1024", "hd"): 0.080,
            ("1024x1792", "standard"): 0.080,
            ("1024x1792", "hd"): 0.120,
            ("1792x1024", "standard"): 0.080,
            ("1792x1024", "hd"): 0.120
        }
        
        logger.info("ImageGenerator initialized with DALL-E integration")

    def _calculate_cost(self, size: str, quality: str, num_images: int) -> float:
        """Calculate estimated cost for image generation"""
        cost_per_image = self.pricing.get((size, quality), 0.080)  # Default fallback
        return cost_per_image * num_images

    def _build_image_prompt(self, request: ImageRequest) -> str:
        """Build comprehensive prompt for image generation"""
        
        if request.prompt:
            base_prompt = request.prompt
        else:
            # Generate prompt from template
            template_data = PromptTemplates.CONTENT_TYPE_TEMPLATES.get(
                request.content_type,
                PromptTemplates.CONTENT_TYPE_TEMPLATES[ContentType.ARTICLE_ILLUSTRATION]
            )
            
            style_desc = PromptTemplates.STYLE_DESCRIPTIONS.get(request.style, "")
            
            base_prompt = template_data["template"].format(
                topic=request.content_topic,
                style_desc=style_desc
            )
        
        # Add style specifications
        prompt_parts = [base_prompt]
        
        # Add color scheme
        if request.color_scheme:
            color_desc = PromptTemplates.COLOR_SCHEME_DESCRIPTIONS.get(request.color_scheme)
            if color_desc:
                prompt_parts.append(f"Use {color_desc}")
        
        # Add brand colors
        if request.brand_colors:
            color_list = ", ".join(request.brand_colors)
            prompt_parts.append(f"Incorporate brand colors: {color_list}")
        
        # Add mood
        if request.mood:
            mood_keywords = PromptTemplates.MOOD_KEYWORDS.get(request.mood.lower(), [request.mood])
            prompt_parts.append(f"Create a {request.mood} mood with {', '.join(mood_keywords[:2])} atmosphere")
        
        # Add text space requirement
        if request.include_text_space:
            prompt_parts.append("Leave space for text overlay in the composition")
        
        # Add context from content
        if request.context_data:
            if "key_points" in request.context_data:
                key_points = request.context_data["key_points"][:3]  # Limit to 3
                prompt_parts.append(f"Visually represent concepts: {', '.join(key_points)}")
            
            if "sentiment" in request.context_data:
                sentiment = request.context_data["sentiment"]
                prompt_parts.append(f"Match the {sentiment} sentiment of the content")
        
        # Add custom instructions
        if request.custom_instructions:
            prompt_parts.append(request.custom_instructions)
        
        # Add elements to avoid
        if request.avoid_elements:
            avoid_list = ", ".join(request.avoid_elements)
            prompt_parts.append(f"Avoid including: {avoid_list}")
        
        # Quality and style modifiers
        quality_modifiers = [
            "high quality",
            "detailed",
            "professionally composed",
            "visually appealing"
        ]
        
        if request.target_audience:
            quality_modifiers.append(f"appealing to {request.target_audience}")
        
        prompt_parts.append(f"Ensure the image is {', '.join(quality_modifiers)}")
        
        final_prompt = ". ".join(prompt_parts) + "."
        
        # Ensure prompt isn't too long (DALL-E has limits)
        if len(final_prompt) > 1000:
            # Trim while preserving core elements
            core_parts = prompt_parts[:3]  # Keep base prompt and key modifiers
            final_prompt = ". ".join(core_parts) + "."
        
        return final_prompt

    @with_retry(max_attempts=3, delay=2.0)
    async def _generate_with_dalle(self, request: ImageRequest) -> List[Dict[str, Any]]:
        """Generate images using DALL-E"""
        
        prompt = self._build_image_prompt(request)
        
        try:
            logger.info(f"Generating {request.num_variants} image(s) for: {request.content_topic}")
            
            # DALL-E 3 only supports n=1, so we need multiple calls for variants
            images = []
            
            for i in range(request.num_variants):
                response = await self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size=request.size.value,
                    quality=request.quality,
                    style=request.style.value if request.style in [ImageStyle.NATURAL, ImageStyle.VIVID] else "natural",
                    n=1
                )
                
                image_data = {
                    "url": response.data[0].url,
                    "prompt_used": prompt,
                    "revised_prompt": response.data[0].revised_prompt,
                    "variant_id": i + 1
                }
                
                images.append(image_data)
                
                # Small delay between generations to avoid rate limits
                if i < request.num_variants - 1:
                    await asyncio.sleep(1)
            
            return images
            
        except Exception as e:
            logger.error(f"DALL-E generation failed: {str(e)}")
            raise ToolError(f"Image generation failed: {str(e)}")

    async def _download_image(self, url: str, filename: str) -> str:
        """Download image from URL to local storage"""
        
        file_path = self.output_dir / filename
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        async with aiofiles.open(file_path, 'wb') as f:
                            await f.write(content)
                        return str(file_path)
                    else:
                        logger.error(f"Failed to download image: HTTP {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Image download failed: {str(e)}")
            return None

    def _assess_image_quality(self, request: ImageRequest, image_data: Dict[str, Any]) -> Tuple[float, float]:
        """Assess content relevance and visual appeal scores (heuristic-based)"""
        
        prompt_used = image_data.get("prompt_used", "")
        revised_prompt = image_data.get("revised_prompt", "")
        
        # Content relevance score (0-100)
        relevance_score = 70  # Base score
        
        # Check if topic keywords are in revised prompt
        topic_words = request.content_topic.lower().split()
        revised_lower = revised_prompt.lower() if revised_prompt else prompt_used.lower()
        
        keyword_matches = sum(1 for word in topic_words if len(word) > 2 and word in revised_lower)
        relevance_score += (keyword_matches / max(1, len(topic_words))) * 20
        
        # Bonus for style consistency
        style_keywords = PromptTemplates.STYLE_DESCRIPTIONS.get(request.style, "").lower().split()
        style_matches = sum(1 for word in style_keywords if word in revised_lower)
        relevance_score += min(10, style_matches * 3)
        
        relevance_score = min(100, relevance_score)
        
        # Visual appeal score (0-100)
        appeal_score = 75  # Base score for DALL-E quality
        
        # Bonus for HD quality
        if request.quality == "hd":
            appeal_score += 10
        
        # Bonus for appropriate size
        if request.content_type in [ContentType.SOCIAL_MEDIA, ContentType.BANNER]:
            if request.size in [ImageSize.LANDSCAPE, ImageSize.SQUARE]:
                appeal_score += 10
        
        # Bonus for color scheme specification
        if request.color_scheme:
            appeal_score += 5
        
        # Penalty for too many avoid elements (might limit creativity)
        if request.avoid_elements and len(request.avoid_elements) > 5:
            appeal_score -= 5
        
        appeal_score = min(100, max(0, appeal_score))
        
        return relevance_score, appeal_score

    def _generate_filename(self, request: ImageRequest, variant_id: int) -> str:
        """Generate unique filename for image"""
        
        # Create hash of content for uniqueness
        content_hash = hashlib.md5(
            f"{request.content_topic}_{request.style.value}_{variant_id}".encode()
        ).hexdigest()[:8]
        
        # Clean topic for filename
        clean_topic = re.sub(r'[^\w\-_]', '_', request.content_topic)[:30]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{clean_topic}_{request.style.value}_{content_hash}_{timestamp}_v{variant_id}.png"

    async def generate_images(self, request: ImageRequest) -> ImageResults:
        """
        Generate images based on request parameters
        
        Args:
            request: Image generation request
            
        Returns:
            Complete image generation results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting image generation for '{request.content_topic}' ({request.content_type.value})")
            
            # Generate images with DALL-E
            image_data_list = await self._generate_with_dalle(request)
            
            # Process each generated image
            generated_images = []
            total_cost = self._calculate_cost(request.size.value, request.quality, len(image_data_list))
            
            for i, image_data in enumerate(image_data_list):
                # Generate filename and download image
                filename = self._generate_filename(request, image_data["variant_id"])
                local_path = await self._download_image(image_data["url"], filename)
                
                # Assess quality
                relevance_score, appeal_score = self._assess_image_quality(request, image_data)
                
                # Calculate individual cost
                individual_cost = self._calculate_cost(request.size.value, request.quality, 1)
                
                generated_image = GeneratedImage(
                    image_url=image_data["url"],
                    image_path=local_path,
                    prompt_used=image_data["prompt_used"],
                    revised_prompt=image_data["revised_prompt"],
                    size=request.size.value,
                    quality=request.quality,
                    style=request.style.value,
                    generation_time=(datetime.now() - start_time).total_seconds() / len(image_data_list),
                    estimated_cost=individual_cost,
                    content_relevance_score=relevance_score,
                    visual_appeal_score=appeal_score,
                    metadata={
                        "content_type": request.content_type.value,
                        "variant_id": image_data["variant_id"],
                        "color_scheme": request.color_scheme.value if request.color_scheme else None,
                        "mood": request.mood,
                        "target_audience": request.target_audience,
                        "include_text_space": request.include_text_space
                    }
                )
                
                generated_images.append(generated_image)
            
            # Find best image (by combined score)
            best_image = max(
                generated_images, 
                key=lambda x: (x.content_relevance_score + x.visual_appeal_score) / 2
            )
            
            # Generate statistics
            total_time = (datetime.now() - start_time).total_seconds()
            
            generation_stats = {
                "total_generated": len(generated_images),
                "avg_relevance_score": sum(img.content_relevance_score for img in generated_images) / len(generated_images),
                "avg_appeal_score": sum(img.visual_appeal_score for img in generated_images) / len(generated_images),
                "successful_downloads": sum(1 for img in generated_images if img.image_path),
                "total_cost": total_cost,
                "avg_generation_time": total_time / len(generated_images)
            }
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(generated_images, request)
            
            result = ImageResults(
                images=generated_images,
                best_image=best_image,
                generation_stats=generation_stats,
                optimization_suggestions=optimization_suggestions,
                total_variants=len(generated_images),
                total_cost=total_cost,
                total_time=total_time
            )
            
            logger.info(f"Generated {len(generated_images)} images successfully. Best relevance: {best_image.content_relevance_score:.1f}/100")
            return result
            
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise ToolError(f"Image generation failed: {str(e)}")

    def _generate_optimization_suggestions(
        self, 
        images: List[GeneratedImage], 
        request: ImageRequest
    ) -> List[str]:
        """Generate suggestions for improving future generations"""
        
        suggestions = []
        
        # Analyze average scores
        avg_relevance = sum(img.content_relevance_score for img in images) / len(images)
        avg_appeal = sum(img.visual_appeal_score for img in images) / len(images)
        
        if avg_relevance < 80:
            suggestions.append("Consider using more specific keywords in the topic description for better relevance")
        
        if avg_appeal < 85:
            suggestions.append("Try different color schemes or styles to improve visual appeal")
        
        if not request.color_scheme:
            suggestions.append("Specify a color scheme to ensure brand consistency")
        
        if not request.mood:
            suggestions.append("Define a mood/atmosphere to guide the visual tone")
        
        if request.content_type in [ContentType.SOCIAL_MEDIA, ContentType.BANNER] and not request.include_text_space:
            suggestions.append("Consider including text space for social media and banner images")
        
        # Check for downloaded vs not downloaded
        downloaded_count = sum(1 for img in images if img.image_path)
        if downloaded_count < len(images):
            suggestions.append("Some images failed to download - check network connection")
        
        return suggestions

    async def optimize_existing_image(self, image_path: str, optimization_type: str = "enhance") -> str:
        """
        Optimize an existing image (basic post-processing)
        
        Args:
            image_path: Path to the image file
            optimization_type: Type of optimization ('enhance', 'sharpen', 'brighten')
            
        Returns:
            Path to optimized image
        """
        try:
            # Load image
            with Image.open(image_path) as img:
                # Apply optimization
                if optimization_type == "enhance":
                    # Enhance color and contrast
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(1.1)
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.05)
                
                elif optimization_type == "sharpen":
                    # Apply sharpening filter
                    img = img.filter(ImageFilter.SHARPEN)
                
                elif optimization_type == "brighten":
                    # Brighten the image
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.1)
                
                # Save optimized version
                optimized_path = image_path.replace('.png', f'_optimized_{optimization_type}.png')
                img.save(optimized_path, 'PNG', quality=95)
                
                logger.info(f"Image optimized: {optimization_type}")
                return optimized_path
                
        except Exception as e:
            logger.error(f"Image optimization failed: {str(e)}")
            raise ToolError(f"Image optimization failed: {str(e)}")


# Initialize tool instance
image_generator_tool = ImageGenerator()


# MCP Functions for external integration
async def mcp_generate_images(
    content_topic: str,
    content_type: str = "article_illustration",
    style: str = "natural",
    size: str = "1792x1024",
    quality: str = "hd",
    num_variants: int = 1,
    color_scheme: Optional[str] = None,
    mood: Optional[str] = None,
    target_audience: Optional[str] = None,
    include_text_space: bool = False,
    custom_prompt: Optional[str] = None,
    avoid_elements: Optional[List[str]] = None,
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    MCP function to generate images with DALL-E
    
    Args:
        content_topic: Main topic or subject for the image
        content_type: Type of content image is for
        style: Visual style (natural, vivid, photographic, etc.)
        size: Image dimensions (1024x1024, 1024x1792, 1792x1024)
        quality: Image quality (standard, hd)
        num_variants: Number of image variants (1-4)
        color_scheme: Preferred color scheme
        mood: Desired mood/atmosphere
        target_audience: Target audience for visual appeal
        include_text_space: Include space for text overlay
        custom_prompt: Custom image prompt
        avoid_elements: Elements to avoid in the image
        custom_instructions: Additional generation instructions
        
    Returns:
        Generated images with analysis and metadata
    """
    try:
        request = ImageRequest(
            content_topic=content_topic,
            content_type=ContentType(content_type),
            prompt=custom_prompt,
            style=ImageStyle(style),
            size=ImageSize(size),
            quality=quality,
            color_scheme=ColorScheme(color_scheme) if color_scheme else None,
            mood=mood,
            target_audience=target_audience,
            avoid_elements=avoid_elements,
            include_text_space=include_text_space,
            custom_instructions=custom_instructions,
            num_variants=num_variants
        )
        
        result = await image_generator_tool.generate_images(request)
        
        return {
            "success": True,
            "images": [
                {
                    "image_url": img.image_url,
                    "image_path": img.image_path,
                    "prompt_used": img.prompt_used,
                    "revised_prompt": img.revised_prompt,
                    "size": img.size,
                    "quality": img.quality,
                    "style": img.style,
                    "content_relevance_score": img.content_relevance_score,
                    "visual_appeal_score": img.visual_appeal_score,
                    "estimated_cost": img.estimated_cost,
                    "generation_time": img.generation_time,
                    "metadata": img.metadata
                }
                for img in result.images
            ],
            "best_image": {
                "image_url": result.best_image.image_url,
                "image_path": result.best_image.image_path,
                "relevance_score": result.best_image.content_relevance_score,
                "appeal_score": result.best_image.visual_appeal_score
            },
            "statistics": result.generation_stats,
            "optimization_suggestions": result.optimization_suggestions,
            "total_cost": result.total_cost,
            "total_time": result.total_time,
            "total_variants": result.total_variants
        }
        
    except Exception as e:
        logger.error(f"MCP image generation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Example usage and testing
    async def test_image_generator():
        """Test the image generator functionality"""
        
        request = ImageRequest(
            content_topic="The Future of Sustainable Energy Technology",
            content_type=ContentType.BLOG_HEADER,
            style=ImageStyle.MODERN,
            size=ImageSize.LANDSCAPE,
            quality="hd",
            color_scheme=ColorScheme.COOL,
            mood="inspiring",
            target_audience="Technology enthusiasts and environmental advocates",
            include_text_space=True,
            num_variants=2
        )
        
        try:
            generator = ImageGenerator()
            result = await generator.generate_images(request)
            
            print(f"Generated {len(result.images)} image(s):")
            print("="*60)
            
            for i, image in enumerate(result.images, 1):
                print(f"{i}. Image URL: {image.image_url}")
                print(f"   Local Path: {image.image_path}")
                print(f"   Relevance: {image.content_relevance_score:.1f}/100")
                print(f"   Appeal: {image.visual_appeal_score:.1f}/100")
                print(f"   Cost: ${image.estimated_cost:.3f}")
                print(f"   Prompt: {image.prompt_used[:100]}...")
                print()
            
            print(f"Best Image: {result.best_image.image_url}")
            print(f"Total Cost: ${result.total_cost:.3f}")
            print(f"Suggestions: {result.optimization_suggestions}")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_image_generator())