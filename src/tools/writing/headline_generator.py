"""
Headline Generator Tool - AI-Powered Headline Creation and A/B Testing

This module provides advanced headline generation capabilities using OpenAI's GPT models.
Features include A/B testing variations, style-specific headlines, and performance optimization.

Key Features:
- Multiple headline generation strategies
- A/B testing with variant generation
- Style-specific headline templates
- Emotional impact scoring
- Click-through rate prediction
- SEO optimization
- Performance analytics
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal, Any, Tuple
from enum import Enum
import re
import json
import random

from pydantic import BaseModel, Field, validator
from openai import AsyncOpenAI

from ...core.errors.exceptions import ToolExecutionError
from ...core.logging.logger import get_logger
from ...utils.retry import with_retry

logger = get_logger(__name__)


class HeadlineStyle(str, Enum):
    """Headline style categories"""
    QUESTION = "question"
    HOW_TO = "how_to"
    LISTICLE = "listicle"
    NEWS = "news"
    BENEFIT = "benefit"
    CURIOSITY = "curiosity"
    URGENT = "urgent"
    STATISTICAL = "statistical"
    STORYTELLING = "storytelling"
    COMPARISON = "comparison"
    PROBLEM_SOLUTION = "problem_solution"
    TESTIMONIAL = "testimonial"


class HeadlineTone(str, Enum):
    """Headline tone options"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    URGENT = "urgent"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    HUMOROUS = "humorous"
    MYSTERIOUS = "mysterious"
    EMOTIONAL = "emotional"
    DIRECT = "direct"
    INSPIRING = "inspiring"


class Platform(str, Enum):
    """Target platform for headlines"""
    BLOG = "blog"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    ADS = "ads"
    YOUTUBE = "youtube"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"


class EmotionalTrigger(str, Enum):
    """Emotional triggers for headlines"""
    FEAR = "fear"
    JOY = "joy"
    ANGER = "anger"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    CURIOSITY = "curiosity"
    URGENCY = "urgency"


class HeadlineRequest(BaseModel):
    """Headline generation request parameters"""
    
    topic: str = Field(..., description="Main topic or content subject")
    style: HeadlineStyle = Field(..., description="Headline style type")
    tone: HeadlineTone = Field(default=HeadlineTone.PROFESSIONAL, description="Tone of the headline")
    platform: Platform = Field(default=Platform.BLOG, description="Target platform")
    
    target_audience: Optional[str] = Field(
        default=None,
        description="Target audience description"
    )
    
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords to include in headlines"
    )
    
    emotional_trigger: Optional[EmotionalTrigger] = Field(
        default=None,
        description="Primary emotional trigger to evoke"
    )
    
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum character length",
        ge=20,
        le=200
    )
    
    include_numbers: bool = Field(default=False, description="Include specific numbers")
    include_year: bool = Field(default=False, description="Include current year")
    
    context_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context (stats, benefits, etc.)"
    )
    
    avoid_words: Optional[List[str]] = Field(
        default=None,
        description="Words to avoid in headlines"
    )
    
    brand_voice: Optional[str] = Field(
        default=None,
        description="Brand voice guidelines"
    )
    
    num_variants: int = Field(default=5, description="Number of headline variants", ge=1, le=20)
    temperature: float = Field(default=0.8, description="Creativity level", ge=0.0, le=2.0)

    @validator('keywords')
    def validate_keywords(cls, v):
        if v and len(v) > 10:
            raise ValueError("Maximum 10 keywords allowed")
        return v

    @validator('avoid_words')  
    def validate_avoid_words(cls, v):
        if v and len(v) > 20:
            raise ValueError("Maximum 20 words to avoid")
        return v


class HeadlineAnalysis(BaseModel):
    """Analysis metrics for a headline"""
    
    emotional_score: float = Field(..., description="Emotional impact score (0-100)")
    clarity_score: float = Field(..., description="Clarity and readability score (0-100)")
    curiosity_score: float = Field(..., description="Curiosity gap score (0-100)")
    urgency_score: float = Field(..., description="Urgency/immediacy score (0-100)")
    seo_score: float = Field(..., description="SEO optimization score (0-100)")
    
    predicted_ctr: float = Field(..., description="Predicted click-through rate (%)")
    
    length: int = Field(..., description="Character length")
    word_count: int = Field(..., description="Word count")
    
    has_numbers: bool = Field(..., description="Contains numbers")
    has_power_words: bool = Field(..., description="Contains power words")
    has_emotional_triggers: List[str] = Field(..., description="Detected emotional triggers")
    
    overall_score: float = Field(..., description="Overall effectiveness score (0-100)")


class GeneratedHeadline(BaseModel):
    """Generated headline with analysis"""
    
    headline: str = Field(..., description="The generated headline")
    style: HeadlineStyle = Field(..., description="Style used")
    analysis: HeadlineAnalysis = Field(..., description="Performance analysis")
    
    variation_notes: Optional[str] = Field(default=None, description="Notes about this variation")
    platform_optimized: bool = Field(..., description="Optimized for target platform")


class HeadlineResults(BaseModel):
    """Complete headline generation results"""
    
    headlines: List[GeneratedHeadline] = Field(..., description="Generated headlines")
    best_headline: GeneratedHeadline = Field(..., description="Top-performing headline")
    
    generation_stats: Dict[str, Any] = Field(..., description="Generation statistics")
    ab_test_recommendations: Dict[str, Any] = Field(..., description="A/B testing recommendations")
    
    total_variants: int = Field(..., description="Total number of variants generated")
    generation_time: float = Field(..., description="Total generation time in seconds")


class HeadlineTemplates:
    """Headline templates and patterns for different styles"""
    
    TEMPLATES = {
        HeadlineStyle.QUESTION: [
            "Why {topic}?",
            "What Makes {topic} So {adjective}?",
            "How Does {topic} Really Work?",
            "Are You Making These {topic} Mistakes?",
            "What If {topic} Could {benefit}?"
        ],
        
        HeadlineStyle.HOW_TO: [
            "How to {action} {topic} in {timeframe}",
            "The Complete Guide to {topic}",
            "How to {action} {topic} Like a Pro",
            "{number} Ways to {action} {topic}",
            "The Ultimate {topic} Tutorial"
        ],
        
        HeadlineStyle.LISTICLE: [
            "{number} {adjective} {topic} Tips",
            "{number} Reasons Why {topic} Matters",
            "{number} {topic} Mistakes to Avoid",
            "Top {number} {topic} Trends for {year}",
            "{number} Surprising Facts About {topic}"
        ],
        
        HeadlineStyle.NEWS: [
            "{topic}: What You Need to Know",
            "Breaking: {topic} Changes Everything",
            "{topic} Update: Latest Developments",
            "Industry Alert: {topic} Impact",
            "{topic} News That Affects You"
        ],
        
        HeadlineStyle.BENEFIT: [
            "Get {benefit} with {topic}",
            "{topic} That Will {positive_outcome}",
            "Why {topic} Is Your {solution}",
            "The {topic} Secret to {desired_result}",
            "Transform Your {area} with {topic}"
        ],
        
        HeadlineStyle.CURIOSITY: [
            "The {topic} Secret Nobody Talks About",
            "What {experts} Don't Want You to Know About {topic}",
            "The Surprising Truth About {topic}",
            "This {topic} Trick Will {amazing_result}",
            "The Hidden {topic} Method That Works"
        ],
        
        HeadlineStyle.URGENT: [
            "Don't Miss This {topic} Opportunity",
            "Last Chance: {topic} Ends {timeframe}",
            "Act Now: {topic} Limited Time",
            "Urgent: Your {topic} Deadline Approaches",
            "Time-Sensitive: {topic} Update"
        ],
        
        HeadlineStyle.STATISTICAL: [
            "{percentage}% of People Don't Know {topic}",
            "{number} Data Points About {topic}",
            "Study Reveals: {topic} Statistics",
            "{topic} by the Numbers",
            "Research Shows: {topic} Facts"
        ]
    }
    
    POWER_WORDS = [
        # Emotional
        "amazing", "incredible", "stunning", "shocking", "remarkable",
        "extraordinary", "revolutionary", "breakthrough", "devastating", "inspiring",
        
        # Urgency  
        "urgent", "immediately", "now", "instantly", "quickly",
        "fast", "rapid", "sudden", "emergency", "critical",
        
        # Value
        "free", "exclusive", "limited", "secret", "hidden",
        "proven", "guaranteed", "ultimate", "complete", "perfect",
        
        # Curiosity
        "surprising", "mysterious", "unknown", "forbidden", "controversial",
        "bizarre", "weird", "strange", "unusual", "unexpected",
        
        # Authority
        "expert", "professional", "master", "guru", "authority",
        "insider", "official", "certified", "approved", "endorsed"
    ]
    
    EMOTIONAL_TRIGGERS = {
        EmotionalTrigger.FEAR: ["avoid", "mistake", "danger", "warning", "risk", "threat"],
        EmotionalTrigger.JOY: ["happy", "joy", "celebration", "success", "achievement", "win"],
        EmotionalTrigger.ANGER: ["outrageous", "unfair", "scandal", "betrayal", "shocking"],
        EmotionalTrigger.SURPRISE: ["amazing", "incredible", "stunning", "unbelievable", "wow"],
        EmotionalTrigger.TRUST: ["proven", "trusted", "reliable", "honest", "authentic"],
        EmotionalTrigger.ANTICIPATION: ["coming", "soon", "preview", "sneak peek", "upcoming"],
        EmotionalTrigger.CURIOSITY: ["secret", "hidden", "mystery", "revealed", "unknown"],
        EmotionalTrigger.URGENCY: ["now", "urgent", "limited", "deadline", "last chance"]
    }


class HeadlineGenerator:
    """AI-powered headline generation and optimization tool"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Headline Generator
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Platform-specific length limits
        self.platform_limits = {
            Platform.TWITTER: 280,
            Platform.FACEBOOK: 125,
            Platform.INSTAGRAM: 125,
            Platform.LINKEDIN: 150,
            Platform.EMAIL: 60,
            Platform.ADS: 30,
            Platform.YOUTUBE: 100,
            Platform.BLOG: 70,
            Platform.NEWS: 80,
            Platform.SOCIAL_MEDIA: 125
        }
        
        logger.info("HeadlineGenerator initialized with OpenAI integration")

    def _get_platform_limit(self, platform: Platform) -> int:
        """Get character limit for platform"""
        return self.platform_limits.get(platform, 100)

    def _build_headline_prompt(self, request: HeadlineRequest) -> str:
        """Build comprehensive prompt for headline generation"""
        
        templates = HeadlineTemplates.TEMPLATES.get(request.style, [])
        
        prompt = f"""Generate compelling headlines for the topic: "{request.topic}"

Requirements:
- Style: {request.style.value}
- Tone: {request.tone.value}  
- Platform: {request.platform.value}
- Number of variations: {request.num_variants}
"""
        
        if request.max_length:
            prompt += f"- Maximum length: {request.max_length} characters\n"
        else:
            platform_limit = self._get_platform_limit(request.platform)
            prompt += f"- Platform character limit: {platform_limit} characters\n"
        
        if request.target_audience:
            prompt += f"- Target audience: {request.target_audience}\n"
        
        if request.keywords:
            prompt += f"- Include keywords: {', '.join(request.keywords)}\n"
        
        if request.emotional_trigger:
            trigger_words = HeadlineTemplates.EMOTIONAL_TRIGGERS.get(request.emotional_trigger, [])
            prompt += f"- Emotional trigger: {request.emotional_trigger.value} (use words like: {', '.join(trigger_words[:3])})\n"
        
        if request.include_numbers:
            prompt += "- Include specific numbers or statistics\n"
        
        if request.include_year:
            current_year = datetime.now().year
            prompt += f"- Include year: {current_year}\n"
        
        if request.avoid_words:
            prompt += f"- Avoid these words: {', '.join(request.avoid_words)}\n"
        
        if request.brand_voice:
            prompt += f"- Brand voice: {request.brand_voice}\n"
        
        if request.context_data:
            prompt += f"- Additional context: {json.dumps(request.context_data, indent=2)}\n"
        
        prompt += f"""
Style Guidelines for {request.style.value}:
"""
        
        if templates:
            prompt += f"- Use patterns like: {', '.join(templates[:3])}\n"
        
        # Add style-specific guidance
        style_guidance = {
            HeadlineStyle.QUESTION: "Use questions that create curiosity gaps and demand answers",
            HeadlineStyle.HOW_TO: "Focus on clear, actionable outcomes and benefits", 
            HeadlineStyle.LISTICLE: "Use specific numbers and promise concrete takeaways",
            HeadlineStyle.NEWS: "Emphasize timeliness, relevance, and impact",
            HeadlineStyle.BENEFIT: "Lead with clear value propositions and outcomes",
            HeadlineStyle.CURIOSITY: "Create information gaps that demand to be filled",
            HeadlineStyle.URGENT: "Use time-sensitive language and scarcity",
            HeadlineStyle.STATISTICAL: "Include specific numbers and research findings"
        }
        
        if request.style in style_guidance:
            prompt += f"- {style_guidance[request.style]}\n"
        
        prompt += f"""
Quality Standards:
- Make each headline unique and compelling
- Optimize for click-through rates
- Ensure clarity and immediate understanding
- Use power words when appropriate
- Create emotional connection
- Match the specified tone throughout

Generate {request.num_variants} distinct, high-quality headlines that follow these requirements.
Format as a simple numbered list, one headline per line."""
        
        return prompt

    def _analyze_headline(self, headline: str, request: HeadlineRequest) -> HeadlineAnalysis:
        """Analyze headline performance factors"""
        
        # Basic metrics
        length = len(headline)
        word_count = len(headline.split())
        
        # Check for numbers
        has_numbers = bool(re.search(r'\d+', headline))
        
        # Check for power words
        headline_lower = headline.lower()
        power_words_found = [word for word in HeadlineTemplates.POWER_WORDS if word in headline_lower]
        has_power_words = len(power_words_found) > 0
        
        # Detect emotional triggers
        emotional_triggers = []
        for trigger, words in HeadlineTemplates.EMOTIONAL_TRIGGERS.items():
            if any(word in headline_lower for word in words):
                emotional_triggers.append(trigger.value)
        
        # Scoring algorithms (simplified heuristic-based)
        
        # Emotional score (0-100)
        emotional_score = 0
        if has_power_words:
            emotional_score += 30
        if emotional_triggers:
            emotional_score += 20
        if request.emotional_trigger and request.emotional_trigger.value in emotional_triggers:
            emotional_score += 25
        # Questions and curiosity words
        if '?' in headline or any(word in headline_lower for word in ['secret', 'mystery', 'hidden', 'surprising']):
            emotional_score += 25
        emotional_score = min(100, emotional_score)
        
        # Clarity score (0-100)
        clarity_score = 90  # Base score
        if word_count > 12:  # Long headlines harder to process
            clarity_score -= 20
        if length > 80:  # Too long for quick comprehension
            clarity_score -= 15
        if re.search(r'[^\w\s\-\?\!]', headline):  # Complex punctuation
            clarity_score -= 10
        clarity_score = max(0, clarity_score)
        
        # Curiosity score (0-100)
        curiosity_score = 0
        curiosity_words = ['secret', 'hidden', 'revealed', 'truth', 'surprising', 'shocking', 'unknown']
        curiosity_score += sum(10 for word in curiosity_words if word in headline_lower)
        if '?' in headline:
            curiosity_score += 20
        if any(phrase in headline_lower for phrase in ['you need to know', 'what you', 'why you']):
            curiosity_score += 15
        curiosity_score = min(100, curiosity_score)
        
        # Urgency score (0-100)
        urgency_score = 0
        urgency_words = ['now', 'urgent', 'immediately', 'fast', 'quick', 'limited', 'deadline', 'last chance']
        urgency_score += sum(15 for word in urgency_words if word in headline_lower)
        if datetime.now().year in headline:
            urgency_score += 10
        urgency_score = min(100, urgency_score)
        
        # SEO score (0-100)
        seo_score = 70  # Base score
        if request.keywords:
            keyword_matches = sum(1 for kw in request.keywords if kw.lower() in headline_lower)
            seo_score += (keyword_matches / len(request.keywords)) * 30
        if 30 <= length <= 60:  # Optimal length for SEO
            seo_score += 10
        seo_score = min(100, seo_score)
        
        # Predicted CTR (simplified model)
        ctr_factors = [
            emotional_score * 0.3,
            curiosity_score * 0.3,
            urgency_score * 0.2,
            (100 - abs(length - 60)) * 0.2  # Length optimization
        ]
        base_ctr = sum(ctr_factors) / 100 * 5  # 0-5% range
        predicted_ctr = max(0.5, min(15.0, base_ctr))  # Realistic CTR range
        
        # Overall score
        overall_score = (emotional_score + clarity_score + curiosity_score + urgency_score + seo_score) / 5
        
        return HeadlineAnalysis(
            emotional_score=emotional_score,
            clarity_score=clarity_score,
            curiosity_score=curiosity_score,
            urgency_score=urgency_score,
            seo_score=seo_score,
            predicted_ctr=predicted_ctr,
            length=length,
            word_count=word_count,
            has_numbers=has_numbers,
            has_power_words=has_power_words,
            has_emotional_triggers=emotional_triggers,
            overall_score=overall_score
        )

    @with_retry(max_attempts=3, delay=1.0)
    async def _generate_with_gpt(self, request: HeadlineRequest) -> List[str]:
        """Generate headlines using OpenAI GPT"""
        
        prompt = self._build_headline_prompt(request)
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",  # Best model for creative tasks
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert copywriter and headline optimization specialist with deep knowledge of psychology, marketing, and platform-specific best practices."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=request.temperature,
                max_tokens=1000,
                top_p=0.9,
                frequency_penalty=0.3,  # Encourage variety
                presence_penalty=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse headlines from numbered list
            lines = content.split('\n')
            headlines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Remove numbering (1., 2., etc.)
                headline = re.sub(r'^\d+\.?\s*', '', line).strip()
                
                # Remove quotes if present
                headline = re.sub(r'^["\'](.+)["\']$', r'\1', headline)
                
                if headline and len(headline) > 10:  # Minimum viable headline
                    headlines.append(headline)
            
            if len(headlines) < request.num_variants:
                logger.warning(f"Generated {len(headlines)} headlines, requested {request.num_variants}")
            
            return headlines[:request.num_variants]  # Limit to requested number
            
        except Exception as e:
            logger.error(f"GPT headline generation failed: {str(e)}")
            raise ToolExecutionError(f"Headline generation failed: {str(e)}")

    def _optimize_for_platform(self, headline: str, platform: Platform) -> str:
        """Optimize headline for specific platform requirements"""
        
        platform_limit = self._get_platform_limit(platform)
        
        if len(headline) <= platform_limit:
            return headline
        
        # Platform-specific optimization strategies
        if platform in [Platform.TWITTER, Platform.INSTAGRAM]:
            # For social media, prioritize impact over completeness
            words = headline.split()
            optimized = ""
            for word in words:
                test_length = len(optimized + " " + word) if optimized else len(word)
                if test_length <= platform_limit - 3:  # Leave room for "..."
                    optimized += " " + word if optimized else word
                else:
                    break
            return optimized + "..."
        
        elif platform == Platform.EMAIL:
            # For email, focus on the key benefit/question
            if '?' in headline:
                # Keep the question intact
                return headline[:platform_limit-3] + "..."
            else:
                # Try to end at a natural break
                truncated = headline[:platform_limit-3]
                last_space = truncated.rfind(' ')
                if last_space > platform_limit * 0.7:  # Only if we don't lose too much
                    return truncated[:last_space] + "..."
                return truncated + "..."
        
        # Default truncation
        return headline[:platform_limit-3] + "..."

    async def generate_headlines(self, request: HeadlineRequest) -> HeadlineResults:
        """
        Generate multiple headline variants with analysis
        
        Args:
            request: Headline generation request
            
        Returns:
            Complete headline results with analysis and recommendations
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Generating {request.num_variants} headlines for '{request.topic}' ({request.style.value})")
            
            # Generate headlines with GPT
            raw_headlines = await self._generate_with_gpt(request)
            
            # Analyze and optimize each headline
            generated_headlines = []
            
            for i, headline in enumerate(raw_headlines):
                # Optimize for platform if needed
                optimized_headline = self._optimize_for_platform(headline, request.platform)
                
                # Analyze performance
                analysis = self._analyze_headline(optimized_headline, request)
                
                # Create variation notes
                variation_notes = None
                if optimized_headline != headline:
                    variation_notes = f"Optimized for {request.platform.value} (original: {len(headline)} chars)"
                
                generated_headline = GeneratedHeadline(
                    headline=optimized_headline,
                    style=request.style,
                    analysis=analysis,
                    variation_notes=variation_notes,
                    platform_optimized=optimized_headline != headline
                )
                
                generated_headlines.append(generated_headline)
            
            # Find best headline
            best_headline = max(generated_headlines, key=lambda x: x.analysis.overall_score)
            
            # Generate statistics
            generation_time = (datetime.now() - start_time).total_seconds()
            
            generation_stats = {
                "total_generated": len(generated_headlines),
                "avg_length": sum(len(h.headline) for h in generated_headlines) / len(generated_headlines),
                "avg_word_count": sum(h.analysis.word_count for h in generated_headlines) / len(generated_headlines),
                "avg_emotional_score": sum(h.analysis.emotional_score for h in generated_headlines) / len(generated_headlines),
                "avg_predicted_ctr": sum(h.analysis.predicted_ctr for h in generated_headlines) / len(generated_headlines),
                "headlines_with_numbers": sum(1 for h in generated_headlines if h.analysis.has_numbers),
                "headlines_with_power_words": sum(1 for h in generated_headlines if h.analysis.has_power_words),
                "platform_optimized_count": sum(1 for h in generated_headlines if h.platform_optimized)
            }
            
            # A/B testing recommendations
            ab_test_recommendations = self._generate_ab_recommendations(generated_headlines, request)
            
            result = HeadlineResults(
                headlines=generated_headlines,
                best_headline=best_headline,
                generation_stats=generation_stats,
                ab_test_recommendations=ab_test_recommendations,
                total_variants=len(generated_headlines),
                generation_time=generation_time
            )
            
            logger.info(f"Generated {len(generated_headlines)} headlines successfully. Best score: {best_headline.analysis.overall_score:.1f}/100")
            return result
            
        except Exception as e:
            logger.error(f"Headline generation failed: {str(e)}")
            raise ToolExecutionError(f"Headline generation failed: {str(e)}")

    def _generate_ab_recommendations(
        self, 
        headlines: List[GeneratedHeadline], 
        request: HeadlineRequest
    ) -> Dict[str, Any]:
        """Generate A/B testing recommendations"""
        
        # Sort by overall score
        sorted_headlines = sorted(headlines, key=lambda x: x.analysis.overall_score, reverse=True)
        
        # Group into tiers for testing
        top_tier = sorted_headlines[:3] if len(sorted_headlines) >= 3 else sorted_headlines[:2]
        
        # Find diverse headlines for testing
        diverse_selection = []
        
        # Add the best overall
        diverse_selection.append(top_tier[0])
        
        # Add best emotional if different
        best_emotional = max(headlines, key=lambda x: x.analysis.emotional_score)
        if best_emotional not in diverse_selection:
            diverse_selection.append(best_emotional)
        
        # Add best curiosity if different
        best_curiosity = max(headlines, key=lambda x: x.analysis.curiosity_score)
        if best_curiosity not in diverse_selection and len(diverse_selection) < 3:
            diverse_selection.append(best_curiosity)
        
        # Add best urgency if different
        best_urgency = max(headlines, key=lambda x: x.analysis.urgency_score)
        if best_urgency not in diverse_selection and len(diverse_selection) < 4:
            diverse_selection.append(best_urgency)
        
        return {
            "recommended_for_testing": [
                {
                    "headline": h.headline,
                    "reason": self._get_selection_reason(h, headlines),
                    "predicted_ctr": h.analysis.predicted_ctr,
                    "overall_score": h.analysis.overall_score
                }
                for h in diverse_selection
            ],
            "testing_strategy": {
                "primary_vs_variants": f"Test top performer against {len(diverse_selection)-1} diverse variants",
                "sample_size_recommendation": "Minimum 1000 impressions per variant for statistical significance",
                "test_duration": "Run for at least 7 days to account for weekly patterns",
                "success_metrics": ["Click-through rate", "Engagement rate", "Conversion rate"]
            },
            "performance_insights": {
                "highest_emotional_impact": max(headlines, key=lambda x: x.analysis.emotional_score).headline,
                "most_curious": max(headlines, key=lambda x: x.analysis.curiosity_score).headline,
                "most_urgent": max(headlines, key=lambda x: x.analysis.urgency_score).headline,
                "best_seo": max(headlines, key=lambda x: x.analysis.seo_score).headline,
                "predicted_ctr_range": f"{min(h.analysis.predicted_ctr for h in headlines):.1f}% - {max(h.analysis.predicted_ctr for h in headlines):.1f}%"
            }
        }

    def _get_selection_reason(self, headline: GeneratedHeadline, all_headlines: List[GeneratedHeadline]) -> str:
        """Get reason why headline was selected for A/B testing"""
        
        if headline.analysis.overall_score == max(h.analysis.overall_score for h in all_headlines):
            return "Highest overall performance score"
        elif headline.analysis.emotional_score == max(h.analysis.emotional_score for h in all_headlines):
            return "Strongest emotional impact"
        elif headline.analysis.curiosity_score == max(h.analysis.curiosity_score for h in all_headlines):
            return "Highest curiosity factor"  
        elif headline.analysis.urgency_score == max(h.analysis.urgency_score for h in all_headlines):
            return "Greatest sense of urgency"
        elif headline.analysis.predicted_ctr == max(h.analysis.predicted_ctr for h in all_headlines):
            return "Highest predicted click-through rate"
        else:
            return "Strong performance across multiple factors"


# Initialize tool instance
headline_generator_tool = HeadlineGenerator()


# MCP Functions for external integration
async def mcp_generate_headlines(
    topic: str,
    style: str = "question",
    tone: str = "professional",
    platform: str = "blog",
    num_variants: int = 5,
    target_audience: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    emotional_trigger: Optional[str] = None,
    max_length: Optional[int] = None,
    include_numbers: bool = False,
    temperature: float = 0.8
) -> Dict[str, Any]:
    """
    MCP function to generate headlines with analysis
    
    Args:
        topic: Main topic or content subject
        style: Headline style (question, how_to, listicle, etc.)
        tone: Headline tone (professional, casual, etc.)
        platform: Target platform (blog, social_media, email, etc.)
        num_variants: Number of headline variants (1-20)
        target_audience: Target audience description
        keywords: Keywords to include
        emotional_trigger: Primary emotional trigger
        max_length: Maximum character length
        include_numbers: Include specific numbers
        temperature: Creativity level (0.0-2.0)
        
    Returns:
        Generated headlines with analysis and recommendations
    """
    try:
        request = HeadlineRequest(
            topic=topic,
            style=HeadlineStyle(style),
            tone=HeadlineTone(tone),
            platform=Platform(platform),
            num_variants=num_variants,
            target_audience=target_audience,
            keywords=keywords,
            emotional_trigger=EmotionalTrigger(emotional_trigger) if emotional_trigger else None,
            max_length=max_length,
            include_numbers=include_numbers,
            temperature=temperature
        )
        
        result = await headline_generator_tool.generate_headlines(request)
        
        return {
            "success": True,
            "headlines": [
                {
                    "headline": h.headline,
                    "style": h.style.value,
                    "analysis": {
                        "overall_score": h.analysis.overall_score,
                        "emotional_score": h.analysis.emotional_score,
                        "clarity_score": h.analysis.clarity_score,
                        "curiosity_score": h.analysis.curiosity_score,
                        "urgency_score": h.analysis.urgency_score,
                        "seo_score": h.analysis.seo_score,
                        "predicted_ctr": h.analysis.predicted_ctr,
                        "length": h.analysis.length,
                        "word_count": h.analysis.word_count,
                        "has_numbers": h.analysis.has_numbers,
                        "has_power_words": h.analysis.has_power_words,
                        "emotional_triggers": h.analysis.has_emotional_triggers
                    },
                    "platform_optimized": h.platform_optimized,
                    "variation_notes": h.variation_notes
                }
                for h in result.headlines
            ],
            "best_headline": {
                "headline": result.best_headline.headline,
                "overall_score": result.best_headline.analysis.overall_score,
                "predicted_ctr": result.best_headline.analysis.predicted_ctr
            },
            "statistics": result.generation_stats,
            "ab_testing": result.ab_test_recommendations,
            "generation_time": result.generation_time,
            "total_variants": result.total_variants
        }
        
    except Exception as e:
        logger.error(f"MCP headline generation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


async def mcp_optimize_headline_for_platform(
    headline: str,
    platform: str,
    max_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    MCP function to optimize a headline for a specific platform
    
    Args:
        headline: Original headline
        platform: Target platform
        max_length: Custom maximum length
        
    Returns:
        Optimized headline with analysis
    """
    try:
        tool = headline_generator_tool
        platform_enum = Platform(platform)
        
        # Get platform limit
        limit = max_length or tool._get_platform_limit(platform_enum)
        
        # Optimize headline
        optimized = tool._optimize_for_platform(headline, platform_enum)
        
        # Create dummy request for analysis
        dummy_request = HeadlineRequest(
            topic="optimization",
            style=HeadlineStyle.QUESTION,
            platform=platform_enum
        )
        
        # Analyze both versions
        original_analysis = tool._analyze_headline(headline, dummy_request)
        optimized_analysis = tool._analyze_headline(optimized, dummy_request)
        
        return {
            "success": True,
            "original": {
                "headline": headline,
                "length": len(headline),
                "analysis": {
                    "overall_score": original_analysis.overall_score,
                    "predicted_ctr": original_analysis.predicted_ctr
                }
            },
            "optimized": {
                "headline": optimized,
                "length": len(optimized),
                "analysis": {
                    "overall_score": optimized_analysis.overall_score,
                    "predicted_ctr": optimized_analysis.predicted_ctr
                }
            },
            "platform_limit": limit,
            "optimization_needed": len(headline) > limit,
            "improvement": optimized_analysis.overall_score - original_analysis.overall_score
        }
        
    except Exception as e:
        logger.error(f"MCP headline optimization failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Example usage and testing
    async def test_headline_generator():
        """Test the headline generator functionality"""
        
        request = HeadlineRequest(
            topic="How to Boost Website Conversion Rates with AI",
            style=HeadlineStyle.HOW_TO,
            tone=HeadlineTone.PROFESSIONAL,
            platform=Platform.BLOG,
            num_variants=5,
            target_audience="Digital marketers and business owners",
            keywords=["conversion rate", "AI", "website optimization"],
            emotional_trigger=EmotionalTrigger.CURIOSITY,
            include_numbers=True,
            temperature=0.8
        )
        
        try:
            generator = HeadlineGenerator()
            result = await generator.generate_headlines(request)
            
            print(f"Generated {len(result.headlines)} headlines:")
            print("="*60)
            
            for i, headline in enumerate(result.headlines, 1):
                print(f"{i}. {headline.headline}")
                print(f"   Score: {headline.analysis.overall_score:.1f}/100")
                print(f"   CTR: {headline.analysis.predicted_ctr:.1f}%")
                print(f"   Length: {headline.analysis.length} chars")
                if headline.platform_optimized:
                    print(f"   Note: {headline.variation_notes}")
                print()
            
            print(f"Best Headline: {result.best_headline.headline}")
            print(f"Best Score: {result.best_headline.analysis.overall_score:.1f}/100")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_headline_generator())