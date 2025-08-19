"""
Content Analysis Tool for sentiment analysis and readability scoring.

Provides comprehensive content analysis including sentiment analysis,
readability scoring, brand voice consistency, and audience alignment.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import math

import textstat
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pydantic import BaseModel, Field

from ...core.config.loader import get_settings
from ...core.errors import ToolError
from ...utils.simple_retry import with_retry


logger = logging.getLogger(__name__)


@dataclass
class SentimentAnalysis:
    """Sentiment analysis results."""
    polarity: float  # -1 (negative) to 1 (positive)
    subjectivity: float  # 0 (objective) to 1 (subjective)
    compound: float  # VADER compound score
    positive: float  # VADER positive score
    negative: float  # VADER negative score
    neutral: float  # VADER neutral score
    sentiment_label: str  # positive, negative, neutral
    confidence: float


@dataclass
class ReadabilityMetrics:
    """Readability analysis results."""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    smog_index: float
    coleman_liau_index: float
    automated_readability_index: float
    avg_sentence_length: float
    avg_syllables_per_word: float
    reading_level: str
    target_audience: str


@dataclass
class StyleAnalysis:
    """Writing style analysis results."""
    formality_score: float  # 0 (informal) to 1 (formal)
    complexity_score: float  # 0 (simple) to 1 (complex)
    tone: str  # professional, casual, academic, etc.
    voice: str  # active, passive
    tense_consistency: float  # 0 to 1
    person_consistency: float  # 0 to 1
    avg_word_length: float
    vocabulary_diversity: float


@dataclass
class ContentQuality:
    """Content quality assessment."""
    overall_score: float  # 0 to 100
    grammar_score: float
    coherence_score: float
    engagement_score: float
    clarity_score: float
    issues: List[str]
    recommendations: List[str]


@dataclass
class ContentAnalysisResult:
    """Complete content analysis result."""
    sentiment: SentimentAnalysis
    readability: ReadabilityMetrics
    style: StyleAnalysis
    quality: ContentQuality
    word_count: int
    character_count: int
    sentence_count: int
    paragraph_count: int
    unique_words: int
    processing_time: float


class ContentAnalysisRequest(BaseModel):
    """Content analysis request configuration."""
    text: str = Field(..., description="Text content to analyze")
    analyze_sentiment: bool = Field(default=True, description="Perform sentiment analysis")
    analyze_readability: bool = Field(default=True, description="Calculate readability metrics")
    analyze_style: bool = Field(default=True, description="Analyze writing style")
    analyze_quality: bool = Field(default=True, description="Assess content quality")
    target_audience: Optional[str] = Field(default=None, description="Target audience level")
    brand_voice: Optional[str] = Field(default=None, description="Expected brand voice")
    language: str = Field(default="en", description="Content language")


class ContentAnalysisResponse(BaseModel):
    """Content analysis response."""
    success: bool
    text_length: int
    result: Optional[ContentAnalysisResult] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class ContentAnalysisTool:
    """
    Content Analysis Tool for comprehensive text quality assessment.
    
    Provides sentiment analysis, readability scoring, style analysis,
    and content quality assessment capabilities.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Formality indicators
        self.formal_indicators = {
            'words': ['therefore', 'furthermore', 'moreover', 'consequently', 'nevertheless', 
                     'however', 'accordingly', 'subsequently', 'additionally', 'specifically'],
            'phrases': ['in conclusion', 'for instance', 'in particular', 'as a result', 
                       'on the other hand', 'it should be noted', 'it is important to'],
            'patterns': [r'\b(shall|ought to|must)\b', r'\b(utilize|commence|terminate)\b']
        }
        
        self.informal_indicators = {
            'words': ['awesome', 'cool', 'great', 'super', 'really', 'pretty', 'kinda', 
                     'sorta', 'gonna', 'wanna', 'yeah', 'ok', 'okay'],
            'contractions': [r"'ll", r"'re", r"'ve", r"'d", r"n't", r"'m"],
            'patterns': [r'\b(lots of|tons of|bunch of)\b', r'\!+']
        }
        
        # Brand voice patterns
        self.brand_voices = {
            'professional': {
                'characteristics': ['formal', 'precise', 'authoritative'],
                'indicators': ['demonstrate', 'implement', 'establish', 'facilitate'],
                'avoid': ['awesome', 'cool', 'super']
            },
            'friendly': {
                'characteristics': ['warm', 'approachable', 'conversational'],
                'indicators': ['help', 'support', 'together', 'community'],
                'avoid': ['terminate', 'cease', 'discontinue']
            },
            'authoritative': {
                'characteristics': ['expert', 'definitive', 'confident'],
                'indicators': ['proven', 'essential', 'critical', 'must'],
                'avoid': ['maybe', 'might', 'possibly']
            },
            'casual': {
                'characteristics': ['relaxed', 'informal', 'conversational'],
                'indicators': ['hey', 'great', 'awesome', 'check out'],
                'avoid': ['shall', 'therefore', 'furthermore']
            }
        }
        
        # Quality assessment criteria
        self.quality_criteria = {
            'grammar_patterns': [
                (r'\b(there|their|they\'re)\b', 'there/their/they\'re confusion'),
                (r'\b(your|you\'re)\b', 'your/you\'re confusion'),
                (r'\b(its|it\'s)\b', 'its/it\'s confusion'),
                (r'\s{2,}', 'excessive whitespace'),
                (r'[.]{3,}', 'excessive ellipsis'),
                (r'[!]{2,}', 'excessive exclamation marks')
            ]
        }
    
    def _analyze_sentiment_textblob(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except Exception as e:
            logger.error(f"TextBlob sentiment analysis failed: {e}")
            return 0.0, 0.0
    
    def _analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return scores
        except Exception as e:
            logger.error(f"VADER sentiment analysis failed: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    
    def _get_sentiment_label(self, polarity: float, compound: float) -> Tuple[str, float]:
        """Determine sentiment label and confidence."""
        # Use compound score for more nuanced classification
        if compound >= 0.05:
            return "positive", abs(compound)
        elif compound <= -0.05:
            return "negative", abs(compound)
        else:
            return "neutral", 1 - abs(compound)
    
    def _analyze_readability(self, text: str) -> ReadabilityMetrics:
        """Calculate comprehensive readability metrics."""
        try:
            # Calculate various readability metrics
            flesch_ease = textstat.flesch_reading_ease(text)
            flesch_grade = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
            smog = textstat.smog_index(text)
            coleman_liau = textstat.coleman_liau_index(text)
            ari = textstat.automated_readability_index(text)
            
            # Calculate additional metrics
            sentence_count = textstat.sentence_count(text)
            word_count = textstat.lexicon_count(text)
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            syllable_count = textstat.syllable_count(text)
            avg_syllables_per_word = syllable_count / max(word_count, 1)
            
            # Determine reading level based on Flesch score
            if flesch_ease >= 90:
                reading_level = "Very Easy (5th grade)"
                target_audience = "Elementary school"
            elif flesch_ease >= 80:
                reading_level = "Easy (6th grade)"
                target_audience = "Middle school"
            elif flesch_ease >= 70:
                reading_level = "Fairly Easy (7th grade)"
                target_audience = "7th grade"
            elif flesch_ease >= 60:
                reading_level = "Standard (8th-9th grade)"
                target_audience = "High school"
            elif flesch_ease >= 50:
                reading_level = "Fairly Difficult (10th-12th grade)"
                target_audience = "High school senior"
            elif flesch_ease >= 30:
                reading_level = "Difficult (College level)"
                target_audience = "College student"
            else:
                reading_level = "Very Difficult (Graduate level)"
                target_audience = "Graduate student"
            
            return ReadabilityMetrics(
                flesch_reading_ease=flesch_ease,
                flesch_kincaid_grade=flesch_grade,
                gunning_fog=gunning_fog,
                smog_index=smog,
                coleman_liau_index=coleman_liau,
                automated_readability_index=ari,
                avg_sentence_length=avg_sentence_length,
                avg_syllables_per_word=avg_syllables_per_word,
                reading_level=reading_level,
                target_audience=target_audience
            )
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return ReadabilityMetrics(
                flesch_reading_ease=0.0,
                flesch_kincaid_grade=0.0,
                gunning_fog=0.0,
                smog_index=0.0,
                coleman_liau_index=0.0,
                automated_readability_index=0.0,
                avg_sentence_length=0.0,
                avg_syllables_per_word=0.0,
                reading_level="Unknown",
                target_audience="Unknown"
            )
    
    def _analyze_formality(self, text: str) -> float:
        """Analyze text formality (0=informal, 1=formal)."""
        try:
            text_lower = text.lower()
            formal_count = 0
            informal_count = 0
            
            # Count formal indicators
            for word in self.formal_indicators['words']:
                formal_count += text_lower.count(word)
            
            for phrase in self.formal_indicators['phrases']:
                formal_count += text_lower.count(phrase)
            
            for pattern in self.formal_indicators['patterns']:
                formal_count += len(re.findall(pattern, text, re.IGNORECASE))
            
            # Count informal indicators
            for word in self.informal_indicators['words']:
                informal_count += text_lower.count(word)
            
            for pattern in self.informal_indicators['contractions']:
                informal_count += len(re.findall(pattern, text))
            
            for pattern in self.informal_indicators['patterns']:
                informal_count += len(re.findall(pattern, text, re.IGNORECASE))
            
            # Calculate formality score
            total_indicators = formal_count + informal_count
            if total_indicators == 0:
                return 0.5  # Neutral
            
            formality_score = formal_count / total_indicators
            return formality_score
            
        except Exception as e:
            logger.error(f"Formality analysis failed: {e}")
            return 0.5
    
    def _analyze_complexity(self, text: str) -> float:
        """Analyze text complexity (0=simple, 1=complex)."""
        try:
            words = re.findall(r'\b\w+\b', text)
            if not words:
                return 0.0
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Sentence length variance
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return 0.0
            
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
            
            # Normalize scores
            word_complexity = min(avg_word_length / 10, 1.0)  # Normalize to 0-1
            sentence_complexity = min(avg_sentence_length / 30, 1.0)  # Normalize to 0-1
            
            # Combined complexity score
            complexity_score = (word_complexity + sentence_complexity) / 2
            return complexity_score
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            return 0.5
    
    def _determine_tone(self, formality: float, sentiment_polarity: float) -> str:
        """Determine overall tone of the text."""
        if formality > 0.7:
            if sentiment_polarity > 0.2:
                return "professional-positive"
            elif sentiment_polarity < -0.2:
                return "professional-critical"
            else:
                return "professional-neutral"
        elif formality < 0.3:
            if sentiment_polarity > 0.2:
                return "casual-friendly"
            elif sentiment_polarity < -0.2:
                return "casual-critical"
            else:
                return "casual-neutral"
        else:
            if sentiment_polarity > 0.2:
                return "conversational-positive"
            elif sentiment_polarity < -0.2:
                return "conversational-concerned"
            else:
                return "conversational-balanced"
    
    def _analyze_voice(self, text: str) -> str:
        """Analyze voice (active vs passive)."""
        try:
            # Simple passive voice detection
            passive_indicators = [
                r'\b(was|were|been|being)\s+\w+ed\b',
                r'\b(is|are|am)\s+\w+ed\b',
                r'\bby\s+\w+\b'
            ]
            
            passive_count = 0
            for pattern in passive_indicators:
                passive_count += len(re.findall(pattern, text, re.IGNORECASE))
            
            sentence_count = len(re.split(r'[.!?]+', text))
            passive_ratio = passive_count / max(sentence_count, 1)
            
            if passive_ratio > 0.3:
                return "predominantly-passive"
            elif passive_ratio > 0.1:
                return "mixed"
            else:
                return "predominantly-active"
                
        except Exception as e:
            logger.error(f"Voice analysis failed: {e}")
            return "mixed"
    
    def _calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate vocabulary diversity (Type-Token Ratio)."""
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            if not words:
                return 0.0
            
            unique_words = set(words)
            diversity = len(unique_words) / len(words)
            return diversity
            
        except Exception as e:
            logger.error(f"Vocabulary diversity calculation failed: {e}")
            return 0.0
    
    def _assess_content_quality(self, text: str, sentiment: SentimentAnalysis, 
                              readability: ReadabilityMetrics, style: StyleAnalysis) -> ContentQuality:
        """Assess overall content quality."""
        try:
            issues = []
            recommendations = []
            scores = {}
            
            # Grammar assessment (simplified)
            grammar_issues = 0
            for pattern, issue_type in self.quality_criteria['grammar_patterns']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    grammar_issues += len(matches)
                    issues.append(f"{issue_type.title()} detected")
            
            # Grammar score (inverse of issues)
            grammar_score = max(0, 100 - (grammar_issues * 5))
            scores['grammar'] = grammar_score
            
            # Readability assessment
            if readability.flesch_reading_ease < 30:
                issues.append("Text is very difficult to read")
                recommendations.append("Simplify sentence structure and vocabulary")
            elif readability.flesch_reading_ease > 90:
                issues.append("Text might be too simplistic")
                recommendations.append("Consider adding more sophisticated vocabulary")
            
            # Coherence assessment (based on sentence length variance)
            sentences = re.split(r'[.!?]+', text)
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            
            if sentence_lengths:
                avg_len = sum(sentence_lengths) / len(sentence_lengths)
                variance = sum((x - avg_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
                coherence_score = max(0, 100 - (variance * 2))
            else:
                coherence_score = 0
            
            scores['coherence'] = coherence_score
            
            # Engagement assessment (based on sentiment and style)
            engagement_factors = []
            if abs(sentiment.polarity) > 0.1:  # Has emotional content
                engagement_factors.append(20)
            if sentiment.subjectivity > 0.3:  # Has personal opinions
                engagement_factors.append(15)
            if style.vocabulary_diversity > 0.5:  # Good vocabulary variety
                engagement_factors.append(25)
            if 10 <= readability.avg_sentence_length <= 20:  # Good sentence length
                engagement_factors.append(20)
            if style.formality > 0.3 and style.formality < 0.7:  # Balanced formality
                engagement_factors.append(20)
            
            engagement_score = sum(engagement_factors)
            scores['engagement'] = min(engagement_score, 100)
            
            # Clarity assessment (based on readability and complexity)
            clarity_score = (readability.flesch_reading_ease + 
                           (100 - style.complexity_score * 100)) / 2
            scores['clarity'] = max(0, clarity_score)
            
            # Overall quality score
            overall_score = sum(scores.values()) / len(scores)
            
            # Generate recommendations
            if grammar_score < 80:
                recommendations.append("Review text for grammar and spelling errors")
            if coherence_score < 70:
                recommendations.append("Improve sentence length consistency")
            if engagement_score < 60:
                recommendations.append("Add more engaging language and varied vocabulary")
            if clarity_score < 70:
                recommendations.append("Simplify complex sentences for better clarity")
            
            return ContentQuality(
                overall_score=overall_score,
                grammar_score=grammar_score,
                coherence_score=coherence_score,
                engagement_score=scores['engagement'],
                clarity_score=scores['clarity'],
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return ContentQuality(
                overall_score=0.0,
                grammar_score=0.0,
                coherence_score=0.0,
                engagement_score=0.0,
                clarity_score=0.0,
                issues=["Quality assessment failed"],
                recommendations=["Please retry analysis"]
            )
    
    async def analyze_content(
        self,
        request: Union[str, ContentAnalysisRequest]
    ) -> ContentAnalysisResponse:
        """
        Analyze content comprehensively for quality, sentiment, and readability.
        
        Args:
            request: Text string or ContentAnalysisRequest object
            
        Returns:
            ContentAnalysisResponse with complete analysis
        """
        start_time = datetime.now()
        
        # Convert string to request object
        if isinstance(request, str):
            request = ContentAnalysisRequest(text=request)
        
        try:
            text = request.text
            text_length = len(text)
            
            # Check minimum text length
            if text_length < 50:
                return ContentAnalysisResponse(
                    success=False,
                    text_length=text_length,
                    error="Text too short for meaningful analysis (minimum 50 characters)",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    timestamp=datetime.now()
                )
            
            logger.info(f"Analyzing content: {text_length} characters")
            
            # Basic text metrics
            word_count = len(re.findall(r'\b\w+\b', text))
            character_count = len(text)
            sentence_count = len(re.split(r'[.!?]+', text.strip()))
            paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
            unique_words = len(set(re.findall(r'\b\w+\b', text.lower())))
            
            # Sentiment analysis
            sentiment = None
            if request.analyze_sentiment:
                polarity, subjectivity = self._analyze_sentiment_textblob(text)
                vader_scores = self._analyze_sentiment_vader(text)
                sentiment_label, confidence = self._get_sentiment_label(polarity, vader_scores['compound'])
                
                sentiment = SentimentAnalysis(
                    polarity=polarity,
                    subjectivity=subjectivity,
                    compound=vader_scores['compound'],
                    positive=vader_scores['pos'],
                    negative=vader_scores['neg'],
                    neutral=vader_scores['neu'],
                    sentiment_label=sentiment_label,
                    confidence=confidence
                )
            
            # Readability analysis
            readability = None
            if request.analyze_readability:
                readability = self._analyze_readability(text)
            
            # Style analysis
            style = None
            if request.analyze_style:
                formality_score = self._analyze_formality(text)
                complexity_score = self._analyze_complexity(text)
                tone = self._determine_tone(formality_score, sentiment.polarity if sentiment else 0.0)
                voice = self._analyze_voice(text)
                vocabulary_diversity = self._calculate_vocabulary_diversity(text)
                
                # Calculate average word length
                words = re.findall(r'\b\w+\b', text)
                avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
                
                style = StyleAnalysis(
                    formality_score=formality_score,
                    complexity_score=complexity_score,
                    tone=tone,
                    voice=voice,
                    tense_consistency=0.8,  # Placeholder - would need more complex analysis
                    person_consistency=0.8,  # Placeholder - would need more complex analysis
                    avg_word_length=avg_word_length,
                    vocabulary_diversity=vocabulary_diversity
                )
            
            # Quality assessment
            quality = None
            if request.analyze_quality and sentiment and readability and style:
                quality = self._assess_content_quality(text, sentiment, readability, style)
            
            # Create result
            result = ContentAnalysisResult(
                sentiment=sentiment or SentimentAnalysis(0, 0, 0, 0, 0, 0, "neutral", 0),
                readability=readability or ReadabilityMetrics(0, 0, 0, 0, 0, 0, 0, 0, "Unknown", "Unknown"),
                style=style or StyleAnalysis(0, 0, "neutral", "mixed", 0, 0, 0, 0),
                quality=quality or ContentQuality(0, 0, 0, 0, 0, [], []),
                word_count=word_count,
                character_count=character_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                unique_words=unique_words,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ContentAnalysisResponse(
                success=True,
                text_length=text_length,
                result=result,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ContentAnalysisResponse(
                success=False,
                text_length=len(request.text),
                error=str(e),
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    async def batch_analyze_content(self, texts: List[str]) -> List[ContentAnalysisResponse]:
        """
        Analyze multiple texts concurrently.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of ContentAnalysisResponse objects
        """
        tasks = [self.analyze_content(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch content analysis failed for text {i}: {result}")
                responses.append(ContentAnalysisResponse(
                    success=False,
                    text_length=len(texts[i]),
                    error=str(result),
                    processing_time=0.0,
                    timestamp=datetime.now()
                ))
            else:
                responses.append(result)
        
        return responses


# Tool instance for MCP integration
content_analysis_tool = ContentAnalysisTool()


# MCP tool function
async def mcp_analyze_content(
    text: str,
    analyze_sentiment: bool = True,
    analyze_readability: bool = True,
    analyze_style: bool = True,
    analyze_quality: bool = True,
    target_audience: Optional[str] = None,
    brand_voice: Optional[str] = None,
    language: str = "en"
) -> Dict:
    """
    MCP-compatible content analysis function.
    
    Args:
        text: Text content to analyze
        analyze_sentiment: Perform sentiment analysis
        analyze_readability: Calculate readability metrics
        analyze_style: Analyze writing style
        analyze_quality: Assess content quality
        target_audience: Target audience level
        brand_voice: Expected brand voice
        language: Content language
    
    Returns:
        Dictionary with content analysis results
    """
    try:
        request = ContentAnalysisRequest(
            text=text,
            analyze_sentiment=analyze_sentiment,
            analyze_readability=analyze_readability,
            analyze_style=analyze_style,
            analyze_quality=analyze_quality,
            target_audience=target_audience,
            brand_voice=brand_voice,
            language=language
        )
        
        response = await content_analysis_tool.analyze_content(request)
        
        if response.success and response.result:
            result = response.result
            
            return {
                "success": True,
                "text_length": response.text_length,
                "basic_metrics": {
                    "word_count": result.word_count,
                    "character_count": result.character_count,
                    "sentence_count": result.sentence_count,
                    "paragraph_count": result.paragraph_count,
                    "unique_words": result.unique_words
                },
                "sentiment": {
                    "polarity": result.sentiment.polarity,
                    "subjectivity": result.sentiment.subjectivity,
                    "compound": result.sentiment.compound,
                    "positive": result.sentiment.positive,
                    "negative": result.sentiment.negative,
                    "neutral": result.sentiment.neutral,
                    "sentiment_label": result.sentiment.sentiment_label,
                    "confidence": result.sentiment.confidence
                } if analyze_sentiment else None,
                "readability": {
                    "flesch_reading_ease": result.readability.flesch_reading_ease,
                    "flesch_kincaid_grade": result.readability.flesch_kincaid_grade,
                    "gunning_fog": result.readability.gunning_fog,
                    "avg_sentence_length": result.readability.avg_sentence_length,
                    "reading_level": result.readability.reading_level,
                    "target_audience": result.readability.target_audience
                } if analyze_readability else None,
                "style": {
                    "formality_score": result.style.formality_score,
                    "complexity_score": result.style.complexity_score,
                    "tone": result.style.tone,
                    "voice": result.style.voice,
                    "avg_word_length": result.style.avg_word_length,
                    "vocabulary_diversity": result.style.vocabulary_diversity
                } if analyze_style else None,
                "quality": {
                    "overall_score": result.quality.overall_score,
                    "grammar_score": result.quality.grammar_score,
                    "coherence_score": result.quality.coherence_score,
                    "engagement_score": result.quality.engagement_score,
                    "clarity_score": result.quality.clarity_score,
                    "issues": result.quality.issues,
                    "recommendations": result.quality.recommendations
                } if analyze_quality else None,
                "processing_time": response.processing_time,
                "timestamp": response.timestamp.isoformat()
            }
        else:
            return {
                "success": False,
                "error": response.error,
                "text_length": response.text_length,
                "processing_time": response.processing_time
            }
            
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }