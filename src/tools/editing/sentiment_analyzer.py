"""
Sentiment Analyzer Tool - Emotional Tone Analysis and Brand Voice Consistency

This module provides comprehensive sentiment analysis capabilities including emotional tone detection,
brand voice consistency checking, and audience sentiment prediction for content optimization.

Key Features:
- Multi-dimensional sentiment analysis (polarity, subjectivity, emotions)
- Brand voice consistency checking
- Audience sentiment prediction
- Emotional tone mapping and scoring
- Content mood analysis
- Competitive sentiment benchmarking
"""

import asyncio
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal, Any, Tuple
from enum import Enum
from collections import Counter, defaultdict
import string

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pydantic import BaseModel, Field, validator

from ...core.errors.exceptions import ToolExecutionError
from ...core.logging.logger import get_logger
from ...utils.retry import with_retry

logger = get_logger(__name__)

# Ensure required NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class EmotionType(str, Enum):
    """Primary emotion types"""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


class SentimentPolarity(str, Enum):
    """Sentiment polarity categories"""
    VERY_POSITIVE = "very_positive"     # > 0.5
    POSITIVE = "positive"               # 0.1 to 0.5
    NEUTRAL = "neutral"                 # -0.1 to 0.1
    NEGATIVE = "negative"               # -0.5 to -0.1
    VERY_NEGATIVE = "very_negative"     # < -0.5


class BrandVoice(str, Enum):
    """Brand voice types"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    PLAYFUL = "playful"
    EMPATHETIC = "empathetic"
    INNOVATIVE = "innovative"
    TRUSTWORTHY = "trustworthy"
    INSPIRING = "inspiring"
    CONVERSATIONAL = "conversational"
    LUXURY = "luxury"


class ContentMood(str, Enum):
    """Overall content mood"""
    OPTIMISTIC = "optimistic"
    CONFIDENT = "confident"
    NEUTRAL = "neutral"
    CONCERNED = "concerned"
    URGENT = "urgent"
    EXCITED = "excited"
    CALM = "calm"
    SERIOUS = "serious"


class SentimentScore(BaseModel):
    """Individual sentiment score"""
    
    polarity: float = Field(..., description="Sentiment polarity (-1 to 1)", ge=-1, le=1)
    subjectivity: float = Field(..., description="Subjectivity score (0 to 1)", ge=0, le=1)
    
    polarity_label: SentimentPolarity = Field(..., description="Polarity category")
    confidence: float = Field(..., description="Confidence in sentiment detection (0-1)")
    
    emotional_intensity: float = Field(..., description="Overall emotional intensity (0-1)")


class EmotionAnalysis(BaseModel):
    """Emotion detection and analysis"""
    
    primary_emotion: EmotionType = Field(..., description="Primary detected emotion")
    emotion_scores: Dict[str, float] = Field(..., description="Scores for all emotions (0-1)")
    
    emotional_stability: float = Field(..., description="Emotional consistency score (0-100)")
    emotional_range: float = Field(..., description="Range of emotions present (0-1)")
    
    dominant_emotions: List[EmotionType] = Field(..., description="Top emotions by strength")


class BrandVoiceAnalysis(BaseModel):
    """Brand voice consistency analysis"""
    
    detected_voice: BrandVoice = Field(..., description="Detected brand voice")
    voice_consistency: float = Field(..., description="Voice consistency score (0-100)")
    
    voice_attributes: Dict[str, float] = Field(..., description="Strength of voice attributes (0-1)")
    
    brand_alignment: Optional[float] = Field(
        default=None, 
        description="Alignment with target brand voice (0-100)"
    )
    
    tone_variations: List[str] = Field(..., description="Detected tone variations")


class AudienceReaction(BaseModel):
    """Predicted audience reaction"""
    
    engagement_prediction: float = Field(..., description="Predicted engagement score (0-100)")
    emotional_resonance: float = Field(..., description="Emotional resonance with audience (0-100)")
    
    appeal_factors: List[str] = Field(..., description="Factors that increase appeal")
    concern_factors: List[str] = Field(..., description="Potential concerns or negatives")
    
    audience_sentiment_match: float = Field(..., description="Match with target audience sentiment (0-100)")


class SentimentAnalysisRequest(BaseModel):
    """Sentiment analysis request parameters"""
    
    text: str = Field(..., description="Text to analyze for sentiment")
    
    target_brand_voice: Optional[BrandVoice] = Field(
        default=None,
        description="Target brand voice for consistency checking"
    )
    
    target_audience: Optional[str] = Field(
        default=None,
        description="Target audience description"
    )
    
    content_context: Optional[str] = Field(
        default=None,
        description="Content context (marketing, support, news, etc.)"
    )
    
    competitor_sentiment: Optional[float] = Field(
        default=None,
        description="Competitor sentiment benchmark (-1 to 1)",
        ge=-1,
        le=1
    )
    
    custom_emotion_keywords: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Custom keywords for specific emotions"
    )
    
    include_sentence_analysis: bool = Field(
        default=True,
        description="Include sentence-by-sentence sentiment analysis"
    )
    
    language: str = Field(default="en", description="Content language")

    @validator('text')
    def validate_text_length(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Text too short for meaningful sentiment analysis")
        if len(v) > 25000:
            raise ValueError("Text too long (max 25,000 characters)")
        return v


class SentimentAnalysisResults(BaseModel):
    """Complete sentiment analysis results"""
    
    overall_sentiment: SentimentScore = Field(..., description="Overall sentiment score")
    content_mood: ContentMood = Field(..., description="Overall content mood")
    
    emotion_analysis: EmotionAnalysis = Field(..., description="Emotion detection results")
    brand_voice_analysis: BrandVoiceAnalysis = Field(..., description="Brand voice analysis")
    audience_reaction: AudienceReaction = Field(..., description="Predicted audience reaction")
    
    sentence_sentiments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Sentence-by-sentence sentiment analysis"
    )
    
    sentiment_trends: Dict[str, float] = Field(..., description="Sentiment trends throughout text")
    key_phrases: List[Tuple[str, float]] = Field(..., description="Key phrases with sentiment scores")
    
    recommendations: List[str] = Field(..., description="Sentiment optimization recommendations")
    warnings: List[str] = Field(..., description="Potential sentiment issues")
    
    processing_time: float = Field(..., description="Analysis processing time")
    confidence_score: float = Field(..., description="Overall analysis confidence (0-100)")


class EmotionKeywords:
    """Emotion keyword mappings for detection"""
    
    EMOTION_KEYWORDS = {
        EmotionType.JOY: [
            "happy", "joy", "delight", "pleasure", "cheerful", "glad", "elated",
            "thrilled", "excited", "wonderful", "amazing", "fantastic", "great",
            "excellent", "brilliant", "awesome", "love", "adore", "celebrate"
        ],
        
        EmotionType.TRUST: [
            "trust", "reliable", "dependable", "honest", "authentic", "genuine",
            "credible", "confident", "secure", "safe", "assured", "certain",
            "proven", "established", "reputable", "trustworthy", "solid", "stable"
        ],
        
        EmotionType.FEAR: [
            "fear", "afraid", "scared", "worried", "anxious", "nervous", "panic",
            "terrified", "frightened", "concerned", "uneasy", "apprehensive",
            "dread", "alarm", "danger", "risk", "threat", "warning", "caution"
        ],
        
        EmotionType.SURPRISE: [
            "surprise", "amazing", "astonishing", "unexpected", "sudden", "shock",
            "startling", "remarkable", "incredible", "unbelievable", "wow",
            "stunning", "extraordinary", "unprecedented", "breakthrough", "discovery"
        ],
        
        EmotionType.SADNESS: [
            "sad", "unhappy", "depressed", "melancholy", "grief", "sorrow",
            "disappointed", "heartbroken", "devastated", "tragic", "unfortunate",
            "regret", "loss", "mourning", "gloomy", "blue", "down"
        ],
        
        EmotionType.DISGUST: [
            "disgusting", "revolting", "repulsive", "awful", "terrible", "horrible",
            "nasty", "gross", "sick", "disturbing", "offensive", "unpleasant",
            "distasteful", "appalling", "shocking", "outrageous"
        ],
        
        EmotionType.ANGER: [
            "angry", "mad", "furious", "rage", "outraged", "irritated", "annoyed",
            "frustrated", "hostile", "aggressive", "violent", "hate", "despise",
            "resentful", "bitter", "infuriated", "livid", "enraged"
        ],
        
        EmotionType.ANTICIPATION: [
            "anticipate", "expect", "hope", "eager", "excited", "looking forward",
            "await", "upcoming", "future", "soon", "planning", "prepare",
            "ready", "waiting", "optimistic", "hopeful", "enthusiastic"
        ]
    }
    
    BRAND_VOICE_KEYWORDS = {
        BrandVoice.PROFESSIONAL: [
            "professional", "expert", "quality", "excellence", "industry", "standard",
            "certified", "qualified", "experienced", "established", "proven"
        ],
        
        BrandVoice.FRIENDLY: [
            "friendly", "welcome", "warm", "approachable", "helpful", "kind",
            "caring", "supportive", "comfortable", "easy", "simple"
        ],
        
        BrandVoice.AUTHORITATIVE: [
            "authority", "expert", "leader", "leading", "premier", "advanced",
            "superior", "dominant", "powerful", "control", "command"
        ],
        
        BrandVoice.PLAYFUL: [
            "fun", "playful", "creative", "innovative", "unique", "colorful",
            "vibrant", "energetic", "dynamic", "fresh", "bold"
        ],
        
        BrandVoice.EMPATHETIC: [
            "understand", "care", "support", "help", "empathy", "compassion",
            "sensitive", "thoughtful", "considerate", "gentle", "patient"
        ],
        
        BrandVoice.INNOVATIVE: [
            "innovative", "cutting-edge", "revolutionary", "breakthrough", "advanced",
            "next-generation", "pioneering", "groundbreaking", "forward-thinking"
        ],
        
        BrandVoice.TRUSTWORTHY: [
            "trust", "reliable", "honest", "transparent", "integrity", "credible",
            "dependable", "secure", "safe", "genuine", "authentic"
        ],
        
        BrandVoice.INSPIRING: [
            "inspire", "motivate", "empower", "achieve", "success", "dream",
            "vision", "aspiration", "potential", "excellence", "greatness"
        ]
    }


class SentimentAnalyzer:
    """Comprehensive sentiment analysis and emotional tone evaluation tool"""
    
    def __init__(self):
        """Initialize Sentiment Analyzer"""
        
        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            self.stopwords = set()
        
        # Sentiment cache
        self.sentiment_cache = {}
        
        logger.info("SentimentAnalyzer initialized")

    def _analyze_textblob_sentiment(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment using TextBlob"""
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except Exception as e:
            logger.warning(f"TextBlob sentiment analysis failed: {str(e)}")
            return 0.0, 0.5

    def _analyze_vader_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        
        try:
            return self.vader_analyzer.polarity_scores(text)
        except Exception as e:
            logger.warning(f"VADER sentiment analysis failed: {str(e)}")
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    def _classify_polarity(self, polarity: float) -> SentimentPolarity:
        """Classify polarity score into categories"""
        
        if polarity > 0.5:
            return SentimentPolarity.VERY_POSITIVE
        elif polarity > 0.1:
            return SentimentPolarity.POSITIVE
        elif polarity > -0.1:
            return SentimentPolarity.NEUTRAL
        elif polarity > -0.5:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.VERY_NEGATIVE

    def _calculate_sentiment_confidence(self, textblob_polarity: float, vader_compound: float) -> float:
        """Calculate confidence in sentiment analysis"""
        
        # Agreement between methods increases confidence
        agreement = 1.0 - abs(textblob_polarity - vader_compound)
        
        # Stronger sentiment values increase confidence
        strength = max(abs(textblob_polarity), abs(vader_compound))
        
        # Combine factors
        confidence = (agreement * 0.6 + strength * 0.4)
        
        return min(1.0, confidence)

    def _analyze_emotions(self, text: str, custom_keywords: Dict[str, List[str]] = None) -> EmotionAnalysis:
        """Analyze emotions in text using keyword matching and intensity"""
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        # Use custom keywords if provided, otherwise use default
        emotion_keywords = custom_keywords or EmotionKeywords.EMOTION_KEYWORDS
        
        emotion_scores = {}
        emotion_counts = {}
        
        # Count emotion keywords
        for emotion, keywords in emotion_keywords.items():
            if isinstance(emotion, str):
                emotion_key = emotion
            else:
                emotion_key = emotion.value
            
            count = 0
            for keyword in keywords:
                # Count exact matches and partial matches
                exact_matches = text_lower.count(keyword.lower())
                count += exact_matches
            
            emotion_counts[emotion_key] = count
            # Normalize by text length (words per 100 words)
            emotion_scores[emotion_key] = (count / max(1, len(words))) * 100
        
        # Find primary emotion
        if emotion_scores:
            primary_emotion_key = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
            try:
                primary_emotion = EmotionType(primary_emotion_key)
            except ValueError:
                primary_emotion = EmotionType.JOY  # Default fallback
        else:
            primary_emotion = EmotionType.JOY
        
        # Calculate emotional stability (variance in emotion scores)
        if len(emotion_scores) > 1:
            scores = list(emotion_scores.values())
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            # Higher variance = lower stability
            emotional_stability = max(0, 100 - (variance * 10))
        else:
            emotional_stability = 100.0
        
        # Calculate emotional range
        max_score = max(emotion_scores.values()) if emotion_scores else 0
        min_score = min(emotion_scores.values()) if emotion_scores else 0
        emotional_range = max_score - min_score
        
        # Find dominant emotions (top 3)
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        dominant_emotions = []
        for emotion_key, score in sorted_emotions[:3]:
            if score > 0:
                try:
                    dominant_emotions.append(EmotionType(emotion_key))
                except ValueError:
                    continue
        
        if not dominant_emotions:
            dominant_emotions = [primary_emotion]
        
        return EmotionAnalysis(
            primary_emotion=primary_emotion,
            emotion_scores=emotion_scores,
            emotional_stability=emotional_stability,
            emotional_range=min(1.0, emotional_range / 100),
            dominant_emotions=dominant_emotions
        )

    def _analyze_brand_voice(self, text: str, target_voice: Optional[BrandVoice] = None) -> BrandVoiceAnalysis:
        """Analyze brand voice characteristics"""
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        voice_scores = {}
        
        # Calculate scores for each brand voice
        for voice, keywords in EmotionKeywords.BRAND_VOICE_KEYWORDS.items():
            count = 0
            for keyword in keywords:
                count += text_lower.count(keyword.lower())
            
            # Normalize score
            voice_scores[voice.value] = (count / max(1, len(words))) * 100
        
        # Detect primary brand voice
        if voice_scores:
            detected_voice_key = max(voice_scores.keys(), key=lambda k: voice_scores[k])
            try:
                detected_voice = BrandVoice(detected_voice_key)
            except ValueError:
                detected_voice = BrandVoice.PROFESSIONAL  # Default
        else:
            detected_voice = BrandVoice.PROFESSIONAL
        
        # Calculate voice consistency
        if len(voice_scores) > 1:
            scores = list(voice_scores.values())
            max_score = max(scores)
            # Consistency is higher when one voice dominates
            consistency = max_score * 2 if max_score > 0 else 50.0
        else:
            consistency = 50.0
        
        voice_consistency = min(100.0, consistency)
        
        # Brand alignment if target voice specified
        brand_alignment = None
        if target_voice:
            target_score = voice_scores.get(target_voice.value, 0)
            total_score = sum(voice_scores.values())
            if total_score > 0:
                brand_alignment = (target_score / total_score) * 100
            else:
                brand_alignment = 0.0
        
        # Detect tone variations
        tone_variations = []
        for voice, score in voice_scores.items():
            if score > 10:  # Threshold for significant presence
                tone_variations.append(voice.replace("_", " ").title())
        
        # Convert scores to 0-1 scale for voice_attributes
        max_score = max(voice_scores.values()) if voice_scores else 1.0
        voice_attributes = {}
        for voice, score in voice_scores.items():
            voice_attributes[voice] = score / max(max_score, 1.0)
        
        return BrandVoiceAnalysis(
            detected_voice=detected_voice,
            voice_consistency=voice_consistency,
            voice_attributes=voice_attributes,
            brand_alignment=brand_alignment,
            tone_variations=tone_variations
        )

    def _predict_audience_reaction(
        self, 
        sentiment: SentimentScore, 
        emotions: EmotionAnalysis,
        brand_voice: BrandVoiceAnalysis,
        target_audience: Optional[str] = None
    ) -> AudienceReaction:
        """Predict audience reaction based on sentiment and emotions"""
        
        # Base engagement prediction on sentiment strength and emotions
        sentiment_engagement = abs(sentiment.polarity) * 50  # Neutral = 0, Strong sentiment = 50
        emotion_engagement = min(50, sum(emotions.emotion_scores.values()))
        
        base_engagement = sentiment_engagement + emotion_engagement
        engagement_prediction = min(100.0, base_engagement)
        
        # Emotional resonance based on emotion diversity and strength
        emotion_strength = max(emotions.emotion_scores.values()) if emotions.emotion_scores else 0
        emotional_resonance = min(100.0, emotion_strength * 10 + emotions.emotional_stability * 0.3)
        
        # Appeal factors
        appeal_factors = []
        if sentiment.polarity > 0.3:
            appeal_factors.append("Positive emotional tone")
        if emotions.primary_emotion in [EmotionType.JOY, EmotionType.TRUST, EmotionType.ANTICIPATION]:
            appeal_factors.append("Engaging emotional content")
        if brand_voice.voice_consistency > 80:
            appeal_factors.append("Consistent brand voice")
        if sentiment.confidence > 0.8:
            appeal_factors.append("Clear emotional messaging")
        
        # Concern factors
        concern_factors = []
        if sentiment.polarity < -0.3:
            concern_factors.append("Negative emotional tone may reduce engagement")
        if emotions.emotional_stability < 60:
            concern_factors.append("Inconsistent emotional messaging")
        if sentiment.confidence < 0.5:
            concern_factors.append("Unclear emotional direction")
        if emotions.primary_emotion in [EmotionType.FEAR, EmotionType.ANGER, EmotionType.DISGUST]:
            concern_factors.append("Negative emotions may alienate audience")
        
        # Audience sentiment match (simplified heuristic)
        if target_audience:
            # Different audiences prefer different sentiment ranges
            if "professional" in target_audience.lower():
                ideal_range = (-0.1, 0.3)  # Neutral to moderately positive
            elif "consumer" in target_audience.lower() or "general" in target_audience.lower():
                ideal_range = (0.1, 0.7)   # Positive
            else:
                ideal_range = (-0.2, 0.5)  # Flexible range
            
            if ideal_range[0] <= sentiment.polarity <= ideal_range[1]:
                audience_sentiment_match = 90.0
            else:
                distance = min(abs(sentiment.polarity - ideal_range[0]), abs(sentiment.polarity - ideal_range[1]))
                audience_sentiment_match = max(0, 90 - (distance * 100))
        else:
            audience_sentiment_match = 75.0  # Default moderate match
        
        return AudienceReaction(
            engagement_prediction=engagement_prediction,
            emotional_resonance=emotional_resonance,
            appeal_factors=appeal_factors,
            concern_factors=concern_factors,
            audience_sentiment_match=audience_sentiment_match
        )

    def _determine_content_mood(self, sentiment: SentimentScore, emotions: EmotionAnalysis) -> ContentMood:
        """Determine overall content mood"""
        
        # Primary mood based on sentiment and dominant emotion
        if sentiment.polarity > 0.4 and emotions.primary_emotion in [EmotionType.JOY, EmotionType.ANTICIPATION]:
            return ContentMood.EXCITED
        elif sentiment.polarity > 0.2:
            return ContentMood.OPTIMISTIC
        elif sentiment.polarity > 0.0:
            return ContentMood.CONFIDENT
        elif sentiment.polarity < -0.3:
            if emotions.primary_emotion == EmotionType.FEAR:
                return ContentMood.CONCERNED
            else:
                return ContentMood.SERIOUS
        elif emotions.primary_emotion == EmotionType.TRUST:
            return ContentMood.CALM
        else:
            return ContentMood.NEUTRAL

    def _analyze_sentence_sentiments(self, text: str) -> List[Dict[str, Any]]:
        """Analyze sentiment for each sentence"""
        
        sentences = sent_tokenize(text)
        sentence_analysis = []
        
        for i, sentence in enumerate(sentences):
            # TextBlob analysis
            tb_polarity, tb_subjectivity = self._analyze_textblob_sentiment(sentence)
            
            # VADER analysis
            vader_scores = self._analyze_vader_sentiment(sentence)
            
            sentence_analysis.append({
                "sentence_index": i,
                "sentence": sentence.strip(),
                "textblob_polarity": tb_polarity,
                "textblob_subjectivity": tb_subjectivity,
                "vader_compound": vader_scores["compound"],
                "vader_positive": vader_scores["pos"],
                "vader_neutral": vader_scores["neu"],
                "vader_negative": vader_scores["neg"],
                "polarity_label": self._classify_polarity(tb_polarity).value,
                "length": len(sentence.split())
            })
        
        return sentence_analysis

    def _calculate_sentiment_trends(self, sentence_analysis: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate sentiment trends throughout the text"""
        
        if not sentence_analysis:
            return {"trend_direction": 0.0, "volatility": 0.0, "consistency": 100.0}
        
        # Extract sentiment values
        polarities = [s["textblob_polarity"] for s in sentence_analysis]
        
        # Trend direction (correlation with sentence position)
        if len(polarities) > 1:
            # Simple linear trend calculation
            n = len(polarities)
            x_values = list(range(n))
            mean_x = sum(x_values) / n
            mean_y = sum(polarities) / n
            
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, polarities))
            denominator = sum((x - mean_x) ** 2 for x in x_values)
            
            if denominator != 0:
                trend_direction = numerator / denominator
            else:
                trend_direction = 0.0
        else:
            trend_direction = 0.0
        
        # Volatility (standard deviation of sentiment)
        if len(polarities) > 1:
            mean_polarity = sum(polarities) / len(polarities)
            variance = sum((p - mean_polarity) ** 2 for p in polarities) / len(polarities)
            volatility = variance ** 0.5
        else:
            volatility = 0.0
        
        # Consistency (inverse of volatility, scaled to 0-100)
        consistency = max(0, 100 - (volatility * 100))
        
        return {
            "trend_direction": trend_direction,
            "volatility": volatility,
            "consistency": consistency
        }

    def _extract_key_phrases(self, text: str, sentiment: SentimentScore) -> List[Tuple[str, float]]:
        """Extract key phrases with their sentiment impact"""
        
        # Simple approach: find phrases around sentiment-heavy words
        sentences = sent_tokenize(text)
        key_phrases = []
        
        # Look for phrases that might be emotionally charged
        emotion_indicators = []
        for emotion_words in EmotionKeywords.EMOTION_KEYWORDS.values():
            emotion_indicators.extend(emotion_words)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            tb_polarity, _ = self._analyze_textblob_sentiment(sentence)
            
            # Find phrases with strong sentiment or emotion keywords
            if abs(tb_polarity) > 0.3 or any(word in sentence_lower for word in emotion_indicators[:20]):
                # Extract meaningful phrases (remove common words)
                words = word_tokenize(sentence)
                meaningful_words = [w for w in words if w.lower() not in self.stopwords and len(w) > 2]
                
                if len(meaningful_words) >= 2:
                    phrase = " ".join(meaningful_words[:5])  # Limit phrase length
                    key_phrases.append((phrase, tb_polarity))
        
        # Sort by sentiment strength and return top phrases
        key_phrases.sort(key=lambda x: abs(x[1]), reverse=True)
        return key_phrases[:10]

    def _generate_recommendations(
        self, 
        sentiment: SentimentScore,
        emotions: EmotionAnalysis, 
        brand_voice: BrandVoiceAnalysis,
        audience_reaction: AudienceReaction,
        target_brand_voice: Optional[BrandVoice] = None
    ) -> List[str]:
        """Generate sentiment optimization recommendations"""
        
        recommendations = []
        
        # Sentiment polarity recommendations
        if sentiment.polarity < -0.2 and audience_reaction.engagement_prediction < 60:
            recommendations.append("Consider adding more positive language to improve audience engagement")
        
        if sentiment.confidence < 0.6:
            recommendations.append("Clarify emotional messaging - current sentiment direction is unclear")
        
        # Emotion recommendations
        if emotions.emotional_stability < 70:
            recommendations.append("Improve emotional consistency throughout the content")
        
        if emotions.primary_emotion in [EmotionType.FEAR, EmotionType.ANGER]:
            recommendations.append("Consider balancing negative emotions with positive elements")
        
        # Brand voice recommendations
        if target_brand_voice and brand_voice.brand_alignment and brand_voice.brand_alignment < 70:
            recommendations.append(f"Strengthen alignment with {target_brand_voice.value} brand voice")
        
        if brand_voice.voice_consistency < 75:
            recommendations.append("Improve brand voice consistency throughout the content")
        
        # Audience reaction recommendations
        if audience_reaction.engagement_prediction < 60:
            recommendations.append("Increase emotional engagement with more dynamic language")
        
        if audience_reaction.emotional_resonance < 60:
            recommendations.append("Add more emotionally resonant content to connect with readers")
        
        # Specific improvement suggestions
        if sentiment.subjectivity < 0.3:
            recommendations.append("Add more personal perspective and opinion to increase engagement")
        
        return recommendations

    def _generate_warnings(
        self, 
        sentiment: SentimentScore,
        emotions: EmotionAnalysis,
        audience_reaction: AudienceReaction
    ) -> List[str]:
        """Generate sentiment-related warnings"""
        
        warnings = []
        
        # Negative sentiment warnings
        if sentiment.polarity < -0.5:
            warnings.append("Very negative sentiment may alienate readers")
        
        # Emotional warnings
        if emotions.primary_emotion in [EmotionType.ANGER, EmotionType.DISGUST]:
            warnings.append(f"Primary emotion ({emotions.primary_emotion.value}) may create negative associations")
        
        if emotions.emotional_stability < 50:
            warnings.append("Highly inconsistent emotional messaging may confuse readers")
        
        # Audience reaction warnings
        if len(audience_reaction.concern_factors) > len(audience_reaction.appeal_factors):
            warnings.append("More potential concerns than appeal factors - review content carefully")
        
        if audience_reaction.audience_sentiment_match < 50:
            warnings.append("Content sentiment may not match target audience preferences")
        
        return warnings

    async def analyze_sentiment(self, request: SentimentAnalysisRequest) -> SentimentAnalysisResults:
        """
        Perform comprehensive sentiment analysis
        
        Args:
            request: Sentiment analysis request parameters
            
        Returns:
            Complete sentiment analysis results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting sentiment analysis for {len(request.text)} characters")
            
            # Basic sentiment analysis
            tb_polarity, tb_subjectivity = self._analyze_textblob_sentiment(request.text)
            vader_scores = self._analyze_vader_sentiment(request.text)
            
            # Calculate combined sentiment score
            confidence = self._calculate_sentiment_confidence(tb_polarity, vader_scores["compound"])
            
            # Use TextBlob polarity as primary, but consider VADER for intensity
            emotional_intensity = max(abs(tb_polarity), abs(vader_scores["compound"]))
            
            overall_sentiment = SentimentScore(
                polarity=tb_polarity,
                subjectivity=tb_subjectivity,
                polarity_label=self._classify_polarity(tb_polarity),
                confidence=confidence,
                emotional_intensity=emotional_intensity
            )
            
            # Emotion analysis
            emotion_analysis = self._analyze_emotions(request.text, request.custom_emotion_keywords)
            
            # Brand voice analysis
            brand_voice_analysis = self._analyze_brand_voice(request.text, request.target_brand_voice)
            
            # Audience reaction prediction
            audience_reaction = self._predict_audience_reaction(
                overall_sentiment, emotion_analysis, brand_voice_analysis, request.target_audience
            )
            
            # Content mood determination
            content_mood = self._determine_content_mood(overall_sentiment, emotion_analysis)
            
            # Sentence-level analysis
            sentence_sentiments = None
            if request.include_sentence_analysis:
                sentence_sentiments = self._analyze_sentence_sentiments(request.text)
            
            # Sentiment trends
            sentiment_trends = self._calculate_sentiment_trends(sentence_sentiments or [])
            
            # Key phrases
            key_phrases = self._extract_key_phrases(request.text, overall_sentiment)
            
            # Recommendations and warnings
            recommendations = self._generate_recommendations(
                overall_sentiment, emotion_analysis, brand_voice_analysis, 
                audience_reaction, request.target_brand_voice
            )
            
            warnings = self._generate_warnings(overall_sentiment, emotion_analysis, audience_reaction)
            
            # Overall confidence score
            confidence_factors = [
                overall_sentiment.confidence * 100,
                emotion_analysis.emotional_stability,
                brand_voice_analysis.voice_consistency,
                audience_reaction.emotional_resonance * 0.8  # Weight emotional resonance
            ]
            
            confidence_score = sum(confidence_factors) / len(confidence_factors)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = SentimentAnalysisResults(
                overall_sentiment=overall_sentiment,
                content_mood=content_mood,
                emotion_analysis=emotion_analysis,
                brand_voice_analysis=brand_voice_analysis,
                audience_reaction=audience_reaction,
                sentence_sentiments=sentence_sentiments,
                sentiment_trends=sentiment_trends,
                key_phrases=key_phrases,
                recommendations=recommendations,
                warnings=warnings,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
            logger.info(f"Sentiment analysis completed: {overall_sentiment.polarity_label.value}, confidence: {confidence_score:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise ToolExecutionError(f"Sentiment analysis failed: {str(e)}")


# Initialize tool instance
sentiment_analyzer_tool = SentimentAnalyzer()


# MCP Functions for external integration
async def mcp_analyze_sentiment(
    text: str,
    target_brand_voice: Optional[str] = None,
    target_audience: Optional[str] = None,
    content_context: Optional[str] = None,
    include_sentence_analysis: bool = True
) -> Dict[str, Any]:
    """
    MCP function to analyze sentiment
    
    Args:
        text: Text to analyze for sentiment
        target_brand_voice: Target brand voice (professional, friendly, etc.)
        target_audience: Target audience description
        content_context: Content context
        include_sentence_analysis: Include sentence-by-sentence analysis
        
    Returns:
        Comprehensive sentiment analysis results
    """
    try:
        request = SentimentAnalysisRequest(
            text=text,
            target_brand_voice=BrandVoice(target_brand_voice) if target_brand_voice else None,
            target_audience=target_audience,
            content_context=content_context,
            include_sentence_analysis=include_sentence_analysis
        )
        
        result = await sentiment_analyzer_tool.analyze_sentiment(request)
        
        return {
            "success": True,
            "overall_sentiment": {
                "polarity": result.overall_sentiment.polarity,
                "polarity_label": result.overall_sentiment.polarity_label.value,
                "subjectivity": result.overall_sentiment.subjectivity,
                "confidence": result.overall_sentiment.confidence,
                "emotional_intensity": result.overall_sentiment.emotional_intensity
            },
            "content_mood": result.content_mood.value,
            "emotion_analysis": {
                "primary_emotion": result.emotion_analysis.primary_emotion.value,
                "emotion_scores": result.emotion_analysis.emotion_scores,
                "emotional_stability": result.emotion_analysis.emotional_stability,
                "dominant_emotions": [e.value for e in result.emotion_analysis.dominant_emotions]
            },
            "brand_voice_analysis": {
                "detected_voice": result.brand_voice_analysis.detected_voice.value,
                "voice_consistency": result.brand_voice_analysis.voice_consistency,
                "brand_alignment": result.brand_voice_analysis.brand_alignment,
                "tone_variations": result.brand_voice_analysis.tone_variations
            },
            "audience_reaction": {
                "engagement_prediction": result.audience_reaction.engagement_prediction,
                "emotional_resonance": result.audience_reaction.emotional_resonance,
                "appeal_factors": result.audience_reaction.appeal_factors,
                "concern_factors": result.audience_reaction.concern_factors,
                "audience_sentiment_match": result.audience_reaction.audience_sentiment_match
            },
            "sentiment_trends": result.sentiment_trends,
            "key_phrases": result.key_phrases[:5],  # Top 5 key phrases
            "recommendations": result.recommendations,
            "warnings": result.warnings,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        logger.error(f"MCP sentiment analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Example usage and testing
    async def test_sentiment_analyzer():
        """Test the sentiment analyzer functionality"""
        
        test_text = """
        We are absolutely thrilled to announce our revolutionary new product that will transform 
        the way you work! This innovative solution combines cutting-edge technology with 
        user-friendly design to deliver an amazing experience. Our team has worked tirelessly 
        to ensure that every feature meets the highest standards of excellence.
        
        However, we understand that change can sometimes be challenging, and we want to address 
        any concerns you might have. Our dedicated support team is here to help you every step 
        of the way, providing expert guidance and reliable assistance whenever you need it.
        
        Join thousands of satisfied customers who have already discovered the incredible benefits 
        of our solution. Don't miss this opportunity to revolutionize your workflow and achieve 
        unprecedented success in your business endeavors!
        """
        
        request = SentimentAnalysisRequest(
            text=test_text,
            target_brand_voice=BrandVoice.PROFESSIONAL,
            target_audience="Business professionals and entrepreneurs",
            content_context="Product launch marketing",
            include_sentence_analysis=True
        )
        
        try:
            analyzer = SentimentAnalyzer()
            result = await analyzer.analyze_sentiment(request)
            
            print(f"Sentiment Analysis Results:")
            print(f"Overall Sentiment: {result.overall_sentiment.polarity_label.value} ({result.overall_sentiment.polarity:.2f})")
            print(f"Content Mood: {result.content_mood.value}")
            print(f"Confidence Score: {result.confidence_score:.1f}%")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print()
            
            print("Emotion Analysis:")
            print(f"- Primary Emotion: {result.emotion_analysis.primary_emotion.value}")
            print(f"- Emotional Stability: {result.emotion_analysis.emotional_stability:.1f}%")
            print(f"- Dominant Emotions: {[e.value for e in result.emotion_analysis.dominant_emotions]}")
            print()
            
            print("Brand Voice Analysis:")
            print(f"- Detected Voice: {result.brand_voice_analysis.detected_voice.value}")
            print(f"- Voice Consistency: {result.brand_voice_analysis.voice_consistency:.1f}%")
            if result.brand_voice_analysis.brand_alignment:
                print(f"- Brand Alignment: {result.brand_voice_analysis.brand_alignment:.1f}%")
            print()
            
            print("Audience Reaction Prediction:")
            print(f"- Engagement Prediction: {result.audience_reaction.engagement_prediction:.1f}%")
            print(f"- Emotional Resonance: {result.audience_reaction.emotional_resonance:.1f}%")
            print()
            
            print("Key Phrases:")
            for phrase, score in result.key_phrases[:3]:
                print(f"- \"{phrase}\" (sentiment: {score:.2f})")
            print()
            
            print("Recommendations:")
            for rec in result.recommendations:
                print(f"- {rec}")
            
            if result.warnings:
                print("\nWarnings:")
                for warning in result.warnings:
                    print(f"- {warning}")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_sentiment_analyzer())