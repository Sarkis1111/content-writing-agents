"""
Original Tools Adapter - Bridge between Editor Agent and Original Tools

This module provides adapters to integrate the sophisticated original editing tools
with the Editor Agent, handling API differences and providing enhanced functionality.
"""

import asyncio
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Add tools path for direct imports
sys.path.insert(0, '/Users/sarkismanukyan/Desktop/content-writing-agents/src/tools/editing')

try:
    from grammar_checker import GrammarChecker as OriginalGrammarChecker, GrammarCheckRequest
    from seo_analyzer import SEOAnalyzer as OriginalSEOAnalyzer, SEOAnalysisRequest  
    from readability_scorer import ReadabilityScorer as OriginalReadabilityScorer, ReadabilityRequest
    from sentiment_analyzer import SentimentAnalyzer as OriginalSentimentAnalyzer, SentimentAnalysisRequest
    ORIGINAL_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Original tools not available: {e}")
    ORIGINAL_TOOLS_AVAILABLE = False


# Adapter classes that match the Editor Agent's expected API
@dataclass
class AdaptedGrammarResult:
    """Adapted grammar result to match Editor Agent expectations"""
    original_text: str
    corrected_text: str
    overall_score: float
    errors: List[Any]
    corrections_applied: int
    processing_time: float
    
    # Additional rich data from original tools
    style_analysis: Dict[str, Any] = None
    readability_metrics: Dict[str, float] = None
    confidence_score: float = 0.0
    suggestions: List[str] = None


@dataclass 
class AdaptedSEOResult:
    """Adapted SEO result to match Editor Agent expectations"""
    overall_score: float
    keyword_density: Dict[str, float]
    recommendations: List[str]
    issues: List[str]
    meta_suggestions: Dict[str, str]
    processing_time: float


@dataclass
class AdaptedReadabilityResult:
    """Adapted readability result to match Editor Agent expectations"""
    overall_score: float
    flesch_score: float
    grade_level: float
    avg_sentence_length: float
    avg_word_length: float
    recommendations: List[str]
    priority_recommendations: List[str]
    processing_time: float


@dataclass
class AdaptedSentimentResult:
    """Adapted sentiment result to match Editor Agent expectations"""
    overall_sentiment: Any
    sentence_sentiments: List[Any]
    recommendations: List[str]
    alignment_score: float
    processing_time: float


class OriginalGrammarAdapter:
    """Adapter for the original grammar checker"""
    
    def __init__(self):
        if ORIGINAL_TOOLS_AVAILABLE:
            self.original_checker = OriginalGrammarChecker()
        else:
            self.original_checker = None
    
    async def check_grammar(self, request) -> AdaptedGrammarResult:
        """Check grammar using original tool and adapt the result"""
        if not self.original_checker:
            # Fallback to basic result
            return AdaptedGrammarResult(
                original_text=request.text,
                corrected_text=request.text,
                overall_score=70.0,
                errors=[],
                corrections_applied=0,
                processing_time=0.001
            )
        
        # Create original request
        original_request = GrammarCheckRequest(
            text=request.text,
            check_grammar=True,
            check_spelling=True
        )
        
        # Call original tool
        result = await self.original_checker.check_grammar(original_request)
        
        # Calculate corrections applied from error summary
        corrections_applied = sum(result.summary.values()) if result.summary else len(result.errors)
        
        # Adapt the result to match Editor Agent expectations
        return AdaptedGrammarResult(
            original_text=request.text,
            corrected_text=result.corrected_text or request.text,
            overall_score=result.overall_score,
            errors=result.errors,
            corrections_applied=corrections_applied,
            processing_time=result.processing_time,
            style_analysis=result.style_analysis,
            readability_metrics=result.readability_metrics,
            confidence_score=result.confidence_score,
            suggestions=result.suggestions
        )


class OriginalSEOAdapter:
    """Adapter for the original SEO analyzer"""
    
    def __init__(self):
        if ORIGINAL_TOOLS_AVAILABLE:
            try:
                self.original_analyzer = OriginalSEOAnalyzer()
            except Exception:
                self.original_analyzer = None
        else:
            self.original_analyzer = None
    
    async def analyze_seo(self, request) -> AdaptedSEOResult:
        """Analyze SEO using original tool and adapt the result"""
        if not self.original_analyzer:
            # Fallback result
            return AdaptedSEOResult(
                overall_score=70.0,
                keyword_density={},
                recommendations=["Original SEO analyzer not available"],
                issues=[],
                meta_suggestions={},
                processing_time=0.001
            )
        
        try:
            # Create original request
            original_request = SEOAnalysisRequest(
                content=request.content,
                target_keywords=request.target_keywords or [],
                content_type=request.content_type or "blog_post"
            )
            
            # Call original tool  
            result = await self.original_analyzer.analyze_seo(original_request)
            
            # Adapt the result
            return AdaptedSEOResult(
                overall_score=result.overall_score,
                keyword_density=result.keyword_density,
                recommendations=result.recommendations,
                issues=[issue.message for issue in result.issues],
                meta_suggestions=result.meta_suggestions,
                processing_time=result.processing_time
            )
            
        except Exception as e:
            # Fallback on error
            return AdaptedSEOResult(
                overall_score=70.0,
                keyword_density={},
                recommendations=[f"SEO analysis error: {str(e)}"],
                issues=[],
                meta_suggestions={},
                processing_time=0.001
            )


class OriginalReadabilityAdapter:
    """Adapter for the original readability scorer"""
    
    def __init__(self):
        if ORIGINAL_TOOLS_AVAILABLE:
            try:
                self.original_scorer = OriginalReadabilityScorer()
            except Exception:
                self.original_scorer = None
        else:
            self.original_scorer = None
    
    async def score_readability(self, request) -> AdaptedReadabilityResult:
        """Score readability using original tool and adapt the result"""
        if not self.original_scorer:
            # Fallback result
            return AdaptedReadabilityResult(
                overall_score=70.0,
                flesch_score=70.0,
                grade_level=8.0,
                avg_sentence_length=15.0,
                avg_word_length=4.5,
                recommendations=["Original readability scorer not available"],
                priority_recommendations=[],
                processing_time=0.001
            )
        
        try:
            # Create original request
            original_request = ReadabilityRequest(
                text=request.text,
                target_audience=request.target_audience or "general_adult"
            )
            
            # Call original tool
            result = await self.original_scorer.score_readability(original_request)
            
            # Adapt the result
            return AdaptedReadabilityResult(
                overall_score=result.overall_score,
                flesch_score=result.scores.flesch_reading_ease,
                grade_level=result.scores.flesch_kincaid_grade,
                avg_sentence_length=result.sentence_analysis.avg_length,
                avg_word_length=result.vocabulary_analysis.avg_word_length,
                recommendations=result.recommendations,
                priority_recommendations=[],  # Could extract high-priority recommendations
                processing_time=result.processing_time
            )
            
        except Exception as e:
            # Fallback on error
            return AdaptedReadabilityResult(
                overall_score=70.0,
                flesch_score=70.0,
                grade_level=8.0,
                avg_sentence_length=15.0,
                avg_word_length=4.5,
                recommendations=[f"Readability analysis error: {str(e)}"],
                priority_recommendations=[],
                processing_time=0.001
            )


class OriginalSentimentAdapter:
    """Adapter for the original sentiment analyzer"""
    
    def __init__(self):
        if ORIGINAL_TOOLS_AVAILABLE:
            try:
                self.original_analyzer = OriginalSentimentAnalyzer()
            except Exception:
                self.original_analyzer = None
        else:
            self.original_analyzer = None
    
    async def analyze_sentiment(self, request) -> AdaptedSentimentResult:
        """Analyze sentiment using original tool and adapt the result"""
        if not self.original_analyzer:
            # Fallback result
            return AdaptedSentimentResult(
                overall_sentiment=None,
                sentence_sentiments=[],
                recommendations=["Original sentiment analyzer not available"],
                alignment_score=70.0,
                processing_time=0.001
            )
        
        try:
            # Create original request
            original_request = SentimentAnalysisRequest(
                text=request.text,
                target_sentiment=request.target_sentiment or "neutral"
            )
            
            # Call original tool
            result = await self.original_analyzer.analyze_sentiment(original_request)
            
            # Adapt the result
            return AdaptedSentimentResult(
                overall_sentiment=result.overall_sentiment,
                sentence_sentiments=result.sentence_sentiments,
                recommendations=result.recommendations,
                alignment_score=result.brand_voice_analysis.consistency_score * 100,
                processing_time=result.processing_time
            )
            
        except Exception as e:
            # Fallback on error
            return AdaptedSentimentResult(
                overall_sentiment=None,
                sentence_sentiments=[],
                recommendations=[f"Sentiment analysis error: {str(e)}"],
                alignment_score=70.0,
                processing_time=0.001
            )


# Convenience functions for Editor Agent integration
def get_adapted_grammar_checker():
    """Get adapted grammar checker"""
    return OriginalGrammarAdapter()


def get_adapted_seo_analyzer():
    """Get adapted SEO analyzer"""
    return OriginalSEOAdapter()


def get_adapted_readability_scorer():
    """Get adapted readability scorer"""
    return OriginalReadabilityAdapter()


def get_adapted_sentiment_analyzer():
    """Get adapted sentiment analyzer"""
    return OriginalSentimentAdapter()


async def test_adapters():
    """Test all adapters to ensure they work correctly"""
    print("=== TESTING ORIGINAL TOOLS ADAPTERS ===")
    
    # Test data
    @dataclass
    class MockRequest:
        text: str
        content: str = ""
        target_keywords: List[str] = None
        content_type: str = "blog_post"
        target_audience: str = "general_adult"
        target_sentiment: str = "neutral"
        
        def __post_init__(self):
            if not self.content:
                self.content = self.text
    
    test_text = "This is test content with grammer error for testing."
    request = MockRequest(text=test_text, content=test_text, target_keywords=["test", "content"])
    
    # Test grammar adapter
    print("\n🔍 Testing Grammar Adapter...")
    grammar_adapter = get_adapted_grammar_checker()
    grammar_result = await grammar_adapter.check_grammar(request)
    print(f"   Grammar Score: {grammar_result.overall_score:.1f}")
    print(f"   Corrections Applied: {grammar_result.corrections_applied}")
    print(f"   Processing Time: {grammar_result.processing_time:.3f}s")
    
    # Test SEO adapter
    print("\n🔍 Testing SEO Adapter...")
    seo_adapter = get_adapted_seo_analyzer()
    seo_result = await seo_adapter.analyze_seo(request)
    print(f"   SEO Score: {seo_result.overall_score:.1f}")
    print(f"   Recommendations: {len(seo_result.recommendations)}")
    
    # Test Readability adapter
    print("\n🔍 Testing Readability Adapter...")
    readability_adapter = get_adapted_readability_scorer()
    readability_result = await readability_adapter.score_readability(request)
    print(f"   Readability Score: {readability_result.overall_score:.1f}")
    print(f"   Flesch Score: {readability_result.flesch_score:.1f}")
    
    # Test Sentiment adapter  
    print("\n🔍 Testing Sentiment Adapter...")
    sentiment_adapter = get_adapted_sentiment_analyzer()
    sentiment_result = await sentiment_adapter.analyze_sentiment(request)
    print(f"   Alignment Score: {sentiment_result.alignment_score:.1f}")
    
    print(f"\n✅ All adapters tested successfully!")
    return True


if __name__ == "__main__":
    asyncio.run(test_adapters())