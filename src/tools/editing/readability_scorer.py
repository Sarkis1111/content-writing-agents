"""
Readability Scorer Tool - Multiple Readability Metrics and Audience Alignment

This module provides comprehensive readability analysis using multiple established metrics,
target audience alignment, and actionable improvement suggestions.

Key Features:
- Multiple readability formulas (Flesch-Kincaid, Gunning Fog, SMOG, etc.)
- Target audience analysis and alignment
- Sentence and word complexity analysis
- Vocabulary diversity assessment
- Improvement suggestions and recommendations
- Grade level and age appropriateness scoring
"""

import asyncio
import os
import re
import math
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal, Any, Tuple
from enum import Enum
from collections import Counter, defaultdict
import string

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from nltk.corpus import stopwords, words
from nltk.tag import pos_tag
import textstat
from pydantic import BaseModel, Field, validator

# Use fallback imports
try:
    from core.errors import ToolError
    from core.logging.logger import get_logger
    from utils.simple_retry import with_retry
except ImportError:
    # Fallback implementations
    class ToolError(Exception): pass
    import logging
    def get_logger(name): return logging.getLogger(name)
    def with_retry(*args, **kwargs):
        def decorator(func): return func
        return decorator

logger = get_logger(__name__)

# Ensure required NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class ReadabilityMetric(str, Enum):
    """Available readability metrics"""
    FLESCH_READING_EASE = "flesch_reading_ease"
    FLESCH_KINCAID_GRADE = "flesch_kincaid_grade"
    GUNNING_FOG_INDEX = "gunning_fog_index"
    SMOG_INDEX = "smog_index"
    COLEMAN_LIAU_INDEX = "coleman_liau_index"
    AUTOMATED_READABILITY_INDEX = "automated_readability_index"
    DALE_CHALL_READABILITY = "dale_chall_readability"
    LINSEAR_WRITE_FORMULA = "linsear_write_formula"


class TargetAudience(str, Enum):
    """Target audience categories"""
    ELEMENTARY_SCHOOL = "elementary_school"    # Ages 6-11, Grades 1-5
    MIDDLE_SCHOOL = "middle_school"           # Ages 11-14, Grades 6-8
    HIGH_SCHOOL = "high_school"               # Ages 14-18, Grades 9-12
    COLLEGE_STUDENTS = "college_students"     # Ages 18-22, College level
    GENERAL_ADULT = "general_adult"           # Ages 18+, General public
    PROFESSIONAL = "professional"            # Working professionals
    ACADEMIC = "academic"                     # Academic/research audience
    TECHNICAL = "technical"                   # Technical specialists
    SENIOR_CITIZENS = "senior_citizens"       # Ages 65+


class ContentPurpose(str, Enum):
    """Content purpose categories"""
    EDUCATIONAL = "educational"
    INFORMATIONAL = "informational"
    MARKETING = "marketing"
    ENTERTAINMENT = "entertainment"
    INSTRUCTIONAL = "instructional"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    NEWS = "news"
    CREATIVE = "creative"


class ReadabilityScore(BaseModel):
    """Individual readability metric score"""
    
    metric: ReadabilityMetric = Field(..., description="Readability metric name")
    score: float = Field(..., description="Metric score")
    grade_level: Optional[float] = Field(default=None, description="Equivalent grade level")
    
    interpretation: str = Field(..., description="Human-readable interpretation")
    target_range: Tuple[float, float] = Field(..., description="Recommended range for metric")
    
    performance: Literal["excellent", "good", "fair", "poor"] = Field(
        ..., description="Performance rating"
    )


class VocabularyAnalysis(BaseModel):
    """Vocabulary complexity analysis"""
    
    total_words: int = Field(..., description="Total word count")
    unique_words: int = Field(..., description="Unique word count")
    vocabulary_diversity: float = Field(..., description="Type-token ratio (0-1)")
    
    avg_word_length: float = Field(..., description="Average word length in characters")
    avg_syllables_per_word: float = Field(..., description="Average syllables per word")
    
    complex_words_count: int = Field(..., description="Number of complex words (3+ syllables)")
    complex_words_percentage: float = Field(..., description="Percentage of complex words")
    
    difficult_words_count: int = Field(..., description="Number of difficult words")
    difficult_words_percentage: float = Field(..., description="Percentage of difficult words")
    
    most_common_words: List[Tuple[str, int]] = Field(..., description="Most frequently used words")
    longest_words: List[str] = Field(..., description="Longest words in text")


class SentenceAnalysis(BaseModel):
    """Sentence structure analysis"""
    
    total_sentences: int = Field(..., description="Total sentence count")
    avg_sentence_length: float = Field(..., description="Average sentence length in words")
    
    sentence_length_distribution: Dict[str, int] = Field(
        ..., description="Distribution of sentence lengths"
    )
    
    longest_sentence_length: int = Field(..., description="Length of longest sentence")
    shortest_sentence_length: int = Field(..., description="Length of shortest sentence")
    
    sentence_variety_score: float = Field(..., description="Sentence length variety (0-100)")
    
    complex_sentences_count: int = Field(..., description="Number of complex sentences")
    simple_sentences_count: int = Field(..., description="Number of simple sentences")


class AudienceAlignment(BaseModel):
    """Target audience alignment analysis"""
    
    target_audience: TargetAudience = Field(..., description="Target audience category")
    alignment_score: float = Field(..., description="Alignment score (0-100)")
    
    recommended_grade_level: float = Field(..., description="Recommended grade level")
    current_grade_level: float = Field(..., description="Current content grade level")
    
    gap_analysis: str = Field(..., description="Analysis of the gap between current and target")
    
    audience_recommendations: List[str] = Field(
        ..., description="Specific recommendations for target audience"
    )


class ReadabilityRequest(BaseModel):
    """Readability analysis request parameters"""
    
    text: str = Field(..., description="Text to analyze")
    
    target_audience: Optional[TargetAudience] = Field(
        default=None, 
        description="Target audience for the content"
    )
    
    content_purpose: ContentPurpose = Field(
        default=ContentPurpose.INFORMATIONAL,
        description="Primary purpose of the content"
    )
    
    metrics_to_include: Optional[List[ReadabilityMetric]] = Field(
        default=None,
        description="Specific metrics to include (all if not specified)"
    )
    
    custom_target_grade: Optional[float] = Field(
        default=None,
        description="Custom target grade level (overrides audience default)",
        ge=1.0,
        le=20.0
    )
    
    include_vocabulary_analysis: bool = Field(
        default=True,
        description="Include detailed vocabulary analysis"
    )
    
    include_sentence_analysis: bool = Field(
        default=True,
        description="Include sentence structure analysis"
    )
    
    language: str = Field(default="en", description="Content language")

    @validator('text')
    def validate_text_length(cls, v):
        if len(v.strip()) < 50:
            raise ValueError("Text too short for meaningful readability analysis")
        if len(v) > 50000:
            raise ValueError("Text too long (max 50,000 characters)")
        return v


class ReadabilityResults(BaseModel):
    """Complete readability analysis results"""
    
    overall_score: float = Field(..., description="Overall readability score (0-100)")
    primary_grade_level: float = Field(..., description="Primary grade level recommendation")
    
    metric_scores: Dict[str, ReadabilityScore] = Field(..., description="Individual metric scores")
    
    vocabulary_analysis: Optional[VocabularyAnalysis] = Field(
        default=None, 
        description="Vocabulary complexity analysis"
    )
    
    sentence_analysis: Optional[SentenceAnalysis] = Field(
        default=None,
        description="Sentence structure analysis"
    )
    
    audience_alignment: Optional[AudienceAlignment] = Field(
        default=None,
        description="Target audience alignment analysis"
    )
    
    improvement_suggestions: List[str] = Field(..., description="Actionable improvement suggestions")
    priority_recommendations: List[str] = Field(..., description="High-priority recommendations")
    
    summary: str = Field(..., description="Overall readability summary")
    
    processing_time: float = Field(..., description="Analysis processing time")


class AudienceProfiles:
    """Target audience profiles and requirements"""
    
    AUDIENCE_PROFILES = {
        TargetAudience.ELEMENTARY_SCHOOL: {
            "age_range": "6-11",
            "grade_level": (1, 5),
            "target_grade": 3.0,
            "max_grade": 5.0,
            "vocabulary_complexity": "simple",
            "sentence_length": "short",
            "characteristics": [
                "Simple vocabulary",
                "Short sentences (8-12 words)",
                "Concrete concepts",
                "Active voice preferred"
            ]
        },
        
        TargetAudience.MIDDLE_SCHOOL: {
            "age_range": "11-14", 
            "grade_level": (6, 8),
            "target_grade": 7.0,
            "max_grade": 8.0,
            "vocabulary_complexity": "moderate",
            "sentence_length": "medium",
            "characteristics": [
                "Expanding vocabulary",
                "Medium sentences (12-18 words)",
                "Abstract thinking emerging",
                "Mix of simple and complex structures"
            ]
        },
        
        TargetAudience.HIGH_SCHOOL: {
            "age_range": "14-18",
            "grade_level": (9, 12),
            "target_grade": 10.0,
            "max_grade": 12.0,
            "vocabulary_complexity": "advanced",
            "sentence_length": "varied",
            "characteristics": [
                "Advanced vocabulary",
                "Varied sentence length",
                "Abstract concepts",
                "Complex sentence structures acceptable"
            ]
        },
        
        TargetAudience.COLLEGE_STUDENTS: {
            "age_range": "18-22",
            "grade_level": (13, 16),
            "target_grade": 14.0,
            "max_grade": 16.0,
            "vocabulary_complexity": "sophisticated",
            "sentence_length": "complex",
            "characteristics": [
                "Sophisticated vocabulary",
                "Complex sentence structures",
                "Academic language",
                "Critical thinking expected"
            ]
        },
        
        TargetAudience.GENERAL_ADULT: {
            "age_range": "18+",
            "grade_level": (8, 12),
            "target_grade": 9.0,
            "max_grade": 12.0,
            "vocabulary_complexity": "accessible",
            "sentence_length": "moderate",
            "characteristics": [
                "Accessible vocabulary",
                "Clear, direct communication",
                "Practical focus",
                "Scannable content preferred"
            ]
        },
        
        TargetAudience.PROFESSIONAL: {
            "age_range": "25-65",
            "grade_level": (12, 16),
            "target_grade": 13.0,
            "max_grade": 16.0,
            "vocabulary_complexity": "professional",
            "sentence_length": "efficient",
            "characteristics": [
                "Professional terminology",
                "Efficient communication",
                "Data-driven content",
                "Industry-specific language acceptable"
            ]
        },
        
        TargetAudience.ACADEMIC: {
            "age_range": "22+",
            "grade_level": (16, 20),
            "target_grade": 18.0,
            "max_grade": 20.0,
            "vocabulary_complexity": "scholarly",
            "sentence_length": "complex",
            "characteristics": [
                "Scholarly vocabulary",
                "Complex argumentation",
                "Precise terminology",
                "Dense information acceptable"
            ]
        },
        
        TargetAudience.TECHNICAL: {
            "age_range": "22+",
            "grade_level": (14, 18),
            "target_grade": 16.0,
            "max_grade": 18.0,
            "vocabulary_complexity": "technical",
            "sentence_length": "precise",
            "characteristics": [
                "Technical terminology",
                "Precise instructions",
                "Logical structure",
                "Detail-oriented"
            ]
        },
        
        TargetAudience.SENIOR_CITIZENS: {
            "age_range": "65+",
            "grade_level": (6, 10),
            "target_grade": 8.0,
            "max_grade": 10.0,
            "vocabulary_complexity": "clear",
            "sentence_length": "simple",
            "characteristics": [
                "Clear, simple language",
                "Larger font considerations",
                "Step-by-step instructions",
                "Respectful tone"
            ]
        }
    }


class ReadabilityScorer:
    """Comprehensive readability analysis and scoring tool"""
    
    def __init__(self):
        """Initialize Readability Scorer"""
        
        # Load English word list for difficulty assessment
        try:
            self.english_words = set(words.words())
        except LookupError:
            self.english_words = set()
        
        # Load stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            self.stopwords = set()
        
        # Dale-Chall easy words list (simplified version)
        self.easy_words = self._load_easy_words()
        
        logger.info("ReadabilityScorer initialized")

    def _load_easy_words(self) -> set:
        """Load Dale-Chall easy words list (simplified)"""
        
        # This is a simplified version of common easy words
        # In production, you'd load the full Dale-Chall word list
        easy_words = {
            'a', 'about', 'after', 'all', 'an', 'and', 'any', 'are', 'as', 'at',
            'be', 'been', 'before', 'being', 'by', 'can', 'come', 'could', 'do',
            'each', 'for', 'from', 'get', 'go', 'had', 'has', 'have', 'he', 'her',
            'him', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its',
            'like', 'make', 'may', 'me', 'my', 'no', 'not', 'now', 'of', 'on',
            'one', 'only', 'or', 'other', 'our', 'out', 'over', 'said', 'same',
            'she', 'some', 'take', 'than', 'that', 'the', 'their', 'them', 'there',
            'they', 'this', 'time', 'to', 'two', 'up', 'use', 'very', 'was',
            'we', 'what', 'when', 'which', 'who', 'will', 'with', 'would', 'you'
        }
        
        return easy_words

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        
        word = word.lower().strip(string.punctuation)
        if not word:
            return 0
        
        # Remove common suffixes that don't add syllables
        if word.endswith(('ed', 'es')):
            word = word[:-2]
        elif word.endswith('e'):
            word = word[:-1]
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        return max(1, syllable_count)

    def _calculate_flesch_reading_ease(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        try:
            return textstat.flesch_reading_ease(text)
        except:
            # Fallback calculation
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            syllables = sum(self._count_syllables(word) for word in words)
            
            if len(sentences) == 0 or len(words) == 0:
                return 0
            
            asl = len(words) / len(sentences)  # Average sentence length
            asw = syllables / len(words)      # Average syllables per word
            
            return 206.835 - (1.015 * asl) - (84.6 * asw)

    def _calculate_flesch_kincaid_grade(self, text: str) -> float:
        """Calculate Flesch-Kincaid Grade Level"""
        try:
            return textstat.flesch_kincaid_grade(text)
        except:
            # Fallback calculation
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            syllables = sum(self._count_syllables(word) for word in words)
            
            if len(sentences) == 0 or len(words) == 0:
                return 0
            
            asl = len(words) / len(sentences)
            asw = syllables / len(words)
            
            return (0.39 * asl) + (11.8 * asw) - 15.59

    def _calculate_gunning_fog_index(self, text: str) -> float:
        """Calculate Gunning Fog Index"""
        try:
            return textstat.gunning_fog(text)
        except:
            # Fallback calculation
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            if len(sentences) == 0 or len(words) == 0:
                return 0
            
            # Count complex words (3+ syllables)
            complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
            
            asl = len(words) / len(sentences)
            percentage_complex = (complex_words / len(words)) * 100
            
            return 0.4 * (asl + percentage_complex)

    def _calculate_smog_index(self, text: str) -> float:
        """Calculate SMOG Index"""
        try:
            return textstat.smog_index(text)
        except:
            # Fallback calculation
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            if len(sentences) < 30:
                # SMOG requires at least 30 sentences, use approximation
                complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
                return 3.0 + math.sqrt(complex_words * (30 / max(1, len(sentences))))
            
            # Standard SMOG calculation for 30+ sentences
            complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
            return 3.0 + math.sqrt(complex_words)

    def _calculate_coleman_liau_index(self, text: str) -> float:
        """Calculate Coleman-Liau Index"""
        try:
            return textstat.coleman_liau_index(text)
        except:
            # Fallback calculation
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            characters = sum(len(word) for word in words if word.isalpha())
            
            if len(words) == 0:
                return 0
            
            l = (characters / len(words)) * 100  # Average letters per 100 words
            s = (len(sentences) / len(words)) * 100  # Average sentences per 100 words
            
            return (0.0588 * l) - (0.296 * s) - 15.8

    def _calculate_automated_readability_index(self, text: str) -> float:
        """Calculate Automated Readability Index"""
        try:
            return textstat.automated_readability_index(text)
        except:
            # Fallback calculation
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            characters = sum(len(word) for word in words if word.isalpha())
            
            if len(sentences) == 0 or len(words) == 0:
                return 0
            
            return (4.71 * (characters / len(words))) + (0.5 * (len(words) / len(sentences))) - 21.43

    def _interpret_flesch_score(self, score: float) -> str:
        """Interpret Flesch Reading Ease score"""
        
        if score >= 90:
            return "Very Easy (5th grade)"
        elif score >= 80:
            return "Easy (6th grade)"
        elif score >= 70:
            return "Fairly Easy (7th grade)"
        elif score >= 60:
            return "Standard (8th-9th grade)"
        elif score >= 50:
            return "Fairly Difficult (10th-12th grade)"
        elif score >= 30:
            return "Difficult (College level)"
        else:
            return "Very Difficult (Graduate level)"

    def _grade_to_performance(self, grade_level: float, target_grade: float = 9.0) -> str:
        """Convert grade level to performance rating"""
        
        diff = abs(grade_level - target_grade)
        
        if diff <= 1.0:
            return "excellent"
        elif diff <= 2.0:
            return "good"
        elif diff <= 3.0:
            return "fair"
        else:
            return "poor"

    def _analyze_vocabulary(self, text: str) -> VocabularyAnalysis:
        """Analyze vocabulary complexity"""
        
        words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
        
        if not words:
            return VocabularyAnalysis(
                total_words=0, unique_words=0, vocabulary_diversity=0,
                avg_word_length=0, avg_syllables_per_word=0,
                complex_words_count=0, complex_words_percentage=0,
                difficult_words_count=0, difficult_words_percentage=0,
                most_common_words=[], longest_words=[]
            )
        
        # Basic metrics
        total_words = len(words)
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / total_words
        
        # Word complexity
        avg_word_length = sum(len(word) for word in words) / total_words
        syllable_counts = [self._count_syllables(word) for word in words]
        avg_syllables = sum(syllable_counts) / len(syllable_counts)
        
        # Complex words (3+ syllables)
        complex_words = [word for word, syllables in zip(words, syllable_counts) if syllables >= 3]
        complex_count = len(complex_words)
        complex_percentage = (complex_count / total_words) * 100
        
        # Difficult words (not in easy words list)
        difficult_words = [word for word in words if word not in self.easy_words and len(word) > 2]
        difficult_count = len(difficult_words)
        difficult_percentage = (difficult_count / total_words) * 100
        
        # Most common words (excluding stopwords)
        content_words = [word for word in words if word not in self.stopwords]
        word_freq = Counter(content_words)
        most_common = word_freq.most_common(10)
        
        # Longest words
        unique_words_list = list(set(words))
        longest_words = sorted(unique_words_list, key=len, reverse=True)[:10]
        
        return VocabularyAnalysis(
            total_words=total_words,
            unique_words=unique_words,
            vocabulary_diversity=vocabulary_diversity,
            avg_word_length=avg_word_length,
            avg_syllables_per_word=avg_syllables,
            complex_words_count=complex_count,
            complex_words_percentage=complex_percentage,
            difficult_words_count=difficult_count,
            difficult_words_percentage=difficult_percentage,
            most_common_words=most_common,
            longest_words=longest_words
        )

    def _analyze_sentences(self, text: str) -> SentenceAnalysis:
        """Analyze sentence structure"""
        
        sentences = sent_tokenize(text)
        
        if not sentences:
            return SentenceAnalysis(
                total_sentences=0, avg_sentence_length=0,
                sentence_length_distribution={}, longest_sentence_length=0,
                shortest_sentence_length=0, sentence_variety_score=0,
                complex_sentences_count=0, simple_sentences_count=0
            )
        
        # Sentence lengths
        sentence_lengths = []
        for sentence in sentences:
            words_in_sentence = len(word_tokenize(sentence))
            sentence_lengths.append(words_in_sentence)
        
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        longest = max(sentence_lengths)
        shortest = min(sentence_lengths)
        
        # Length distribution
        distribution = {
            "short (1-10 words)": sum(1 for length in sentence_lengths if length <= 10),
            "medium (11-20 words)": sum(1 for length in sentence_lengths if 11 <= length <= 20),
            "long (21-30 words)": sum(1 for length in sentence_lengths if 21 <= length <= 30),
            "very long (30+ words)": sum(1 for length in sentence_lengths if length > 30)
        }
        
        # Sentence variety (based on standard deviation of lengths)
        if len(sentence_lengths) > 1:
            mean_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - mean_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            std_dev = math.sqrt(variance)
            variety_score = min(100, std_dev * 10)  # Scale to 0-100
        else:
            variety_score = 0
        
        # Complex vs simple sentences (simplified heuristic)
        complex_count = sum(1 for sentence in sentences if len(word_tokenize(sentence)) > 20)
        simple_count = len(sentences) - complex_count
        
        return SentenceAnalysis(
            total_sentences=len(sentences),
            avg_sentence_length=avg_length,
            sentence_length_distribution=distribution,
            longest_sentence_length=longest,
            shortest_sentence_length=shortest,
            sentence_variety_score=variety_score,
            complex_sentences_count=complex_count,
            simple_sentences_count=simple_count
        )

    def _analyze_audience_alignment(
        self, 
        primary_grade_level: float, 
        target_audience: TargetAudience,
        vocabulary_analysis: VocabularyAnalysis,
        sentence_analysis: SentenceAnalysis
    ) -> AudienceAlignment:
        """Analyze alignment with target audience"""
        
        profile = AudienceProfiles.AUDIENCE_PROFILES[target_audience]
        target_grade = profile["target_grade"]
        max_grade = profile["max_grade"]
        
        # Calculate alignment score
        grade_diff = abs(primary_grade_level - target_grade)
        
        if primary_grade_level <= max_grade:
            grade_score = max(0, 100 - (grade_diff * 20))
        else:
            grade_score = max(0, 100 - (grade_diff * 30))  # Penalty for exceeding max
        
        # Vocabulary alignment
        vocab_score = 100
        if vocabulary_analysis.complex_words_percentage > 20 and target_audience in [
            TargetAudience.ELEMENTARY_SCHOOL, TargetAudience.MIDDLE_SCHOOL
        ]:
            vocab_score -= 20
        
        # Sentence length alignment
        sentence_score = 100
        if sentence_analysis.avg_sentence_length > 20 and target_audience in [
            TargetAudience.ELEMENTARY_SCHOOL, TargetAudience.MIDDLE_SCHOOL, TargetAudience.SENIOR_CITIZENS
        ]:
            sentence_score -= 15
        
        # Overall alignment score
        alignment_score = (grade_score + vocab_score + sentence_score) / 3
        
        # Gap analysis
        if primary_grade_level > max_grade:
            gap_analysis = f"Content is too advanced for {target_audience.value}. Current grade level {primary_grade_level:.1f} exceeds maximum {max_grade}."
        elif primary_grade_level < target_grade - 2:
            gap_analysis = f"Content may be too simple for {target_audience.value}. Consider adding complexity."
        else:
            gap_analysis = f"Content grade level {primary_grade_level:.1f} is appropriate for {target_audience.value}."
        
        # Generate audience-specific recommendations
        recommendations = self._generate_audience_recommendations(
            target_audience, primary_grade_level, target_grade, vocabulary_analysis, sentence_analysis
        )
        
        return AudienceAlignment(
            target_audience=target_audience,
            alignment_score=alignment_score,
            recommended_grade_level=target_grade,
            current_grade_level=primary_grade_level,
            gap_analysis=gap_analysis,
            audience_recommendations=recommendations
        )

    def _generate_audience_recommendations(
        self,
        target_audience: TargetAudience,
        current_grade: float,
        target_grade: float,
        vocab: VocabularyAnalysis,
        sentences: SentenceAnalysis
    ) -> List[str]:
        """Generate audience-specific recommendations"""
        
        recommendations = []
        
        # Grade level recommendations
        if current_grade > target_grade + 1:
            recommendations.append(f"Simplify language to better match {target_audience.value} reading level")
            
            if vocab.complex_words_percentage > 15:
                recommendations.append("Reduce complex words (3+ syllables) - consider simpler alternatives")
            
            if sentences.avg_sentence_length > 18:
                recommendations.append("Break up long sentences into shorter, clearer statements")
        
        elif current_grade < target_grade - 1:
            recommendations.append("Content may be too simple - consider adding appropriate complexity")
        
        # Audience-specific recommendations
        if target_audience == TargetAudience.ELEMENTARY_SCHOOL:
            recommendations.extend([
                "Use simple, everyday vocabulary",
                "Keep sentences under 12 words when possible",
                "Use active voice and concrete examples"
            ])
        
        elif target_audience == TargetAudience.PROFESSIONAL:
            if vocab.vocabulary_diversity < 0.4:
                recommendations.append("Increase vocabulary diversity with industry-specific terms")
        
        elif target_audience == TargetAudience.SENIOR_CITIZENS:
            recommendations.extend([
                "Use clear, direct language",
                "Avoid jargon and abbreviations",
                "Consider larger font and spacing for readability"
            ])
        
        return recommendations

    def _calculate_metric_scores(self, text: str, metrics: List[ReadabilityMetric] = None) -> Dict[str, ReadabilityScore]:
        """Calculate all requested readability metrics"""
        
        if metrics is None:
            metrics = list(ReadabilityMetric)
        
        scores = {}
        
        for metric in metrics:
            try:
                if metric == ReadabilityMetric.FLESCH_READING_EASE:
                    score = self._calculate_flesch_reading_ease(text)
                    interpretation = self._interpret_flesch_score(score)
                    target_range = (60.0, 70.0)  # Standard range
                    performance = "excellent" if 60 <= score <= 70 else "good" if 50 <= score <= 80 else "fair"
                    grade_level = None  # Flesch is not grade-based
                
                elif metric == ReadabilityMetric.FLESCH_KINCAID_GRADE:
                    score = self._calculate_flesch_kincaid_grade(text)
                    grade_level = score
                    interpretation = f"Grade {score:.1f} level"
                    target_range = (8.0, 10.0)
                    performance = self._grade_to_performance(score, 9.0)
                
                elif metric == ReadabilityMetric.GUNNING_FOG_INDEX:
                    score = self._calculate_gunning_fog_index(text)
                    grade_level = score
                    interpretation = f"Grade {score:.1f} level"
                    target_range = (8.0, 12.0)
                    performance = self._grade_to_performance(score, 10.0)
                
                elif metric == ReadabilityMetric.SMOG_INDEX:
                    score = self._calculate_smog_index(text)
                    grade_level = score
                    interpretation = f"Grade {score:.1f} level"
                    target_range = (8.0, 12.0)
                    performance = self._grade_to_performance(score, 10.0)
                
                elif metric == ReadabilityMetric.COLEMAN_LIAU_INDEX:
                    score = self._calculate_coleman_liau_index(text)
                    grade_level = score
                    interpretation = f"Grade {score:.1f} level"
                    target_range = (8.0, 12.0)
                    performance = self._grade_to_performance(score, 10.0)
                
                elif metric == ReadabilityMetric.AUTOMATED_READABILITY_INDEX:
                    score = self._calculate_automated_readability_index(text)
                    grade_level = score
                    interpretation = f"Grade {score:.1f} level"
                    target_range = (8.0, 12.0)
                    performance = self._grade_to_performance(score, 10.0)
                
                else:
                    # For metrics not implemented, use textstat library
                    score = getattr(textstat, metric.value)(text)
                    grade_level = score if "grade" in metric.value else None
                    interpretation = f"Score: {score:.1f}"
                    target_range = (0.0, 100.0)
                    performance = "good"
                
                scores[metric.value] = ReadabilityScore(
                    metric=metric,
                    score=score,
                    grade_level=grade_level,
                    interpretation=interpretation,
                    target_range=target_range,
                    performance=performance
                )
                
            except Exception as e:
                logger.warning(f"Failed to calculate {metric.value}: {str(e)}")
                # Provide default score for failed calculations
                scores[metric.value] = ReadabilityScore(
                    metric=metric,
                    score=0.0,
                    interpretation="Calculation failed",
                    target_range=(0.0, 0.0),
                    performance="poor"
                )
        
        return scores

    def _generate_improvement_suggestions(
        self, 
        metric_scores: Dict[str, ReadabilityScore],
        vocabulary_analysis: VocabularyAnalysis,
        sentence_analysis: SentenceAnalysis,
        target_audience: Optional[TargetAudience] = None
    ) -> Tuple[List[str], List[str]]:
        """Generate improvement suggestions and priority recommendations"""
        
        suggestions = []
        priorities = []
        
        # Grade level analysis
        grade_levels = [score.grade_level for score in metric_scores.values() if score.grade_level is not None]
        if grade_levels:
            avg_grade = sum(grade_levels) / len(grade_levels)
            
            if avg_grade > 12:
                priorities.append("Simplify language complexity - current grade level is too high for most audiences")
                suggestions.extend([
                    "Replace complex words with simpler alternatives",
                    "Break long sentences into shorter ones",
                    "Use more common vocabulary"
                ])
        
        # Vocabulary suggestions
        if vocabulary_analysis.complex_words_percentage > 25:
            suggestions.append("Reduce complex words (3+ syllables) to improve readability")
        
        if vocabulary_analysis.vocabulary_diversity < 0.3:
            suggestions.append("Increase vocabulary variety to maintain reader interest")
        
        if vocabulary_analysis.avg_word_length > 5.5:
            suggestions.append("Use shorter words when possible")
        
        # Sentence structure suggestions
        if sentence_analysis.avg_sentence_length > 20:
            priorities.append("Shorten average sentence length for better readability")
            suggestions.extend([
                "Break up sentences longer than 20 words",
                "Use active voice to reduce sentence length",
                "Eliminate unnecessary words and phrases"
            ])
        
        if sentence_analysis.sentence_variety_score < 30:
            suggestions.append("Vary sentence lengths to improve reading flow")
        
        # Performance-based suggestions
        poor_performers = [name for name, score in metric_scores.items() if score.performance == "poor"]
        if len(poor_performers) > 2:
            priorities.append("Multiple readability metrics show poor performance - comprehensive revision needed")
        
        return suggestions, priorities

    async def score_readability(self, request: ReadabilityRequest) -> ReadabilityResults:
        """
        Perform comprehensive readability analysis
        
        Args:
            request: Readability analysis request
            
        Returns:
            Complete readability analysis results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting readability analysis for {len(request.text)} characters")
            
            # Calculate readability metrics
            metrics_to_use = request.metrics_to_include or list(ReadabilityMetric)
            metric_scores = self._calculate_metric_scores(request.text, metrics_to_use)
            
            # Calculate primary grade level (average of grade-based metrics)
            grade_scores = [score.grade_level for score in metric_scores.values() if score.grade_level is not None]
            primary_grade_level = sum(grade_scores) / len(grade_scores) if grade_scores else 9.0
            
            # Vocabulary analysis
            vocabulary_analysis = None
            if request.include_vocabulary_analysis:
                vocabulary_analysis = self._analyze_vocabulary(request.text)
            
            # Sentence analysis
            sentence_analysis = None
            if request.include_sentence_analysis:
                sentence_analysis = self._analyze_sentences(request.text)
            
            # Audience alignment analysis
            audience_alignment = None
            if request.target_audience:
                audience_alignment = self._analyze_audience_alignment(
                    primary_grade_level, request.target_audience, vocabulary_analysis, sentence_analysis
                )
            
            # Calculate overall readability score
            performance_scores = []
            for score in metric_scores.values():
                if score.performance == "excellent":
                    performance_scores.append(90)
                elif score.performance == "good":
                    performance_scores.append(75)
                elif score.performance == "fair":
                    performance_scores.append(60)
                else:
                    performance_scores.append(40)
            
            overall_score = sum(performance_scores) / len(performance_scores) if performance_scores else 50
            
            # Adjust based on audience alignment
            if audience_alignment and audience_alignment.alignment_score < 70:
                overall_score = max(0, overall_score - (100 - audience_alignment.alignment_score) * 0.2)
            
            # Generate suggestions
            suggestions, priorities = self._generate_improvement_suggestions(
                metric_scores, vocabulary_analysis, sentence_analysis, request.target_audience
            )
            
            # Generate summary
            if overall_score >= 80:
                summary = f"Excellent readability (Grade {primary_grade_level:.1f}). Content is well-suited for the target audience."
            elif overall_score >= 60:
                summary = f"Good readability (Grade {primary_grade_level:.1f}). Minor improvements could enhance accessibility."
            elif overall_score >= 40:
                summary = f"Fair readability (Grade {primary_grade_level:.1f}). Several areas need improvement."
            else:
                summary = f"Poor readability (Grade {primary_grade_level:.1f}). Significant revision recommended."
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ReadabilityResults(
                overall_score=overall_score,
                primary_grade_level=primary_grade_level,
                metric_scores=metric_scores,
                vocabulary_analysis=vocabulary_analysis,
                sentence_analysis=sentence_analysis,
                audience_alignment=audience_alignment,
                improvement_suggestions=suggestions,
                priority_recommendations=priorities,
                summary=summary,
                processing_time=processing_time
            )
            
            logger.info(f"Readability analysis completed: score {overall_score:.1f}/100, grade {primary_grade_level:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {str(e)}")
            raise ToolError(f"Readability analysis failed: {str(e)}")


# Initialize tool instance
readability_scorer_tool = ReadabilityScorer()


# MCP Functions for external integration
async def mcp_score_readability(
    text: str,
    target_audience: Optional[str] = None,
    content_purpose: str = "informational",
    custom_target_grade: Optional[float] = None,
    include_vocabulary_analysis: bool = True,
    include_sentence_analysis: bool = True
) -> Dict[str, Any]:
    """
    MCP function to score readability
    
    Args:
        text: Text to analyze
        target_audience: Target audience (elementary_school, general_adult, etc.)
        content_purpose: Content purpose (educational, marketing, etc.)
        custom_target_grade: Custom target grade level
        include_vocabulary_analysis: Include vocabulary analysis
        include_sentence_analysis: Include sentence analysis
        
    Returns:
        Comprehensive readability analysis results
    """
    try:
        request = ReadabilityRequest(
            text=text,
            target_audience=TargetAudience(target_audience) if target_audience else None,
            content_purpose=ContentPurpose(content_purpose),
            custom_target_grade=custom_target_grade,
            include_vocabulary_analysis=include_vocabulary_analysis,
            include_sentence_analysis=include_sentence_analysis
        )
        
        result = await readability_scorer_tool.score_readability(request)
        
        return {
            "success": True,
            "overall_score": result.overall_score,
            "primary_grade_level": result.primary_grade_level,
            "metric_scores": {
                name: {
                    "score": score.score,
                    "grade_level": score.grade_level,
                    "interpretation": score.interpretation,
                    "performance": score.performance
                }
                for name, score in result.metric_scores.items()
            },
            "vocabulary_analysis": {
                "total_words": result.vocabulary_analysis.total_words,
                "vocabulary_diversity": result.vocabulary_analysis.vocabulary_diversity,
                "complex_words_percentage": result.vocabulary_analysis.complex_words_percentage,
                "avg_word_length": result.vocabulary_analysis.avg_word_length,
                "most_common_words": result.vocabulary_analysis.most_common_words[:5]
            } if result.vocabulary_analysis else None,
            "sentence_analysis": {
                "total_sentences": result.sentence_analysis.total_sentences,
                "avg_sentence_length": result.sentence_analysis.avg_sentence_length,
                "sentence_variety_score": result.sentence_analysis.sentence_variety_score,
                "sentence_length_distribution": result.sentence_analysis.sentence_length_distribution
            } if result.sentence_analysis else None,
            "audience_alignment": {
                "alignment_score": result.audience_alignment.alignment_score,
                "gap_analysis": result.audience_alignment.gap_analysis,
                "recommendations": result.audience_alignment.audience_recommendations
            } if result.audience_alignment else None,
            "improvement_suggestions": result.improvement_suggestions,
            "priority_recommendations": result.priority_recommendations,
            "summary": result.summary,
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        logger.error(f"MCP readability scoring failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Example usage and testing
    async def test_readability_scorer():
        """Test the readability scorer functionality"""
        
        test_text = """
        Artificial intelligence represents a transformative technological paradigm that fundamentally 
        revolutionizes computational methodologies. Contemporary implementations demonstrate sophisticated 
        algorithmic architectures capable of emulating cognitive processes traditionally associated 
        with human intellectual capabilities. These systems utilize complex mathematical frameworks 
        to analyze multidimensional datasets and generate predictive models with unprecedented accuracy.
        
        The implications of these developments extend across numerous professional domains, including 
        healthcare diagnostics, financial analysis, and autonomous transportation systems. Organizations 
        are increasingly recognizing the strategic importance of incorporating AI-driven solutions 
        into their operational frameworks to maintain competitive advantages in evolving market conditions.
        """
        
        request = ReadabilityRequest(
            text=test_text,
            target_audience=TargetAudience.GENERAL_ADULT,
            content_purpose=ContentPurpose.INFORMATIONAL,
            include_vocabulary_analysis=True,
            include_sentence_analysis=True
        )
        
        try:
            scorer = ReadabilityScorer()
            result = await scorer.score_readability(request)
            
            print(f"Readability Analysis Results:")
            print(f"Overall Score: {result.overall_score:.1f}/100")
            print(f"Primary Grade Level: {result.primary_grade_level:.1f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print()
            
            print("Metric Scores:")
            for name, score in result.metric_scores.items():
                print(f"- {name}: {score.score:.1f} ({score.performance}) - {score.interpretation}")
            print()
            
            if result.vocabulary_analysis:
                print("Vocabulary Analysis:")
                print(f"- Total words: {result.vocabulary_analysis.total_words}")
                print(f"- Vocabulary diversity: {result.vocabulary_analysis.vocabulary_diversity:.2f}")
                print(f"- Complex words: {result.vocabulary_analysis.complex_words_percentage:.1f}%")
            print()
            
            if result.audience_alignment:
                print("Audience Alignment:")
                print(f"- Alignment score: {result.audience_alignment.alignment_score:.1f}/100")
                print(f"- Gap analysis: {result.audience_alignment.gap_analysis}")
            print()
            
            print("Priority Recommendations:")
            for rec in result.priority_recommendations:
                print(f"- {rec}")
            
            print(f"\nSummary: {result.summary}")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_readability_scorer())