"""
Grammar Checker Tool - Language Correctness and Style Consistency

This module provides comprehensive grammar and style checking capabilities using multiple
approaches including rule-based checking, statistical analysis, and AI-powered validation.

Key Features:
- Grammar and spelling correction
- Style consistency checking
- Language-specific rules
- Contextual error detection
- Writing style analysis
- Automated corrections and suggestions
"""

import asyncio
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal, Any, Tuple
from enum import Enum
import string
from collections import Counter, defaultdict

import language_tool_python
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from pydantic import BaseModel, Field, validator

from ...core.errors.exceptions import ToolExecutionError
from ...core.logging.logger import get_logger
from ...utils.retry import with_retry

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
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class ErrorType(str, Enum):
    """Types of grammar and style errors"""
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    STYLE = "style"
    WORD_CHOICE = "word_choice"
    SENTENCE_STRUCTURE = "sentence_structure"
    REDUNDANCY = "redundancy"
    CLARITY = "clarity"
    CONSISTENCY = "consistency"
    FORMATTING = "formatting"


class ErrorSeverity(str, Enum):
    """Severity levels for errors"""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    SUGGESTION = "suggestion"


class WritingStyle(str, Enum):
    """Writing style preferences"""
    ACADEMIC = "academic"
    BUSINESS = "business"
    CASUAL = "casual"
    CREATIVE = "creative"
    JOURNALISTIC = "journalistic"
    TECHNICAL = "technical"
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"


class Language(str, Enum):
    """Supported languages"""
    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-GB"
    ENGLISH_CA = "en-CA"
    ENGLISH_AU = "en-AU"


class GrammarError(BaseModel):
    """Individual grammar or style error"""
    
    error_type: ErrorType = Field(..., description="Type of error")
    severity: ErrorSeverity = Field(..., description="Error severity level")
    
    message: str = Field(..., description="Error description")
    suggestion: Optional[str] = Field(default=None, description="Suggested correction")
    
    start_pos: int = Field(..., description="Start position in text")
    end_pos: int = Field(..., description="End position in text")
    
    original_text: str = Field(..., description="Original problematic text")
    context: str = Field(..., description="Surrounding context")
    
    rule_id: Optional[str] = Field(default=None, description="Rule identifier")
    confidence: float = Field(default=1.0, description="Confidence in error detection (0-1)")
    
    auto_correctable: bool = Field(default=False, description="Can be automatically corrected")


class GrammarCheckRequest(BaseModel):
    """Grammar check request parameters"""
    
    text: str = Field(..., description="Text to check for grammar and style issues")
    
    language: Language = Field(default=Language.ENGLISH_US, description="Text language")
    style: WritingStyle = Field(default=WritingStyle.BUSINESS, description="Expected writing style")
    
    check_spelling: bool = Field(default=True, description="Check spelling errors")
    check_grammar: bool = Field(default=True, description="Check grammar errors")
    check_style: bool = Field(default=True, description="Check style consistency")
    check_punctuation: bool = Field(default=True, description="Check punctuation errors")
    
    min_severity: ErrorSeverity = Field(default=ErrorSeverity.MINOR, description="Minimum error severity to report")
    
    target_audience: Optional[str] = Field(
        default=None,
        description="Target audience for style checking"
    )
    
    brand_guidelines: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Brand-specific writing guidelines"
    )
    
    custom_dictionary: Optional[List[str]] = Field(
        default=None,
        description="Custom words to ignore (brand names, technical terms, etc.)"
    )
    
    auto_correct: bool = Field(default=False, description="Apply automatic corrections")
    
    context_sensitive: bool = Field(default=True, description="Use context for error detection")

    @validator('text')
    def validate_text_length(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(v) > 100000:
            raise ValueError("Text too long (max 100,000 characters)")
        return v


class GrammarCheckResults(BaseModel):
    """Complete grammar check results"""
    
    errors: List[GrammarError] = Field(..., description="Detected errors")
    corrected_text: Optional[str] = Field(default=None, description="Auto-corrected text")
    
    summary: Dict[str, int] = Field(..., description="Error count summary by type")
    overall_score: float = Field(..., description="Overall grammar score (0-100)")
    
    style_analysis: Dict[str, Any] = Field(..., description="Writing style analysis")
    readability_metrics: Dict[str, float] = Field(..., description="Basic readability metrics")
    
    processing_time: float = Field(..., description="Processing time in seconds")
    confidence_score: float = Field(..., description="Overall confidence in analysis")
    
    suggestions: List[str] = Field(..., description="General improvement suggestions")


class StylePatterns:
    """Style patterns and rules for different writing styles"""
    
    STYLE_RULES = {
        WritingStyle.ACADEMIC: {
            "avoid_contractions": True,
            "prefer_passive_voice": False,
            "min_sentence_length": 15,
            "max_sentence_length": 35,
            "avoid_first_person": True,
            "formal_vocabulary": True,
            "citation_format": True
        },
        
        WritingStyle.BUSINESS: {
            "avoid_contractions": False,
            "prefer_passive_voice": False,
            "min_sentence_length": 10,
            "max_sentence_length": 25,
            "avoid_first_person": False,
            "formal_vocabulary": True,
            "action_oriented": True
        },
        
        WritingStyle.CASUAL: {
            "avoid_contractions": False,
            "prefer_passive_voice": False,
            "min_sentence_length": 5,
            "max_sentence_length": 20,
            "avoid_first_person": False,
            "formal_vocabulary": False,
            "conversational_tone": True
        },
        
        WritingStyle.TECHNICAL: {
            "avoid_contractions": True,
            "prefer_passive_voice": True,
            "min_sentence_length": 12,
            "max_sentence_length": 30,
            "avoid_first_person": True,
            "formal_vocabulary": True,
            "precise_terminology": True
        }
    }
    
    COMMON_STYLE_ISSUES = {
        "redundant_phrases": [
            ("in order to", "to"),
            ("due to the fact that", "because"),
            ("at this point in time", "now"),
            ("for the purpose of", "for"),
            ("with regard to", "about"),
            ("in the event that", "if"),
            ("take into consideration", "consider")
        ],
        
        "weak_words": [
            "very", "really", "quite", "rather", "somewhat", "pretty",
            "just", "only", "actually", "basically", "literally"
        ],
        
        "overused_words": [
            "thing", "stuff", "nice", "good", "bad", "great", "awesome",
            "amazing", "incredible", "unbelievable"
        ],
        
        "filler_words": [
            "um", "uh", "like", "you know", "I mean", "sort of", "kind of"
        ]
    }


class GrammarChecker:
    """Comprehensive grammar and style checking tool"""
    
    def __init__(self, language: Language = Language.ENGLISH_US):
        """
        Initialize Grammar Checker
        
        Args:
            language: Default language for checking
        """
        self.language = language
        
        # Initialize LanguageTool
        try:
            self.language_tool = language_tool_python.LanguageTool(language.value)
        except Exception as e:
            logger.warning(f"Failed to initialize LanguageTool: {e}")
            self.language_tool = None
        
        # Load English stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            self.stopwords = set()
        
        # Common grammar patterns
        self.grammar_patterns = self._load_grammar_patterns()
        
        # Style rule cache
        self.style_cache = {}
        
        logger.info("GrammarChecker initialized")

    def _load_grammar_patterns(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load common grammar error patterns"""
        
        patterns = {
            "subject_verb_agreement": [
                (r'\b(he|she|it)\s+(are|were)\b', r'\1 is/was'),
                (r'\b(they|we|you)\s+(is|was)\b', r'\1 are/were'),
                (r'\b(\w+s)\s+(is|are)\s+(\w+)\b', r'Check singular/plural agreement')
            ],
            
            "article_errors": [
                (r'\ba\s+([aeiouAEIOU]\w+)', r'an \1'),
                (r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w+)', r'a \1')
            ],
            
            "common_mistakes": [
                (r'\byour\s+welcome\b', 'you\'re welcome'),
                (r'\bits\s+own\b', 'its own'),
                (r'\bit\'s\s+(own|color|time)\b', r'its \1'),
                (r'\bthere\s+(own|car|house)\b', r'their \1'),
                (r'\bwho\'s\s+(car|book)\b', r'whose \1')
            ],
            
            "punctuation": [
                (r'\s+([,.!?;:])', r'\1'),  # Remove space before punctuation
                (r'([.!?])\s*([a-z])', r'\1 \2'),  # Space after sentence end
                (r'\s{2,}', ' ')  # Multiple spaces
            ]
        }
        
        return patterns

    def _check_with_language_tool(self, text: str) -> List[GrammarError]:
        """Check text using LanguageTool"""
        
        if not self.language_tool:
            return []
        
        errors = []
        
        try:
            matches = self.language_tool.check(text)
            
            for match in matches:
                # Map LanguageTool categories to our ErrorType
                error_type = self._map_language_tool_category(match.category)
                
                # Determine severity
                if match.category in ['TYPOS', 'GRAMMAR']:
                    severity = ErrorSeverity.MAJOR
                elif match.category in ['STYLE', 'REDUNDANCY']:
                    severity = ErrorSeverity.MINOR
                else:
                    severity = ErrorSeverity.SUGGESTION
                
                error = GrammarError(
                    error_type=error_type,
                    severity=severity,
                    message=match.message,
                    suggestion=match.replacements[0] if match.replacements else None,
                    start_pos=match.offset,
                    end_pos=match.offset + match.errorLength,
                    original_text=text[match.offset:match.offset + match.errorLength],
                    context=self._get_context(text, match.offset, match.errorLength),
                    rule_id=match.ruleId,
                    confidence=0.85,  # LanguageTool confidence
                    auto_correctable=bool(match.replacements)
                )
                
                errors.append(error)
                
        except Exception as e:
            logger.error(f"LanguageTool check failed: {str(e)}")
        
        return errors

    def _map_language_tool_category(self, category: str) -> ErrorType:
        """Map LanguageTool categories to our ErrorType enum"""
        
        category_mapping = {
            'TYPOS': ErrorType.SPELLING,
            'GRAMMAR': ErrorType.GRAMMAR,
            'PUNCTUATION': ErrorType.PUNCTUATION,
            'STYLE': ErrorType.STYLE,
            'REDUNDANCY': ErrorType.REDUNDANCY,
            'WORD_CHOICE': ErrorType.WORD_CHOICE,
            'CLARITY': ErrorType.CLARITY,
            'CONSISTENCY': ErrorType.CONSISTENCY
        }
        
        return category_mapping.get(category, ErrorType.GRAMMAR)

    def _check_spelling_with_textblob(self, text: str) -> List[GrammarError]:
        """Check spelling using TextBlob"""
        
        errors = []
        
        try:
            blob = TextBlob(text)
            corrected = blob.correct()
            
            # Find differences
            words_original = text.split()
            words_corrected = str(corrected).split()
            
            pos = 0
            for i, (original, corrected_word) in enumerate(zip(words_original, words_corrected)):
                if original.lower() != corrected_word.lower():
                    # Find position in text
                    start_pos = text.find(original, pos)
                    if start_pos != -1:
                        error = GrammarError(
                            error_type=ErrorType.SPELLING,
                            severity=ErrorSeverity.MAJOR,
                            message=f"Possible spelling error: '{original}'",
                            suggestion=corrected_word,
                            start_pos=start_pos,
                            end_pos=start_pos + len(original),
                            original_text=original,
                            context=self._get_context(text, start_pos, len(original)),
                            confidence=0.75,
                            auto_correctable=True
                        )
                        errors.append(error)
                        pos = start_pos + len(original)
                
        except Exception as e:
            logger.error(f"TextBlob spelling check failed: {str(e)}")
        
        return errors

    def _check_grammar_patterns(self, text: str) -> List[GrammarError]:
        """Check text using regex patterns"""
        
        errors = []
        
        for category, patterns in self.grammar_patterns.items():
            for pattern, suggestion in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    error_type = self._pattern_to_error_type(category)
                    
                    error = GrammarError(
                        error_type=error_type,
                        severity=ErrorSeverity.MINOR,
                        message=f"{category.replace('_', ' ').title()} issue detected",
                        suggestion=suggestion if isinstance(suggestion, str) else suggestion.replace('\\1', match.group(1)) if '\\1' in suggestion else None,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        original_text=match.group(),
                        context=self._get_context(text, match.start(), match.end() - match.start()),
                        rule_id=f"pattern_{category}",
                        confidence=0.60,
                        auto_correctable=isinstance(suggestion, str)
                    )
                    
                    errors.append(error)
        
        return errors

    def _pattern_to_error_type(self, category: str) -> ErrorType:
        """Convert pattern category to ErrorType"""
        
        mapping = {
            'subject_verb_agreement': ErrorType.GRAMMAR,
            'article_errors': ErrorType.GRAMMAR,
            'common_mistakes': ErrorType.SPELLING,
            'punctuation': ErrorType.PUNCTUATION
        }
        
        return mapping.get(category, ErrorType.GRAMMAR)

    def _check_style_consistency(self, text: str, style: WritingStyle) -> List[GrammarError]:
        """Check style consistency against writing style rules"""
        
        errors = []
        style_rules = StylePatterns.STYLE_RULES.get(style, {})
        
        # Check contractions
        if style_rules.get("avoid_contractions", False):
            contractions = re.finditer(r"\b\w+'\w+\b", text)
            for match in contractions:
                error = GrammarError(
                    error_type=ErrorType.STYLE,
                    severity=ErrorSeverity.MINOR,
                    message=f"Avoid contractions in {style.value} writing",
                    suggestion=self._expand_contraction(match.group()),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    original_text=match.group(),
                    context=self._get_context(text, match.start(), len(match.group())),
                    confidence=0.80,
                    auto_correctable=True
                )
                errors.append(error)
        
        # Check sentence length
        sentences = sent_tokenize(text)
        min_length = style_rules.get("min_sentence_length", 0)
        max_length = style_rules.get("max_sentence_length", float('inf'))
        
        for sentence in sentences:
            word_count = len(word_tokenize(sentence))
            
            if word_count < min_length:
                pos = text.find(sentence.strip())
                if pos != -1:
                    error = GrammarError(
                        error_type=ErrorType.STYLE,
                        severity=ErrorSeverity.SUGGESTION,
                        message=f"Sentence too short for {style.value} style (minimum {min_length} words)",
                        start_pos=pos,
                        end_pos=pos + len(sentence),
                        original_text=sentence.strip(),
                        context=sentence,
                        confidence=0.70,
                        auto_correctable=False
                    )
                    errors.append(error)
            
            elif word_count > max_length:
                pos = text.find(sentence.strip())
                if pos != -1:
                    error = GrammarError(
                        error_type=ErrorType.STYLE,
                        severity=ErrorSeverity.MINOR,
                        message=f"Sentence too long for {style.value} style (maximum {max_length} words)",
                        start_pos=pos,
                        end_pos=pos + len(sentence),
                        original_text=sentence.strip(),
                        context=sentence,
                        confidence=0.70,
                        auto_correctable=False
                    )
                    errors.append(error)
        
        # Check for redundant phrases
        for redundant, replacement in StylePatterns.COMMON_STYLE_ISSUES["redundant_phrases"]:
            pattern = r'\b' + re.escape(redundant) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                error = GrammarError(
                    error_type=ErrorType.REDUNDANCY,
                    severity=ErrorSeverity.MINOR,
                    message=f"Redundant phrase: '{redundant}'",
                    suggestion=replacement,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    original_text=match.group(),
                    context=self._get_context(text, match.start(), len(match.group())),
                    confidence=0.85,
                    auto_correctable=True
                )
                errors.append(error)
        
        # Check for weak words
        for weak_word in StylePatterns.COMMON_STYLE_ISSUES["weak_words"]:
            pattern = r'\b' + re.escape(weak_word) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                error = GrammarError(
                    error_type=ErrorType.WORD_CHOICE,
                    severity=ErrorSeverity.SUGGESTION,
                    message=f"Consider replacing weak word: '{weak_word}'",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    original_text=match.group(),
                    context=self._get_context(text, match.start(), len(match.group())),
                    confidence=0.60,
                    auto_correctable=False
                )
                errors.append(error)
        
        return errors

    def _expand_contraction(self, contraction: str) -> str:
        """Expand common contractions"""
        
        contractions_map = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "'s": " is"  # Context-dependent, simplified
        }
        
        lower_contraction = contraction.lower()
        for contracted, expanded in contractions_map.items():
            if contracted in lower_contraction:
                return contraction.replace(contracted, expanded)
        
        return contraction

    def _get_context(self, text: str, start_pos: int, length: int, context_size: int = 50) -> str:
        """Get context around an error"""
        
        context_start = max(0, start_pos - context_size)
        context_end = min(len(text), start_pos + length + context_size)
        
        context = text[context_start:context_end]
        
        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."
        
        return context

    def _calculate_style_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate style analysis metrics"""
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Basic metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Vocabulary diversity (Type-Token Ratio)
        unique_words = set(word for word in words if word.isalpha())
        vocabulary_diversity = len(unique_words) / len([w for w in words if w.isalpha()]) if words else 0
        
        # Passive voice detection (simplified)
        pos_tags = pos_tag(words)
        passive_indicators = sum(1 for word, tag in pos_tags if tag in ['VBN', 'VBG'] and word in ['been', 'being'])
        passive_voice_ratio = passive_indicators / len(sentences) if sentences else 0
        
        # Readability approximation (simplified Flesch Reading Ease)
        if sentences and words:
            avg_sentence_len = len(words) / len(sentences)
            avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
            flesch_score = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)
        else:
            flesch_score = 0
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "vocabulary_diversity": vocabulary_diversity,
            "passive_voice_ratio": passive_voice_ratio,
            "flesch_reading_ease": max(0, min(100, flesch_score)),
            "sentence_count": len(sentences),
            "word_count": len(words),
            "unique_word_count": len(unique_words)
        }

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximate)"""
        
        word = word.lower().strip(string.punctuation)
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)  # Every word has at least 1 syllable

    def _apply_auto_corrections(self, text: str, errors: List[GrammarError]) -> str:
        """Apply automatic corrections to text"""
        
        # Sort errors by position (reverse order to maintain positions)
        correctable_errors = [e for e in errors if e.auto_correctable and e.suggestion]
        correctable_errors.sort(key=lambda x: x.start_pos, reverse=True)
        
        corrected_text = text
        
        for error in correctable_errors:
            # Apply correction
            before = corrected_text[:error.start_pos]
            after = corrected_text[error.end_pos:]
            corrected_text = before + error.suggestion + after
        
        return corrected_text

    def _filter_errors(self, errors: List[GrammarError], request: GrammarCheckRequest) -> List[GrammarError]:
        """Filter errors based on request criteria"""
        
        filtered_errors = []
        
        # Severity mapping for filtering
        severity_order = [ErrorSeverity.SUGGESTION, ErrorSeverity.MINOR, ErrorSeverity.MAJOR, ErrorSeverity.CRITICAL]
        min_severity_index = severity_order.index(request.min_severity)
        
        for error in errors:
            # Check severity
            error_severity_index = severity_order.index(error.severity)
            if error_severity_index < min_severity_index:
                continue
            
            # Check error types based on request flags
            if error.error_type == ErrorType.SPELLING and not request.check_spelling:
                continue
            if error.error_type == ErrorType.GRAMMAR and not request.check_grammar:
                continue
            if error.error_type == ErrorType.STYLE and not request.check_style:
                continue
            if error.error_type == ErrorType.PUNCTUATION and not request.check_punctuation:
                continue
            
            # Check custom dictionary
            if request.custom_dictionary and error.original_text.lower() in [w.lower() for w in request.custom_dictionary]:
                continue
            
            filtered_errors.append(error)
        
        return filtered_errors

    def _generate_suggestions(self, errors: List[GrammarError], style_metrics: Dict[str, Any]) -> List[str]:
        """Generate general improvement suggestions"""
        
        suggestions = []
        
        # Error-based suggestions
        error_counts = Counter(error.error_type for error in errors)
        
        if error_counts.get(ErrorType.SPELLING, 0) > 3:
            suggestions.append("Consider using spell-check tools or proofreading more carefully")
        
        if error_counts.get(ErrorType.SENTENCE_STRUCTURE, 0) > 2:
            suggestions.append("Review sentence structure for clarity and flow")
        
        if error_counts.get(ErrorType.REDUNDANCY, 0) > 2:
            suggestions.append("Eliminate redundant phrases to improve conciseness")
        
        # Style-based suggestions
        if style_metrics.get("avg_sentence_length", 0) > 25:
            suggestions.append("Consider breaking up long sentences for better readability")
        
        if style_metrics.get("vocabulary_diversity", 0) < 0.3:
            suggestions.append("Try using more varied vocabulary to improve engagement")
        
        if style_metrics.get("passive_voice_ratio", 0) > 0.3:
            suggestions.append("Consider using more active voice for stronger writing")
        
        return suggestions

    async def check_grammar(self, request: GrammarCheckRequest) -> GrammarCheckResults:
        """
        Perform comprehensive grammar and style checking
        
        Args:
            request: Grammar check request parameters
            
        Returns:
            Complete grammar check results with errors and suggestions
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting grammar check for {len(request.text)} characters")
            
            all_errors = []
            
            # LanguageTool checking
            if request.check_grammar or request.check_spelling:
                lt_errors = self._check_with_language_tool(request.text)
                all_errors.extend(lt_errors)
            
            # TextBlob spelling check
            if request.check_spelling:
                tb_errors = self._check_spelling_with_textblob(request.text)
                all_errors.extend(tb_errors)
            
            # Pattern-based grammar checking
            if request.check_grammar:
                pattern_errors = self._check_grammar_patterns(request.text)
                all_errors.extend(pattern_errors)
            
            # Style consistency checking
            if request.check_style:
                style_errors = self._check_style_consistency(request.text, request.style)
                all_errors.extend(style_errors)
            
            # Remove duplicates based on position and type
            all_errors = self._deduplicate_errors(all_errors)
            
            # Filter errors based on request criteria
            filtered_errors = self._filter_errors(all_errors, request)
            
            # Apply auto-corrections if requested
            corrected_text = None
            if request.auto_correct:
                corrected_text = self._apply_auto_corrections(request.text, filtered_errors)
            
            # Calculate style metrics
            style_metrics = self._calculate_style_metrics(request.text)
            
            # Calculate overall score
            total_words = len(word_tokenize(request.text))
            error_density = len(filtered_errors) / max(1, total_words / 100)  # Errors per 100 words
            overall_score = max(0, min(100, 100 - (error_density * 10)))
            
            # Generate error summary
            error_summary = Counter(error.error_type for error in filtered_errors)
            
            # Calculate confidence
            confidence_scores = [error.confidence for error in filtered_errors]
            average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 1.0
            
            # Generate suggestions
            suggestions = self._generate_suggestions(filtered_errors, style_metrics)
            
            # Processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = GrammarCheckResults(
                errors=filtered_errors,
                corrected_text=corrected_text,
                summary=dict(error_summary),
                overall_score=overall_score,
                style_analysis=style_metrics,
                readability_metrics={
                    "flesch_reading_ease": style_metrics["flesch_reading_ease"],
                    "avg_sentence_length": style_metrics["avg_sentence_length"],
                    "vocabulary_diversity": style_metrics["vocabulary_diversity"]
                },
                processing_time=processing_time,
                confidence_score=average_confidence,
                suggestions=suggestions
            )
            
            logger.info(f"Grammar check completed: {len(filtered_errors)} errors found, score: {overall_score:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Grammar check failed: {str(e)}")
            raise ToolExecutionError(f"Grammar check failed: {str(e)}")

    def _deduplicate_errors(self, errors: List[GrammarError]) -> List[GrammarError]:
        """Remove duplicate errors based on position and similarity"""
        
        if not errors:
            return []
        
        # Sort by position
        errors.sort(key=lambda x: (x.start_pos, x.end_pos))
        
        deduplicated = []
        
        for error in errors:
            # Check if this error overlaps significantly with any existing error
            is_duplicate = False
            
            for existing in deduplicated:
                # Check position overlap
                if (error.start_pos < existing.end_pos and error.end_pos > existing.start_pos):
                    # Overlapping positions
                    overlap_ratio = min(error.end_pos, existing.end_pos) - max(error.start_pos, existing.start_pos)
                    overlap_ratio /= max(error.end_pos - error.start_pos, existing.end_pos - existing.start_pos)
                    
                    if overlap_ratio > 0.5:  # More than 50% overlap
                        is_duplicate = True
                        # Keep the error with higher confidence
                        if error.confidence > existing.confidence:
                            deduplicated.remove(existing)
                            break
                        else:
                            break
            
            if not is_duplicate:
                deduplicated.append(error)
        
        return deduplicated


# Initialize tool instance
grammar_checker_tool = GrammarChecker()


# MCP Functions for external integration
async def mcp_check_grammar(
    text: str,
    language: str = "en-US",
    style: str = "business",
    check_spelling: bool = True,
    check_grammar: bool = True,
    check_style: bool = True,
    check_punctuation: bool = True,
    min_severity: str = "minor",
    auto_correct: bool = False,
    custom_dictionary: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    MCP function to check grammar and style
    
    Args:
        text: Text to check
        language: Language code (en-US, en-GB, etc.)
        style: Writing style (academic, business, casual, etc.)
        check_spelling: Check spelling errors
        check_grammar: Check grammar errors
        check_style: Check style consistency
        check_punctuation: Check punctuation errors
        min_severity: Minimum error severity (critical, major, minor, suggestion)
        auto_correct: Apply automatic corrections
        custom_dictionary: Custom words to ignore
        
    Returns:
        Grammar check results with errors and suggestions
    """
    try:
        request = GrammarCheckRequest(
            text=text,
            language=Language(language),
            style=WritingStyle(style),
            check_spelling=check_spelling,
            check_grammar=check_grammar,
            check_style=check_style,
            check_punctuation=check_punctuation,
            min_severity=ErrorSeverity(min_severity),
            auto_correct=auto_correct,
            custom_dictionary=custom_dictionary
        )
        
        result = await grammar_checker_tool.check_grammar(request)
        
        return {
            "success": True,
            "errors": [
                {
                    "type": error.error_type.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "suggestion": error.suggestion,
                    "start_pos": error.start_pos,
                    "end_pos": error.end_pos,
                    "original_text": error.original_text,
                    "context": error.context,
                    "confidence": error.confidence,
                    "auto_correctable": error.auto_correctable
                }
                for error in result.errors
            ],
            "corrected_text": result.corrected_text,
            "summary": result.summary,
            "overall_score": result.overall_score,
            "style_analysis": result.style_analysis,
            "readability_metrics": result.readability_metrics,
            "processing_time": result.processing_time,
            "confidence_score": result.confidence_score,
            "suggestions": result.suggestions
        }
        
    except Exception as e:
        logger.error(f"MCP grammar check failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Example usage and testing
    async def test_grammar_checker():
        """Test the grammar checker functionality"""
        
        test_text = """
        This is a test text with some erors. It has grammer mistakes and spelling issues.
        The sentences is too short. And some sentence are way too long because they contain multiple clauses that could be separated into different sentences for better readability and understanding.
        There's also redundant phrases like "in order to" and weak words like "very good".
        """
        
        request = GrammarCheckRequest(
            text=test_text,
            style=WritingStyle.BUSINESS,
            check_spelling=True,
            check_grammar=True,
            check_style=True,
            auto_correct=True
        )
        
        try:
            checker = GrammarChecker()
            result = await checker.check_grammar(request)
            
            print(f"Grammar Check Results:")
            print(f"Overall Score: {result.overall_score}/100")
            print(f"Errors Found: {len(result.errors)}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print()
            
            for i, error in enumerate(result.errors[:5], 1):  # Show first 5 errors
                print(f"{i}. {error.error_type.value.title()}: {error.message}")
                print(f"   Text: '{error.original_text}'")
                if error.suggestion:
                    print(f"   Suggestion: '{error.suggestion}'")
                print()
            
            if result.corrected_text:
                print("Corrected Text:")
                print(result.corrected_text)
            
            print("Suggestions:")
            for suggestion in result.suggestions:
                print(f"- {suggestion}")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_grammar_checker())