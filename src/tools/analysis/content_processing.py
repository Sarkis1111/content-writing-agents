"""
Content Processing Tool for text cleaning, normalization, and language detection.

Provides comprehensive text processing capabilities including cleaning,
normalization, language detection, and content deduplication.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Union, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from langdetect import detect, detect_langs, LangDetectError
from textblob import TextBlob
from pydantic import BaseModel, Field

from ...core.config.loader import get_settings
from ...core.errors.exceptions import ToolExecutionError
from ...utils.retry import with_retry


logger = logging.getLogger(__name__)


@dataclass
class ProcessedContent:
    """Container for processed content."""
    original_text: str
    cleaned_text: str
    normalized_text: str
    language: str
    language_confidence: float
    word_count: int
    sentence_count: int
    paragraph_count: int
    tokens: List[str]
    sentences: List[str]
    unique_words: Set[str]
    readability_metrics: Dict[str, float]
    processing_time: float


@dataclass
class LanguageInfo:
    """Language detection information."""
    language: str
    confidence: float
    detected_languages: List[Tuple[str, float]]


class ContentProcessingRequest(BaseModel):
    """Content processing request configuration."""
    text: str = Field(..., description="Text content to process")
    clean_text: bool = Field(default=True, description="Perform text cleaning")
    normalize_text: bool = Field(default=True, description="Normalize text")
    detect_language: bool = Field(default=True, description="Detect content language")
    tokenize: bool = Field(default=True, description="Tokenize text")
    remove_stopwords: bool = Field(default=False, description="Remove stop words")
    stem_words: bool = Field(default=False, description="Apply stemming")
    lemmatize_words: bool = Field(default=False, description="Apply lemmatization")
    calculate_readability: bool = Field(default=True, description="Calculate readability metrics")
    target_language: Optional[str] = Field(default=None, description="Expected language (for validation)")


class ContentProcessingResponse(BaseModel):
    """Content processing response."""
    success: bool
    original_length: int
    processed_content: Optional[ProcessedContent] = None
    language_info: Optional[LanguageInfo] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class ContentProcessingTool:
    """
    Content Processing Tool for comprehensive text analysis and cleaning.
    
    Provides text cleaning, normalization, language detection, tokenization,
    and various NLP preprocessing capabilities.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize NLTK components
        self._init_nltk()
        
        # Text cleaning patterns
        self.cleaning_patterns = {
            'html_tags': re.compile(r'<[^>]+>'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_numbers': re.compile(r'(\+\d{1,3}[- ]?)?\d{10}'),
            'social_handles': re.compile(r'@\w+'),
            'hashtags': re.compile(r'#\w+'),
            'excessive_whitespace': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']+'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),  # 3+ repeated characters
            'numeric_only': re.compile(r'^\d+$'),
            'punctuation_spam': re.compile(r'[!.?]{3,}')
        }
        
        # Normalization patterns
        self.normalization_patterns = {
            'quotes': [('"', '"'), ('"', '"'), (''', "'"), (''', "'")],
            'dashes': [('–', '-'), ('—', '-'), ('−', '-')],
            'ellipsis': [('…', '...'), ('⋯', '...')],
            'whitespace': [(r'\s+', ' '), (r'\n+', '\n'), (r'\t+', ' ')],
        }
        
        # Initialize stemmer and lemmatizer
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Supported languages for stopwords
        self.supported_stopword_languages = {
            'en': 'english',
            'es': 'spanish', 
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'ru': 'russian',
            'ar': 'arabic',
            'zh': 'chinese'
        }
    
    def _init_nltk(self):
        """Initialize NLTK data."""
        try:
            # Download required NLTK data
            nltk_downloads = [
                'punkt',
                'stopwords', 
                'wordnet',
                'averaged_perceptron_tagger',
                'omw-1.4'
            ]
            
            for download in nltk_downloads:
                try:
                    nltk.data.find(f'tokenizers/{download}' if download == 'punkt' else 
                                 f'corpora/{download}' if download in ['stopwords', 'wordnet', 'omw-1.4'] else
                                 f'taggers/{download}')
                except LookupError:
                    logger.info(f"Downloading NLTK data: {download}")
                    nltk.download(download, quiet=True)
            
            logger.info("NLTK initialization complete")
            
        except Exception as e:
            logger.error(f"NLTK initialization failed: {e}")
    
    def _detect_language(self, text: str) -> LanguageInfo:
        """Detect text language with confidence scores."""
        try:
            # Get all detected languages with confidence
            detected_langs = detect_langs(text)
            
            # Primary language (highest confidence)
            primary_lang = detected_langs[0]
            
            # Convert to standard format
            lang_info = LanguageInfo(
                language=primary_lang.lang,
                confidence=primary_lang.prob,
                detected_languages=[(lang.lang, lang.prob) for lang in detected_langs[:3]]
            )
            
            return lang_info
            
        except LangDetectError:
            # Fallback for very short or problematic text
            return LanguageInfo(
                language='unknown',
                confidence=0.0,
                detected_languages=[('unknown', 0.0)]
            )
    
    def _clean_text(self, text: str) -> str:
        """Comprehensive text cleaning."""
        cleaned = text
        
        # Remove HTML tags
        cleaned = self.cleaning_patterns['html_tags'].sub('', cleaned)
        
        # Remove URLs
        cleaned = self.cleaning_patterns['urls'].sub(' [URL] ', cleaned)
        
        # Remove email addresses
        cleaned = self.cleaning_patterns['emails'].sub(' [EMAIL] ', cleaned)
        
        # Remove phone numbers
        cleaned = self.cleaning_patterns['phone_numbers'].sub(' [PHONE] ', cleaned)
        
        # Handle social media elements
        cleaned = self.cleaning_patterns['social_handles'].sub(' [MENTION] ', cleaned)
        cleaned = self.cleaning_patterns['hashtags'].sub(' [HASHTAG] ', cleaned)
        
        # Fix repeated characters (e.g., "sooooo" -> "soo")
        cleaned = self.cleaning_patterns['repeated_chars'].sub(r'\1\1', cleaned)
        
        # Clean up punctuation spam
        cleaned = self.cleaning_patterns['punctuation_spam'].sub(lambda m: m.group()[0] * 3, cleaned)
        
        # Remove excessive whitespace
        cleaned = self.cleaning_patterns['excessive_whitespace'].sub(' ', cleaned)
        
        # Remove lines that are just numbers
        lines = cleaned.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not self.cleaning_patterns['numeric_only'].match(line):
                cleaned_lines.append(line)
        
        cleaned = '\n'.join(cleaned_lines)
        
        return cleaned.strip()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistency."""
        normalized = text
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKC', normalized)
        
        # Fix quotes and dashes
        for original, replacement in self.normalization_patterns['quotes']:
            normalized = normalized.replace(original, replacement)
        
        for original, replacement in self.normalization_patterns['dashes']:
            normalized = normalized.replace(original, replacement)
        
        for original, replacement in self.normalization_patterns['ellipsis']:
            normalized = normalized.replace(original, replacement)
        
        # Normalize whitespace
        for pattern, replacement in self.normalization_patterns['whitespace']:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Fix common encoding issues
        normalized = normalized.replace('â€™', "'")
        normalized = normalized.replace('â€œ', '"')
        normalized = normalized.replace('â€', '"')
        
        return normalized.strip()
    
    def _tokenize_text(self, text: str, language: str = 'en') -> Tuple[List[str], List[str]]:
        """Tokenize text into words and sentences."""
        try:
            # Sentence tokenization
            sentences = sent_tokenize(text)
            
            # Word tokenization
            words = word_tokenize(text.lower())
            
            # Filter out pure punctuation and very short tokens
            words = [word for word in words if len(word) > 1 and word.isalnum()]
            
            return words, sentences
            
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            # Fallback tokenization
            sentences = text.split('. ')
            words = re.findall(r'\b\w+\b', text.lower())
            return words, sentences
    
    def _remove_stopwords(self, tokens: List[str], language: str = 'en') -> List[str]:
        """Remove stop words from tokens."""
        try:
            # Get appropriate stopwords set
            lang_code = self.supported_stopword_languages.get(language, 'english')
            stop_words = set(stopwords.words(lang_code))
            
            # Filter out stopwords
            filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
            
            return filtered_tokens
            
        except Exception as e:
            logger.error(f"Stopword removal failed: {e}")
            return tokens
    
    def _stem_words(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens."""
        try:
            return [self.stemmer.stem(token) for token in tokens]
        except Exception as e:
            logger.error(f"Stemming failed: {e}")
            return tokens
    
    def _lemmatize_words(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens."""
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception as e:
            logger.error(f"Lemmatization failed: {e}")
            return tokens
    
    def _calculate_basic_readability(self, text: str, word_count: int, sentence_count: int) -> Dict[str, float]:
        """Calculate basic readability metrics."""
        metrics = {}
        
        try:
            # Average sentence length
            avg_sentence_length = word_count / max(sentence_count, 1)
            metrics['avg_sentence_length'] = round(avg_sentence_length, 2)
            
            # Average word length
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
            metrics['avg_word_length'] = round(avg_word_length, 2)
            
            # Simple Flesch Reading Ease approximation
            # Formula: 206.835 - (1.015 × ASL) - (84.6 × ASW)
            # ASL = Average Sentence Length, ASW = Average Syllables per Word
            
            # Rough syllable estimation
            syllable_count = sum(max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in words)
            avg_syllables_per_word = syllable_count / max(len(words), 1)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            metrics['flesch_reading_ease'] = round(max(0, min(100, flesch_score)), 2)
            
            # Reading level classification
            if flesch_score >= 90:
                metrics['reading_level'] = 'Very Easy'
            elif flesch_score >= 80:
                metrics['reading_level'] = 'Easy'
            elif flesch_score >= 70:
                metrics['reading_level'] = 'Fairly Easy'
            elif flesch_score >= 60:
                metrics['reading_level'] = 'Standard'
            elif flesch_score >= 50:
                metrics['reading_level'] = 'Fairly Difficult'
            elif flesch_score >= 30:
                metrics['reading_level'] = 'Difficult'
            else:
                metrics['reading_level'] = 'Very Difficult'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Readability calculation failed: {e}")
            return {
                'avg_sentence_length': 0.0,
                'avg_word_length': 0.0,
                'flesch_reading_ease': 0.0,
                'reading_level': 'Unknown'
            }
    
    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs in text."""
        # Split by double newlines or more
        paragraphs = re.split(r'\n\s*\n', text.strip())
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return len(paragraphs)
    
    async def process_content(
        self,
        request: Union[str, ContentProcessingRequest]
    ) -> ContentProcessingResponse:
        """
        Process content with comprehensive text analysis and cleaning.
        
        Args:
            request: Text string or ContentProcessingRequest object
            
        Returns:
            ContentProcessingResponse with processed content and metadata
        """
        start_time = datetime.now()
        
        # Convert string to request object
        if isinstance(request, str):
            request = ContentProcessingRequest(text=request)
        
        try:
            original_text = request.text
            original_length = len(original_text)
            
            # Skip processing if text is too short
            if original_length < 10:
                return ContentProcessingResponse(
                    success=False,
                    original_length=original_length,
                    error="Text too short for processing (minimum 10 characters)",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    timestamp=datetime.now()
                )
            
            logger.info(f"Processing content: {original_length} characters")
            
            # Initialize processed text
            processed_text = original_text
            
            # Language detection
            language_info = None
            if request.detect_language:
                language_info = self._detect_language(original_text)
                logger.info(f"Detected language: {language_info.language} (confidence: {language_info.confidence:.2f})")
            
            # Text cleaning
            cleaned_text = original_text
            if request.clean_text:
                cleaned_text = self._clean_text(original_text)
                processed_text = cleaned_text
            
            # Text normalization
            normalized_text = processed_text
            if request.normalize_text:
                normalized_text = self._normalize_text(processed_text)
                processed_text = normalized_text
            
            # Tokenization
            tokens = []
            sentences = []
            if request.tokenize:
                detected_lang = language_info.language if language_info else 'en'
                tokens, sentences = self._tokenize_text(processed_text, detected_lang)
            
            # Stopword removal
            if request.remove_stopwords and tokens:
                detected_lang = language_info.language if language_info else 'en'
                tokens = self._remove_stopwords(tokens, detected_lang)
            
            # Stemming
            if request.stem_words and tokens:
                tokens = self._stem_words(tokens)
            
            # Lemmatization
            if request.lemmatize_words and tokens:
                tokens = self._lemmatize_words(tokens)
            
            # Calculate metrics
            word_count = len(processed_text.split())
            sentence_count = len(sentences) if sentences else len(processed_text.split('.'))
            paragraph_count = self._count_paragraphs(processed_text)
            unique_words = set(tokens) if tokens else set(processed_text.lower().split())
            
            # Readability metrics
            readability_metrics = {}
            if request.calculate_readability:
                readability_metrics = self._calculate_basic_readability(
                    processed_text, word_count, sentence_count
                )
            
            # Create processed content object
            processed_content = ProcessedContent(
                original_text=original_text,
                cleaned_text=cleaned_text,
                normalized_text=normalized_text,
                language=language_info.language if language_info else 'unknown',
                language_confidence=language_info.confidence if language_info else 0.0,
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                tokens=tokens,
                sentences=sentences,
                unique_words=unique_words,
                readability_metrics=readability_metrics,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ContentProcessingResponse(
                success=True,
                original_length=original_length,
                processed_content=processed_content,
                language_info=language_info,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ContentProcessingResponse(
                success=False,
                original_length=len(request.text),
                error=str(e),
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    async def batch_process(self, texts: List[str]) -> List[ContentProcessingResponse]:
        """
        Process multiple texts concurrently.
        
        Args:
            texts: List of text strings to process
            
        Returns:
            List of ContentProcessingResponse objects
        """
        tasks = [self.process_content(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed for text {i}: {result}")
                responses.append(ContentProcessingResponse(
                    success=False,
                    original_length=len(texts[i]),
                    error=str(result),
                    processing_time=0.0,
                    timestamp=datetime.now()
                ))
            else:
                responses.append(result)
        
        return responses
    
    def detect_duplicates(self, texts: List[str], similarity_threshold: float = 0.8) -> Dict[str, List[int]]:
        """
        Detect duplicate or near-duplicate content.
        
        Args:
            texts: List of texts to check for duplicates
            similarity_threshold: Similarity threshold for duplicate detection
            
        Returns:
            Dictionary mapping content hash to list of indices
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            if len(texts) < 2:
                return {}
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find duplicates
            duplicates = {}
            processed_indices = set()
            
            for i in range(len(texts)):
                if i in processed_indices:
                    continue
                
                similar_indices = []
                for j in range(i + 1, len(texts)):
                    if similarity_matrix[i][j] >= similarity_threshold:
                        similar_indices.append(j)
                        processed_indices.add(j)
                
                if similar_indices:
                    # Use content hash as key
                    content_hash = hashlib.md5(texts[i].encode()).hexdigest()[:8]
                    duplicates[content_hash] = [i] + similar_indices
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            return {}


# Tool instance for MCP integration
content_processing_tool = ContentProcessingTool()


# MCP tool function
async def mcp_process_content(
    text: str,
    clean_text: bool = True,
    normalize_text: bool = True,
    detect_language: bool = True,
    tokenize: bool = True,
    remove_stopwords: bool = False,
    stem_words: bool = False,
    lemmatize_words: bool = False,
    calculate_readability: bool = True,
    target_language: Optional[str] = None
) -> Dict:
    """
    MCP-compatible content processing function.
    
    Args:
        text: Text content to process
        clean_text: Perform text cleaning
        normalize_text: Normalize text
        detect_language: Detect content language
        tokenize: Tokenize text
        remove_stopwords: Remove stop words
        stem_words: Apply stemming
        lemmatize_words: Apply lemmatization
        calculate_readability: Calculate readability metrics
        target_language: Expected language (for validation)
    
    Returns:
        Dictionary with processed content and analysis
    """
    try:
        request = ContentProcessingRequest(
            text=text,
            clean_text=clean_text,
            normalize_text=normalize_text,
            detect_language=detect_language,
            tokenize=tokenize,
            remove_stopwords=remove_stopwords,
            stem_words=stem_words,
            lemmatize_words=lemmatize_words,
            calculate_readability=calculate_readability,
            target_language=target_language
        )
        
        response = await content_processing_tool.process_content(request)
        
        if response.success and response.processed_content:
            content = response.processed_content
            
            return {
                "success": True,
                "original_length": response.original_length,
                "processed_content": {
                    "cleaned_text": content.cleaned_text,
                    "normalized_text": content.normalized_text,
                    "language": content.language,
                    "language_confidence": content.language_confidence,
                    "word_count": content.word_count,
                    "sentence_count": content.sentence_count,
                    "paragraph_count": content.paragraph_count,
                    "tokens": content.tokens[:100],  # Limit tokens in response
                    "unique_words_count": len(content.unique_words),
                    "readability_metrics": content.readability_metrics
                },
                "language_info": {
                    "language": response.language_info.language,
                    "confidence": response.language_info.confidence,
                    "detected_languages": response.language_info.detected_languages
                } if response.language_info else None,
                "processing_time": response.processing_time,
                "timestamp": response.timestamp.isoformat()
            }
        else:
            return {
                "success": False,
                "error": response.error,
                "original_length": response.original_length,
                "processing_time": response.processing_time
            }
            
    except Exception as e:
        logger.error(f"Content processing failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# MCP function for duplicate detection
async def mcp_detect_duplicates(
    texts: List[str],
    similarity_threshold: float = 0.8
) -> Dict:
    """
    MCP-compatible duplicate detection function.
    
    Args:
        texts: List of texts to check for duplicates
        similarity_threshold: Similarity threshold for duplicate detection
    
    Returns:
        Dictionary with duplicate detection results
    """
    try:
        duplicates = content_processing_tool.detect_duplicates(texts, similarity_threshold)
        
        return {
            "success": True,
            "total_texts": len(texts),
            "duplicate_groups": len(duplicates),
            "duplicates": duplicates,
            "similarity_threshold": similarity_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Duplicate detection failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "total_texts": len(texts)
        }