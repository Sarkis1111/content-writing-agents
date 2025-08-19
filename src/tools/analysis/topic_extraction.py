"""
Topic Extraction Tool for NLP-based topic modeling and keyword extraction.

Provides comprehensive topic modeling capabilities including keyword extraction,
topic clustering, theme identification, and content categorization.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import math

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from textblob import TextBlob
from pydantic import BaseModel, Field

# Flexible imports to handle both package and direct imports
try:
    from core.config.loader import get_settings
    from core.errors import ToolError
    from utils.simple_retry import with_retry
except ImportError:
    # Mock implementations for testing
    def get_settings():
        return {}
    
    class ToolError(Exception):
        pass
    
    def with_retry(max_attempts=3, delay=1.0, backoff=2.0):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return decorator


logger = logging.getLogger(__name__)


@dataclass
class ExtractedKeyword:
    """Individual extracted keyword."""
    word: str
    score: float
    frequency: int
    pos_tag: Optional[str] = None
    category: str = "general"


@dataclass
class ExtractedTopic:
    """Individual topic from topic modeling."""
    topic_id: int
    keywords: List[str]
    weights: List[float]
    coherence_score: float
    description: str


@dataclass
class NamedEntity:
    """Named entity extraction result."""
    entity: str
    label: str
    confidence: float
    start_pos: int
    end_pos: int


@dataclass
class TopicExtractionResult:
    """Complete topic extraction result."""
    keywords: List[ExtractedKeyword]
    topics: List[ExtractedTopic]
    named_entities: List[NamedEntity]
    key_phrases: List[str]
    themes: List[str]
    categories: List[str]
    document_summary: str
    processing_time: float


class TopicExtractionRequest(BaseModel):
    """Topic extraction request configuration."""
    text: str = Field(..., description="Text content for topic extraction")
    num_keywords: int = Field(default=20, ge=5, le=100, description="Number of keywords to extract")
    num_topics: int = Field(default=5, ge=2, le=20, description="Number of topics for modeling")
    extract_named_entities: bool = Field(default=True, description="Extract named entities")
    extract_key_phrases: bool = Field(default=True, description="Extract key phrases")
    use_tfidf: bool = Field(default=True, description="Use TF-IDF for keyword extraction")
    use_topic_modeling: bool = Field(default=True, description="Perform topic modeling")
    min_keyword_length: int = Field(default=3, description="Minimum keyword length")
    language: str = Field(default="en", description="Content language")
    remove_stopwords: bool = Field(default=True, description="Remove stopwords")


class TopicExtractionResponse(BaseModel):
    """Topic extraction response."""
    success: bool
    text_length: int
    result: Optional[TopicExtractionResult] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class TopicExtractionTool:
    """
    Topic Extraction Tool for comprehensive NLP-based content analysis.
    
    Provides keyword extraction, topic modeling, named entity recognition,
    and theme identification capabilities.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize NLP models
        self._init_nlp_models()
        
        # Keyword extraction methods
        self.keyword_extractors = {
            'tfidf': self._extract_keywords_tfidf,
            'frequency': self._extract_keywords_frequency,
            'textrank': self._extract_keywords_textrank
        }
        
        # Topic modeling algorithms
        self.topic_models = {
            'lda': LatentDirichletAllocation,
            'nmf': NMF
        }
        
        # POS tags that are typically good keywords
        self.relevant_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBG'}
        
        # Common categories for keyword classification
        self.keyword_categories = {
            'technology': ['software', 'hardware', 'computer', 'digital', 'tech', 'ai', 'ml', 'data'],
            'business': ['market', 'company', 'business', 'revenue', 'profit', 'sales', 'customer'],
            'science': ['research', 'study', 'analysis', 'experiment', 'scientific', 'theory'],
            'health': ['health', 'medical', 'treatment', 'disease', 'patient', 'doctor', 'medicine'],
            'education': ['learn', 'student', 'teacher', 'school', 'education', 'course', 'training'],
            'finance': ['money', 'investment', 'financial', 'bank', 'economy', 'price', 'cost'],
        }
    
    def _init_nlp_models(self):
        """Initialize NLP models and download required data."""
        try:
            # Initialize spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy English model loaded")
            except OSError:
                logger.warning("spaCy English model not found. Named entity recognition may be limited.")
                self.nlp = None
            
            # Ensure NLTK data is available
            nltk_downloads = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
            
            for download in nltk_downloads:
                try:
                    nltk.data.find(f'tokenizers/{download}' if download == 'punkt' else 
                                 f'corpora/{download}' if download in ['stopwords', 'words'] else
                                 f'taggers/{download}' if download == 'averaged_perceptron_tagger' else
                                 f'chunkers/{download}')
                except LookupError:
                    logger.info(f"Downloading NLTK data: {download}")
                    nltk.download(download, quiet=True)
            
            logger.info("NLP models initialization complete")
            
        except Exception as e:
            logger.error(f"NLP models initialization failed: {e}")
    
    def _preprocess_text(self, text: str, language: str = 'en', remove_stopwords: bool = True) -> List[str]:
        """Preprocess text for topic extraction."""
        try:
            # Convert to lowercase and tokenize
            tokens = word_tokenize(text.lower())
            
            # Filter tokens (remove punctuation, numbers, short words)
            tokens = [token for token in tokens if token.isalpha() and len(token) >= 3]
            
            # Remove stopwords
            if remove_stopwords:
                stop_words = set(stopwords.words('english' if language == 'en' else language))
                tokens = [token for token in tokens if token not in stop_words]
            
            # POS tagging to keep relevant words
            pos_tags = pos_tag(tokens)
            relevant_tokens = [token for token, pos in pos_tags if pos in self.relevant_pos_tags]
            
            return relevant_tokens
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            # Fallback preprocessing
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return tokens[:1000]  # Limit tokens
    
    def _extract_keywords_tfidf(self, text: str, num_keywords: int = 20) -> List[ExtractedKeyword]:
        """Extract keywords using TF-IDF."""
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=1,
                max_df=0.8
            )
            
            # Fit and transform text (split into sentences for better TF-IDF)
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) < 2:
                sentences = [text]  # Use full text if not enough sentences
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF scores
            mean_scores = tfidf_matrix.mean(axis=0).A1
            
            # Create keyword-score pairs
            keyword_scores = list(zip(feature_names, mean_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Create ExtractedKeyword objects
            keywords = []
            for word, score in keyword_scores[:num_keywords]:
                # Calculate frequency in original text
                frequency = text.lower().count(word)
                
                # Categorize keyword
                category = self._categorize_keyword(word)
                
                keyword = ExtractedKeyword(
                    word=word,
                    score=float(score),
                    frequency=frequency,
                    category=category
                )
                keywords.append(keyword)
            
            return keywords
            
        except Exception as e:
            logger.error(f"TF-IDF keyword extraction failed: {e}")
            return []
    
    def _extract_keywords_frequency(self, text: str, num_keywords: int = 20) -> List[ExtractedKeyword]:
        """Extract keywords using frequency analysis."""
        try:
            # Preprocess text
            tokens = self._preprocess_text(text)
            
            # Count frequencies
            word_freq = Counter(tokens)
            
            # Calculate normalized scores
            max_freq = max(word_freq.values()) if word_freq else 1
            
            keywords = []
            for word, freq in word_freq.most_common(num_keywords):
                score = freq / max_freq
                category = self._categorize_keyword(word)
                
                keyword = ExtractedKeyword(
                    word=word,
                    score=score,
                    frequency=freq,
                    category=category
                )
                keywords.append(keyword)
            
            return keywords
            
        except Exception as e:
            logger.error(f"Frequency keyword extraction failed: {e}")
            return []
    
    def _extract_keywords_textrank(self, text: str, num_keywords: int = 20) -> List[ExtractedKeyword]:
        """Extract keywords using TextRank algorithm (simplified)."""
        try:
            # Preprocess text
            tokens = self._preprocess_text(text)
            
            if not tokens:
                return []
            
            # Create word co-occurrence matrix
            window_size = 5
            word_graph = {}
            
            for i, word in enumerate(tokens):
                if word not in word_graph:
                    word_graph[word] = {}
                
                # Add connections within window
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        neighbor = tokens[j]
                        if neighbor not in word_graph[word]:
                            word_graph[word][neighbor] = 0
                        word_graph[word][neighbor] += 1
            
            # Simple PageRank-like scoring
            scores = {word: 1.0 for word in word_graph}
            
            # Iterate to convergence (simplified)
            for _ in range(10):
                new_scores = {}
                for word in word_graph:
                    score = 0.15  # Damping factor
                    for neighbor, weight in word_graph[word].items():
                        if neighbor in scores:
                            neighbor_links = sum(word_graph[neighbor].values()) or 1
                            score += 0.85 * (weight / neighbor_links) * scores[neighbor]
                    new_scores[word] = score
                scores = new_scores
            
            # Sort and return top keywords
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            keywords = []
            for word, score in sorted_words[:num_keywords]:
                frequency = tokens.count(word)
                category = self._categorize_keyword(word)
                
                keyword = ExtractedKeyword(
                    word=word,
                    score=score,
                    frequency=frequency,
                    category=category
                )
                keywords.append(keyword)
            
            return keywords
            
        except Exception as e:
            logger.error(f"TextRank keyword extraction failed: {e}")
            return []
    
    def _categorize_keyword(self, word: str) -> str:
        """Categorize keyword into semantic categories."""
        word_lower = word.lower()
        
        for category, keywords in self.keyword_categories.items():
            if any(kw in word_lower for kw in keywords):
                return category
        
        return "general"
    
    def _extract_topics_lda(self, text: str, num_topics: int = 5) -> List[ExtractedTopic]:
        """Extract topics using Latent Dirichlet Allocation."""
        try:
            # Prepare text for topic modeling
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) < num_topics:
                # Not enough sentences for meaningful topic modeling
                return []
            
            # Vectorize text
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Fit LDA model
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=20
            )
            
            lda.fit(doc_term_matrix)
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                # Get top words for this topic
                top_indices = topic.argsort()[-10:][::-1]
                topic_keywords = [feature_names[i] for i in top_indices]
                topic_weights = [float(topic[i]) for i in top_indices]
                
                # Calculate coherence (simplified)
                coherence_score = float(topic.max() / topic.sum())
                
                # Generate topic description
                description = f"Topic {topic_idx + 1}: {', '.join(topic_keywords[:3])}"
                
                extracted_topic = ExtractedTopic(
                    topic_id=topic_idx,
                    keywords=topic_keywords,
                    weights=topic_weights,
                    coherence_score=coherence_score,
                    description=description
                )
                
                topics.append(extracted_topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"LDA topic extraction failed: {e}")
            return []
    
    def _extract_named_entities(self, text: str) -> List[NamedEntity]:
        """Extract named entities from text."""
        entities = []
        
        try:
            # Try spaCy first (more accurate)
            if self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entity = NamedEntity(
                        entity=ent.text,
                        label=ent.label_,
                        confidence=1.0,  # spaCy doesn't provide confidence scores directly
                        start_pos=ent.start_char,
                        end_pos=ent.end_char
                    )
                    entities.append(entity)
            else:
                # Fallback to NLTK
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                current_pos = 0
                for chunk in chunks:
                    if isinstance(chunk, Tree):
                        entity_name = ' '.join([token for token, pos in chunk.leaves()])
                        entity_label = chunk.label()
                        
                        # Find position in original text
                        start_pos = text.lower().find(entity_name.lower(), current_pos)
                        end_pos = start_pos + len(entity_name) if start_pos != -1 else 0
                        
                        entity = NamedEntity(
                            entity=entity_name,
                            label=entity_label,
                            confidence=0.8,  # NLTK approximation
                            start_pos=start_pos,
                            end_pos=end_pos
                        )
                        entities.append(entity)
                        current_pos = end_pos
            
            return entities
            
        except Exception as e:
            logger.error(f"Named entity extraction failed: {e}")
            return []
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 15) -> List[str]:
        """Extract key phrases from text."""
        try:
            # Use TextBlob for noun phrase extraction
            blob = TextBlob(text)
            noun_phrases = list(blob.noun_phrases)
            
            # Filter and score phrases
            phrase_scores = {}
            for phrase in noun_phrases:
                if len(phrase.split()) >= 2 and len(phrase) >= 5:  # Multi-word phrases
                    # Simple scoring based on frequency and length
                    frequency = text.lower().count(phrase.lower())
                    length_bonus = min(len(phrase.split()), 4) / 4  # Favor longer phrases
                    score = frequency * (1 + length_bonus)
                    phrase_scores[phrase] = score
            
            # Sort and return top phrases
            sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
            key_phrases = [phrase for phrase, score in sorted_phrases[:max_phrases]]
            
            return key_phrases
            
        except Exception as e:
            logger.error(f"Key phrase extraction failed: {e}")
            # Fallback: extract common bigrams and trigrams
            words = re.findall(r'\b\w+\b', text.lower())
            phrases = []
            
            # Extract bigrams
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 5:
                    phrases.append(phrase)
            
            # Extract trigrams
            for i in range(len(words) - 2):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(phrase) > 8:
                    phrases.append(phrase)
            
            # Return most frequent phrases
            phrase_freq = Counter(phrases)
            return [phrase for phrase, freq in phrase_freq.most_common(max_phrases)]
    
    def _identify_themes(self, keywords: List[ExtractedKeyword], topics: List[ExtractedTopic]) -> List[str]:
        """Identify main themes from keywords and topics."""
        themes = []
        
        try:
            # Group keywords by category
            category_keywords = {}
            for keyword in keywords[:10]:  # Top keywords only
                cat = keyword.category
                if cat not in category_keywords:
                    category_keywords[cat] = []
                category_keywords[cat].append(keyword.word)
            
            # Create themes from categories with multiple keywords
            for category, words in category_keywords.items():
                if len(words) >= 2:
                    theme = f"{category.title()}: {', '.join(words[:3])}"
                    themes.append(theme)
            
            # Add themes from topic modeling
            for topic in topics[:3]:  # Top topics only
                if topic.coherence_score > 0.1:
                    theme = f"Topic: {', '.join(topic.keywords[:3])}"
                    themes.append(theme)
            
            return themes[:5]  # Limit to top 5 themes
            
        except Exception as e:
            logger.error(f"Theme identification failed: {e}")
            return []
    
    def _generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a brief summary of the document."""
        try:
            # Simple extractive summary using sentence scoring
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) <= max_sentences:
                return '. '.join(sentences) + '.'
            
            # Score sentences based on word frequency
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = Counter(words)
            
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                sentence_words = re.findall(r'\b\w+\b', sentence.lower())
                score = sum(word_freq[word] for word in sentence_words)
                sentence_scores[i] = score / max(len(sentence_words), 1)
            
            # Select top scoring sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            selected_indices = sorted([idx for idx, score in top_sentences[:max_sentences]])
            
            summary_sentences = [sentences[i] for i in selected_indices]
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Fallback: return first few sentences
            sentences = text.split('.')[:max_sentences]
            return '. '.join(s.strip() for s in sentences if s.strip()) + '.'
    
    async def extract_topics(
        self,
        request: Union[str, TopicExtractionRequest]
    ) -> TopicExtractionResponse:
        """
        Extract topics, keywords, and themes from text.
        
        Args:
            request: Text string or TopicExtractionRequest object
            
        Returns:
            TopicExtractionResponse with extracted topics and analysis
        """
        start_time = datetime.now()
        
        # Convert string to request object
        if isinstance(request, str):
            request = TopicExtractionRequest(text=request)
        
        try:
            text = request.text
            text_length = len(text)
            
            # Check minimum text length
            if text_length < 100:
                return TopicExtractionResponse(
                    success=False,
                    text_length=text_length,
                    error="Text too short for meaningful topic extraction (minimum 100 characters)",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    timestamp=datetime.now()
                )
            
            logger.info(f"Extracting topics from text: {text_length} characters")
            
            # Extract keywords using multiple methods
            keywords = []
            if request.use_tfidf:
                tfidf_keywords = self._extract_keywords_tfidf(text, request.num_keywords // 2)
                keywords.extend(tfidf_keywords)
            
            freq_keywords = self._extract_keywords_frequency(text, request.num_keywords // 2)
            keywords.extend(freq_keywords)
            
            # Deduplicate and sort keywords
            keyword_dict = {kw.word: kw for kw in keywords}
            keywords = sorted(keyword_dict.values(), key=lambda x: x.score, reverse=True)[:request.num_keywords]
            
            # Extract topics using LDA
            topics = []
            if request.use_topic_modeling:
                topics = self._extract_topics_lda(text, request.num_topics)
            
            # Extract named entities
            named_entities = []
            if request.extract_named_entities:
                named_entities = self._extract_named_entities(text)
            
            # Extract key phrases
            key_phrases = []
            if request.extract_key_phrases:
                key_phrases = self._extract_key_phrases(text)
            
            # Identify themes
            themes = self._identify_themes(keywords, topics)
            
            # Categorize content
            categories = list(set([kw.category for kw in keywords if kw.category != "general"]))
            
            # Generate summary
            document_summary = self._generate_summary(text)
            
            # Create result
            result = TopicExtractionResult(
                keywords=keywords,
                topics=topics,
                named_entities=named_entities,
                key_phrases=key_phrases,
                themes=themes,
                categories=categories,
                document_summary=document_summary,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TopicExtractionResponse(
                success=True,
                text_length=text_length,
                result=result,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TopicExtractionResponse(
                success=False,
                text_length=len(request.text),
                error=str(e),
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    async def batch_extract_topics(self, texts: List[str]) -> List[TopicExtractionResponse]:
        """
        Extract topics from multiple texts concurrently.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of TopicExtractionResponse objects
        """
        tasks = [self.extract_topics(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch topic extraction failed for text {i}: {result}")
                responses.append(TopicExtractionResponse(
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
topic_extraction_tool = TopicExtractionTool()


# MCP tool function
async def mcp_extract_topics(
    text: str,
    num_keywords: int = 20,
    num_topics: int = 5,
    extract_named_entities: bool = True,
    extract_key_phrases: bool = True,
    use_tfidf: bool = True,
    use_topic_modeling: bool = True,
    min_keyword_length: int = 3,
    language: str = "en",
    remove_stopwords: bool = True
) -> Dict:
    """
    MCP-compatible topic extraction function.
    
    Args:
        text: Text content for topic extraction
        num_keywords: Number of keywords to extract
        num_topics: Number of topics for modeling
        extract_named_entities: Extract named entities
        extract_key_phrases: Extract key phrases
        use_tfidf: Use TF-IDF for keyword extraction
        use_topic_modeling: Perform topic modeling
        min_keyword_length: Minimum keyword length
        language: Content language
        remove_stopwords: Remove stopwords
    
    Returns:
        Dictionary with topic extraction results
    """
    try:
        request = TopicExtractionRequest(
            text=text,
            num_keywords=num_keywords,
            num_topics=num_topics,
            extract_named_entities=extract_named_entities,
            extract_key_phrases=extract_key_phrases,
            use_tfidf=use_tfidf,
            use_topic_modeling=use_topic_modeling,
            min_keyword_length=min_keyword_length,
            language=language,
            remove_stopwords=remove_stopwords
        )
        
        response = await topic_extraction_tool.extract_topics(request)
        
        if response.success and response.result:
            result = response.result
            
            return {
                "success": True,
                "text_length": response.text_length,
                "keywords": [
                    {
                        "word": kw.word,
                        "score": kw.score,
                        "frequency": kw.frequency,
                        "category": kw.category
                    } for kw in result.keywords
                ],
                "topics": [
                    {
                        "topic_id": topic.topic_id,
                        "keywords": topic.keywords[:5],  # Top 5 keywords per topic
                        "weights": topic.weights[:5],
                        "coherence_score": topic.coherence_score,
                        "description": topic.description
                    } for topic in result.topics
                ],
                "named_entities": [
                    {
                        "entity": ne.entity,
                        "label": ne.label,
                        "confidence": ne.confidence
                    } for ne in result.named_entities
                ],
                "key_phrases": result.key_phrases,
                "themes": result.themes,
                "categories": result.categories,
                "document_summary": result.document_summary,
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
        logger.error(f"Topic extraction failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }