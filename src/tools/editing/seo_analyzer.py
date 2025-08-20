"""
SEO Analyzer Tool - Search Engine Optimization Analysis and Recommendations

This module provides comprehensive SEO analysis capabilities including keyword optimization,
content structure analysis, meta tag validation, and search ranking improvement suggestions.

Key Features:
- Keyword density analysis and optimization
- Meta tag optimization (title, description, keywords)
- Content structure recommendations
- Internal/external link analysis
- Image SEO optimization
- Technical SEO checks
- Competitor analysis insights
"""

import asyncio
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal, Any, Tuple
from enum import Enum
from collections import Counter, defaultdict
from urllib.parse import urlparse, urljoin
import string

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textstat import flesch_reading_ease
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
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class SEOIssueType(str, Enum):
    """Types of SEO issues"""
    KEYWORD_DENSITY = "keyword_density"
    META_TAGS = "meta_tags"
    HEADINGS = "headings"
    CONTENT_LENGTH = "content_length"
    INTERNAL_LINKS = "internal_links"
    EXTERNAL_LINKS = "external_links"
    IMAGE_OPTIMIZATION = "image_optimization"
    URL_STRUCTURE = "url_structure"
    READABILITY = "readability"
    DUPLICATE_CONTENT = "duplicate_content"


class SEOPriority(str, Enum):
    """Priority levels for SEO improvements"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ContentType(str, Enum):
    """Content types for SEO optimization"""
    BLOG_POST = "blog_post"
    PRODUCT_PAGE = "product_page"
    LANDING_PAGE = "landing_page"
    CATEGORY_PAGE = "category_page"
    NEWS_ARTICLE = "news_article"
    HOME_PAGE = "home_page"
    ABOUT_PAGE = "about_page"


class SEOIssue(BaseModel):
    """Individual SEO issue or recommendation"""
    
    issue_type: SEOIssueType = Field(..., description="Type of SEO issue")
    priority: SEOPriority = Field(..., description="Priority level")
    
    title: str = Field(..., description="Issue title")
    description: str = Field(..., description="Detailed description")
    recommendation: str = Field(..., description="Specific recommendation")
    
    current_value: Optional[Union[str, int, float]] = Field(default=None, description="Current value")
    target_value: Optional[Union[str, int, float]] = Field(default=None, description="Recommended target value")
    
    impact_score: float = Field(..., description="Potential impact score (0-100)", ge=0, le=100)
    difficulty_score: float = Field(..., description="Implementation difficulty (0-100)", ge=0, le=100)


class KeywordAnalysis(BaseModel):
    """Keyword analysis results"""
    
    keyword: str = Field(..., description="The analyzed keyword")
    density: float = Field(..., description="Keyword density percentage")
    frequency: int = Field(..., description="Number of occurrences")
    
    positions: List[int] = Field(..., description="Character positions in content")
    context_quality: float = Field(..., description="Quality of keyword context (0-100)")
    
    prominence: Dict[str, int] = Field(..., description="Keyword prominence in different sections")
    variations_found: List[str] = Field(..., description="Keyword variations found")
    
    optimization_score: float = Field(..., description="Overall keyword optimization score (0-100)")


class ContentStructure(BaseModel):
    """Content structure analysis"""
    
    word_count: int = Field(..., description="Total word count")
    paragraph_count: int = Field(..., description="Number of paragraphs")
    sentence_count: int = Field(..., description="Number of sentences")
    
    headings: Dict[str, List[str]] = Field(..., description="Headings by level (h1, h2, etc.)")
    heading_structure_score: float = Field(..., description="Heading structure quality (0-100)")
    
    avg_paragraph_length: float = Field(..., description="Average paragraph length in words")
    avg_sentence_length: float = Field(..., description="Average sentence length in words")
    
    readability_score: float = Field(..., description="Flesch reading ease score")
    readability_grade: str = Field(..., description="Reading grade level")


class MetaTagsAnalysis(BaseModel):
    """Meta tags analysis results"""
    
    title_tag: Optional[str] = Field(default=None, description="Page title")
    title_length: int = Field(default=0, description="Title character length")
    title_score: float = Field(..., description="Title optimization score (0-100)")
    
    meta_description: Optional[str] = Field(default=None, description="Meta description")
    description_length: int = Field(default=0, description="Description character length")
    description_score: float = Field(..., description="Description optimization score (0-100)")
    
    meta_keywords: List[str] = Field(default=[], description="Meta keywords (if present)")
    
    og_tags: Dict[str, str] = Field(default={}, description="Open Graph tags")
    twitter_tags: Dict[str, str] = Field(default={}, description="Twitter Card tags")
    
    schema_markup: List[str] = Field(default=[], description="Detected schema markup types")


class LinkAnalysis(BaseModel):
    """Link analysis results"""
    
    internal_links: List[Dict[str, str]] = Field(..., description="Internal links found")
    external_links: List[Dict[str, str]] = Field(..., description="External links found")
    
    internal_link_count: int = Field(..., description="Number of internal links")
    external_link_count: int = Field(..., description="Number of external links")
    
    nofollow_links: int = Field(..., description="Number of nofollow links")
    broken_links: List[str] = Field(default=[], description="Potentially broken links")
    
    anchor_text_analysis: Dict[str, int] = Field(..., description="Anchor text distribution")
    link_density: float = Field(..., description="Link density percentage")


class SEOAnalysisRequest(BaseModel):
    """SEO analysis request parameters"""
    
    content: str = Field(..., description="Content to analyze (HTML or plain text)")
    target_keywords: List[str] = Field(..., description="Primary keywords to optimize for")
    
    content_type: ContentType = Field(default=ContentType.BLOG_POST, description="Type of content")
    target_url: Optional[str] = Field(default=None, description="Target URL for the content")
    
    secondary_keywords: Optional[List[str]] = Field(
        default=None,
        description="Secondary keywords to consider"
    )
    
    competitor_urls: Optional[List[str]] = Field(
        default=None,
        description="Competitor URLs for comparison"
    )
    
    target_audience: Optional[str] = Field(
        default=None,
        description="Target audience description"
    )
    
    language: str = Field(default="en", description="Content language")
    
    include_technical_seo: bool = Field(default=True, description="Include technical SEO checks")
    include_content_gaps: bool = Field(default=True, description="Analyze content gaps")
    
    custom_rules: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom SEO rules and thresholds"
    )

    @validator('target_keywords')
    def validate_keywords(cls, v):
        if not v:
            raise ValueError("At least one target keyword is required")
        if len(v) > 20:
            raise ValueError("Maximum 20 target keywords allowed")
        return v

    @validator('content')
    def validate_content_length(cls, v):
        if len(v.strip()) < 50:
            raise ValueError("Content too short for meaningful SEO analysis")
        return v


class SEOAnalysisResults(BaseModel):
    """Complete SEO analysis results"""
    
    overall_score: float = Field(..., description="Overall SEO score (0-100)")
    issues: List[SEOIssue] = Field(..., description="Identified SEO issues")
    
    keyword_analysis: Dict[str, KeywordAnalysis] = Field(..., description="Keyword analysis results")
    content_structure: ContentStructure = Field(..., description="Content structure analysis")
    meta_tags: MetaTagsAnalysis = Field(..., description="Meta tags analysis")
    link_analysis: LinkAnalysis = Field(..., description="Link analysis")
    
    recommendations: List[str] = Field(..., description="Priority recommendations")
    quick_wins: List[str] = Field(..., description="Easy improvements")
    
    competitor_insights: Optional[Dict[str, Any]] = Field(default=None, description="Competitor comparison")
    
    processing_time: float = Field(..., description="Analysis processing time")
    analysis_timestamp: datetime = Field(..., description="When analysis was performed")


class SEOBestPractices:
    """SEO best practices and thresholds"""
    
    KEYWORD_DENSITY_RANGES = {
        "primary": (1.0, 3.0),      # 1-3% for primary keywords
        "secondary": (0.5, 2.0),    # 0.5-2% for secondary keywords
    }
    
    CONTENT_LENGTH_TARGETS = {
        ContentType.BLOG_POST: (1000, 3000),
        ContentType.PRODUCT_PAGE: (300, 1000),
        ContentType.LANDING_PAGE: (500, 2000),
        ContentType.CATEGORY_PAGE: (200, 800),
        ContentType.NEWS_ARTICLE: (400, 1200),
        ContentType.HOME_PAGE: (200, 800),
        ContentType.ABOUT_PAGE: (300, 800)
    }
    
    META_TAG_LIMITS = {
        "title_min": 30,
        "title_max": 60,
        "description_min": 120,
        "description_max": 160
    }
    
    READABILITY_TARGETS = {
        "min_score": 60,  # Flesch Reading Ease
        "max_grade": 10   # Grade level
    }
    
    HEADING_STRUCTURE = {
        "h1_count": (1, 1),     # Exactly one H1
        "h2_min": 2,            # At least 2 H2s for long content
        "max_heading_length": 70  # Characters
    }
    
    LINK_GUIDELINES = {
        "internal_links_per_1000_words": (2, 5),
        "external_links_per_1000_words": (1, 3),
        "max_link_density": 5.0  # Percentage
    }


class SEOAnalyzer:
    """Comprehensive SEO analysis and optimization tool"""
    
    def __init__(self):
        """Initialize SEO Analyzer"""
        
        # Load English stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            self.stopwords = set()
        
        # Initialize stemmer for keyword variations
        self.stemmer = PorterStemmer()
        
        # SEO rules cache
        self.rules_cache = {}
        
        logger.info("SEOAnalyzer initialized")

    def _extract_text_from_html(self, content: str) -> Tuple[str, BeautifulSoup]:
        """Extract text content from HTML and return both text and parsed HTML"""
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text, soup
            
        except Exception:
            # If HTML parsing fails, treat as plain text
            return content, None

    def _analyze_keywords(self, text: str, keywords: List[str], secondary_keywords: List[str] = None) -> Dict[str, KeywordAnalysis]:
        """Analyze keyword usage and optimization"""
        
        results = {}
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        text_length = len(words)
        
        # Analyze primary keywords
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Count occurrences (exact and partial matches)
            exact_matches = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', text_lower))
            
            # Find positions
            positions = [m.start() for m in re.finditer(r'\b' + re.escape(keyword_lower) + r'\b', text_lower)]
            
            # Calculate density
            keyword_words = len(keyword_lower.split())
            density = (exact_matches * keyword_words / text_length * 100) if text_length > 0 else 0
            
            # Find variations (stemmed versions)
            variations = self._find_keyword_variations(text_lower, keyword_lower)
            
            # Analyze context quality
            context_quality = self._analyze_keyword_context(text, keyword, positions)
            
            # Analyze prominence in different sections
            prominence = self._analyze_keyword_prominence(text, keyword)
            
            # Calculate optimization score
            optimization_score = self._calculate_keyword_optimization_score(
                density, context_quality, prominence, len(positions)
            )
            
            results[keyword] = KeywordAnalysis(
                keyword=keyword,
                density=density,
                frequency=exact_matches,
                positions=positions,
                context_quality=context_quality,
                prominence=prominence,
                variations_found=variations,
                optimization_score=optimization_score
            )
        
        # Analyze secondary keywords if provided
        if secondary_keywords:
            for keyword in secondary_keywords:
                if keyword not in results:  # Avoid duplicates
                    keyword_lower = keyword.lower()
                    exact_matches = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', text_lower))
                    positions = [m.start() for m in re.finditer(r'\b' + re.escape(keyword_lower) + r'\b', text_lower)]
                    
                    keyword_words = len(keyword_lower.split())
                    density = (exact_matches * keyword_words / text_length * 100) if text_length > 0 else 0
                    
                    context_quality = self._analyze_keyword_context(text, keyword, positions)
                    prominence = self._analyze_keyword_prominence(text, keyword)
                    optimization_score = self._calculate_keyword_optimization_score(
                        density, context_quality, prominence, len(positions), is_secondary=True
                    )
                    
                    results[keyword] = KeywordAnalysis(
                        keyword=keyword,
                        density=density,
                        frequency=exact_matches,
                        positions=positions,
                        context_quality=context_quality,
                        prominence=prominence,
                        variations_found=self._find_keyword_variations(text_lower, keyword_lower),
                        optimization_score=optimization_score
                    )
        
        return results

    def _find_keyword_variations(self, text: str, keyword: str) -> List[str]:
        """Find variations of a keyword in the text"""
        
        variations = []
        keyword_words = keyword.split()
        
        # Look for stemmed versions and common variations
        for word in keyword_words:
            stem = self.stemmer.stem(word)
            
            # Find words with the same stem
            text_words = word_tokenize(text)
            for text_word in set(text_words):
                if self.stemmer.stem(text_word) == stem and text_word != word:
                    variations.append(text_word)
        
        return list(set(variations))

    def _analyze_keyword_context(self, text: str, keyword: str, positions: List[int]) -> float:
        """Analyze the quality of keyword context"""
        
        if not positions:
            return 0.0
        
        context_scores = []
        
        for pos in positions:
            # Get surrounding context (50 chars before and after)
            start = max(0, pos - 50)
            end = min(len(text), pos + len(keyword) + 50)
            context = text[start:end].lower()
            
            # Score based on context quality
            score = 50.0  # Base score
            
            # Bonus for being in headings or emphasized text
            if any(tag in context for tag in ['<h1', '<h2', '<h3', '<strong', '<b>', '<em>']):
                score += 20
            
            # Bonus for being near related keywords
            related_words = ['tips', 'guide', 'how', 'best', 'top', 'benefits', 'advantages']
            if any(word in context for word in related_words):
                score += 10
            
            # Penalty for keyword stuffing (nearby duplicate)
            keyword_lower = keyword.lower()
            if context.count(keyword_lower) > 1:
                score -= 15
            
            context_scores.append(min(100.0, score))
        
        return sum(context_scores) / len(context_scores)

    def _analyze_keyword_prominence(self, text: str, keyword: str) -> Dict[str, int]:
        """Analyze keyword prominence in different content sections"""
        
        prominence = {
            "title": 0,
            "headings": 0,
            "first_paragraph": 0,
            "last_paragraph": 0,
            "meta_description": 0
        }
        
        keyword_lower = keyword.lower()
        
        # Check in title (first line or H1)
        lines = text.split('\n')
        if lines and keyword_lower in lines[0].lower():
            prominence["title"] = 1
        
        # Check in headings (lines that might be headings based on caps or length)
        for line in lines:
            if len(line) < 100 and (line.isupper() or line.istitle()):
                if keyword_lower in line.lower():
                    prominence["headings"] += 1
        
        # Check in first paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs and keyword_lower in paragraphs[0].lower():
            prominence["first_paragraph"] = 1
        
        # Check in last paragraph
        if len(paragraphs) > 1 and keyword_lower in paragraphs[-1].lower():
            prominence["last_paragraph"] = 1
        
        return prominence

    def _calculate_keyword_optimization_score(
        self, 
        density: float, 
        context_quality: float, 
        prominence: Dict[str, int], 
        frequency: int,
        is_secondary: bool = False
    ) -> float:
        """Calculate overall keyword optimization score"""
        
        score = 0.0
        
        # Density score (40% weight)
        target_range = SEOBestPractices.KEYWORD_DENSITY_RANGES["secondary" if is_secondary else "primary"]
        if target_range[0] <= density <= target_range[1]:
            density_score = 100.0
        elif density < target_range[0]:
            density_score = (density / target_range[0]) * 100
        else:
            # Penalty for over-optimization
            density_score = max(0, 100 - (density - target_range[1]) * 20)
        
        score += density_score * 0.4
        
        # Context quality score (30% weight)
        score += context_quality * 0.3
        
        # Prominence score (20% weight)
        prominence_score = 0
        if prominence["title"]: prominence_score += 30
        if prominence["headings"]: prominence_score += 20
        if prominence["first_paragraph"]: prominence_score += 15
        if prominence["last_paragraph"]: prominence_score += 10
        
        score += min(100, prominence_score) * 0.2
        
        # Frequency score (10% weight)
        frequency_score = min(100, frequency * 25) if frequency > 0 else 0
        score += frequency_score * 0.1
        
        return min(100.0, score)

    def _analyze_content_structure(self, text: str, soup: BeautifulSoup = None) -> ContentStructure:
        """Analyze content structure and organization"""
        
        # Basic metrics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        word_count = len(words)
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        
        # Calculate averages
        avg_paragraph_length = word_count / max(1, paragraph_count)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Headings analysis
        headings = {"h1": [], "h2": [], "h3": [], "h4": [], "h5": [], "h6": []}
        
        if soup:
            # Extract from HTML
            for level in range(1, 7):
                heading_tags = soup.find_all(f'h{level}')
                headings[f'h{level}'] = [tag.get_text().strip() for tag in heading_tags]
        else:
            # Heuristic heading detection for plain text
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) < 100 and len(line) > 0:
                    if line.isupper():
                        headings["h1"].append(line)
                    elif line.istitle() and len(line) < 80:
                        headings["h2"].append(line)
        
        # Heading structure score
        heading_structure_score = self._score_heading_structure(headings, word_count)
        
        # Readability
        try:
            readability_score = flesch_reading_ease(text)
            readability_grade = self._flesch_to_grade_level(readability_score)
        except:
            readability_score = 50.0
            readability_grade = "Unknown"
        
        return ContentStructure(
            word_count=word_count,
            paragraph_count=paragraph_count,
            sentence_count=sentence_count,
            headings=headings,
            heading_structure_score=heading_structure_score,
            avg_paragraph_length=avg_paragraph_length,
            avg_sentence_length=avg_sentence_length,
            readability_score=readability_score,
            readability_grade=readability_grade
        )

    def _score_heading_structure(self, headings: Dict[str, List[str]], word_count: int) -> float:
        """Score heading structure quality"""
        
        score = 100.0
        
        # Check H1 count (should have exactly 1)
        h1_count = len(headings["h1"])
        if h1_count == 0:
            score -= 30
        elif h1_count > 1:
            score -= 20
        
        # Check H2 presence for longer content
        if word_count > 500 and len(headings["h2"]) < 2:
            score -= 15
        
        # Check heading hierarchy
        for level in range(2, 7):
            h_level = f"h{level}"
            prev_level = f"h{level-1}"
            
            if len(headings[h_level]) > 0 and len(headings[prev_level]) == 0:
                score -= 10  # Skipped hierarchy level
        
        # Check heading length
        for level_headings in headings.values():
            for heading in level_headings:
                if len(heading) > SEOBestPractices.HEADING_STRUCTURE["max_heading_length"]:
                    score -= 5
        
        return max(0.0, score)

    def _flesch_to_grade_level(self, flesch_score: float) -> str:
        """Convert Flesch Reading Ease score to grade level"""
        
        if flesch_score >= 90:
            return "5th grade"
        elif flesch_score >= 80:
            return "6th grade"
        elif flesch_score >= 70:
            return "7th grade"
        elif flesch_score >= 60:
            return "8th-9th grade"
        elif flesch_score >= 50:
            return "10th-12th grade"
        elif flesch_score >= 30:
            return "College level"
        else:
            return "Graduate level"

    def _analyze_meta_tags(self, soup: BeautifulSoup, keywords: List[str]) -> MetaTagsAnalysis:
        """Analyze meta tags for SEO optimization"""
        
        if not soup:
            return MetaTagsAnalysis(
                title_score=0.0,
                description_score=0.0
            )
        
        # Title tag analysis
        title_tag = soup.find('title')
        title_text = title_tag.get_text().strip() if title_tag else None
        title_length = len(title_text) if title_text else 0
        
        title_score = self._score_title_tag(title_text, keywords, title_length)
        
        # Meta description analysis
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        desc_text = meta_desc.get('content').strip() if meta_desc and meta_desc.get('content') else None
        desc_length = len(desc_text) if desc_text else 0
        
        description_score = self._score_meta_description(desc_text, keywords, desc_length)
        
        # Meta keywords (deprecated but still check)
        meta_keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        meta_keywords = []
        if meta_keywords_tag and meta_keywords_tag.get('content'):
            meta_keywords = [kw.strip() for kw in meta_keywords_tag.get('content').split(',')]
        
        # Open Graph tags
        og_tags = {}
        og_metas = soup.find_all('meta', attrs={'property': re.compile(r'^og:')})
        for meta in og_metas:
            prop = meta.get('property')
            content = meta.get('content')
            if prop and content:
                og_tags[prop] = content
        
        # Twitter Card tags
        twitter_tags = {}
        twitter_metas = soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')})
        for meta in twitter_metas:
            name = meta.get('name')
            content = meta.get('content')
            if name and content:
                twitter_tags[name] = content
        
        # Schema markup detection
        schema_markup = []
        script_tags = soup.find_all('script', attrs={'type': 'application/ld+json'})
        for script in script_tags:
            try:
                import json
                data = json.loads(script.string)
                if '@type' in data:
                    schema_markup.append(data['@type'])
            except:
                pass
        
        return MetaTagsAnalysis(
            title_tag=title_text,
            title_length=title_length,
            title_score=title_score,
            meta_description=desc_text,
            description_length=desc_length,
            description_score=description_score,
            meta_keywords=meta_keywords,
            og_tags=og_tags,
            twitter_tags=twitter_tags,
            schema_markup=schema_markup
        )

    def _score_title_tag(self, title: str, keywords: List[str], length: int) -> float:
        """Score title tag optimization"""
        
        if not title:
            return 0.0
        
        score = 100.0
        title_lower = title.lower()
        
        # Length check
        if length < SEOBestPractices.META_TAG_LIMITS["title_min"]:
            score -= 20
        elif length > SEOBestPractices.META_TAG_LIMITS["title_max"]:
            score -= 15
        
        # Keyword inclusion
        primary_keyword_found = False
        for keyword in keywords:
            if keyword.lower() in title_lower:
                primary_keyword_found = True
                break
        
        if not primary_keyword_found:
            score -= 30
        
        # Keyword position (earlier is better)
        for keyword in keywords:
            pos = title_lower.find(keyword.lower())
            if pos != -1:
                if pos == 0:
                    score += 10  # Keyword at the beginning
                elif pos < len(title) / 2:
                    score += 5   # Keyword in first half
                break
        
        return max(0.0, score)

    def _score_meta_description(self, description: str, keywords: List[str], length: int) -> float:
        """Score meta description optimization"""
        
        if not description:
            return 0.0
        
        score = 100.0
        desc_lower = description.lower()
        
        # Length check
        if length < SEOBestPractices.META_TAG_LIMITS["description_min"]:
            score -= 25
        elif length > SEOBestPractices.META_TAG_LIMITS["description_max"]:
            score -= 20
        
        # Keyword inclusion
        keywords_found = 0
        for keyword in keywords:
            if keyword.lower() in desc_lower:
                keywords_found += 1
        
        if keywords_found == 0:
            score -= 30
        elif keywords_found >= 2:
            score += 10  # Bonus for multiple keywords
        
        # Call-to-action words
        cta_words = ['learn', 'discover', 'find out', 'get', 'try', 'start', 'buy', 'download']
        if any(word in desc_lower for word in cta_words):
            score += 5
        
        return max(0.0, score)

    def _analyze_links(self, soup: BeautifulSoup, base_url: str = None) -> LinkAnalysis:
        """Analyze internal and external links"""
        
        if not soup:
            return LinkAnalysis(
                internal_links=[],
                external_links=[],
                internal_link_count=0,
                external_link_count=0,
                nofollow_links=0,
                anchor_text_analysis={},
                link_density=0.0
            )
        
        all_links = soup.find_all('a', href=True)
        internal_links = []
        external_links = []
        nofollow_count = 0
        anchor_texts = []
        
        base_domain = None
        if base_url:
            base_domain = urlparse(base_url).netloc
        
        for link in all_links:
            href = link.get('href', '').strip()
            anchor_text = link.get_text().strip()
            rel = link.get('rel', [])
            
            if not href or href.startswith('#'):
                continue
            
            anchor_texts.append(anchor_text)
            
            # Check for nofollow
            if 'nofollow' in rel:
                nofollow_count += 1
            
            # Determine if internal or external
            parsed_url = urlparse(href)
            
            link_info = {
                'url': href,
                'anchor_text': anchor_text,
                'rel': rel,
                'title': link.get('title', '')
            }
            
            if parsed_url.netloc == '' or (base_domain and parsed_url.netloc == base_domain):
                internal_links.append(link_info)
            else:
                external_links.append(link_info)
        
        # Analyze anchor text distribution
        anchor_text_counter = Counter(anchor_texts)
        anchor_text_analysis = dict(anchor_text_counter.most_common(10))
        
        # Calculate link density
        text_content = soup.get_text()
        text_words = len(word_tokenize(text_content))
        link_density = (len(all_links) / max(1, text_words)) * 100
        
        return LinkAnalysis(
            internal_links=internal_links,
            external_links=external_links,
            internal_link_count=len(internal_links),
            external_link_count=len(external_links),
            nofollow_links=nofollow_count,
            anchor_text_analysis=anchor_text_analysis,
            link_density=link_density
        )

    def _generate_seo_issues(
        self, 
        request: SEOAnalysisRequest,
        keyword_analysis: Dict[str, KeywordAnalysis],
        content_structure: ContentStructure,
        meta_tags: MetaTagsAnalysis,
        link_analysis: LinkAnalysis
    ) -> List[SEOIssue]:
        """Generate list of SEO issues and recommendations"""
        
        issues = []
        
        # Keyword density issues
        for keyword, analysis in keyword_analysis.items():
            if analysis.density < 1.0:
                issues.append(SEOIssue(
                    issue_type=SEOIssueType.KEYWORD_DENSITY,
                    priority=SEOPriority.HIGH,
                    title=f"Low keyword density for '{keyword}'",
                    description=f"The keyword '{keyword}' appears only {analysis.frequency} times ({analysis.density:.1f}% density)",
                    recommendation=f"Increase keyword usage to 1-3% density. Add {keyword} naturally in headings, first paragraph, and throughout content.",
                    current_value=analysis.density,
                    target_value=2.0,
                    impact_score=80,
                    difficulty_score=20
                ))
            elif analysis.density > 4.0:
                issues.append(SEOIssue(
                    issue_type=SEOIssueType.KEYWORD_DENSITY,
                    priority=SEOPriority.MEDIUM,
                    title=f"High keyword density for '{keyword}'",
                    description=f"The keyword '{keyword}' density is {analysis.density:.1f}%, which may be considered keyword stuffing",
                    recommendation=f"Reduce keyword usage and use variations or synonyms. Aim for 1-3% density.",
                    current_value=analysis.density,
                    target_value=2.0,
                    impact_score=60,
                    difficulty_score=30
                ))
        
        # Meta tags issues
        if meta_tags.title_score < 70:
            issues.append(SEOIssue(
                issue_type=SEOIssueType.META_TAGS,
                priority=SEOPriority.CRITICAL,
                title="Title tag optimization needed",
                description=f"Title tag score is {meta_tags.title_score:.0f}/100",
                recommendation="Optimize title tag: include primary keyword near the beginning, keep 30-60 characters, make it compelling",
                current_value=meta_tags.title_length,
                target_value=50,
                impact_score=95,
                difficulty_score=15
            ))
        
        if meta_tags.description_score < 70:
            issues.append(SEOIssue(
                issue_type=SEOIssueType.META_TAGS,
                priority=SEOPriority.HIGH,
                title="Meta description optimization needed",
                description=f"Meta description score is {meta_tags.description_score:.0f}/100",
                recommendation="Optimize meta description: include keywords, add call-to-action, keep 120-160 characters",
                current_value=meta_tags.description_length,
                target_value=150,
                impact_score=75,
                difficulty_score=20
            ))
        
        # Content length issues
        target_length = SEOBestPractices.CONTENT_LENGTH_TARGETS.get(request.content_type, (800, 2000))
        if content_structure.word_count < target_length[0]:
            issues.append(SEOIssue(
                issue_type=SEOIssueType.CONTENT_LENGTH,
                priority=SEOPriority.HIGH,
                title="Content too short",
                description=f"Content has {content_structure.word_count} words, below recommended minimum of {target_length[0]}",
                recommendation=f"Expand content to at least {target_length[0]} words. Add more detailed explanations, examples, or sections.",
                current_value=content_structure.word_count,
                target_value=target_length[0],
                impact_score=70,
                difficulty_score=50
            ))
        
        # Heading structure issues
        if content_structure.heading_structure_score < 80:
            h1_count = len(content_structure.headings["h1"])
            if h1_count == 0:
                issues.append(SEOIssue(
                    issue_type=SEOIssueType.HEADINGS,
                    priority=SEOPriority.HIGH,
                    title="Missing H1 heading",
                    description="No H1 heading found in content",
                    recommendation="Add exactly one H1 heading that includes your primary keyword",
                    current_value=0,
                    target_value=1,
                    impact_score=85,
                    difficulty_score=15
                ))
            elif h1_count > 1:
                issues.append(SEOIssue(
                    issue_type=SEOIssueType.HEADINGS,
                    priority=SEOPriority.MEDIUM,
                    title="Multiple H1 headings",
                    description=f"Found {h1_count} H1 headings, should have exactly one",
                    recommendation="Use only one H1 heading per page. Convert additional H1s to H2 or H3",
                    current_value=h1_count,
                    target_value=1,
                    impact_score=60,
                    difficulty_score=25
                ))
        
        # Readability issues
        if content_structure.readability_score < SEOBestPractices.READABILITY_TARGETS["min_score"]:
            issues.append(SEOIssue(
                issue_type=SEOIssueType.READABILITY,
                priority=SEOPriority.MEDIUM,
                title="Poor readability score",
                description=f"Flesch Reading Ease score is {content_structure.readability_score:.1f}, below recommended minimum of 60",
                recommendation="Improve readability: use shorter sentences, simpler words, more paragraphs, and bullet points",
                current_value=content_structure.readability_score,
                target_value=70.0,
                impact_score=55,
                difficulty_score=40
            ))
        
        # Link issues
        words_per_thousand = content_structure.word_count / 1000
        expected_internal_links = int(words_per_thousand * 3)  # ~3 internal links per 1000 words
        
        if link_analysis.internal_link_count < expected_internal_links:
            issues.append(SEOIssue(
                issue_type=SEOIssueType.INTERNAL_LINKS,
                priority=SEOPriority.MEDIUM,
                title="Insufficient internal links",
                description=f"Only {link_analysis.internal_link_count} internal links found, recommend ~{expected_internal_links} for this content length",
                recommendation="Add more internal links to related content, use descriptive anchor text",
                current_value=link_analysis.internal_link_count,
                target_value=expected_internal_links,
                impact_score=50,
                difficulty_score=30
            ))
        
        return issues

    async def analyze_seo(self, request: SEOAnalysisRequest) -> SEOAnalysisResults:
        """
        Perform comprehensive SEO analysis
        
        Args:
            request: SEO analysis request parameters
            
        Returns:
            Complete SEO analysis results with issues and recommendations
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting SEO analysis for {len(request.content)} characters of content")
            
            # Extract text and parse HTML if applicable
            text, soup = self._extract_text_from_html(request.content)
            
            # Perform analysis components
            keyword_analysis = self._analyze_keywords(
                text, 
                request.target_keywords, 
                request.secondary_keywords or []
            )
            
            content_structure = self._analyze_content_structure(text, soup)
            
            meta_tags = self._analyze_meta_tags(soup, request.target_keywords)
            
            link_analysis = self._analyze_links(soup, request.target_url)
            
            # Generate SEO issues and recommendations
            issues = self._generate_seo_issues(
                request, keyword_analysis, content_structure, meta_tags, link_analysis
            )
            
            # Calculate overall SEO score
            component_scores = [
                sum(ka.optimization_score for ka in keyword_analysis.values()) / max(1, len(keyword_analysis)),
                content_structure.heading_structure_score,
                (meta_tags.title_score + meta_tags.description_score) / 2,
                min(100, content_structure.readability_score * 1.5),  # Readability contribution
            ]
            
            overall_score = sum(component_scores) / len(component_scores)
            
            # Adjust score based on critical issues
            critical_issues = [i for i in issues if i.priority == SEOPriority.CRITICAL]
            overall_score = max(0, overall_score - len(critical_issues) * 15)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues, overall_score)
            quick_wins = self._generate_quick_wins(issues)
            
            # Processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = SEOAnalysisResults(
                overall_score=overall_score,
                issues=issues,
                keyword_analysis=keyword_analysis,
                content_structure=content_structure,
                meta_tags=meta_tags,
                link_analysis=link_analysis,
                recommendations=recommendations,
                quick_wins=quick_wins,
                processing_time=processing_time,
                analysis_timestamp=datetime.now()
            )
            
            logger.info(f"SEO analysis completed: score {overall_score:.1f}/100, {len(issues)} issues found")
            return result
            
        except Exception as e:
            logger.error(f"SEO analysis failed: {str(e)}")
            raise ToolError(f"SEO analysis failed: {str(e)}")

    def _generate_recommendations(self, issues: List[SEOIssue], overall_score: float) -> List[str]:
        """Generate priority recommendations based on issues"""
        
        recommendations = []
        
        # Sort issues by impact score
        high_impact_issues = sorted(
            [i for i in issues if i.impact_score >= 70], 
            key=lambda x: x.impact_score, 
            reverse=True
        )
        
        for issue in high_impact_issues[:5]:  # Top 5 high-impact issues
            recommendations.append(f"{issue.title}: {issue.recommendation}")
        
        # Add general recommendations based on score
        if overall_score < 50:
            recommendations.insert(0, "Focus on fundamental SEO: optimize title tags, add primary keywords, improve content structure")
        elif overall_score < 70:
            recommendations.insert(0, "Good foundation, now focus on advanced optimization: internal linking, content depth, user experience")
        
        return recommendations

    def _generate_quick_wins(self, issues: List[SEOIssue]) -> List[str]:
        """Generate quick win recommendations (high impact, low difficulty)"""
        
        quick_wins = []
        
        for issue in issues:
            if issue.impact_score >= 60 and issue.difficulty_score <= 30:
                quick_wins.append(f"{issue.title}: {issue.recommendation}")
        
        # Limit to most impactful quick wins
        return quick_wins[:3]


# Initialize tool instance
seo_analyzer_tool = SEOAnalyzer()


# MCP Functions for external integration
async def mcp_analyze_seo(
    content: str,
    target_keywords: List[str],
    content_type: str = "blog_post",
    target_url: Optional[str] = None,
    secondary_keywords: Optional[List[str]] = None,
    include_technical_seo: bool = True
) -> Dict[str, Any]:
    """
    MCP function to perform SEO analysis
    
    Args:
        content: Content to analyze (HTML or plain text)
        target_keywords: Primary keywords to optimize for
        content_type: Type of content (blog_post, product_page, etc.)
        target_url: Target URL for the content
        secondary_keywords: Secondary keywords to consider
        include_technical_seo: Include technical SEO checks
        
    Returns:
        Complete SEO analysis results
    """
    try:
        request = SEOAnalysisRequest(
            content=content,
            target_keywords=target_keywords,
            content_type=ContentType(content_type),
            target_url=target_url,
            secondary_keywords=secondary_keywords,
            include_technical_seo=include_technical_seo
        )
        
        result = await seo_analyzer_tool.analyze_seo(request)
        
        return {
            "success": True,
            "overall_score": result.overall_score,
            "issues": [
                {
                    "type": issue.issue_type.value,
                    "priority": issue.priority.value,
                    "title": issue.title,
                    "description": issue.description,
                    "recommendation": issue.recommendation,
                    "current_value": issue.current_value,
                    "target_value": issue.target_value,
                    "impact_score": issue.impact_score,
                    "difficulty_score": issue.difficulty_score
                }
                for issue in result.issues
            ],
            "keyword_analysis": {
                keyword: {
                    "density": analysis.density,
                    "frequency": analysis.frequency,
                    "optimization_score": analysis.optimization_score,
                    "context_quality": analysis.context_quality,
                    "prominence": analysis.prominence
                }
                for keyword, analysis in result.keyword_analysis.items()
            },
            "content_structure": {
                "word_count": result.content_structure.word_count,
                "readability_score": result.content_structure.readability_score,
                "readability_grade": result.content_structure.readability_grade,
                "heading_structure_score": result.content_structure.heading_structure_score,
                "headings": result.content_structure.headings
            },
            "meta_tags": {
                "title_score": result.meta_tags.title_score,
                "description_score": result.meta_tags.description_score,
                "title_tag": result.meta_tags.title_tag,
                "meta_description": result.meta_tags.meta_description
            },
            "link_analysis": {
                "internal_link_count": result.link_analysis.internal_link_count,
                "external_link_count": result.link_analysis.external_link_count,
                "link_density": result.link_analysis.link_density
            },
            "recommendations": result.recommendations,
            "quick_wins": result.quick_wins,
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        logger.error(f"MCP SEO analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Example usage and testing
    async def test_seo_analyzer():
        """Test the SEO analyzer functionality"""
        
        test_content = """
        <html>
        <head>
            <title>Best SEO Practices for Content Marketing</title>
            <meta name="description" content="Learn the most effective SEO strategies for content marketing in 2024. Improve your search rankings with proven techniques.">
        </head>
        <body>
            <h1>SEO Best Practices for Content Marketing</h1>
            <p>Search engine optimization is crucial for content marketing success. In this comprehensive guide, we'll explore the most effective SEO strategies.</p>
            
            <h2>Keyword Research and Optimization</h2>
            <p>Proper keyword research forms the foundation of successful SEO. Content marketing requires understanding what your audience searches for.</p>
            
            <h2>Content Structure and Organization</h2>
            <p>Well-structured content helps both users and search engines understand your message. Use headings, bullet points, and clear paragraphs.</p>
            
            <p>Remember to focus on SEO best practices while maintaining high-quality, valuable content for your readers.</p>
        </body>
        </html>
        """
        
        request = SEOAnalysisRequest(
            content=test_content,
            target_keywords=["SEO", "content marketing", "search engine optimization"],
            content_type=ContentType.BLOG_POST,
            secondary_keywords=["SEO strategies", "keyword research"]
        )
        
        try:
            analyzer = SEOAnalyzer()
            result = await analyzer.analyze_seo(request)
            
            print(f"SEO Analysis Results:")
            print(f"Overall Score: {result.overall_score:.1f}/100")
            print(f"Issues Found: {len(result.issues)}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print()
            
            print("Keyword Analysis:")
            for keyword, analysis in result.keyword_analysis.items():
                print(f"- {keyword}: {analysis.density:.1f}% density, {analysis.optimization_score:.1f}/100 score")
            print()
            
            print("Top Issues:")
            for i, issue in enumerate(result.issues[:3], 1):
                print(f"{i}. {issue.title} ({issue.priority.value})")
                print(f"   {issue.recommendation}")
            print()
            
            print("Quick Wins:")
            for win in result.quick_wins:
                print(f"- {win}")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_seo_analyzer())