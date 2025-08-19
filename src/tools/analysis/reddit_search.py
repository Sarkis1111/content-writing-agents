"""
Reddit Search Tool for Reddit API integration and subreddit analysis.

Provides comprehensive Reddit data extraction including subreddit analysis,
comment sentiment tracking, and community insights for content research.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Union, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from collections import Counter, defaultdict

import praw
from praw.models import Submission, Comment, Subreddit
from pydantic import BaseModel, Field

from ...core.config.loader import get_settings
from ...core.errors import ToolError, APIError
from ...utils.simple_retry import with_retry


logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Reddit post/submission data."""
    id: str
    title: str
    content: str
    author: str
    subreddit: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    url: str
    flair: Optional[str] = None
    is_self: bool = True
    selftext: str = ""
    domain: str = ""
    permalink: str = ""


@dataclass
class RedditComment:
    """Reddit comment data."""
    id: str
    body: str
    author: str
    score: int
    created_utc: datetime
    parent_id: str
    is_submitter: bool = False
    depth: int = 0
    permalink: str = ""


@dataclass
class SubredditInfo:
    """Subreddit information and statistics."""
    name: str
    display_name: str
    title: str
    description: str
    subscribers: int
    active_users: int
    created_utc: datetime
    is_over_18: bool
    public_description: str
    subreddit_type: str
    lang: str


@dataclass
class SubredditAnalysis:
    """Comprehensive subreddit analysis."""
    subreddit_info: SubredditInfo
    top_posts: List[RedditPost]
    trending_topics: List[str]
    common_keywords: List[str]
    sentiment_distribution: Dict[str, int]
    engagement_metrics: Dict[str, float]
    posting_patterns: Dict[str, int]
    user_activity: Dict[str, int]


class RedditSearchRequest(BaseModel):
    """Reddit search request configuration."""
    query: Optional[str] = Field(default=None, description="Search query")
    subreddit: Optional[str] = Field(default=None, description="Specific subreddit to search")
    limit: int = Field(default=25, ge=1, le=100, description="Number of results to retrieve")
    sort: str = Field(default="relevance", description="Sort method (relevance, hot, new, top)")
    time_filter: str = Field(default="all", description="Time filter (all, day, week, month, year)")
    include_comments: bool = Field(default=False, description="Include comments in results")
    min_score: int = Field(default=0, description="Minimum post score")
    max_age_days: Optional[int] = Field(default=None, description="Maximum age in days")


class RedditSearchResponse(BaseModel):
    """Reddit search response."""
    success: bool
    query: Optional[str]
    subreddit: Optional[str]
    posts: List[RedditPost]
    comments: List[RedditComment]
    total_results: int
    search_time: float
    timestamp: datetime
    error: Optional[str] = None


class RedditSearchTool:
    """
    Reddit Search Tool for comprehensive Reddit data analysis.
    
    Provides subreddit exploration, post/comment analysis, sentiment tracking,
    and community insights for content research and trend analysis.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize Reddit API client
        self._init_reddit_api()
        
        # Analysis parameters
        self.sentiment_keywords = {
            'positive': ['great', 'awesome', 'amazing', 'excellent', 'love', 'best', 'good', 
                        'fantastic', 'wonderful', 'perfect', 'happy', 'glad', 'thanks'],
            'negative': ['bad', 'terrible', 'awful', 'worst', 'hate', 'sucks', 'horrible', 
                        'disappointed', 'angry', 'frustrated', 'useless', 'stupid', 'annoying'],
            'neutral': ['ok', 'okay', 'fine', 'alright', 'decent', 'average', 'normal', 'standard']
        }
        
        # Common stop words for keyword extraction
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'reddit', 'post',
            'comment', 'thread', 'subreddit'
        }
        
        # Cache for subreddit info
        self.subreddit_cache: Dict[str, SubredditInfo] = {}
        self.cache_ttl = 3600  # 1 hour
    
    def _init_reddit_api(self):
        """Initialize Reddit API client (PRAW)."""
        try:
            # Get Reddit API credentials
            client_id = self.settings.get("REDDIT_CLIENT_ID")
            client_secret = self.settings.get("REDDIT_CLIENT_SECRET")
            user_agent = self.settings.get("REDDIT_USER_AGENT", "ContentWritingAgent/1.0")
            
            if not client_id or not client_secret:
                logger.warning("Reddit API credentials not found. Reddit search will be unavailable.")
                self.reddit = None
                return
            
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            # Test the connection
            self.reddit.auth.read_only()
            logger.info("Reddit API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Reddit API initialization failed: {e}")
            self.reddit = None
    
    def _parse_submission(self, submission: Submission) -> RedditPost:
        """Parse Reddit submission into RedditPost object."""
        try:
            return RedditPost(
                id=submission.id,
                title=submission.title,
                content=submission.selftext or "",
                author=str(submission.author) if submission.author else "[deleted]",
                subreddit=str(submission.subreddit),
                score=submission.score,
                upvote_ratio=submission.upvote_ratio,
                num_comments=submission.num_comments,
                created_utc=datetime.fromtimestamp(submission.created_utc),
                url=submission.url,
                flair=submission.link_flair_text,
                is_self=submission.is_self,
                selftext=submission.selftext or "",
                domain=submission.domain,
                permalink=f"https://reddit.com{submission.permalink}"
            )
        except Exception as e:
            logger.error(f"Failed to parse submission {submission.id}: {e}")
            return None
    
    def _parse_comment(self, comment: Comment, depth: int = 0) -> Optional[RedditComment]:
        """Parse Reddit comment into RedditComment object."""
        try:
            if hasattr(comment, 'body') and comment.body != "[deleted]":
                return RedditComment(
                    id=comment.id,
                    body=comment.body,
                    author=str(comment.author) if comment.author else "[deleted]",
                    score=comment.score,
                    created_utc=datetime.fromtimestamp(comment.created_utc),
                    parent_id=comment.parent_id,
                    is_submitter=comment.is_submitter,
                    depth=depth,
                    permalink=f"https://reddit.com{comment.permalink}"
                )
            return None
        except Exception as e:
            logger.error(f"Failed to parse comment: {e}")
            return None
    
    def _get_subreddit_info(self, subreddit_name: str) -> Optional[SubredditInfo]:
        """Get detailed subreddit information."""
        try:
            # Check cache first
            if subreddit_name in self.subreddit_cache:
                cached_info = self.subreddit_cache[subreddit_name]
                # Simple cache validation (in production, add timestamp checking)
                return cached_info
            
            subreddit = self.reddit.subreddit(subreddit_name)
            
            info = SubredditInfo(
                name=subreddit.name,
                display_name=subreddit.display_name,
                title=subreddit.title,
                description=subreddit.description or "",
                subscribers=subreddit.subscribers or 0,
                active_users=subreddit.active_user_count or 0,
                created_utc=datetime.fromtimestamp(subreddit.created_utc),
                is_over_18=subreddit.over18,
                public_description=subreddit.public_description or "",
                subreddit_type=subreddit.subreddit_type,
                lang=subreddit.lang or "en"
            )
            
            # Cache the result
            self.subreddit_cache[subreddit_name] = info
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get subreddit info for {subreddit_name}: {e}")
            return None
    
    def _extract_keywords(self, text: str, limit: int = 10) -> List[str]:
        """Extract keywords from text."""
        try:
            # Clean and tokenize text
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Filter out stop words and short words
            filtered_words = [
                word for word in words 
                if len(word) >= 3 and word not in self.stop_words
            ]
            
            # Count word frequencies
            word_freq = Counter(filtered_words)
            
            # Return most common words
            return [word for word, count in word_freq.most_common(limit)]
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis based on keyword matching."""
        try:
            text_lower = text.lower()
            
            positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
            negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
            
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _calculate_engagement_metrics(self, posts: List[RedditPost]) -> Dict[str, float]:
        """Calculate engagement metrics for posts."""
        try:
            if not posts:
                return {}
            
            scores = [post.score for post in posts]
            comment_counts = [post.num_comments for post in posts]
            upvote_ratios = [post.upvote_ratio for post in posts]
            
            return {
                'avg_score': sum(scores) / len(scores),
                'avg_comments': sum(comment_counts) / len(comment_counts),
                'avg_upvote_ratio': sum(upvote_ratios) / len(upvote_ratios),
                'engagement_rate': (sum(comment_counts) / max(sum(scores), 1)) * 100
            }
            
        except Exception as e:
            logger.error(f"Engagement metrics calculation failed: {e}")
            return {}
    
    @with_retry(max_attempts=3, delay=1.0)
    async def search_reddit(
        self,
        request: Union[str, RedditSearchRequest]
    ) -> RedditSearchResponse:
        """
        Search Reddit posts and comments.
        
        Args:
            request: Search query string or RedditSearchRequest object
            
        Returns:
            RedditSearchResponse with posts and comments
        """
        start_time = datetime.now()
        
        # Convert string to request object
        if isinstance(request, str):
            request = RedditSearchRequest(query=request)
        
        if not self.reddit:
            return RedditSearchResponse(
                success=False,
                query=request.query,
                subreddit=request.subreddit,
                posts=[],
                comments=[],
                total_results=0,
                search_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(),
                error="Reddit API not configured"
            )
        
        try:
            logger.info(f"Searching Reddit - Query: {request.query}, Subreddit: {request.subreddit}")
            
            posts = []
            comments = []
            
            if request.subreddit:
                # Search within specific subreddit
                subreddit = self.reddit.subreddit(request.subreddit)
                
                if request.query:
                    # Search with query in subreddit
                    submissions = subreddit.search(
                        request.query,
                        sort=request.sort,
                        time_filter=request.time_filter,
                        limit=request.limit
                    )
                else:
                    # Get posts based on sort method
                    if request.sort == "hot":
                        submissions = subreddit.hot(limit=request.limit)
                    elif request.sort == "new":
                        submissions = subreddit.new(limit=request.limit)
                    elif request.sort == "top":
                        submissions = subreddit.top(time_filter=request.time_filter, limit=request.limit)
                    else:
                        submissions = subreddit.hot(limit=request.limit)
            else:
                # Search all of Reddit
                if not request.query:
                    return RedditSearchResponse(
                        success=False,
                        query=request.query,
                        subreddit=request.subreddit,
                        posts=[],
                        comments=[],
                        total_results=0,
                        search_time=(datetime.now() - start_time).total_seconds(),
                        timestamp=datetime.now(),
                        error="Query required when not searching specific subreddit"
                    )
                
                submissions = self.reddit.subreddit("all").search(
                    request.query,
                    sort=request.sort,
                    time_filter=request.time_filter,
                    limit=request.limit
                )
            
            # Process submissions
            for submission in submissions:
                try:
                    # Apply filters
                    if submission.score < request.min_score:
                        continue
                    
                    if request.max_age_days:
                        post_age = datetime.now() - datetime.fromtimestamp(submission.created_utc)
                        if post_age.days > request.max_age_days:
                            continue
                    
                    # Parse post
                    post = self._parse_submission(submission)
                    if post:
                        posts.append(post)
                    
                    # Get comments if requested
                    if request.include_comments and len(comments) < 50:  # Limit comments
                        try:
                            submission.comments.replace_more(limit=0)  # Don't load "load more" comments
                            for comment in submission.comments.list()[:10]:  # Top 10 comments per post
                                parsed_comment = self._parse_comment(comment)
                                if parsed_comment:
                                    comments.append(parsed_comment)
                        except Exception as e:
                            logger.warning(f"Failed to load comments for {submission.id}: {e}")
                
                except Exception as e:
                    logger.warning(f"Failed to process submission: {e}")
                    continue
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return RedditSearchResponse(
                success=True,
                query=request.query,
                subreddit=request.subreddit,
                posts=posts,
                comments=comments,
                total_results=len(posts),
                search_time=search_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
            search_time = (datetime.now() - start_time).total_seconds()
            
            return RedditSearchResponse(
                success=False,
                query=request.query,
                subreddit=request.subreddit,
                posts=[],
                comments=[],
                total_results=0,
                search_time=search_time,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def analyze_subreddit(
        self,
        subreddit_name: str,
        analysis_limit: int = 50
    ) -> Optional[SubredditAnalysis]:
        """
        Perform comprehensive analysis of a subreddit.
        
        Args:
            subreddit_name: Name of the subreddit to analyze
            analysis_limit: Number of posts to analyze
            
        Returns:
            SubredditAnalysis with comprehensive insights
        """
        try:
            if not self.reddit:
                raise ToolError("Reddit API not configured")
            
            logger.info(f"Analyzing subreddit: {subreddit_name}")
            
            # Get subreddit info
            subreddit_info = self._get_subreddit_info(subreddit_name)
            if not subreddit_info:
                return None
            
            # Get top posts for analysis
            request = RedditSearchRequest(
                subreddit=subreddit_name,
                limit=analysis_limit,
                sort="top",
                time_filter="month",
                include_comments=True
            )
            
            search_response = await self.search_reddit(request)
            
            if not search_response.success:
                return None
            
            posts = search_response.posts
            comments = search_response.comments
            
            # Analyze trending topics (from post titles)
            all_titles = " ".join([post.title for post in posts])
            trending_topics = self._extract_keywords(all_titles, limit=15)
            
            # Extract common keywords from content
            all_content = " ".join([post.content for post in posts if post.content])
            common_keywords = self._extract_keywords(all_content, limit=20)
            
            # Analyze sentiment distribution
            sentiment_counts = defaultdict(int)
            for post in posts:
                sentiment = self._analyze_sentiment(post.title + " " + post.content)
                sentiment_counts[sentiment] += 1
            
            for comment in comments:
                sentiment = self._analyze_sentiment(comment.body)
                sentiment_counts[sentiment] += 1
            
            # Calculate engagement metrics
            engagement_metrics = self._calculate_engagement_metrics(posts)
            
            # Analyze posting patterns (by hour of day)
            posting_patterns = defaultdict(int)
            for post in posts:
                hour = post.created_utc.hour
                posting_patterns[f"hour_{hour:02d}"] += 1
            
            # Analyze user activity
            user_posts = defaultdict(int)
            for post in posts:
                if post.author != "[deleted]":
                    user_posts[post.author] += 1
            
            # Get top active users
            top_users = dict(Counter(user_posts).most_common(10))
            
            return SubredditAnalysis(
                subreddit_info=subreddit_info,
                top_posts=posts[:10],  # Top 10 posts
                trending_topics=trending_topics,
                common_keywords=common_keywords,
                sentiment_distribution=dict(sentiment_counts),
                engagement_metrics=engagement_metrics,
                posting_patterns=dict(posting_patterns),
                user_activity=top_users
            )
            
        except Exception as e:
            logger.error(f"Subreddit analysis failed for {subreddit_name}: {e}")
            return None
    
    async def get_trending_discussions(
        self,
        subreddit_name: str,
        hours: int = 24,
        min_score: int = 10
    ) -> List[RedditPost]:
        """
        Get trending discussions from the last N hours.
        
        Args:
            subreddit_name: Subreddit to search
            hours: Hours to look back
            min_score: Minimum score threshold
            
        Returns:
            List of trending RedditPost objects
        """
        try:
            request = RedditSearchRequest(
                subreddit=subreddit_name,
                limit=50,
                sort="hot",
                min_score=min_score
            )
            
            response = await self.search_reddit(request)
            
            if not response.success:
                return []
            
            # Filter by time
            cutoff_time = datetime.now() - timedelta(hours=hours)
            trending_posts = [
                post for post in response.posts 
                if post.created_utc >= cutoff_time
            ]
            
            # Sort by engagement (score + comments)
            trending_posts.sort(
                key=lambda x: x.score + (x.num_comments * 2),
                reverse=True
            )
            
            return trending_posts[:15]  # Top 15 trending
            
        except Exception as e:
            logger.error(f"Failed to get trending discussions: {e}")
            return []


# Tool instance for MCP integration
reddit_search_tool = RedditSearchTool()


# MCP tool function
async def mcp_search_reddit(
    query: Optional[str] = None,
    subreddit: Optional[str] = None,
    limit: int = 25,
    sort: str = "relevance",
    time_filter: str = "all",
    include_comments: bool = False,
    min_score: int = 0,
    max_age_days: Optional[int] = None
) -> Dict:
    """
    MCP-compatible Reddit search function.
    
    Args:
        query: Search query
        subreddit: Specific subreddit to search
        limit: Number of results to retrieve
        sort: Sort method (relevance, hot, new, top)
        time_filter: Time filter (all, day, week, month, year)
        include_comments: Include comments in results
        min_score: Minimum post score
        max_age_days: Maximum age in days
    
    Returns:
        Dictionary with Reddit search results
    """
    try:
        request = RedditSearchRequest(
            query=query,
            subreddit=subreddit,
            limit=limit,
            sort=sort,
            time_filter=time_filter,
            include_comments=include_comments,
            min_score=min_score,
            max_age_days=max_age_days
        )
        
        response = await reddit_search_tool.search_reddit(request)
        
        if response.success:
            return {
                "success": True,
                "query": response.query,
                "subreddit": response.subreddit,
                "posts": [
                    {
                        "id": post.id,
                        "title": post.title,
                        "content": post.content[:500] + "..." if len(post.content) > 500 else post.content,
                        "author": post.author,
                        "subreddit": post.subreddit,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": post.created_utc.isoformat(),
                        "url": post.url,
                        "permalink": post.permalink
                    } for post in response.posts
                ],
                "comments": [
                    {
                        "id": comment.id,
                        "body": comment.body[:300] + "..." if len(comment.body) > 300 else comment.body,
                        "author": comment.author,
                        "score": comment.score,
                        "created_utc": comment.created_utc.isoformat()
                    } for comment in response.comments
                ] if include_comments else [],
                "total_results": response.total_results,
                "search_time": response.search_time,
                "timestamp": response.timestamp.isoformat()
            }
        else:
            return {
                "success": False,
                "error": response.error,
                "query": response.query,
                "subreddit": response.subreddit
            }
            
    except Exception as e:
        logger.error(f"Reddit search failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# MCP function for subreddit analysis
async def mcp_analyze_subreddit(
    subreddit_name: str,
    analysis_limit: int = 50
) -> Dict:
    """
    MCP-compatible subreddit analysis function.
    
    Args:
        subreddit_name: Name of the subreddit to analyze
        analysis_limit: Number of posts to analyze
    
    Returns:
        Dictionary with subreddit analysis results
    """
    try:
        analysis = await reddit_search_tool.analyze_subreddit(subreddit_name, analysis_limit)
        
        if analysis:
            return {
                "success": True,
                "subreddit_info": {
                    "name": analysis.subreddit_info.name,
                    "title": analysis.subreddit_info.title,
                    "description": analysis.subreddit_info.description[:500] + "..." if len(analysis.subreddit_info.description) > 500 else analysis.subreddit_info.description,
                    "subscribers": analysis.subreddit_info.subscribers,
                    "active_users": analysis.subreddit_info.active_users,
                    "created_utc": analysis.subreddit_info.created_utc.isoformat(),
                    "is_over_18": analysis.subreddit_info.is_over_18
                },
                "trending_topics": analysis.trending_topics,
                "common_keywords": analysis.common_keywords,
                "sentiment_distribution": analysis.sentiment_distribution,
                "engagement_metrics": analysis.engagement_metrics,
                "top_posts": [
                    {
                        "title": post.title,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "author": post.author,
                        "permalink": post.permalink
                    } for post in analysis.top_posts[:5]  # Top 5 posts only
                ],
                "user_activity": analysis.user_activity,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"Failed to analyze subreddit: {subreddit_name}",
                "subreddit_name": subreddit_name
            }
            
    except Exception as e:
        logger.error(f"Subreddit analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "subreddit_name": subreddit_name
        }


# MCP function for trending discussions
async def mcp_get_trending_discussions(
    subreddit_name: str,
    hours: int = 24,
    min_score: int = 10
) -> Dict:
    """
    MCP-compatible function to get trending discussions.
    
    Args:
        subreddit_name: Subreddit to search
        hours: Hours to look back
        min_score: Minimum score threshold
    
    Returns:
        Dictionary with trending discussions
    """
    try:
        trending_posts = await reddit_search_tool.get_trending_discussions(
            subreddit_name, hours, min_score
        )
        
        return {
            "success": True,
            "subreddit": subreddit_name,
            "hours_back": hours,
            "min_score": min_score,
            "trending_posts": [
                {
                    "title": post.title,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "author": post.author,
                    "created_utc": post.created_utc.isoformat(),
                    "permalink": post.permalink
                } for post in trending_posts
            ],
            "count": len(trending_posts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get trending discussions: {e}")
        return {
            "success": False,
            "error": str(e),
            "subreddit": subreddit_name
        }