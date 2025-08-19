"""
Strategy Agent Data Models and Schemas.

Defines data structures for strategy requests, responses, and analysis results
used by the AutoGen-powered Strategy Agent.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class StrategyType(Enum):
    """Types of content strategies."""
    CONTENT_MARKETING = "content_marketing"
    BRAND_AWARENESS = "brand_awareness"
    LEAD_GENERATION = "lead_generation"
    THOUGHT_LEADERSHIP = "thought_leadership"
    CUSTOMER_EDUCATION = "customer_education"
    SEO_FOCUSED = "seo_focused"
    SOCIAL_MEDIA = "social_media"
    EMAIL_MARKETING = "email_marketing"
    PRODUCT_LAUNCH = "product_launch"
    CRISIS_COMMUNICATION = "crisis_communication"


class ContentType(Enum):
    """Types of content for strategy planning."""
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"
    SOCIAL_POST = "social_post"
    EMAIL = "email"
    VIDEO_SCRIPT = "video_script"
    INFOGRAPHIC = "infographic"
    LANDING_PAGE = "landing_page"
    PRESS_RELEASE = "press_release"


class AudienceSegment(Enum):
    """Target audience segments."""
    B2B_EXECUTIVES = "b2b_executives"
    TECHNICAL_PROFESSIONALS = "technical_professionals"
    SMALL_BUSINESS_OWNERS = "small_business_owners"
    CONSUMERS = "consumers"
    STUDENTS = "students"
    ENTERPRISE_BUYERS = "enterprise_buyers"
    INDUSTRY_EXPERTS = "industry_experts"
    GENERAL_PUBLIC = "general_public"


class CompetitivePosition(Enum):
    """Competitive positioning strategies."""
    MARKET_LEADER = "market_leader"
    CHALLENGER = "challenger"
    FOLLOWER = "follower"
    NICHE_SPECIALIST = "niche_specialist"
    DISRUPTOR = "disruptor"


@dataclass
class StrategyContext:
    """Context information for strategy development."""
    topic: str
    industry: Optional[str] = None
    target_audience: List[AudienceSegment] = field(default_factory=list)
    business_objectives: List[str] = field(default_factory=list)
    brand_voice: Optional[str] = None
    competitive_landscape: Optional[str] = None
    content_goals: List[str] = field(default_factory=list)
    budget_constraints: Optional[str] = None
    timeline: Optional[str] = None
    channels: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)


@dataclass
class ContentStrategistPerspective:
    """Content strategist's analysis and recommendations."""
    overall_strategy: str
    content_pillars: List[str]
    messaging_framework: Dict[str, str]
    content_mix_recommendations: Dict[str, float]  # content type -> percentage
    editorial_calendar_structure: Dict[str, Any]
    brand_alignment_score: float
    strategic_recommendations: List[str]
    success_metrics: List[str]
    confidence_level: float


@dataclass
class AudienceAnalystPerspective:
    """Audience analyst's insights and targeting recommendations."""
    primary_audience_profile: Dict[str, Any]
    secondary_audiences: List[Dict[str, Any]]
    audience_pain_points: List[str]
    content_preferences: Dict[str, Any]
    engagement_patterns: Dict[str, Any]
    persona_mapping: Dict[str, Dict[str, Any]]
    channel_preferences: Dict[str, float]  # channel -> preference score
    messaging_preferences: Dict[str, str]
    audience_journey_insights: Dict[str, Any]
    targeting_recommendations: List[str]
    confidence_level: float


@dataclass
class CompetitiveAnalystPerspective:
    """Competitive analyst's market positioning insights."""
    competitive_landscape_overview: str
    key_competitors: List[Dict[str, Any]]
    market_positioning: CompetitivePosition
    differentiation_opportunities: List[str]
    competitive_gaps: List[str]
    market_trends: List[str]
    positioning_strategy: str
    unique_value_propositions: List[str]
    competitive_advantages: List[str]
    market_share_insights: Dict[str, Any]
    strategic_threats: List[str]
    opportunities: List[str]
    confidence_level: float


@dataclass
class PerformanceOptimizerPerspective:
    """Performance optimizer's metrics and improvement recommendations."""
    key_performance_indicators: List[str]
    measurement_framework: Dict[str, Any]
    optimization_opportunities: List[str]
    performance_benchmarks: Dict[str, float]
    conversion_optimization: Dict[str, Any]
    channel_performance_analysis: Dict[str, Dict[str, Any]]
    roi_projections: Dict[str, float]
    testing_recommendations: List[str]
    attribution_model: str
    success_criteria: Dict[str, Any]
    improvement_tactics: List[str]
    confidence_level: float


@dataclass
class StrategyConsensus:
    """Consensus reached by the strategy council."""
    agreed_strategy_type: StrategyType
    consensus_score: float  # 0.0 to 1.0
    unified_approach: str
    key_decisions: List[str]
    trade_offs_accepted: List[str]
    implementation_priorities: List[str]
    resource_requirements: Dict[str, Any]
    timeline_agreement: Dict[str, Any]
    success_metrics_consensus: List[str]
    risks_identified: List[str]
    mitigation_strategies: List[str]


@dataclass
class StrategyResult:
    """Complete strategy development result."""
    strategy_id: str
    context: StrategyContext
    content_strategist: ContentStrategistPerspective
    audience_analyst: AudienceAnalystPerspective
    competitive_analyst: CompetitiveAnalystPerspective
    performance_optimizer: PerformanceOptimizerPerspective
    consensus: StrategyConsensus
    overall_confidence: float
    discussion_rounds: int
    processing_time: float
    created_at: datetime
    agent_contributions: Dict[str, int]  # agent -> message count


class StrategyRequest(BaseModel):
    """Strategy development request."""
    topic: str = Field(..., description="Main topic or subject for strategy development")
    strategy_type: Optional[StrategyType] = Field(default=None, description="Preferred strategy type")
    content_types: List[ContentType] = Field(default_factory=list, description="Target content types")
    target_audience: List[AudienceSegment] = Field(default_factory=list, description="Target audience segments")
    industry: Optional[str] = Field(default=None, description="Industry or sector")
    business_objectives: List[str] = Field(default_factory=list, description="Business objectives")
    brand_voice: Optional[str] = Field(default=None, description="Brand voice and tone guidelines")
    competitive_context: Optional[str] = Field(default=None, description="Competitive landscape context")
    budget_level: Optional[str] = Field(default=None, description="Budget constraints (high/medium/low)")
    timeline: Optional[str] = Field(default=None, description="Timeline for implementation")
    channels: List[str] = Field(default_factory=list, description="Distribution channels")
    success_metrics: List[str] = Field(default_factory=list, description="Preferred success metrics")
    
    # Discussion settings
    max_discussion_rounds: int = Field(default=15, ge=5, le=30, description="Maximum discussion rounds")
    consensus_threshold: float = Field(default=0.75, ge=0.5, le=1.0, description="Required consensus score")
    include_competitive_analysis: bool = Field(default=True, description="Include competitive analysis")
    include_audience_research: bool = Field(default=True, description="Include audience research")
    prioritize_performance_metrics: bool = Field(default=True, description="Emphasize performance optimization")
    
    # Advanced options
    research_depth: str = Field(default="standard", description="Research depth (quick/standard/deep)")
    creative_freedom: float = Field(default=0.7, ge=0.0, le=1.0, description="Creative freedom level")
    risk_tolerance: str = Field(default="medium", description="Risk tolerance (low/medium/high)")


class StrategyResponse(BaseModel):
    """Strategy development response."""
    success: bool
    request_id: str
    strategy_result: Optional[StrategyResult] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    processing_time: float
    agent_session_id: Optional[str] = None
    
    # Performance metrics
    discussion_quality_score: Optional[float] = None
    consensus_achieved: bool = False
    rounds_completed: int = 0
    agent_participation: Dict[str, float] = Field(default_factory=dict)  # agent -> participation score


@dataclass
class AgentContribution:
    """Individual agent contribution tracking."""
    agent_name: str
    role: str
    messages_sent: int
    words_contributed: int
    key_insights: List[str]
    consensus_alignment: float
    contribution_quality: float
    unique_perspectives: List[str]


@dataclass
class DiscussionMetrics:
    """Metrics for strategy discussion analysis."""
    total_rounds: int
    total_messages: int
    total_words: int
    consensus_progression: List[float]  # consensus score over time
    agent_contributions: List[AgentContribution]
    key_decision_points: List[Dict[str, Any]]
    conflict_resolution_instances: int
    creative_breakthroughs: List[str]
    time_per_round: List[float]
    engagement_score: float
    discussion_efficiency: float


class StrategySessionConfig(BaseModel):
    """Configuration for strategy development sessions."""
    session_name: str = Field(default="Strategy Development Session")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=2000, ge=500, le=4000, description="Max tokens per response")
    timeout_seconds: int = Field(default=300, ge=60, le=900, description="Session timeout")
    
    # Agent configuration
    enable_content_strategist: bool = Field(default=True)
    enable_audience_analyst: bool = Field(default=True)
    enable_competitive_analyst: bool = Field(default=True)
    enable_performance_optimizer: bool = Field(default=True)
    
    # Discussion rules
    prevent_agent_domination: bool = Field(default=True)
    require_all_perspectives: bool = Field(default=True)
    enable_creative_tangents: bool = Field(default=False)
    force_consensus_building: bool = Field(default=True)
    
    # Quality gates
    min_perspective_depth: int = Field(default=3, description="Minimum depth for each perspective")
    require_data_backing: bool = Field(default=False, description="Require data to support claims")
    enforce_audience_focus: bool = Field(default=True, description="Keep discussion audience-focused")


@dataclass 
class StrategyWorkflowState:
    """State tracking for strategy development workflows."""
    session_id: str
    current_phase: str  # "initiation", "analysis", "synthesis", "consensus", "finalization"
    rounds_completed: int
    perspectives_gathered: Dict[str, bool]  # agent_role -> has_contributed
    consensus_items: List[str]
    unresolved_items: List[str]
    current_consensus_score: float
    last_activity: datetime
    agent_states: Dict[str, Dict[str, Any]]
    decisions_made: List[Dict[str, Any]]
    next_actions: List[str]


# Tool integration data models
@dataclass
class ToolAnalysisInput:
    """Input for tool analysis integration."""
    content_context: str
    analysis_type: str  # "sentiment", "topic_extraction", "trend_analysis", etc.
    parameters: Dict[str, Any]


@dataclass
class ToolAnalysisResult:
    """Result from tool analysis."""
    tool_name: str
    analysis_type: str
    result_data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    insights: List[str]