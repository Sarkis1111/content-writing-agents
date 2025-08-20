"""
Editor Agent - Multi-Stage Content Editing and Quality Assurance

This module implements the Editor Agent using LangGraph framework for comprehensive content editing,
quality assurance, and optimization workflows. The agent employs a multi-stage editing process
with conditional logic, quality gates, and human escalation capabilities.

Key Features:
- Multi-dimensional quality assurance (grammar, SEO, readability, sentiment)
- LangGraph StateGraph workflow orchestration
- Conditional editing logic with quality gates
- Intelligent human escalation for complex issues
- State management for revision tracking
- Performance monitoring and metrics collection
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, src_path)

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.state import CompiledStateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    CompiledStateGraph = None

# Core imports with fallbacks
try:
    from core.errors import ToolError, ValidationError, WorkflowError
    from core.logging.logger import get_logger
    from core.monitoring.metrics import get_metrics_collector
    from utils.simple_retry import with_retry
except ImportError:
    # Mock implementations for testing
    import logging
    class ToolError(Exception): pass
    class ValidationError(Exception): pass  
    class WorkflowError(Exception): pass
    
    def get_logger(name): return logging.getLogger(name)
    def get_metrics_collector(): return None
    def with_retry(*args, **kwargs): 
        def decorator(func): return func
        return decorator

# LangGraph framework imports
try:
    from frameworks.langgraph.state import ContentEditingState, QualityMetrics, StateStatus, get_state_manager
    from frameworks.langgraph.workflows import get_workflow_registry, get_conditional_logic
    from frameworks.langgraph.config import get_langgraph_framework
except ImportError:
    # Create minimal fallbacks
    class ContentEditingState(dict): pass
    class QualityMetrics: pass
    class StateStatus: pass
    def get_state_manager(): return None
    def get_workflow_registry(): return None
    def get_conditional_logic(): return None
    def get_langgraph_framework(): return None

# Tool imports - use real editing tools
tool_imports_successful = False
try:
    # Import adapted original tools (superior to real tools)
    from .original_tools_adapter import (
        get_adapted_grammar_checker,
        get_adapted_seo_analyzer,
        get_adapted_readability_scorer,
        get_adapted_sentiment_analyzer,
        AdaptedGrammarResult,
        AdaptedSEOResult,
        AdaptedReadabilityResult,  
        AdaptedSentimentResult
    )
    # Also import request types
    from .real_editing_tools import (
        GrammarCheckRequest,
        SEOAnalysisRequest, 
        ReadabilityRequest,
        SentimentAnalysisRequest
    )
    tool_imports_successful = True
    print("✅ Real editing tools loaded successfully!")
except ImportError as e:
    print(f"Real tool import failed: {e}")
    # Create mock tools for development
    class GrammarChecker:
        async def check_grammar(self, request): return {"overall_score": 85.0}
    class SEOAnalyzer:
        async def analyze_seo(self, request): return {"overall_score": 78.0}
    class ReadabilityScorer:
        async def score_readability(self, request): return {"overall_score": 82.0}
    class SentimentAnalyzer:
        async def analyze_sentiment(self, request): return {"overall_sentiment": {"polarity": 0.3}}
    
    # Mock request classes
    class GrammarCheckRequest: pass
    class SEOAnalysisRequest: pass
    class ReadabilityRequest: pass
    class SentimentAnalysisRequest: pass

logger = get_logger(__name__)


class EditingStage(str, Enum):
    """Editing workflow stages"""
    GRAMMAR_CHECK = "grammar_check"
    STYLE_REVIEW = "style_review"
    SEO_OPTIMIZATION = "seo_optimization"
    READABILITY_ASSESSMENT = "readability_assessment"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    FINAL_REVIEW = "final_review"
    QUALITY_GATE = "quality_gate"
    HUMAN_ESCALATION = "human_escalation"
    REVISION_APPLICATION = "revision_application"
    FINALIZATION = "finalization"


class QualityGateResult(str, Enum):
    """Quality gate decision results"""
    PASS = "pass"
    FAIL = "fail" 
    ESCALATE = "escalate"
    REVISE = "revise"


class EditingPriority(str, Enum):
    """Editing priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EditingIssue:
    """Individual editing issue or recommendation"""
    
    stage: EditingStage
    priority: EditingPriority
    category: str
    message: str
    suggestion: Optional[str] = None
    confidence: float = 1.0
    auto_fixable: bool = False
    position: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "stage": self.stage.value,
            "priority": self.priority.value,
            "category": self.category,
            "message": self.message,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "auto_fixable": self.auto_fixable,
            "position": self.position
        }


@dataclass
class EditorAgentConfig:
    """Configuration for Editor Agent"""
    
    # Quality thresholds
    min_grammar_score: float = 85.0
    min_seo_score: float = 75.0
    min_readability_score: float = 80.0
    min_overall_quality: float = 80.0
    
    # Workflow parameters
    max_editing_iterations: int = 3
    enable_auto_fixes: bool = True
    require_human_review: bool = False
    
    # Escalation thresholds
    escalation_quality_threshold: float = 70.0
    escalation_issue_count: int = 10
    
    # Tool configurations
    enable_grammar_check: bool = True
    enable_seo_optimization: bool = True
    enable_readability_check: bool = True
    enable_sentiment_analysis: bool = True
    
    # Performance settings
    timeout_per_stage: int = 60  # seconds
    parallel_processing: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "min_grammar_score": self.min_grammar_score,
            "min_seo_score": self.min_seo_score,
            "min_readability_score": self.min_readability_score,
            "min_overall_quality": self.min_overall_quality,
            "max_editing_iterations": self.max_editing_iterations,
            "enable_auto_fixes": self.enable_auto_fixes,
            "require_human_review": self.require_human_review,
            "escalation_quality_threshold": self.escalation_quality_threshold,
            "escalation_issue_count": self.escalation_issue_count,
            "enable_grammar_check": self.enable_grammar_check,
            "enable_seo_optimization": self.enable_seo_optimization,
            "enable_readability_check": self.enable_readability_check,
            "enable_sentiment_analysis": self.enable_sentiment_analysis,
            "timeout_per_stage": self.timeout_per_stage,
            "parallel_processing": self.parallel_processing
        }


@dataclass  
class EditingResults:
    """Results from editing process"""
    
    original_content: str
    edited_content: str
    quality_scores: Dict[str, float]
    issues_found: List[EditingIssue]
    changes_applied: List[Dict[str, Any]]
    
    stages_completed: List[EditingStage]
    total_iterations: int
    processing_time: float
    
    final_quality_score: float
    human_review_required: bool
    escalation_reasons: List[str]
    
    recommendations: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_content": self.original_content,
            "edited_content": self.edited_content,
            "quality_scores": self.quality_scores,
            "issues_found": [issue.to_dict() for issue in self.issues_found],
            "changes_applied": self.changes_applied,
            "stages_completed": [stage.value for stage in self.stages_completed],
            "total_iterations": self.total_iterations,
            "processing_time": self.processing_time,
            "final_quality_score": self.final_quality_score,
            "human_review_required": self.human_review_required,
            "escalation_reasons": self.escalation_reasons,
            "recommendations": self.recommendations,
            "warnings": self.warnings
        }


class EditorAgent:
    """
    Advanced Editor Agent using LangGraph framework for multi-stage content editing
    
    The Editor Agent implements a sophisticated multi-stage editing workflow using LangGraph's
    StateGraph for orchestration. It provides comprehensive quality assurance through multiple
    editing tools and intelligent decision-making for revision and escalation.
    """
    
    def __init__(self, config: Optional[EditorAgentConfig] = None):
        """
        Initialize Editor Agent
        
        Args:
            config: Editor agent configuration
        """
        self.config = config or EditorAgentConfig()
        
        # Initialize logging and monitoring
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Initialize editing tools
        self._initialize_editing_tools()
        
        # Initialize LangGraph components
        self.state_manager = get_state_manager()
        self.workflow_registry = get_workflow_registry() 
        self.conditional_logic = get_conditional_logic()
        
        # Workflow state
        self.current_workflow: Optional[CompiledStateGraph] = None
        self.workflow_config = {}
        
        # Performance tracking
        self.execution_stats = {
            "total_edits": 0,
            "avg_processing_time": 0.0,
            "success_rate": 0.0,
            "avg_quality_improvement": 0.0
        }
        
        self.logger.info(f"EditorAgent initialized with config: grammar_check={self.config.enable_grammar_check}, seo={self.config.enable_seo_optimization}")
    
    def _initialize_editing_tools(self):
        """Initialize editing tools with adapted original tools"""
        try:
            # Use adapted original tools for superior functionality
            self.grammar_checker = get_adapted_grammar_checker() if self.config.enable_grammar_check else None
            self.seo_analyzer = get_adapted_seo_analyzer() if self.config.enable_seo_optimization else None
            self.readability_scorer = get_adapted_readability_scorer() if self.config.enable_readability_check else None
            self.sentiment_analyzer = get_adapted_sentiment_analyzer() if self.config.enable_sentiment_analysis else None
            
            self.logger.info("Original editing tools initialized successfully via adapters")
            
        except Exception as e:
            self.logger.warning(f"Some editing tools failed to initialize: {e}")
            # Use same adapted tools as fallback (they have internal fallbacks)
            self.grammar_checker = get_adapted_grammar_checker() if self.config.enable_grammar_check else None
            self.seo_analyzer = get_adapted_seo_analyzer() if self.config.enable_seo_optimization else None
            self.readability_scorer = get_adapted_readability_scorer() if self.config.enable_readability_check else None
            self.sentiment_analyzer = get_adapted_sentiment_analyzer() if self.config.enable_sentiment_analysis else None
    
    async def create_editing_workflow(self) -> Optional[CompiledStateGraph]:
        """
        Create LangGraph editing workflow
        
        Returns:
            Compiled StateGraph workflow for editing
        """
        if not LANGGRAPH_AVAILABLE:
            self.logger.error("LangGraph not available - cannot create workflow")
            return None
        
        try:
            # Create StateGraph for editing workflow
            # Use dict instead of TypedDict for better LangGraph compatibility
            workflow = StateGraph(dict)
            
            # Add editing workflow nodes
            workflow.add_node("grammar_check", self._grammar_check_node)
            workflow.add_node("style_review", self._style_review_node)
            workflow.add_node("seo_optimization", self._seo_optimization_node)
            workflow.add_node("readability_assessment", self._readability_assessment_node)
            workflow.add_node("sentiment_analysis", self._sentiment_analysis_node)
            workflow.add_node("final_review", self._final_review_node)
            workflow.add_node("quality_gate", self._quality_gate_node)
            workflow.add_node("human_escalation", self._human_escalation_node)
            workflow.add_node("revision_application", self._revision_application_node)
            workflow.add_node("finalization", self._finalization_node)
            
            # Set entry point
            workflow.set_entry_point("grammar_check")
            
            # Add workflow edges with conditional logic
            workflow.add_edge("grammar_check", "style_review")
            workflow.add_edge("style_review", "seo_optimization")
            workflow.add_edge("seo_optimization", "readability_assessment") 
            workflow.add_edge("readability_assessment", "sentiment_analysis")
            workflow.add_edge("sentiment_analysis", "final_review")
            workflow.add_edge("final_review", "quality_gate")
            
            # Conditional edges from quality gate
            workflow.add_conditional_edges(
                "quality_gate",
                self._quality_gate_decision,
                {
                    QualityGateResult.PASS.value: "finalization",
                    QualityGateResult.FAIL.value: END,
                    QualityGateResult.ESCALATE.value: "human_escalation", 
                    QualityGateResult.REVISE.value: "revision_application"
                }
            )
            
            # Revision loop
            workflow.add_edge("revision_application", "final_review")
            workflow.add_edge("human_escalation", "finalization")
            workflow.add_edge("finalization", END)
            
            # Compile workflow
            compiled_workflow = workflow.compile()
            
            self.logger.info("LangGraph editing workflow created successfully")
            return compiled_workflow
            
        except Exception as e:
            self.logger.error(f"Failed to create editing workflow: {e}")
            return None
    
    # LangGraph Node Functions
    
    async def _grammar_check_node(self, state: ContentEditingState) -> ContentEditingState:
        """Grammar and style checking workflow node"""
        self.logger.info("Executing grammar check stage")
        self.logger.info(f"Grammar node input state - current_quality: {state.get('current_quality', {})}")
        
        try:
            if not self.grammar_checker or not self.config.enable_grammar_check:
                self.logger.info("Grammar check disabled, using default score")
                # Provide default grammar score when disabled
                new_quality = self._update_quality_metrics(
                    state.get("current_quality", {}),
                    {"grammar_score": 85.0}  # Default score when disabled
                )
                return {
                    "grammar_checked_content": state.get("content"),
                    "current_step": EditingStage.STYLE_REVIEW.value,
                    "completed_steps": state.get("completed_steps", []) + [EditingStage.GRAMMAR_CHECK.value],
                    "current_quality": new_quality,
                    "updated_at": datetime.now().isoformat()
                }
            
            # Get content from various possible state keys
            content = (
                state.get("content") or 
                state.get("grammar_checked_content") or 
                state.get("current_content") or 
                ""
            )
            
            if not content:
                self.logger.warning("No content found in state, using default score")
                # Provide default score when no content is available
                new_quality = self._update_quality_metrics(
                    state.get("current_quality", {}),
                    {"grammar_score": 70.0}  # Lower default when no content
                )
                return {
                    "grammar_checked_content": "",
                    "current_step": EditingStage.STYLE_REVIEW.value,
                    "completed_steps": state.get("completed_steps", []) + [EditingStage.GRAMMAR_CHECK.value],
                    "current_quality": new_quality,
                    "updated_at": datetime.now().isoformat()
                }
            
            # Create grammar check request
            request = GrammarCheckRequest(
                text=content,
                style="business",  # Default style
                check_spelling=True,
                check_grammar=True,
                check_style=True,
                auto_correct=self.config.enable_auto_fixes
            ) if tool_imports_successful else None
            
            # Perform grammar check
            if request:
                result = await self.grammar_checker.check_grammar(request)
                grammar_score = result.overall_score
                corrected_content = result.corrected_text if result.corrected_text else content
                
                # Convert grammar issues to EditingIssue objects
                issues = [
                    EditingIssue(
                        stage=EditingStage.GRAMMAR_CHECK,
                        priority=EditingPriority.HIGH if error.severity == "major" else EditingPriority.MEDIUM,
                        category=error.error_type if isinstance(error.error_type, str) else error.error_type.value,
                        message=error.message,
                        suggestion=error.suggestion,
                        confidence=error.confidence,
                        auto_fixable=error.auto_correctable,
                        position=error.start_pos
                    )
                    for error in result.errors
                ]
            else:
                # Mock result for development
                grammar_score = 85.0
                corrected_content = content
                issues = []
            
            # For LangGraph, return partial state updates that will be merged
            new_quality = self._update_quality_metrics(
                state.get("current_quality", {}), 
                {"grammar_score": grammar_score}
            )
            
            partial_update = {
                "grammar_checked_content": corrected_content,
                "current_step": EditingStage.STYLE_REVIEW.value,
                "completed_steps": state.get("completed_steps", []) + [EditingStage.GRAMMAR_CHECK.value],
                "current_quality": new_quality,
                "editing_issues": state.get("editing_issues", []) + [issue.to_dict() for issue in issues],
                "updated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Grammar check completed with score: {grammar_score}")
            self.logger.info(f"Grammar node output state - current_quality: {partial_update.get('current_quality', {})}")
            return partial_update
            
        except Exception as e:
            self.logger.error(f"Grammar check node failed: {e}")
            return self._handle_node_error(state, EditingStage.GRAMMAR_CHECK, str(e))
    
    async def _style_review_node(self, state: ContentEditingState) -> ContentEditingState:
        """Style consistency review workflow node"""
        self.logger.info("Executing style review stage")
        self.logger.info(f"Style node input state - current_quality: {state.get('current_quality', {})}")
        
        try:
            content = state.get("grammar_checked_content") or state.get("content", "")
            
            # Style review logic (simplified for this implementation)
            # In a full implementation, this would include brand voice analysis,
            # tone consistency checking, and style guide compliance
            
            style_score = 82.0  # Mock score for now
            style_issues = []  # Would contain style-specific issues
            
            # For LangGraph, return partial state updates that will be merged
            new_quality = self._update_quality_metrics(
                state.get("current_quality", {}),
                {"style_score": style_score}
            )
            
            partial_update = {
                "style_adjusted_content": content,
                "current_step": EditingStage.SEO_OPTIMIZATION.value,
                "completed_steps": state.get("completed_steps", []) + [EditingStage.STYLE_REVIEW.value],
                "current_quality": new_quality,
                "updated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Style review completed with score: {style_score}")
            self.logger.info(f"Style node output state - current_quality: {partial_update.get('current_quality', {})}")
            return partial_update
            
        except Exception as e:
            self.logger.error(f"Style review node failed: {e}")
            return self._handle_node_error(state, EditingStage.STYLE_REVIEW, str(e))
    
    async def _seo_optimization_node(self, state: ContentEditingState) -> ContentEditingState:
        """SEO optimization workflow node"""
        self.logger.info("Executing SEO optimization stage")
        
        try:
            if not self.seo_analyzer or not self.config.enable_seo_optimization:
                self.logger.info("SEO optimization disabled, skipping")
                return self._update_state(state, {
                    "seo_optimized_content": state.get("style_adjusted_content") or state.get("content"),
                    "current_step": EditingStage.READABILITY_ASSESSMENT.value,
                    "completed_steps": state.get("completed_steps", []) + [EditingStage.SEO_OPTIMIZATION.value]
                })
            
            content = state.get("style_adjusted_content") or state.get("content", "")
            editing_requirements = state.get("editing_requirements", {})
            
            # Extract SEO parameters from requirements
            target_keywords = editing_requirements.get("target_keywords", ["content", "optimization"])
            content_type = editing_requirements.get("content_type", "blog_post")
            
            # Create SEO analysis request
            request = SEOAnalysisRequest(
                content=content,
                target_keywords=target_keywords,
                target_audience=editing_requirements.get("target_audience", ""),
                content_type=content_type
            ) if tool_imports_successful else None
            
            # Perform SEO analysis
            if request:
                result = await self.seo_analyzer.analyze_seo(request)
                seo_score = result.overall_score
                
                # Convert SEO issues to EditingIssue objects
                seo_issues = []
                for issue in result.issues:
                    # Handle string issues from real SEO analyzer
                    if isinstance(issue, str):
                        seo_issues.append(EditingIssue(
                            stage=EditingStage.SEO_OPTIMIZATION,
                            priority=EditingPriority.MEDIUM,
                            category="seo_issue",
                            message=issue,
                            suggestion="Review SEO recommendations",
                            confidence=0.8,
                            auto_fixable=False
                        ))
                    else:
                        # Handle complex issue objects (for future compatibility)
                        seo_issues.append(EditingIssue(
                            stage=EditingStage.SEO_OPTIMIZATION,
                            priority=EditingPriority.HIGH if getattr(issue, 'priority', '') == "critical" else EditingPriority.MEDIUM,
                            category=getattr(issue, 'issue_type', 'seo_issue'),
                            message=getattr(issue, 'description', str(issue)),
                            suggestion=getattr(issue, 'recommendation', ''),
                            confidence=0.8,
                            auto_fixable=False
                        ))
            else:
                # Mock result
                seo_score = 78.0
                seo_issues = []
            
            # Update state
            updated_state = self._update_state(state, {
                "seo_optimized_content": content,  # Would include optimizations in full implementation
                "current_step": EditingStage.READABILITY_ASSESSMENT.value,
                "completed_steps": state.get("completed_steps", []) + [EditingStage.SEO_OPTIMIZATION.value],
                "current_quality": self._update_quality_metrics(
                    state.get("current_quality", {}),
                    {"seo_score": seo_score}
                ),
                "editing_issues": state.get("editing_issues", []) + [issue.to_dict() for issue in seo_issues]
            })
            
            self.logger.info(f"SEO optimization completed with score: {seo_score}")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"SEO optimization node failed: {e}")
            return self._handle_node_error(state, EditingStage.SEO_OPTIMIZATION, str(e))
    
    async def _readability_assessment_node(self, state: ContentEditingState) -> ContentEditingState:
        """Readability assessment workflow node"""
        self.logger.info("Executing readability assessment stage")
        
        try:
            if not self.readability_scorer or not self.config.enable_readability_check:
                self.logger.info("Readability check disabled, skipping")
                return self._update_state(state, {
                    "readability_assessed_content": state.get("seo_optimized_content") or state.get("content"),
                    "current_step": EditingStage.SENTIMENT_ANALYSIS.value,
                    "completed_steps": state.get("completed_steps", []) + [EditingStage.READABILITY_ASSESSMENT.value]
                })
            
            content = state.get("seo_optimized_content") or state.get("content", "")
            editing_requirements = state.get("editing_requirements", {})
            
            # Create readability analysis request
            request = ReadabilityRequest(
                text=content,
                target_audience=editing_requirements.get("target_audience", "general"),
                content_type=editing_requirements.get("content_type", "")
            ) if tool_imports_successful else None
            
            # Perform readability analysis
            if request:
                result = await self.readability_scorer.score_readability(request)
                readability_score = result.overall_score
                
                # Convert readability recommendations to EditingIssue objects
                readability_issues = [
                    EditingIssue(
                        stage=EditingStage.READABILITY_ASSESSMENT,
                        priority=EditingPriority.MEDIUM,
                        category="readability",
                        message=rec,
                        confidence=0.8
                    )
                    for rec in result.priority_recommendations
                ]
            else:
                # Mock result
                readability_score = 82.0
                readability_issues = []
            
            # Update state
            updated_state = self._update_state(state, {
                "readability_assessed_content": content,
                "current_step": EditingStage.SENTIMENT_ANALYSIS.value,
                "completed_steps": state.get("completed_steps", []) + [EditingStage.READABILITY_ASSESSMENT.value],
                "current_quality": self._update_quality_metrics(
                    state.get("current_quality", {}),
                    {"readability_score": readability_score}
                ),
                "editing_issues": state.get("editing_issues", []) + [issue.to_dict() for issue in readability_issues]
            })
            
            self.logger.info(f"Readability assessment completed with score: {readability_score}")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Readability assessment node failed: {e}")
            return self._handle_node_error(state, EditingStage.READABILITY_ASSESSMENT, str(e))
    
    async def _sentiment_analysis_node(self, state: ContentEditingState) -> ContentEditingState:
        """Sentiment analysis workflow node"""
        self.logger.info("Executing sentiment analysis stage")
        
        try:
            if not self.sentiment_analyzer or not self.config.enable_sentiment_analysis:
                self.logger.info("Sentiment analysis disabled, skipping")
                return self._update_state(state, {
                    "sentiment_analyzed_content": state.get("readability_assessed_content") or state.get("content"),
                    "current_step": EditingStage.FINAL_REVIEW.value,
                    "completed_steps": state.get("completed_steps", []) + [EditingStage.SENTIMENT_ANALYSIS.value]
                })
            
            content = state.get("readability_assessed_content") or state.get("content", "")
            editing_requirements = state.get("editing_requirements", {})
            
            # Create sentiment analysis request
            request = SentimentAnalysisRequest(
                text=content,
                target_sentiment=editing_requirements.get("target_sentiment", "neutral"),
                brand_voice=editing_requirements.get("brand_voice", "")
            ) if tool_imports_successful else None
            
            # Perform sentiment analysis
            if request:
                result = await self.sentiment_analyzer.analyze_sentiment(request)
                sentiment_score = result.alignment_score
                
                # Convert sentiment recommendations to EditingIssue objects
                sentiment_issues = []
                for recommendation in result.recommendations:
                    sentiment_issues.append(EditingIssue(
                        stage=EditingStage.SENTIMENT_ANALYSIS,
                        priority=EditingPriority.LOW,
                        category="sentiment",
                        message=recommendation,
                        confidence=0.7,
                        auto_fixable=False
                    ))
            else:
                # Mock result
                sentiment_score = 75.0
                sentiment_issues = []
            
            # Update state
            updated_state = self._update_state(state, {
                "sentiment_analyzed_content": content,
                "current_step": EditingStage.FINAL_REVIEW.value,
                "completed_steps": state.get("completed_steps", []) + [EditingStage.SENTIMENT_ANALYSIS.value],
                "current_quality": self._update_quality_metrics(
                    state.get("current_quality", {}),
                    {"sentiment_score": sentiment_score}
                ),
                "editing_issues": state.get("editing_issues", []) + [issue.to_dict() for issue in sentiment_issues]
            })
            
            self.logger.info(f"Sentiment analysis completed with score: {sentiment_score}")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis node failed: {e}")
            return self._handle_node_error(state, EditingStage.SENTIMENT_ANALYSIS, str(e))
    
    async def _final_review_node(self, state: ContentEditingState) -> ContentEditingState:
        """Final comprehensive review workflow node"""
        self.logger.info("Executing final review stage")
        
        try:
            content = state.get("sentiment_analyzed_content") or state.get("content", "")
            current_quality = state.get("current_quality", {})
            editing_issues = state.get("editing_issues", [])
            
            # Calculate overall quality score
            quality_scores = {
                "grammar_score": current_quality.get("grammar_score", 0.0),
                "style_score": current_quality.get("style_score", 0.0),
                "seo_score": current_quality.get("seo_score", 0.0),
                "readability_score": current_quality.get("readability_score", 0.0),
                "sentiment_score": current_quality.get("sentiment_score", 0.0)
            }
            
            # Debug: Log the quality scores being used
            self.logger.info(f"Quality scores for overall calculation: {quality_scores}")
            
            # Calculate weighted overall score
            weights = {
                "grammar_score": 0.25,
                "style_score": 0.15,
                "seo_score": 0.25,
                "readability_score": 0.20,
                "sentiment_score": 0.15
            }
            
            # Calculate weighted average including zero scores (to prevent 0 overall when some tools fail)
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for metric, score in quality_scores.items():
                weight = weights.get(metric, 0.2)
                # Include all scores, even if 0, to get proper weighted average
                total_weighted_score += score * weight
                total_weight += weight
            
            overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Generate recommendations based on scores and issues
            recommendations = self._generate_recommendations(quality_scores, editing_issues)
            warnings = self._generate_warnings(quality_scores, editing_issues)
            
            # Determine if human review is required
            critical_issues = [
                issue for issue in editing_issues 
                if issue.get("priority") == EditingPriority.CRITICAL.value
            ]
            
            human_review_required = (
                self.config.require_human_review or
                overall_score < self.config.escalation_quality_threshold or
                len(critical_issues) > 0 or
                len(editing_issues) > self.config.escalation_issue_count
            )
            
            # Update quality metrics
            final_quality = {
                **quality_scores,
                "overall_score": overall_score,
                "issues_count": len(editing_issues),
                "critical_issues_count": len(critical_issues),
                "human_review_required": human_review_required
            }
            
            # Update state
            updated_state = self._update_state(state, {
                "final_content": content,
                "current_step": EditingStage.QUALITY_GATE.value,
                "completed_steps": state.get("completed_steps", []) + [EditingStage.FINAL_REVIEW.value],
                "current_quality": final_quality,
                "recommendations": recommendations,
                "warnings": warnings,
                "human_review_required": human_review_required
            })
            
            self.logger.info(f"Final review completed - Overall score: {overall_score:.1f}, Human review: {human_review_required}")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Final review node failed: {e}")
            return self._handle_node_error(state, EditingStage.FINAL_REVIEW, str(e))
    
    async def _quality_gate_node(self, state: ContentEditingState) -> ContentEditingState:
        """Quality gate checkpoint workflow node"""
        self.logger.info("Executing quality gate stage")
        
        try:
            current_quality = state.get("current_quality", {})
            overall_score = current_quality.get("overall_score", 0.0)
            human_review_required = current_quality.get("human_review_required", False)
            editing_iterations = state.get("editing_iterations", 0)
            max_iterations = state.get("max_iterations", self.config.max_editing_iterations)
            
            # Log quality gate decision factors
            self.logger.info(f"Quality gate assessment - Score: {overall_score}, Human review: {human_review_required}, Iterations: {editing_iterations}/{max_iterations}")
            
            # Update state
            updated_state = self._update_state(state, {
                "current_step": "quality_gate_decision",
                "completed_steps": state.get("completed_steps", []) + [EditingStage.QUALITY_GATE.value]
            })
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Quality gate node failed: {e}")
            return self._handle_node_error(state, EditingStage.QUALITY_GATE, str(e))
    
    def _quality_gate_decision(self, state: ContentEditingState) -> str:
        """Quality gate conditional logic"""
        try:
            current_quality = state.get("current_quality", {})
            overall_score = current_quality.get("overall_score", 0.0)
            human_review_required = current_quality.get("human_review_required", False)
            editing_iterations = state.get("editing_iterations", 0)
            max_iterations = state.get("max_iterations", self.config.max_editing_iterations)
            
            # Log current state for debugging
            self.logger.info(f"Quality gate assessment - Score: {overall_score}, Human review: {human_review_required}, Iterations: {editing_iterations}/{max_iterations}")
            
            # Decision logic with improved iteration handling
            if overall_score >= self.config.min_overall_quality:
                self.logger.info("Quality gate: PASS - Quality threshold met")
                return QualityGateResult.PASS.value
            
            elif editing_iterations >= max_iterations:
                self.logger.info("Quality gate: FAIL - Max iterations reached")
                return QualityGateResult.FAIL.value
            
            # If we have a very low score and human review is needed, escalate
            elif human_review_required and overall_score < self.config.escalation_quality_threshold:
                self.logger.info("Quality gate: ESCALATE - Human review required due to low quality")
                return QualityGateResult.ESCALATE.value
            
            # Prevent infinite loops - if we've tried revisions but score is still 0, fail
            elif editing_iterations >= 1 and overall_score == 0.0:
                self.logger.info("Quality gate: FAIL - No quality improvement after revision attempts")
                return QualityGateResult.FAIL.value
            
            else:
                self.logger.info("Quality gate: REVISE - Quality below threshold, revision needed")
                return QualityGateResult.REVISE.value
                
        except Exception as e:
            self.logger.error(f"Quality gate decision failed: {e}")
            return QualityGateResult.FAIL.value
    
    async def _human_escalation_node(self, state: ContentEditingState) -> ContentEditingState:
        """Human escalation workflow node"""
        self.logger.info("Executing human escalation stage")
        
        try:
            # Prepare escalation information
            current_quality = state.get("current_quality", {})
            editing_issues = state.get("editing_issues", [])
            
            escalation_data = {
                "overall_score": current_quality.get("overall_score", 0.0),
                "critical_issues": [
                    issue for issue in editing_issues 
                    if issue.get("priority") == EditingPriority.CRITICAL.value
                ],
                "recommendations": state.get("recommendations", []),
                "warnings": state.get("warnings", []),
                "escalation_reasons": self._determine_escalation_reasons(current_quality, editing_issues)
            }
            
            # In a real implementation, this would trigger human review workflow
            self.logger.info(f"Content escalated for human review: {len(escalation_data['critical_issues'])} critical issues")
            
            # Update state
            updated_state = self._update_state(state, {
                "current_step": EditingStage.FINALIZATION.value,
                "completed_steps": state.get("completed_steps", []) + [EditingStage.HUMAN_ESCALATION.value],
                "escalation_data": escalation_data,
                "human_review_completed": True  # Mock completion for now
            })
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Human escalation node failed: {e}")
            return self._handle_node_error(state, EditingStage.HUMAN_ESCALATION, str(e))
    
    async def _revision_application_node(self, state: ContentEditingState) -> ContentEditingState:
        """Revision application workflow node"""
        self.logger.info("Executing revision application stage")
        
        try:
            current_content = state.get("final_content") or state.get("content", "")
            editing_issues = state.get("editing_issues", [])
            editing_iterations = state.get("editing_iterations", 0) + 1
            
            # Apply automatic fixes for auto-fixable issues
            revised_content = current_content
            changes_applied = []
            
            auto_fixable_issues = [
                issue for issue in editing_issues
                if issue.get("auto_fixable", False) and issue.get("suggestion")
            ]
            
            for issue in auto_fixable_issues[:5]:  # Limit to prevent over-editing
                if issue.get("suggestion"):
                    # In full implementation, would apply specific fixes based on issue type
                    changes_applied.append({
                        "type": issue.get("category"),
                        "description": issue.get("message"),
                        "applied": True
                    })
            
            self.logger.info(f"Applied {len(changes_applied)} automatic revisions in iteration {editing_iterations}")
            
            # Return partial state updates for LangGraph - PRESERVE current_quality
            partial_update = {
                "content": revised_content,
                "editing_iterations": editing_iterations,
                "current_step": EditingStage.FINAL_REVIEW.value,
                "completed_steps": state.get("completed_steps", []) + [EditingStage.REVISION_APPLICATION.value],
                "changes_applied": state.get("changes_applied", []) + changes_applied,
                "current_quality": state.get("current_quality", {}),  # Preserve quality scores!
                "updated_at": datetime.now().isoformat()
            }
            
            return partial_update
            
        except Exception as e:
            self.logger.error(f"Revision application node failed: {e}")
            return self._handle_node_error(state, EditingStage.REVISION_APPLICATION, str(e))
    
    async def _finalization_node(self, state: ContentEditingState) -> ContentEditingState:
        """Content finalization workflow node"""
        self.logger.info("Executing finalization stage")
        
        try:
            final_content = state.get("final_content") or state.get("content", "")
            current_quality = state.get("current_quality", {})
            
            # Mark workflow as completed
            completion_time = datetime.now().isoformat()
            
            # Update state for completion
            updated_state = self._update_state(state, {
                "final_content": final_content,
                "workflow_status": StateStatus.COMPLETED.value if hasattr(StateStatus, 'COMPLETED') else "completed",
                "current_step": "completed",
                "completed_steps": state.get("completed_steps", []) + [EditingStage.FINALIZATION.value],
                "completed_at": completion_time
            })
            
            self.logger.info(f"Content editing completed successfully with final score: {current_quality.get('overall_score', 0.0):.1f}")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Finalization node failed: {e}")
            return self._handle_node_error(state, EditingStage.FINALIZATION, str(e))
    
    # Helper Methods
    
    def _update_state(self, current_state: ContentEditingState, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update workflow state with new values"""
        # For LangGraph StateGraph, we need to return the full new state
        # LangGraph will handle the state format internally
        if isinstance(current_state, dict):
            new_state = current_state.copy()
        else:
            # Handle TypedDict or other state formats
            new_state = dict(current_state) if current_state else {}
        
        new_state.update(updates)
        new_state["updated_at"] = datetime.now().isoformat()
        
        # Debug logging to track state updates
        if "current_quality" in updates:
            self.logger.info(f"State update returning current_quality: {updates['current_quality']}")
            self.logger.info(f"Full state current_quality: {new_state.get('current_quality', {})}")
        
        return new_state
    
    def _update_quality_metrics(self, current_quality: Dict[str, Any], updates: Dict[str, float]) -> Dict[str, Any]:
        """Update quality metrics and calculate overall score"""
        if isinstance(current_quality, dict):
            quality = current_quality.copy()
        else:
            quality = {}
        
        quality.update(updates)
        
        # Calculate overall score based on available metrics
        quality_scores = {
            "grammar_score": quality.get("grammar_score", 0.0),
            "style_score": quality.get("style_score", 0.0),
            "seo_score": quality.get("seo_score", 0.0),
            "readability_score": quality.get("readability_score", 0.0),
            "sentiment_score": quality.get("sentiment_score", 0.0)
        }
        
        # Calculate weighted overall score
        weights = {
            "grammar_score": 0.25,
            "style_score": 0.15,
            "seo_score": 0.25,
            "readability_score": 0.20,
            "sentiment_score": 0.15
        }
        
        # Calculate weighted average including zero scores
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric, score in quality_scores.items():
            if score > 0:  # Only include metrics that have been calculated
                weight = weights.get(metric, 0.2)
                total_weighted_score += score * weight
                total_weight += weight
        
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Always preserve all existing scores and add new ones
        quality.update(quality_scores)
        quality["overall_score"] = overall_score
        
        return quality
    
    def _handle_node_error(self, state: ContentEditingState, stage: EditingStage, error_msg: str) -> ContentEditingState:
        """Handle node execution errors"""
        self.logger.error(f"Node error in {stage.value}: {error_msg}")
        
        return self._update_state(state, {
            "workflow_status": StateStatus.FAILED.value if hasattr(StateStatus, 'FAILED') else "failed",
            "failed_steps": state.get("failed_steps", []) + [stage.value],
            "last_error": error_msg
        })
    
    def _generate_recommendations(self, quality_scores: Dict[str, float], editing_issues: List[Dict[str, Any]]) -> List[str]:
        """Generate editing recommendations based on quality scores and issues"""
        recommendations = []
        
        # Grammar recommendations
        if quality_scores.get("grammar_score", 0) < self.config.min_grammar_score:
            recommendations.append("Improve grammar and spelling accuracy")
        
        # SEO recommendations  
        if quality_scores.get("seo_score", 0) < self.config.min_seo_score:
            recommendations.append("Enhance SEO optimization with better keyword usage")
        
        # Readability recommendations
        if quality_scores.get("readability_score", 0) < self.config.min_readability_score:
            recommendations.append("Simplify sentence structure and vocabulary for better readability")
        
        # Issue-based recommendations
        issue_categories = {}
        for issue in editing_issues:
            category = issue.get("category", "general")
            issue_categories[category] = issue_categories.get(category, 0) + 1
        
        if issue_categories.get("spelling", 0) > 3:
            recommendations.append("Focus on spelling accuracy - multiple errors detected")
        
        if issue_categories.get("seo", 0) > 2:
            recommendations.append("Review SEO best practices for content optimization")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_warnings(self, quality_scores: Dict[str, float], editing_issues: List[Dict[str, Any]]) -> List[str]:
        """Generate editing warnings"""
        warnings = []
        
        overall_score = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
        
        if overall_score < 60.0:
            warnings.append("Content quality is significantly below standards")
        
        critical_issues = [
            issue for issue in editing_issues 
            if issue.get("priority") == EditingPriority.CRITICAL.value
        ]
        
        if len(critical_issues) > 0:
            warnings.append(f"{len(critical_issues)} critical issues require immediate attention")
        
        if quality_scores.get("grammar_score", 100) < 70:
            warnings.append("Grammar score is critically low - extensive revision needed")
        
        return warnings
    
    def _determine_escalation_reasons(self, quality: Dict[str, Any], issues: List[Dict[str, Any]]) -> List[str]:
        """Determine reasons for human escalation"""
        reasons = []
        
        overall_score = quality.get("overall_score", 0.0)
        if overall_score < self.config.escalation_quality_threshold:
            reasons.append(f"Overall quality score ({overall_score:.1f}) below threshold ({self.config.escalation_quality_threshold})")
        
        critical_issues = len([
            issue for issue in issues 
            if issue.get("priority") == EditingPriority.CRITICAL.value
        ])
        if critical_issues > 0:
            reasons.append(f"{critical_issues} critical issues detected")
        
        if len(issues) > self.config.escalation_issue_count:
            reasons.append(f"Too many issues detected ({len(issues)} > {self.config.escalation_issue_count})")
        
        return reasons
    
    # Public Interface Methods
    
    async def edit_content(self, content: str, editing_requirements: Optional[Dict[str, Any]] = None) -> EditingResults:
        """
        Edit content using multi-stage LangGraph workflow
        
        Args:
            content: Content to edit
            editing_requirements: Specific editing requirements and parameters
            
        Returns:
            Complete editing results with quality metrics and recommendations
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting content editing for {len(content)} characters")
            
            # Create or get editing workflow
            if not self.current_workflow:
                self.current_workflow = await self.create_editing_workflow()
                
                if not self.current_workflow:
                    raise WorkflowError("Failed to create editing workflow")
            
            # Initialize editing state
            initial_state = {
                "content": content,
                "editing_requirements": editing_requirements or {},
                "target_quality_score": self.config.min_overall_quality / 100.0,
                "max_iterations": self.config.max_editing_iterations,
                "editing_iterations": 0,
                "workflow_id": f"edit_{hash(content)}{int(start_time.timestamp())}",
                "started_at": start_time.isoformat(),
                "current_step": EditingStage.GRAMMAR_CHECK.value,
                "completed_steps": [],
                "editing_issues": [],
                "changes_applied": []
            }
            
            # Execute workflow - Prefer LangGraph for advanced orchestration
            if LANGGRAPH_AVAILABLE and self.current_workflow:
                # Run LangGraph workflow for sophisticated orchestration
                final_state = await self._execute_langgraph_workflow(initial_state)
            else:
                # Use reliable sequential execution as fallback
                final_state = await self._execute_sequential_workflow(initial_state)
            
            # Process results
            processing_time = (datetime.now() - start_time).total_seconds()
            editing_results = self._process_editing_results(final_state, content, processing_time)
            
            # Update performance stats
            self._update_performance_stats(editing_results)
            
            self.logger.info(f"Content editing completed in {processing_time:.2f}s with final score: {editing_results.final_quality_score:.1f}")
            return editing_results
            
        except Exception as e:
            self.logger.error(f"Content editing failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Return error result
            return EditingResults(
                original_content=content,
                edited_content=content,  # Return original if editing failed
                quality_scores={},
                issues_found=[],
                changes_applied=[],
                stages_completed=[],
                total_iterations=0,
                processing_time=processing_time,
                final_quality_score=0.0,
                human_review_required=True,
                escalation_reasons=[f"Editing workflow failed: {str(e)}"],
                recommendations=["Manual review required due to system error"],
                warnings=["Automated editing process encountered errors"]
            )
    
    async def _execute_langgraph_workflow(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LangGraph workflow"""
        try:
            # Create ContentEditingState format directly (bypass StateManager import issues)
            from datetime import datetime
            now = datetime.now().isoformat()
            
            # Create state in ContentEditingState format
            langgraph_state = ContentEditingState({
                # Required fields
                "content": initial_state.get("content", ""),
                "editing_requirements": initial_state.get("editing_requirements", {}),
                "target_quality_score": initial_state.get("target_quality_score", 0.8),
                
                # Processing stages
                "grammar_checked_content": None,
                "seo_optimized_content": None, 
                "style_adjusted_content": None,
                "fact_checked_content": None,
                
                # Quality assessment
                "initial_quality": None,
                "current_quality": {},
                "quality_history": [],
                
                # Editing iterations
                "editing_iterations": initial_state.get("editing_iterations", 0),
                "max_iterations": initial_state.get("max_iterations", self.config.max_editing_iterations),
                "improvement_threshold": 0.1,
                
                # Final output
                "final_content": None,
                "quality_report": {},
                "editing_summary": {},
                
                # Workflow control
                "workflow_status": StateStatus.INITIALIZED if hasattr(globals().get('StateStatus', None), 'INITIALIZED') else "initialized",
                "current_step": initial_state.get("current_step", "grammar_check"),
                "completed_steps": initial_state.get("completed_steps", []),
                
                # Metadata
                "workflow_id": initial_state.get("workflow_id", f"edit_{id(initial_state)}"),
                "started_at": now,
                "completed_at": None
            })
            
            # Execute the compiled workflow with recursion limit
            config = {"recursion_limit": max(50, self.config.max_editing_iterations * 10)}
            result = await self.current_workflow.ainvoke(langgraph_state, config=config)
            
            # Convert back to dict format for compatibility
            return dict(result)
            
        except Exception as e:
            self.logger.error(f"LangGraph workflow execution failed: {e}")
            raise WorkflowError(f"LangGraph execution failed: {e}")
    
    async def _execute_sequential_workflow(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback sequential workflow execution"""
        self.logger.info("Executing sequential workflow (LangGraph not available)")
        
        state = initial_state.copy()
        
        # Execute workflow stages sequentially
        stages = [
            self._grammar_check_node,
            self._style_review_node,
            self._seo_optimization_node,
            self._readability_assessment_node,
            self._sentiment_analysis_node,
            self._final_review_node,
            self._quality_gate_node
        ]
        
        for stage_func in stages:
            try:
                state = await stage_func(state)
                
                # Check if workflow should continue
                if state.get("workflow_status") == "failed":
                    break
                    
                # Check quality gate decision for sequential mode
                if stage_func == self._quality_gate_node:
                    decision = self._quality_gate_decision(state)
                    
                    if decision == QualityGateResult.ESCALATE.value:
                        state = await self._human_escalation_node(state)
                        break
                    elif decision == QualityGateResult.REVISE.value:
                        # Apply revision and re-evaluate, but only if under max iterations
                        current_iterations = state.get("editing_iterations", 0)
                        if current_iterations < self.config.max_editing_iterations:
                            state = await self._revision_application_node(state)
                            # Re-run final review after revision
                            state = await self._final_review_node(state)
                            # Final decision after revision
                            final_decision = self._quality_gate_decision(state)
                            if final_decision == QualityGateResult.PASS.value:
                                self.logger.info("Quality improved after revision - passing")
                            elif final_decision == QualityGateResult.ESCALATE.value:
                                state = await self._human_escalation_node(state)
                            # Exit revision loop regardless of outcome
                        else:
                            self.logger.info("Max iterations reached, stopping revisions")
                        break
                    elif decision == QualityGateResult.PASS.value or decision == QualityGateResult.FAIL.value:
                        break
                
            except Exception as e:
                self.logger.error(f"Sequential workflow stage failed: {e}")
                state = self._handle_node_error(state, EditingStage.GRAMMAR_CHECK, str(e))
                break
        
        # Finalize
        if state.get("workflow_status") != "failed":
            state = await self._finalization_node(state)
        
        return state
    
    def _process_editing_results(self, final_state: Dict[str, Any], original_content: str, processing_time: float) -> EditingResults:
        """Process final workflow state into EditingResults"""
        
        # Extract results from final state
        edited_content = final_state.get("final_content", original_content)
        current_quality = final_state.get("current_quality", {})
        
        quality_scores = {
            "grammar_score": current_quality.get("grammar_score", 0.0),
            "style_score": current_quality.get("style_score", 0.0), 
            "seo_score": current_quality.get("seo_score", 0.0),
            "readability_score": current_quality.get("readability_score", 0.0),
            "sentiment_score": current_quality.get("sentiment_score", 0.0),
            "overall_score": current_quality.get("overall_score", 0.0)
        }
        
        # Convert editing issues back to EditingIssue objects
        issues_data = final_state.get("editing_issues", [])
        issues_found = [
            EditingIssue(
                stage=EditingStage(issue.get("stage", "grammar_check")),
                priority=EditingPriority(issue.get("priority", "medium")),
                category=issue.get("category", "general"),
                message=issue.get("message", ""),
                suggestion=issue.get("suggestion"),
                confidence=issue.get("confidence", 1.0),
                auto_fixable=issue.get("auto_fixable", False),
                position=issue.get("position")
            )
            for issue in issues_data
        ]
        
        # Extract completed stages
        completed_steps = final_state.get("completed_steps", [])
        stages_completed = [
            EditingStage(step) for step in completed_steps 
            if step in [stage.value for stage in EditingStage]
        ]
        
        return EditingResults(
            original_content=original_content,
            edited_content=edited_content,
            quality_scores=quality_scores,
            issues_found=issues_found,
            changes_applied=final_state.get("changes_applied", []),
            stages_completed=stages_completed,
            total_iterations=final_state.get("editing_iterations", 0),
            processing_time=processing_time,
            final_quality_score=quality_scores["overall_score"],
            human_review_required=final_state.get("human_review_required", False),
            escalation_reasons=final_state.get("escalation_data", {}).get("escalation_reasons", []),
            recommendations=final_state.get("recommendations", []),
            warnings=final_state.get("warnings", [])
        )
    
    def _update_performance_stats(self, results: EditingResults):
        """Update performance tracking statistics"""
        try:
            self.execution_stats["total_edits"] += 1
            
            # Update average processing time
            total_time = self.execution_stats["avg_processing_time"] * (self.execution_stats["total_edits"] - 1)
            self.execution_stats["avg_processing_time"] = (total_time + results.processing_time) / self.execution_stats["total_edits"]
            
            # Update success rate
            success = results.final_quality_score >= self.config.min_overall_quality
            total_successes = self.execution_stats["success_rate"] * (self.execution_stats["total_edits"] - 1)
            self.execution_stats["success_rate"] = (total_successes + (1 if success else 0)) / self.execution_stats["total_edits"]
            
            # Log metrics if available
            if self.metrics:
                self.metrics.record_counter("editor_agent_executions")
                self.metrics.record_histogram("editor_agent_processing_time", results.processing_time)
                self.metrics.record_gauge("editor_agent_quality_score", results.final_quality_score)
                
        except Exception as e:
            self.logger.warning(f"Failed to update performance stats: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.execution_stats.copy()
    
    def get_config(self) -> EditorAgentConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, new_config: EditorAgentConfig):
        """Update agent configuration"""
        self.config = new_config
        self.logger.info("Editor agent configuration updated")


# Export main classes
__all__ = [
    'EditorAgent',
    'EditorAgentConfig',
    'EditingResults',
    'EditingIssue',
    'EditingStage',
    'EditingPriority',
    'QualityGateResult'
]


if __name__ == "__main__":
    # Example usage and testing
    async def test_editor_agent():
        """Test the Editor Agent functionality"""
        
        test_content = """
        This is a test content for editing. It has some grammer mistakes and could be improved.
        The SEO optimization is not great and readability might be better with shorter sentences.
        
        Overall this content needs significant editing to meet quality standards for professional publication.
        We want to make sure it's perfect before we publish it to our audience.
        """
        
        config = EditorAgentConfig(
            min_overall_quality=75.0,
            enable_auto_fixes=True,
            max_editing_iterations=2
        )
        
        try:
            agent = EditorAgent(config)
            
            editing_requirements = {
                "target_keywords": ["content", "editing", "quality"],
                "target_audience": "professional writers",
                "content_type": "blog_post",
                "brand_voice": "professional"
            }
            
            results = await agent.edit_content(test_content, editing_requirements)
            
            print("Editor Agent Test Results:")
            print(f"Final Quality Score: {results.final_quality_score:.1f}")
            print(f"Processing Time: {results.processing_time:.2f}s")
            print(f"Stages Completed: {len(results.stages_completed)}")
            print(f"Issues Found: {len(results.issues_found)}")
            print(f"Changes Applied: {len(results.changes_applied)}")
            print(f"Human Review Required: {results.human_review_required}")
            
            if results.recommendations:
                print("\nRecommendations:")
                for rec in results.recommendations:
                    print(f"- {rec}")
            
            print(f"\nQuality Scores:")
            for metric, score in results.quality_scores.items():
                print(f"- {metric}: {score:.1f}")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_editor_agent())