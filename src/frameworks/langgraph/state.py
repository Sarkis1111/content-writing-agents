"""LangGraph state management schemas and utilities."""

from typing import Dict, Any, List, Optional, Union, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

try:
    from ...core.logging import get_framework_logger
    from ...core.errors import LangGraphError, ValidationError
except ImportError:
    # Handle direct import case
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from core.logging import get_framework_logger
    from core.errors import LangGraphError, ValidationError


class StateStatus(Enum):
    """Workflow state status."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class ContentType(Enum):
    """Content types for content creation workflows."""
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    PRESS_RELEASE = "press_release"
    DOCUMENTATION = "documentation"


@dataclass
class ResearchData:
    """Research data structure."""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    trends: List[Dict[str, Any]] = field(default_factory=list)
    competitive_analysis: Dict[str, Any] = field(default_factory=dict)
    fact_checks: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sources": self.sources,
            "key_findings": self.key_findings,
            "trends": self.trends,
            "competitive_analysis": self.competitive_analysis,
            "fact_checks": self.fact_checks
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchData':
        """Create from dictionary."""
        return cls(
            sources=data.get("sources", []),
            key_findings=data.get("key_findings", []),
            trends=data.get("trends", []),
            competitive_analysis=data.get("competitive_analysis", {}),
            fact_checks=data.get("fact_checks", [])
        )


@dataclass
class ContentStrategy:
    """Content strategy structure."""
    objectives: List[str] = field(default_factory=list)
    target_audience: Dict[str, Any] = field(default_factory=dict)
    key_messages: List[str] = field(default_factory=list)
    tone_and_style: Dict[str, Any] = field(default_factory=dict)
    distribution_channels: List[str] = field(default_factory=list)
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "objectives": self.objectives,
            "target_audience": self.target_audience,
            "key_messages": self.key_messages,
            "tone_and_style": self.tone_and_style,
            "distribution_channels": self.distribution_channels,
            "success_metrics": self.success_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentStrategy':
        """Create from dictionary."""
        return cls(
            objectives=data.get("objectives", []),
            target_audience=data.get("target_audience", {}),
            key_messages=data.get("key_messages", []),
            tone_and_style=data.get("tone_and_style", {}),
            distribution_channels=data.get("distribution_channels", []),
            success_metrics=data.get("success_metrics", {})
        )


@dataclass
class QualityMetrics:
    """Quality assessment metrics."""
    grammar_score: float = 0.0
    readability_score: float = 0.0
    seo_score: float = 0.0
    sentiment_score: float = 0.0
    brand_alignment_score: float = 0.0
    factual_accuracy_score: float = 0.0
    overall_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def calculate_overall_score(self):
        """Calculate overall quality score."""
        scores = [
            self.grammar_score,
            self.readability_score,
            self.seo_score,
            self.sentiment_score,
            self.brand_alignment_score,
            self.factual_accuracy_score
        ]
        self.overall_score = sum(scores) / len(scores) if scores else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grammar_score": self.grammar_score,
            "readability_score": self.readability_score,
            "seo_score": self.seo_score,
            "sentiment_score": self.sentiment_score,
            "brand_alignment_score": self.brand_alignment_score,
            "factual_accuracy_score": self.factual_accuracy_score,
            "overall_score": self.overall_score,
            "issues": self.issues,
            "recommendations": self.recommendations
        }


# Content Creation Workflow State
class ContentCreationState(TypedDict, total=False):
    """State schema for content creation workflows."""
    
    # Input parameters
    topic: str
    content_type: ContentType
    requirements: Dict[str, Any]
    
    # Research phase
    research_data: Optional[ResearchData]
    research_status: str
    
    # Strategy phase
    strategy: Optional[ContentStrategy]
    strategy_status: str
    
    # Content creation phase
    outline: Optional[str]
    content: Optional[str]
    headlines: List[str]
    images: List[Dict[str, Any]]
    
    # Review and editing phase
    revisions: List[Dict[str, Any]]
    quality_metrics: Optional[QualityMetrics]
    final_content: Optional[str]
    
    # Workflow control
    workflow_status: StateStatus
    current_step: str
    completed_steps: List[str]
    failed_steps: List[str]
    retry_count: int
    
    # Metadata
    workflow_id: str
    created_at: str
    updated_at: str
    execution_time: float
    
    # Human interaction
    human_feedback: List[Dict[str, Any]]
    approval_status: str
    review_required: bool


# Content Editing Workflow State
class ContentEditingState(TypedDict, total=False):
    """State schema for content editing workflows."""
    
    # Input
    content: str
    editing_requirements: Dict[str, Any]
    target_quality_score: float
    
    # Processing stages
    grammar_checked_content: Optional[str]
    seo_optimized_content: Optional[str]
    style_adjusted_content: Optional[str]
    fact_checked_content: Optional[str]
    
    # Quality assessment
    initial_quality: Optional[QualityMetrics]
    current_quality: Optional[QualityMetrics]
    quality_history: List[QualityMetrics]
    
    # Editing iterations
    editing_iterations: int
    max_iterations: int
    improvement_threshold: float
    
    # Final output
    final_content: Optional[str]
    quality_report: Dict[str, Any]
    editing_summary: Dict[str, Any]
    
    # Workflow control
    workflow_status: StateStatus
    current_step: str
    completed_steps: List[str]
    
    # Metadata
    workflow_id: str
    started_at: str
    completed_at: Optional[str]


# Research Workflow State
class ResearchState(TypedDict, total=False):
    """State schema for research workflows."""
    
    # Input
    topic: str
    research_scope: Dict[str, Any]
    depth_level: str  # surface, medium, deep
    
    # Research phases
    web_research_results: List[Dict[str, Any]]
    trend_analysis_results: Dict[str, Any]
    competitive_research: Dict[str, Any]
    fact_verification_results: List[Dict[str, Any]]
    
    # Aggregated results
    consolidated_findings: Dict[str, Any]
    research_summary: str
    source_credibility_scores: Dict[str, float]
    
    # Quality control
    information_gaps: List[str]
    conflicting_information: List[Dict[str, Any]]
    verification_status: Dict[str, str]
    
    # Workflow control
    workflow_status: StateStatus
    research_progress: float
    completed_research_types: List[str]
    
    # Metadata
    workflow_id: str
    research_started: str
    estimated_completion: Optional[str]


class StateManager:
    """Manages LangGraph workflow states."""
    
    def __init__(self):
        self.logger = get_framework_logger("LangGraph")
        self.state_schemas = {
            "content_creation": ContentCreationState,
            "content_editing": ContentEditingState,
            "research": ResearchState
        }
    
    def create_initial_state(
        self,
        workflow_type: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create initial state for a workflow."""
        
        if workflow_type not in self.state_schemas:
            raise LangGraphError(f"Unknown workflow type: {workflow_type}")
        
        try:
            if workflow_type == "content_creation":
                return self._create_content_creation_state(inputs)
            elif workflow_type == "content_editing":
                return self._create_content_editing_state(inputs)
            elif workflow_type == "research":
                return self._create_research_state(inputs)
            else:
                raise LangGraphError(f"Unsupported workflow type: {workflow_type}")
                
        except Exception as e:
            raise LangGraphError(f"Failed to create initial state: {e}")
    
    def _create_content_creation_state(self, inputs: Dict[str, Any]) -> ContentCreationState:
        """Create initial content creation state."""
        now = datetime.now().isoformat()
        
        return ContentCreationState(
            topic=inputs.get("topic", ""),
            content_type=ContentType(inputs.get("content_type", "blog_post")),
            requirements=inputs.get("requirements", {}),
            research_data=None,
            research_status="pending",
            strategy=None,
            strategy_status="pending",
            outline=None,
            content=None,
            headlines=[],
            images=[],
            revisions=[],
            quality_metrics=None,
            final_content=None,
            workflow_status=StateStatus.INITIALIZED,
            current_step="research",
            completed_steps=[],
            failed_steps=[],
            retry_count=0,
            workflow_id=inputs.get("workflow_id", f"content_{id(inputs)}"),
            created_at=now,
            updated_at=now,
            execution_time=0.0,
            human_feedback=[],
            approval_status="pending",
            review_required=inputs.get("review_required", False)
        )
    
    def _create_content_editing_state(self, inputs: Dict[str, Any]) -> ContentEditingState:
        """Create initial content editing state."""
        now = datetime.now().isoformat()
        
        return ContentEditingState(
            content=inputs.get("content", ""),
            editing_requirements=inputs.get("editing_requirements", {}),
            target_quality_score=inputs.get("target_quality_score", 0.8),
            grammar_checked_content=None,
            seo_optimized_content=None,
            style_adjusted_content=None,
            fact_checked_content=None,
            initial_quality=None,
            current_quality=None,
            quality_history=[],
            editing_iterations=0,
            max_iterations=inputs.get("max_iterations", 3),
            improvement_threshold=inputs.get("improvement_threshold", 0.1),
            final_content=None,
            quality_report={},
            editing_summary={},
            workflow_status=StateStatus.INITIALIZED,
            current_step="grammar_check",
            completed_steps=[],
            workflow_id=inputs.get("workflow_id", f"editing_{id(inputs)}"),
            started_at=now,
            completed_at=None
        )
    
    def _create_research_state(self, inputs: Dict[str, Any]) -> ResearchState:
        """Create initial research state."""
        now = datetime.now().isoformat()
        
        return ResearchState(
            topic=inputs.get("topic", ""),
            research_scope=inputs.get("research_scope", {}),
            depth_level=inputs.get("depth_level", "medium"),
            web_research_results=[],
            trend_analysis_results={},
            competitive_research={},
            fact_verification_results=[],
            consolidated_findings={},
            research_summary="",
            source_credibility_scores={},
            information_gaps=[],
            conflicting_information=[],
            verification_status={},
            workflow_status=StateStatus.INITIALIZED,
            research_progress=0.0,
            completed_research_types=[],
            workflow_id=inputs.get("workflow_id", f"research_{id(inputs)}"),
            research_started=now,
            estimated_completion=None
        )
    
    def validate_state(self, workflow_type: str, state: Dict[str, Any]) -> bool:
        """Validate state against schema."""
        try:
            if workflow_type not in self.state_schemas:
                return False
            
            schema = self.state_schemas[workflow_type]
            
            # Basic type checking (simplified validation)
            for key, value in state.items():
                if key in schema.__annotations__:
                    # More comprehensive validation could be added here
                    pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"State validation failed: {e}")
            return False
    
    def update_state(
        self,
        current_state: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update state with new values."""
        new_state = current_state.copy()
        new_state.update(updates)
        new_state["updated_at"] = datetime.now().isoformat()
        
        return new_state
    
    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "workflow_id": state.get("workflow_id"),
            "workflow_status": state.get("workflow_status"),
            "current_step": state.get("current_step"),
            "completed_steps": state.get("completed_steps", []),
            "failed_steps": state.get("failed_steps", []),
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "progress": len(state.get("completed_steps", [])) / max(len(state.get("completed_steps", [])) + len(state.get("failed_steps", [])) + 1, 1)
        }
    
    def serialize_state(self, state: Dict[str, Any]) -> str:
        """Serialize state to JSON string."""
        def serialize_custom_objects(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return str(obj)
        
        try:
            return json.dumps(state, default=serialize_custom_objects, indent=2)
        except Exception as e:
            raise LangGraphError(f"Failed to serialize state: {e}")
    
    def deserialize_state(self, state_json: str) -> Dict[str, Any]:
        """Deserialize state from JSON string."""
        try:
            return json.loads(state_json)
        except Exception as e:
            raise LangGraphError(f"Failed to deserialize state: {e}")


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager