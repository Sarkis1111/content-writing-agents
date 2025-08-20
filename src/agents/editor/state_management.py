"""
Editor Agent State Management System

This module implements comprehensive state management for the Editor Agent,
providing revision tracking, quality gate state persistence, workflow state management,
and detailed audit trails for all editing operations.

Key Features:
- Comprehensive revision tracking and versioning
- Quality gate state persistence and history
- Workflow state management with checkpoint support
- Detailed audit trails and change tracking
- Performance metrics and analytics collection
- State recovery and rollback capabilities
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import hashlib
import copy

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, src_path)

# Core imports with fallbacks
try:
    from core.errors import StateManagementError, ValidationError
    from core.logging.logger import get_logger
    from core.monitoring.metrics import get_metrics_collector
except ImportError:
    # Mock implementations
    import logging
    class StateManagementError(Exception): pass
    class ValidationError(Exception): pass
    def get_logger(name): return logging.getLogger(name)
    def get_metrics_collector(): return None

# Import editor components
try:
    from .quality_assurance import QualityAssessmentResult, QualityDimension
    from .human_escalation import EscalationRequest
except ImportError:
    # Create minimal fallbacks
    class QualityAssessmentResult:
        def __init__(self): 
            self.overall_score = 0.0
            self.to_dict = lambda: {}
    class QualityDimension: pass
    class EscalationRequest: 
        def __init__(self): 
            self.escalation_id = "test"
            self.to_dict = lambda: {}

logger = get_logger(__name__)


class RevisionType(str, Enum):
    """Types of revisions"""
    GRAMMAR_FIX = "grammar_fix"
    STYLE_ADJUSTMENT = "style_adjustment"
    SEO_OPTIMIZATION = "seo_optimization"
    READABILITY_IMPROVEMENT = "readability_improvement"
    SENTIMENT_ADJUSTMENT = "sentiment_adjustment"
    HUMAN_EDIT = "human_edit"
    AUTO_CORRECTION = "auto_correction"
    QUALITY_IMPROVEMENT = "quality_improvement"
    USER_FEEDBACK = "user_feedback"
    ROLLBACK = "rollback"


class WorkflowState(str, Enum):
    """Workflow execution states"""
    INITIALIZED = "initialized"
    GRAMMAR_CHECK = "grammar_check"
    STYLE_REVIEW = "style_review"
    SEO_OPTIMIZATION = "seo_optimization"
    READABILITY_ASSESSMENT = "readability_assessment"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    QUALITY_GATE = "quality_gate"
    HUMAN_ESCALATION = "human_escalation"
    REVISION_APPLICATION = "revision_application"
    FINAL_REVIEW = "final_review"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class QualityGateDecision(str, Enum):
    """Quality gate decisions"""
    PASS = "pass"
    CONDITIONAL_PASS = "conditional_pass"
    FAIL = "fail"
    ESCALATE = "escalate"
    REVISE = "revise"


@dataclass
class ContentRevision:
    """Individual content revision record"""
    
    revision_id: str = field(default_factory=lambda: str(abs(hash(datetime.now().isoformat()))))
    
    # Revision metadata
    timestamp: datetime = field(default_factory=datetime.now)
    revision_type: RevisionType = RevisionType.AUTO_CORRECTION
    stage: str = ""
    
    # Content changes
    content_before: str = ""
    content_after: str = ""
    content_hash_before: str = ""
    content_hash_after: str = ""
    
    # Change details
    changes_made: List[Dict[str, Any]] = field(default_factory=list)
    change_summary: str = ""
    
    # Quality impact
    quality_before: float = 0.0
    quality_after: float = 0.0
    quality_improvement: float = 0.0
    
    # Metadata
    tool_used: Optional[str] = None
    human_reviewer: Optional[str] = None
    confidence: float = 1.0
    processing_time: float = 0.0
    
    def calculate_content_hash(self, content: str) -> str:
        """Calculate content hash for change detection"""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def __post_init__(self):
        """Calculate hashes after initialization"""
        if not self.content_hash_before and self.content_before:
            self.content_hash_before = self.calculate_content_hash(self.content_before)
        if not self.content_hash_after and self.content_after:
            self.content_hash_after = self.calculate_content_hash(self.content_after)
        
        # Calculate quality improvement
        if self.quality_before > 0 and self.quality_after > 0:
            self.quality_improvement = self.quality_after - self.quality_before
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "revision_id": self.revision_id,
            "timestamp": self.timestamp.isoformat(),
            "revision_type": self.revision_type.value,
            "stage": self.stage,
            "content_before": self.content_before,
            "content_after": self.content_after,
            "content_hash_before": self.content_hash_before,
            "content_hash_after": self.content_hash_after,
            "changes_made": self.changes_made,
            "change_summary": self.change_summary,
            "quality_before": self.quality_before,
            "quality_after": self.quality_after,
            "quality_improvement": self.quality_improvement,
            "tool_used": self.tool_used,
            "human_reviewer": self.human_reviewer,
            "confidence": self.confidence,
            "processing_time": self.processing_time
        }


@dataclass
class QualityGateRecord:
    """Quality gate execution record"""
    
    gate_id: str = field(default_factory=lambda: str(abs(hash(datetime.now().isoformat()))))
    
    # Gate metadata
    timestamp: datetime = field(default_factory=datetime.now)
    stage: str = ""
    iteration: int = 1
    
    # Assessment data
    quality_assessment: Optional[QualityAssessmentResult] = None
    overall_score: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    
    # Gate decision
    decision: QualityGateDecision = QualityGateDecision.FAIL
    decision_reasons: List[str] = field(default_factory=list)
    decision_confidence: float = 1.0
    
    # Criteria evaluation
    criteria_met: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    threshold_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Next actions
    recommended_actions: List[str] = field(default_factory=list)
    escalation_recommended: bool = False
    human_review_required: bool = False
    
    # Performance
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "gate_id": self.gate_id,
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage,
            "iteration": self.iteration,
            "quality_assessment": self.quality_assessment.to_dict() if self.quality_assessment else None,
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "decision": self.decision.value,
            "decision_reasons": self.decision_reasons,
            "decision_confidence": self.decision_confidence,
            "criteria_met": self.criteria_met,
            "criteria_failed": self.criteria_failed,
            "threshold_comparisons": self.threshold_comparisons,
            "recommended_actions": self.recommended_actions,
            "escalation_recommended": self.escalation_recommended,
            "human_review_required": self.human_review_required,
            "processing_time": self.processing_time
        }


@dataclass
class WorkflowStateCheckpoint:
    """Workflow state checkpoint"""
    
    checkpoint_id: str = field(default_factory=lambda: str(abs(hash(datetime.now().isoformat()))))
    
    # Checkpoint metadata
    timestamp: datetime = field(default_factory=datetime.now)
    current_state: WorkflowState = WorkflowState.INITIALIZED
    previous_state: Optional[WorkflowState] = None
    
    # Content state
    current_content: str = ""
    content_hash: str = ""
    
    # Workflow context
    completed_stages: List[str] = field(default_factory=list)
    failed_stages: List[str] = field(default_factory=list)
    current_iteration: int = 1
    max_iterations: int = 3
    
    # Quality state
    current_quality_scores: Dict[str, float] = field(default_factory=dict)
    quality_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Escalation state
    escalation_requests: List[str] = field(default_factory=list)  # escalation IDs
    human_feedback_pending: bool = False
    
    # Performance metrics
    total_processing_time: float = 0.0
    stage_processing_times: Dict[str, float] = field(default_factory=dict)
    
    def calculate_content_hash(self) -> str:
        """Calculate content hash"""
        if self.current_content:
            return hashlib.md5(self.current_content.encode()).hexdigest()[:12]
        return ""
    
    def __post_init__(self):
        """Calculate content hash after initialization"""
        if not self.content_hash:
            self.content_hash = self.calculate_content_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp.isoformat(),
            "current_state": self.current_state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "current_content": self.current_content,
            "content_hash": self.content_hash,
            "completed_stages": self.completed_stages,
            "failed_stages": self.failed_stages,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "current_quality_scores": self.current_quality_scores,
            "quality_history": self.quality_history,
            "escalation_requests": self.escalation_requests,
            "human_feedback_pending": self.human_feedback_pending,
            "total_processing_time": self.total_processing_time,
            "stage_processing_times": self.stage_processing_times
        }


@dataclass
class EditingSessionState:
    """Complete editing session state"""
    
    session_id: str = field(default_factory=lambda: str(abs(hash(datetime.now().isoformat()))))
    
    # Session metadata
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Content evolution
    original_content: str = ""
    current_content: str = ""
    final_content: Optional[str] = None
    
    # Session configuration
    editing_requirements: Dict[str, Any] = field(default_factory=dict)
    quality_criteria: Dict[str, Any] = field(default_factory=dict)
    
    # Revision tracking
    revisions: List[ContentRevision] = field(default_factory=list)
    total_revisions: int = 0
    
    # Quality gate tracking
    quality_gates: List[QualityGateRecord] = field(default_factory=list)
    quality_gate_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Workflow state
    current_checkpoint: Optional[WorkflowStateCheckpoint] = None
    checkpoint_history: List[WorkflowStateCheckpoint] = field(default_factory=list)
    workflow_status: str = "initialized"
    
    # Performance metrics
    total_processing_time: float = 0.0
    tool_execution_times: Dict[str, float] = field(default_factory=dict)
    quality_progression: List[float] = field(default_factory=list)
    
    # Human interaction
    escalation_history: List[str] = field(default_factory=list)  # escalation IDs
    human_feedback_count: int = 0
    
    def update_timestamp(self):
        """Update the last updated timestamp"""
        self.updated_at = datetime.now()
    
    def add_revision(self, revision: ContentRevision):
        """Add a revision to the session"""
        self.revisions.append(revision)
        self.total_revisions = len(self.revisions)
        self.update_timestamp()
    
    def add_quality_gate(self, gate_record: QualityGateRecord):
        """Add a quality gate record"""
        self.quality_gates.append(gate_record)
        self.quality_gate_history.append(gate_record.to_dict())
        self.update_timestamp()
    
    def create_checkpoint(self, state: WorkflowState, content: str) -> WorkflowStateCheckpoint:
        """Create a workflow checkpoint"""
        previous_state = self.current_checkpoint.current_state if self.current_checkpoint else None
        
        checkpoint = WorkflowStateCheckpoint(
            current_state=state,
            previous_state=previous_state,
            current_content=content,
            current_iteration=len(self.checkpoint_history) + 1
        )
        
        self.current_checkpoint = checkpoint
        self.checkpoint_history.append(checkpoint)
        self.update_timestamp()
        
        return checkpoint
    
    def get_revision_summary(self) -> Dict[str, Any]:
        """Get summary of all revisions"""
        if not self.revisions:
            return {"total_revisions": 0}
        
        revision_types = {}
        total_quality_improvement = 0.0
        total_processing_time = 0.0
        
        for revision in self.revisions:
            rev_type = revision.revision_type.value
            revision_types[rev_type] = revision_types.get(rev_type, 0) + 1
            total_quality_improvement += revision.quality_improvement
            total_processing_time += revision.processing_time
        
        return {
            "total_revisions": len(self.revisions),
            "revision_types": revision_types,
            "total_quality_improvement": total_quality_improvement,
            "avg_quality_improvement": total_quality_improvement / len(self.revisions),
            "total_revision_time": total_processing_time,
            "avg_revision_time": total_processing_time / len(self.revisions)
        }
    
    def get_quality_progression(self) -> List[Dict[str, Any]]:
        """Get quality score progression over time"""
        progression = []
        
        for i, gate in enumerate(self.quality_gates):
            progression.append({
                "iteration": i + 1,
                "timestamp": gate.timestamp.isoformat(),
                "overall_score": gate.overall_score,
                "dimension_scores": gate.dimension_scores,
                "decision": gate.decision.value
            })
        
        return progression
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "original_content": self.original_content,
            "current_content": self.current_content,
            "final_content": self.final_content,
            "editing_requirements": self.editing_requirements,
            "quality_criteria": self.quality_criteria,
            "revisions": [revision.to_dict() for revision in self.revisions],
            "total_revisions": self.total_revisions,
            "quality_gates": [gate.to_dict() for gate in self.quality_gates],
            "quality_gate_history": self.quality_gate_history,
            "current_checkpoint": self.current_checkpoint.to_dict() if self.current_checkpoint else None,
            "checkpoint_history": [checkpoint.to_dict() for checkpoint in self.checkpoint_history],
            "workflow_status": self.workflow_status,
            "total_processing_time": self.total_processing_time,
            "tool_execution_times": self.tool_execution_times,
            "quality_progression": self.quality_progression,
            "escalation_history": self.escalation_history,
            "human_feedback_count": self.human_feedback_count,
            "revision_summary": self.get_revision_summary(),
            "quality_progression_data": self.get_quality_progression()
        }


class StateManager:
    """Central state management system"""
    
    def __init__(self, max_sessions: int = 1000, session_ttl_hours: int = 24):
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Configuration
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(hours=session_ttl_hours)
        
        # Active sessions
        self.active_sessions: Dict[str, EditingSessionState] = {}
        
        # Performance tracking
        self.state_metrics = {
            "total_sessions": 0,
            "active_sessions": 0,
            "avg_session_duration": 0.0,
            "avg_revisions_per_session": 0.0,
            "avg_quality_improvement": 0.0
        }
    
    def create_session(
        self,
        original_content: str,
        editing_requirements: Optional[Dict[str, Any]] = None,
        quality_criteria: Optional[Dict[str, Any]] = None
    ) -> EditingSessionState:
        """
        Create new editing session
        
        Args:
            original_content: Original content to edit
            editing_requirements: Editing requirements and preferences
            quality_criteria: Quality assessment criteria
            
        Returns:
            New editing session state
        """
        
        try:
            # Clean up old sessions first
            self._cleanup_old_sessions()
            
            # Create new session
            session = EditingSessionState(
                original_content=original_content,
                current_content=original_content,
                editing_requirements=editing_requirements or {},
                quality_criteria=quality_criteria or {}
            )
            
            # Create initial checkpoint
            session.create_checkpoint(WorkflowState.INITIALIZED, original_content)
            
            # Store session
            self.active_sessions[session.session_id] = session
            
            # Update metrics
            self.state_metrics["total_sessions"] += 1
            self.state_metrics["active_sessions"] = len(self.active_sessions)
            
            # Log metrics
            if self.metrics:
                self.metrics.record_counter("editing_sessions_created")
                self.metrics.record_gauge("active_editing_sessions", len(self.active_sessions))
            
            self.logger.info(f"Created editing session: {session.session_id}")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise StateManagementError(f"Session creation failed: {str(e)}")
    
    def get_session(self, session_id: str) -> Optional[EditingSessionState]:
        """Get existing session by ID"""
        return self.active_sessions.get(session_id)
    
    def update_session_content(
        self,
        session_id: str,
        new_content: str,
        revision_type: RevisionType = RevisionType.AUTO_CORRECTION,
        stage: str = "",
        tool_used: Optional[str] = None,
        changes_made: Optional[List[Dict[str, Any]]] = None,
        quality_before: float = 0.0,
        quality_after: float = 0.0
    ) -> Optional[ContentRevision]:
        """
        Update session content with revision tracking
        
        Returns:
            Created revision record or None if session not found
        """
        
        try:
            session = self.get_session(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return None
            
            # Create revision record
            revision = ContentRevision(
                revision_type=revision_type,
                stage=stage,
                content_before=session.current_content,
                content_after=new_content,
                changes_made=changes_made or [],
                quality_before=quality_before,
                quality_after=quality_after,
                tool_used=tool_used
            )
            
            # Generate change summary
            revision.change_summary = self._generate_change_summary(revision)
            
            # Update session
            session.current_content = new_content
            session.add_revision(revision)
            
            self.logger.debug(f"Updated content for session {session_id}: {revision.revision_type.value}")
            return revision
            
        except Exception as e:
            self.logger.error(f"Failed to update session content: {e}")
            return None
    
    def create_quality_gate_record(
        self,
        session_id: str,
        stage: str,
        quality_assessment: QualityAssessmentResult,
        decision: QualityGateDecision,
        decision_reasons: List[str],
        criteria_met: List[str],
        criteria_failed: List[str],
        recommended_actions: List[str]
    ) -> Optional[QualityGateRecord]:
        """
        Create quality gate record
        
        Returns:
            Created quality gate record or None if session not found
        """
        
        try:
            session = self.get_session(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return None
            
            # Create quality gate record
            gate_record = QualityGateRecord(
                stage=stage,
                iteration=len(session.quality_gates) + 1,
                quality_assessment=quality_assessment,
                overall_score=quality_assessment.overall_score,
                decision=decision,
                decision_reasons=decision_reasons,
                criteria_met=criteria_met,
                criteria_failed=criteria_failed,
                recommended_actions=recommended_actions,
                escalation_recommended=decision == QualityGateDecision.ESCALATE,
                human_review_required=decision in [QualityGateDecision.ESCALATE, QualityGateDecision.FAIL]
            )
            
            # Extract dimension scores
            if hasattr(quality_assessment, 'dimension_metrics'):
                for dimension, metrics in quality_assessment.dimension_metrics.items():
                    if hasattr(dimension, 'value'):
                        gate_record.dimension_scores[dimension.value] = metrics.score
            
            # Add to session
            session.add_quality_gate(gate_record)
            
            self.logger.debug(f"Created quality gate record for session {session_id}: {decision.value}")
            return gate_record
            
        except Exception as e:
            self.logger.error(f"Failed to create quality gate record: {e}")
            return None
    
    def update_workflow_state(
        self,
        session_id: str,
        new_state: WorkflowState,
        completed_stages: Optional[List[str]] = None,
        failed_stages: Optional[List[str]] = None
    ) -> Optional[WorkflowStateCheckpoint]:
        """
        Update workflow state with checkpoint
        
        Returns:
            Created checkpoint or None if session not found
        """
        
        try:
            session = self.get_session(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return None
            
            # Create checkpoint
            checkpoint = session.create_checkpoint(new_state, session.current_content)
            
            # Update stage lists
            if completed_stages:
                checkpoint.completed_stages.extend(completed_stages)
            if failed_stages:
                checkpoint.failed_stages.extend(failed_stages)
            
            # Update session workflow status
            session.workflow_status = new_state.value
            
            self.logger.debug(f"Updated workflow state for session {session_id}: {new_state.value}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to update workflow state: {e}")
            return None
    
    def add_escalation_to_session(
        self,
        session_id: str,
        escalation_request: EscalationRequest
    ) -> bool:
        """
        Add escalation request to session
        
        Returns:
            True if successful, False otherwise
        """
        
        try:
            session = self.get_session(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return False
            
            # Add escalation ID to session
            session.escalation_history.append(escalation_request.escalation_id)
            
            # Update current checkpoint if exists
            if session.current_checkpoint:
                session.current_checkpoint.escalation_requests.append(escalation_request.escalation_id)
                session.current_checkpoint.human_feedback_pending = True
            
            session.update_timestamp()
            
            self.logger.debug(f"Added escalation {escalation_request.escalation_id} to session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add escalation to session: {e}")
            return False
    
    def complete_session(
        self,
        session_id: str,
        final_content: str,
        final_quality_score: float
    ) -> bool:
        """
        Mark session as completed
        
        Returns:
            True if successful, False otherwise
        """
        
        try:
            session = self.get_session(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return False
            
            # Mark as completed
            session.completed_at = datetime.now()
            session.final_content = final_content
            session.current_content = final_content
            session.workflow_status = "completed"
            
            # Create final checkpoint
            session.create_checkpoint(WorkflowState.COMPLETED, final_content)
            
            # Calculate final metrics
            session_duration = (session.completed_at - session.started_at).total_seconds()
            session.total_processing_time = session_duration
            
            # Add final quality score to progression
            session.quality_progression.append(final_quality_score)
            
            # Update global metrics
            self._update_completion_metrics(session)
            
            self.logger.info(f"Completed editing session {session_id}: {len(session.revisions)} revisions, {session_duration:.1f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to complete session: {e}")
            return False
    
    def rollback_to_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str
    ) -> bool:
        """
        Rollback session to a specific checkpoint
        
        Returns:
            True if successful, False otherwise
        """
        
        try:
            session = self.get_session(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return False
            
            # Find checkpoint
            target_checkpoint = None
            for checkpoint in session.checkpoint_history:
                if checkpoint.checkpoint_id == checkpoint_id:
                    target_checkpoint = checkpoint
                    break
            
            if not target_checkpoint:
                self.logger.warning(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            # Create rollback revision
            rollback_revision = ContentRevision(
                revision_type=RevisionType.ROLLBACK,
                stage="rollback",
                content_before=session.current_content,
                content_after=target_checkpoint.current_content,
                change_summary=f"Rolled back to checkpoint {checkpoint_id}"
            )
            
            # Update session state
            session.current_content = target_checkpoint.current_content
            session.add_revision(rollback_revision)
            session.current_checkpoint = target_checkpoint
            
            self.logger.info(f"Rolled back session {session_id} to checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback session: {e}")
            return False
    
    def get_session_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session analytics"""
        
        try:
            session = self.get_session(session_id)
            if not session:
                return None
            
            # Basic metrics
            analytics = {
                "session_id": session_id,
                "started_at": session.started_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "workflow_status": session.workflow_status,
                "total_processing_time": session.total_processing_time
            }
            
            # Content metrics
            analytics["content_metrics"] = {
                "original_length": len(session.original_content),
                "current_length": len(session.current_content),
                "length_change": len(session.current_content) - len(session.original_content),
                "content_similarity": self._calculate_content_similarity(
                    session.original_content, session.current_content
                )
            }
            
            # Revision analytics
            analytics["revision_analytics"] = session.get_revision_summary()
            
            # Quality progression
            analytics["quality_analytics"] = {
                "quality_progression": session.get_quality_progression(),
                "total_quality_gates": len(session.quality_gates),
                "gate_decisions": self._analyze_gate_decisions(session.quality_gates)
            }
            
            # Workflow analytics
            analytics["workflow_analytics"] = {
                "total_checkpoints": len(session.checkpoint_history),
                "completed_stages": session.current_checkpoint.completed_stages if session.current_checkpoint else [],
                "failed_stages": session.current_checkpoint.failed_stages if session.current_checkpoint else [],
                "current_iteration": session.current_checkpoint.current_iteration if session.current_checkpoint else 0
            }
            
            # Human interaction analytics
            analytics["human_interaction"] = {
                "escalation_count": len(session.escalation_history),
                "human_feedback_count": session.human_feedback_count,
                "escalation_rate": len(session.escalation_history) / max(1, len(session.quality_gates))
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get session analytics: {e}")
            return None
    
    def _generate_change_summary(self, revision: ContentRevision) -> str:
        """Generate human-readable change summary"""
        
        if not revision.changes_made:
            content_before_len = len(revision.content_before)
            content_after_len = len(revision.content_after)
            
            if content_after_len > content_before_len:
                return f"Added {content_after_len - content_before_len} characters"
            elif content_after_len < content_before_len:
                return f"Removed {content_before_len - content_after_len} characters"
            else:
                return "Content modified without length change"
        
        # Summarize specific changes
        change_types = {}
        for change in revision.changes_made:
            change_type = change.get("type", "unknown")
            change_types[change_type] = change_types.get(change_type, 0) + 1
        
        summary_parts = []
        for change_type, count in change_types.items():
            if count == 1:
                summary_parts.append(f"1 {change_type}")
            else:
                summary_parts.append(f"{count} {change_type}s")
        
        return "; ".join(summary_parts) if summary_parts else "Content modified"
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity (simplified)"""
        
        if not content1 and not content2:
            return 1.0
        
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _analyze_gate_decisions(self, gates: List[QualityGateRecord]) -> Dict[str, int]:
        """Analyze quality gate decisions"""
        
        decisions = {}
        for gate in gates:
            decision = gate.decision.value
            decisions[decision] = decisions.get(decision, 0) + 1
        
        return decisions
    
    def _cleanup_old_sessions(self):
        """Remove old expired sessions"""
        
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                # Check if session is expired
                session_age = current_time - session.updated_at
                
                if session_age > self.session_ttl:
                    expired_sessions.append(session_id)
                elif len(self.active_sessions) > self.max_sessions:
                    # Remove oldest sessions if over limit
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions[:max(0, len(expired_sessions) - 10)]:  # Keep at least 10
                del self.active_sessions[session_id]
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            # Update metrics
            self.state_metrics["active_sessions"] = len(self.active_sessions)
            
        except Exception as e:
            self.logger.warning(f"Session cleanup failed: {e}")
    
    def _update_completion_metrics(self, completed_session: EditingSessionState):
        """Update metrics when session completes"""
        
        try:
            # Calculate session duration
            if completed_session.completed_at and completed_session.started_at:
                duration = (completed_session.completed_at - completed_session.started_at).total_seconds()
            else:
                duration = 0.0
            
            # Update averages
            total_completed = self.state_metrics.get("completed_sessions", 0) + 1
            current_avg_duration = self.state_metrics.get("avg_session_duration", 0.0)
            
            self.state_metrics["avg_session_duration"] = (
                (current_avg_duration * (total_completed - 1) + duration) / total_completed
            )
            
            # Update revision averages
            current_avg_revisions = self.state_metrics.get("avg_revisions_per_session", 0.0)
            self.state_metrics["avg_revisions_per_session"] = (
                (current_avg_revisions * (total_completed - 1) + completed_session.total_revisions) / total_completed
            )
            
            # Calculate quality improvement
            if completed_session.quality_progression:
                quality_improvement = (
                    completed_session.quality_progression[-1] - completed_session.quality_progression[0]
                    if len(completed_session.quality_progression) > 1 else 0.0
                )
                
                current_avg_improvement = self.state_metrics.get("avg_quality_improvement", 0.0)
                self.state_metrics["avg_quality_improvement"] = (
                    (current_avg_improvement * (total_completed - 1) + quality_improvement) / total_completed
                )
            
            self.state_metrics["completed_sessions"] = total_completed
            
            # Log metrics
            if self.metrics:
                self.metrics.record_counter("editing_sessions_completed")
                self.metrics.record_histogram("editing_session_duration", duration)
                self.metrics.record_histogram("editing_session_revisions", completed_session.total_revisions)
                if completed_session.quality_progression:
                    self.metrics.record_histogram("editing_quality_improvement", quality_improvement)
            
        except Exception as e:
            self.logger.warning(f"Failed to update completion metrics: {e}")
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global state management metrics"""
        
        return {
            **self.state_metrics,
            "active_sessions": len(self.active_sessions),
            "memory_usage_estimate": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage (simplified)"""
        
        total_sessions = len(self.active_sessions)
        total_revisions = sum(len(session.revisions) for session in self.active_sessions.values())
        total_checkpoints = sum(len(session.checkpoint_history) for session in self.active_sessions.values())
        
        return {
            "total_sessions": total_sessions,
            "total_revisions": total_revisions,
            "total_checkpoints": total_checkpoints,
            "estimated_mb": (total_sessions * 0.1 + total_revisions * 0.05 + total_checkpoints * 0.02)
        }


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the global state manager instance"""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


# Export main classes
__all__ = [
    'RevisionType',
    'WorkflowState',
    'QualityGateDecision',
    'ContentRevision',
    'QualityGateRecord',
    'WorkflowStateCheckpoint',
    'EditingSessionState',
    'StateManager',
    'get_state_manager'
]


if __name__ == "__main__":
    # Example usage and testing
    async def test_state_management():
        """Test the state management system"""
        
        original_content = "This is the original content that needs editing for grammer and style improvements."
        
        try:
            manager = get_state_manager()
            
            # Create session
            session = manager.create_session(
                original_content,
                editing_requirements={
                    "target_keywords": ["content", "editing"],
                    "writing_style": "professional"
                },
                quality_criteria={"min_score": 80.0}
            )
            
            print("State Management Test Results:")
            print(f"Session ID: {session.session_id}")
            print(f"Initial checkpoints: {len(session.checkpoint_history)}")
            
            # Update content with revisions
            revised_content = "This is the original content that needs editing for grammar and style improvements."
            
            revision = manager.update_session_content(
                session.session_id,
                revised_content,
                RevisionType.GRAMMAR_FIX,
                stage="grammar_check",
                tool_used="grammar_checker",
                quality_before=65.0,
                quality_after=82.0
            )
            
            if revision:
                print(f"Revision created: {revision.revision_type.value}")
                print(f"Quality improvement: +{revision.quality_improvement:.1f}")
            
            # Update workflow state
            checkpoint = manager.update_workflow_state(
                session.session_id,
                WorkflowState.SEO_OPTIMIZATION,
                completed_stages=["grammar_check"],
                failed_stages=[]
            )
            
            if checkpoint:
                print(f"Workflow updated: {checkpoint.current_state.value}")
            
            # Get analytics
            analytics = manager.get_session_analytics(session.session_id)
            if analytics:
                print(f"Content length change: {analytics['content_metrics']['length_change']}")
                print(f"Total revisions: {analytics['revision_analytics']['total_revisions']}")
                print(f"Content similarity: {analytics['content_metrics']['content_similarity']:.2f}")
            
            # Complete session
            completed = manager.complete_session(
                session.session_id,
                revised_content,
                final_quality_score=85.0
            )
            
            print(f"Session completed: {completed}")
            
            # Get global metrics
            global_metrics = manager.get_global_metrics()
            print(f"Total sessions created: {global_metrics['total_sessions']}")
            print(f"Average session duration: {global_metrics.get('avg_session_duration', 0):.1f}s")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_state_management())