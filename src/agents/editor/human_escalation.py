"""
Editor Agent Human Escalation System

This module implements intelligent human escalation workflows for the Editor Agent.
It provides sophisticated escalation decision-making, human-in-the-loop integration,
escalation prioritization, and workflow management for complex editing scenarios.

Key Features:
- Intelligent escalation decision algorithms
- Human-in-the-loop workflow integration
- Escalation prioritization and routing
- Expert reviewer assignment and matching
- Escalation tracking and analytics
- Automated follow-up and resolution workflows
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, src_path)

# Core imports with fallbacks
try:
    from core.errors import EscalationError, WorkflowError
    from core.logging.logger import get_logger
    from core.monitoring.metrics import get_metrics_collector
except ImportError:
    # Mock implementations
    import logging
    class EscalationError(Exception): pass
    class WorkflowError(Exception): pass
    def get_logger(name): return logging.getLogger(name)
    def get_metrics_collector(): return None

# Import quality assurance components
try:
    from .quality_assurance import QualityAssessmentResult, QualityDimension, QualityIssue, AssessmentResult
except ImportError:
    # Create minimal fallbacks for development
    class QualityAssessmentResult:
        def __init__(self): 
            self.all_issues = []
            self.overall_score = 0.0
            self.assessment_result = "fail"
    class QualityDimension: pass
    class QualityIssue: pass
    class AssessmentResult: pass

logger = get_logger(__name__)


class EscalationReason(str, Enum):
    """Reasons for human escalation"""
    QUALITY_FAILURE = "quality_failure"             # Quality standards not met
    CRITICAL_ISSUES = "critical_issues"             # Critical issues detected
    COMPLEX_CONTENT = "complex_content"             # Content too complex for automated editing
    DOMAIN_EXPERTISE = "domain_expertise"          # Requires domain-specific knowledge
    BRAND_COMPLIANCE = "brand_compliance"          # Brand compliance concerns
    LEGAL_REVIEW = "legal_review"                  # Legal or regulatory review needed
    TECHNICAL_ERROR = "technical_error"            # Technical errors in automated processing
    USER_REQUEST = "user_request"                  # User explicitly requested human review
    AMBIGUOUS_CONTEXT = "ambiguous_context"        # Context is ambiguous or unclear
    CREATIVE_JUDGMENT = "creative_judgment"        # Requires creative/subjective judgment


class EscalationPriority(str, Enum):
    """Escalation priority levels"""
    URGENT = "urgent"           # <2 hours response needed
    HIGH = "high"               # <8 hours response needed  
    MEDIUM = "medium"           # <24 hours response needed
    LOW = "low"                 # <72 hours response needed


class EscalationStatus(str, Enum):
    """Escalation workflow status"""
    PENDING = "pending"         # Waiting for assignment
    ASSIGNED = "assigned"       # Assigned to reviewer
    IN_REVIEW = "in_review"     # Under human review
    FEEDBACK_PROVIDED = "feedback_provided"  # Human feedback received
    RESOLVED = "resolved"       # Escalation resolved
    CANCELLED = "cancelled"     # Escalation cancelled
    EXPIRED = "expired"         # Escalation expired without resolution


class ReviewerExpertise(str, Enum):
    """Reviewer expertise domains"""
    GENERAL_EDITING = "general_editing"
    TECHNICAL_WRITING = "technical_writing"
    MARKETING_COPY = "marketing_copy"
    LEGAL_COMPLIANCE = "legal_compliance"
    BRAND_VOICE = "brand_voice"
    SEO_SPECIALIST = "seo_specialist"
    CONTENT_STRATEGY = "content_strategy"
    CREATIVE_WRITING = "creative_writing"
    ACADEMIC_WRITING = "academic_writing"


@dataclass
class EscalationContext:
    """Context information for escalation decision"""
    
    # Content information
    content_type: str = ""
    content_length: int = 0
    domain: str = ""
    
    # Quality assessment context
    quality_score: float = 0.0
    failed_dimensions: List[str] = field(default_factory=list)
    critical_issues_count: int = 0
    
    # Processing context
    editing_iterations: int = 0
    processing_time: float = 0.0
    previous_escalations: int = 0
    
    # User context
    user_requirements: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    importance_level: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content_type": self.content_type,
            "content_length": self.content_length,
            "domain": self.domain,
            "quality_score": self.quality_score,
            "failed_dimensions": self.failed_dimensions,
            "critical_issues_count": self.critical_issues_count,
            "editing_iterations": self.editing_iterations,
            "processing_time": self.processing_time,
            "previous_escalations": self.previous_escalations,
            "user_requirements": self.user_requirements,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "importance_level": self.importance_level
        }


@dataclass
class HumanReviewer:
    """Human reviewer profile"""
    
    reviewer_id: str
    name: str
    email: str
    
    # Expertise and capabilities
    expertise_domains: List[ReviewerExpertise] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    max_content_length: int = 10000
    
    # Availability and workload
    available: bool = True
    current_workload: int = 0
    max_concurrent_reviews: int = 5
    
    # Performance metrics
    avg_response_time: timedelta = field(default_factory=lambda: timedelta(hours=4))
    quality_rating: float = 4.5  # 1-5 scale
    completion_rate: float = 0.95
    
    # Preferences
    preferred_content_types: List[str] = field(default_factory=list)
    timezone: str = "UTC"
    
    def can_handle_escalation(self, escalation: 'EscalationRequest') -> bool:
        """Check if reviewer can handle an escalation"""
        
        # Check availability
        if not self.available or self.current_workload >= self.max_concurrent_reviews:
            return False
        
        # Check content length limits
        if escalation.content_length > self.max_content_length:
            return False
        
        # Check expertise match
        required_expertise = escalation.required_expertise
        if required_expertise and required_expertise not in self.expertise_domains:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "reviewer_id": self.reviewer_id,
            "name": self.name,
            "email": self.email,
            "expertise_domains": [domain.value for domain in self.expertise_domains],
            "languages": self.languages,
            "max_content_length": self.max_content_length,
            "available": self.available,
            "current_workload": self.current_workload,
            "max_concurrent_reviews": self.max_concurrent_reviews,
            "avg_response_time_hours": self.avg_response_time.total_seconds() / 3600,
            "quality_rating": self.quality_rating,
            "completion_rate": self.completion_rate,
            "preferred_content_types": self.preferred_content_types,
            "timezone": self.timezone
        }


@dataclass
class EscalationRequest:
    """Human escalation request"""
    
    escalation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Content and context
    content: str = ""
    original_content: str = ""
    content_length: int = 0
    
    # Escalation details
    reasons: List[EscalationReason] = field(default_factory=list)
    priority: EscalationPriority = EscalationPriority.MEDIUM
    required_expertise: Optional[ReviewerExpertise] = None
    
    # Quality assessment
    quality_assessment: Optional[QualityAssessmentResult] = None
    specific_issues: List[QualityIssue] = field(default_factory=list)
    
    # Request metadata
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    escalation_context: Optional[EscalationContext] = None
    
    # Assignment and status
    status: EscalationStatus = EscalationStatus.PENDING
    assigned_reviewer_id: Optional[str] = None
    assigned_at: Optional[datetime] = None
    
    # Resolution
    human_feedback: Optional[str] = None
    revised_content: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "escalation_id": self.escalation_id,
            "content": self.content,
            "original_content": self.original_content,
            "content_length": self.content_length,
            "reasons": [reason.value for reason in self.reasons],
            "priority": self.priority.value,
            "required_expertise": self.required_expertise.value if self.required_expertise else None,
            "quality_assessment": self.quality_assessment.to_dict() if self.quality_assessment else None,
            "specific_issues": [issue.to_dict() for issue in self.specific_issues],
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "escalation_context": self.escalation_context.to_dict() if self.escalation_context else None,
            "status": self.status.value,
            "assigned_reviewer_id": self.assigned_reviewer_id,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "human_feedback": self.human_feedback,
            "revised_content": self.revised_content,
            "resolution_notes": self.resolution_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class EscalationDecisionEngine:
    """Intelligent escalation decision engine"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Decision thresholds and weights
        self.escalation_thresholds = {
            "quality_score_threshold": 60.0,      # Below this score triggers escalation
            "critical_issues_threshold": 1,       # Number of critical issues
            "failed_dimensions_threshold": 2,     # Number of failed quality dimensions
            "iteration_limit": 3,                 # Max automated editing iterations
            "processing_time_threshold": 300.0    # Max processing time in seconds
        }
        
        # Scoring weights for escalation decision
        self.decision_weights = {
            "quality_score": 0.3,
            "issue_severity": 0.25,
            "content_complexity": 0.2,
            "domain_specificity": 0.15,
            "user_context": 0.1
        }
    
    def should_escalate(
        self,
        quality_assessment: QualityAssessmentResult,
        context: EscalationContext
    ) -> Tuple[bool, List[EscalationReason], EscalationPriority]:
        """
        Determine if content should be escalated to human review
        
        Returns:
            Tuple of (should_escalate, reasons, priority)
        """
        
        escalation_reasons = []
        escalation_score = 0.0
        
        # Quality-based escalation
        quality_score, quality_reasons = self._assess_quality_escalation(quality_assessment)
        escalation_score += quality_score * self.decision_weights["quality_score"]
        escalation_reasons.extend(quality_reasons)
        
        # Issue severity-based escalation
        severity_score, severity_reasons = self._assess_issue_severity(quality_assessment)
        escalation_score += severity_score * self.decision_weights["issue_severity"]
        escalation_reasons.extend(severity_reasons)
        
        # Content complexity-based escalation
        complexity_score, complexity_reasons = self._assess_content_complexity(context)
        escalation_score += complexity_score * self.decision_weights["content_complexity"]
        escalation_reasons.extend(complexity_reasons)
        
        # Domain specificity escalation
        domain_score, domain_reasons = self._assess_domain_specificity(context)
        escalation_score += domain_score * self.decision_weights["domain_specificity"]
        escalation_reasons.extend(domain_reasons)
        
        # User context escalation
        user_score, user_reasons = self._assess_user_context(context)
        escalation_score += user_score * self.decision_weights["user_context"]
        escalation_reasons.extend(user_reasons)
        
        # Determine if escalation is needed
        should_escalate = escalation_score > 0.6 or len(escalation_reasons) > 0
        
        # Determine priority based on escalation score and context
        priority = self._determine_priority(escalation_score, escalation_reasons, context)
        
        self.logger.info(f"Escalation decision: {should_escalate}, score: {escalation_score:.2f}, priority: {priority.value if should_escalate else 'N/A'}")
        
        return should_escalate, escalation_reasons, priority
    
    def _assess_quality_escalation(self, assessment: QualityAssessmentResult) -> Tuple[float, List[EscalationReason]]:
        """Assess quality-based escalation factors"""
        
        score = 0.0
        reasons = []
        
        # Overall quality score
        if assessment.overall_score < self.escalation_thresholds["quality_score_threshold"]:
            score += 0.8
            reasons.append(EscalationReason.QUALITY_FAILURE)
        
        # Failed dimensions
        if len(assessment.failed_dimensions) >= self.escalation_thresholds["failed_dimensions_threshold"]:
            score += 0.6
            if EscalationReason.QUALITY_FAILURE not in reasons:
                reasons.append(EscalationReason.QUALITY_FAILURE)
        
        # Assessment confidence
        if assessment.confidence < 0.6:
            score += 0.4
            reasons.append(EscalationReason.AMBIGUOUS_CONTEXT)
        
        return min(1.0, score), reasons
    
    def _assess_issue_severity(self, assessment: QualityAssessmentResult) -> Tuple[float, List[EscalationReason]]:
        """Assess issue severity for escalation"""
        
        score = 0.0
        reasons = []
        
        # Critical issues
        critical_issues = [issue for issue in assessment.all_issues if hasattr(issue, 'issue_type') and issue.issue_type == 'critical']
        if len(critical_issues) >= self.escalation_thresholds["critical_issues_threshold"]:
            score += 1.0
            reasons.append(EscalationReason.CRITICAL_ISSUES)
        
        # High-impact issues
        high_impact_issues = [
            issue for issue in assessment.all_issues 
            if hasattr(issue, 'impact_score') and issue.impact_score > 80
        ]
        if len(high_impact_issues) > 5:
            score += 0.7
            if EscalationReason.CRITICAL_ISSUES not in reasons:
                reasons.append(EscalationReason.CRITICAL_ISSUES)
        
        return min(1.0, score), reasons
    
    def _assess_content_complexity(self, context: EscalationContext) -> Tuple[float, List[EscalationReason]]:
        """Assess content complexity for escalation"""
        
        score = 0.0
        reasons = []
        
        # Length-based complexity
        if context.content_length > 5000:
            score += 0.3
        
        # Multiple editing iterations
        if context.editing_iterations >= self.escalation_thresholds["iteration_limit"]:
            score += 0.8
            reasons.append(EscalationReason.COMPLEX_CONTENT)
        
        # Long processing time
        if context.processing_time > self.escalation_thresholds["processing_time_threshold"]:
            score += 0.5
            if EscalationReason.COMPLEX_CONTENT not in reasons:
                reasons.append(EscalationReason.COMPLEX_CONTENT)
        
        # Technical content indicators
        technical_domains = ["technical", "scientific", "legal", "medical", "academic"]
        if any(domain in context.domain.lower() for domain in technical_domains):
            score += 0.6
            reasons.append(EscalationReason.DOMAIN_EXPERTISE)
        
        return min(1.0, score), reasons
    
    def _assess_domain_specificity(self, context: EscalationContext) -> Tuple[float, List[EscalationReason]]:
        """Assess domain-specific escalation needs"""
        
        score = 0.0
        reasons = []
        
        # Legal content
        if "legal" in context.domain.lower() or "compliance" in context.content_type.lower():
            score += 0.9
            reasons.append(EscalationReason.LEGAL_REVIEW)
        
        # Brand-sensitive content
        brand_sensitive_types = ["marketing", "brand", "press_release", "public_statement"]
        if any(content_type in context.content_type.lower() for content_type in brand_sensitive_types):
            score += 0.7
            reasons.append(EscalationReason.BRAND_COMPLIANCE)
        
        # Creative content
        creative_types = ["creative", "story", "narrative", "artistic"]
        if any(creative_type in context.content_type.lower() for creative_type in creative_types):
            score += 0.6
            reasons.append(EscalationReason.CREATIVE_JUDGMENT)
        
        return min(1.0, score), reasons
    
    def _assess_user_context(self, context: EscalationContext) -> Tuple[float, List[EscalationReason]]:
        """Assess user context for escalation"""
        
        score = 0.0
        reasons = []
        
        # Explicit user request
        if context.user_requirements.get("require_human_review", False):
            score += 1.0
            reasons.append(EscalationReason.USER_REQUEST)
        
        # High importance
        if context.importance_level == "critical":
            score += 0.8
        elif context.importance_level == "high":
            score += 0.5
        
        # Tight deadline
        if context.deadline and context.deadline < datetime.now() + timedelta(hours=24):
            score += 0.4
        
        # Previous escalations
        if context.previous_escalations > 0:
            score += 0.6
            reasons.append(EscalationReason.COMPLEX_CONTENT)
        
        return min(1.0, score), reasons
    
    def _determine_priority(
        self,
        escalation_score: float,
        reasons: List[EscalationReason],
        context: EscalationContext
    ) -> EscalationPriority:
        """Determine escalation priority"""
        
        # High priority reasons
        high_priority_reasons = [
            EscalationReason.CRITICAL_ISSUES,
            EscalationReason.LEGAL_REVIEW,
            EscalationReason.BRAND_COMPLIANCE
        ]
        
        # Urgent conditions
        if (any(reason in high_priority_reasons for reason in reasons) or
            escalation_score > 0.9 or
            context.importance_level == "critical" or
            (context.deadline and context.deadline < datetime.now() + timedelta(hours=4))):
            return EscalationPriority.URGENT
        
        # High priority conditions
        elif (escalation_score > 0.7 or
              context.importance_level == "high" or
              (context.deadline and context.deadline < datetime.now() + timedelta(hours=12))):
            return EscalationPriority.HIGH
        
        # Medium priority (default)
        elif escalation_score > 0.5:
            return EscalationPriority.MEDIUM
        
        # Low priority
        else:
            return EscalationPriority.LOW


class HumanReviewerPool:
    """Manages pool of human reviewers"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.reviewers: Dict[str, HumanReviewer] = {}
        
        # Initialize with some mock reviewers for development
        self._initialize_mock_reviewers()
    
    def _initialize_mock_reviewers(self):
        """Initialize mock reviewers for development"""
        
        # General editor
        general_editor = HumanReviewer(
            reviewer_id="reviewer_001",
            name="Alex Johnson",
            email="alex.johnson@company.com",
            expertise_domains=[ReviewerExpertise.GENERAL_EDITING, ReviewerExpertise.CONTENT_STRATEGY],
            max_concurrent_reviews=8,
            quality_rating=4.7
        )
        self.add_reviewer(general_editor)
        
        # Technical writer
        tech_writer = HumanReviewer(
            reviewer_id="reviewer_002",
            name="Sarah Chen",
            email="sarah.chen@company.com",
            expertise_domains=[ReviewerExpertise.TECHNICAL_WRITING, ReviewerExpertise.SEO_SPECIALIST],
            max_content_length=15000,
            quality_rating=4.8
        )
        self.add_reviewer(tech_writer)
        
        # Marketing specialist
        marketing_specialist = HumanReviewer(
            reviewer_id="reviewer_003",
            name="Marcus Williams",
            email="marcus.williams@company.com",
            expertise_domains=[ReviewerExpertise.MARKETING_COPY, ReviewerExpertise.BRAND_VOICE],
            preferred_content_types=["marketing", "advertising", "social_media"],
            quality_rating=4.6
        )
        self.add_reviewer(marketing_specialist)
        
        # Legal reviewer
        legal_reviewer = HumanReviewer(
            reviewer_id="reviewer_004",
            name="Dr. Patricia Lee",
            email="patricia.lee@company.com",
            expertise_domains=[ReviewerExpertise.LEGAL_COMPLIANCE],
            max_concurrent_reviews=3,
            avg_response_time=timedelta(hours=8),
            quality_rating=4.9
        )
        self.add_reviewer(legal_reviewer)
    
    def add_reviewer(self, reviewer: HumanReviewer):
        """Add reviewer to pool"""
        self.reviewers[reviewer.reviewer_id] = reviewer
        self.logger.info(f"Added reviewer: {reviewer.name} ({reviewer.reviewer_id})")
    
    def remove_reviewer(self, reviewer_id: str):
        """Remove reviewer from pool"""
        if reviewer_id in self.reviewers:
            reviewer = self.reviewers.pop(reviewer_id)
            self.logger.info(f"Removed reviewer: {reviewer.name} ({reviewer_id})")
    
    def find_best_reviewer(self, escalation: EscalationRequest) -> Optional[HumanReviewer]:
        """Find the best reviewer for an escalation"""
        
        available_reviewers = [
            reviewer for reviewer in self.reviewers.values()
            if reviewer.can_handle_escalation(escalation)
        ]
        
        if not available_reviewers:
            self.logger.warning("No available reviewers found for escalation")
            return None
        
        # Score reviewers based on suitability
        reviewer_scores = []
        for reviewer in available_reviewers:
            score = self._calculate_reviewer_score(reviewer, escalation)
            reviewer_scores.append((reviewer, score))
        
        # Sort by score (highest first)
        reviewer_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_reviewer = reviewer_scores[0][0]
        self.logger.info(f"Selected reviewer: {best_reviewer.name} (score: {reviewer_scores[0][1]:.2f})")
        
        return best_reviewer
    
    def _calculate_reviewer_score(self, reviewer: HumanReviewer, escalation: EscalationRequest) -> float:
        """Calculate suitability score for reviewer"""
        
        score = 0.0
        
        # Expertise match
        if escalation.required_expertise in reviewer.expertise_domains:
            score += 40.0
        
        # Workload consideration (lower workload = higher score)
        workload_ratio = reviewer.current_workload / reviewer.max_concurrent_reviews
        score += 20.0 * (1.0 - workload_ratio)
        
        # Quality rating
        score += reviewer.quality_rating * 8.0  # Max 40 points
        
        # Response time consideration (faster = higher score)
        max_response_hours = 24.0
        response_hours = reviewer.avg_response_time.total_seconds() / 3600
        score += 20.0 * (1.0 - min(response_hours / max_response_hours, 1.0))
        
        # Content type preference
        if (escalation.escalation_context and 
            escalation.escalation_context.content_type in reviewer.preferred_content_types):
            score += 10.0
        
        # Priority matching (urgent cases get experienced reviewers)
        if escalation.priority == EscalationPriority.URGENT and reviewer.quality_rating >= 4.5:
            score += 15.0
        
        return score
    
    def get_reviewer_stats(self) -> Dict[str, Any]:
        """Get reviewer pool statistics"""
        
        total_reviewers = len(self.reviewers)
        available_reviewers = sum(1 for r in self.reviewers.values() if r.available)
        total_workload = sum(r.current_workload for r in self.reviewers.values())
        avg_quality = sum(r.quality_rating for r in self.reviewers.values()) / total_reviewers if total_reviewers > 0 else 0
        
        return {
            "total_reviewers": total_reviewers,
            "available_reviewers": available_reviewers,
            "total_current_workload": total_workload,
            "average_quality_rating": avg_quality,
            "expertise_coverage": self._get_expertise_coverage()
        }
    
    def _get_expertise_coverage(self) -> Dict[str, int]:
        """Get expertise coverage statistics"""
        
        expertise_counts = {}
        for reviewer in self.reviewers.values():
            for expertise in reviewer.expertise_domains:
                expertise_counts[expertise.value] = expertise_counts.get(expertise.value, 0) + 1
        
        return expertise_counts


class EscalationManager:
    """Manages human escalation workflows"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Components
        self.decision_engine = EscalationDecisionEngine()
        self.reviewer_pool = HumanReviewerPool()
        
        # Active escalations
        self.active_escalations: Dict[str, EscalationRequest] = {}
        
        # Performance tracking
        self.escalation_stats = {
            "total_escalations": 0,
            "resolved_escalations": 0,
            "avg_resolution_time": 0.0,
            "escalation_rate": 0.0
        }
    
    async def create_escalation(
        self,
        content: str,
        original_content: str,
        quality_assessment: QualityAssessmentResult,
        context: EscalationContext
    ) -> EscalationRequest:
        """
        Create new escalation request
        
        Args:
            content: Current content
            original_content: Original content
            quality_assessment: Quality assessment results
            context: Escalation context
            
        Returns:
            Created escalation request
        """
        
        try:
            self.logger.info("Creating escalation request")
            
            # Make escalation decision
            should_escalate, reasons, priority = self.decision_engine.should_escalate(
                quality_assessment, context
            )
            
            if not should_escalate:
                raise EscalationError("Content does not meet escalation criteria")
            
            # Determine required expertise
            required_expertise = self._determine_required_expertise(reasons, context)
            
            # Create escalation request
            escalation = EscalationRequest(
                content=content,
                original_content=original_content,
                content_length=len(content),
                reasons=reasons,
                priority=priority,
                required_expertise=required_expertise,
                quality_assessment=quality_assessment,
                specific_issues=quality_assessment.all_issues,
                escalation_context=context
            )
            
            # Set deadline based on priority
            escalation.deadline = self._calculate_deadline(priority, context)
            
            # Store escalation
            self.active_escalations[escalation.escalation_id] = escalation
            
            # Update statistics
            self.escalation_stats["total_escalations"] += 1
            
            # Log metrics
            if self.metrics:
                self.metrics.record_counter("escalations_created", priority=priority.value)
                for reason in reasons:
                    self.metrics.record_counter("escalation_reasons", reason=reason.value)
            
            self.logger.info(f"Created escalation {escalation.escalation_id} with priority {priority.value}")
            return escalation
            
        except Exception as e:
            self.logger.error(f"Failed to create escalation: {e}")
            raise EscalationError(f"Escalation creation failed: {str(e)}")
    
    async def assign_escalation(self, escalation_id: str) -> bool:
        """
        Assign escalation to best available reviewer
        
        Returns:
            True if assignment successful, False otherwise
        """
        
        try:
            if escalation_id not in self.active_escalations:
                raise EscalationError(f"Escalation not found: {escalation_id}")
            
            escalation = self.active_escalations[escalation_id]
            
            # Find best reviewer
            reviewer = self.reviewer_pool.find_best_reviewer(escalation)
            
            if not reviewer:
                self.logger.warning(f"No available reviewer for escalation {escalation_id}")
                return False
            
            # Assign escalation
            escalation.assigned_reviewer_id = reviewer.reviewer_id
            escalation.assigned_at = datetime.now()
            escalation.status = EscalationStatus.ASSIGNED
            
            # Update reviewer workload
            reviewer.current_workload += 1
            
            # In a real system, would send notification to reviewer
            self.logger.info(f"Assigned escalation {escalation_id} to reviewer {reviewer.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to assign escalation: {e}")
            return False
    
    async def process_human_feedback(
        self,
        escalation_id: str,
        feedback: str,
        revised_content: Optional[str] = None,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """
        Process feedback from human reviewer
        
        Returns:
            True if feedback processed successfully
        """
        
        try:
            if escalation_id not in self.active_escalations:
                raise EscalationError(f"Escalation not found: {escalation_id}")
            
            escalation = self.active_escalations[escalation_id]
            
            # Update escalation with feedback
            escalation.human_feedback = feedback
            escalation.revised_content = revised_content
            escalation.resolution_notes = resolution_notes
            escalation.status = EscalationStatus.FEEDBACK_PROVIDED
            
            self.logger.info(f"Received human feedback for escalation {escalation_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process human feedback: {e}")
            return False
    
    async def resolve_escalation(self, escalation_id: str) -> bool:
        """
        Resolve escalation and update statistics
        
        Returns:
            True if resolution successful
        """
        
        try:
            if escalation_id not in self.active_escalations:
                raise EscalationError(f"Escalation not found: {escalation_id}")
            
            escalation = self.active_escalations[escalation_id]
            
            # Mark as resolved
            escalation.status = EscalationStatus.RESOLVED
            escalation.resolved_at = datetime.now()
            
            # Calculate resolution time
            if escalation.assigned_at:
                resolution_time = (escalation.resolved_at - escalation.assigned_at).total_seconds()
            else:
                resolution_time = (escalation.resolved_at - escalation.created_at).total_seconds()
            
            # Update reviewer workload
            if escalation.assigned_reviewer_id:
                reviewer = self.reviewer_pool.reviewers.get(escalation.assigned_reviewer_id)
                if reviewer:
                    reviewer.current_workload = max(0, reviewer.current_workload - 1)
            
            # Update statistics
            self.escalation_stats["resolved_escalations"] += 1
            total_resolutions = self.escalation_stats["resolved_escalations"]
            avg_time = self.escalation_stats["avg_resolution_time"]
            
            # Update moving average
            self.escalation_stats["avg_resolution_time"] = (
                (avg_time * (total_resolutions - 1) + resolution_time) / total_resolutions
            )
            
            # Log metrics
            if self.metrics:
                self.metrics.record_counter("escalations_resolved")
                self.metrics.record_histogram("escalation_resolution_time", resolution_time)
            
            # Remove from active escalations
            del self.active_escalations[escalation_id]
            
            self.logger.info(f"Resolved escalation {escalation_id} in {resolution_time:.0f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resolve escalation: {e}")
            return False
    
    def _determine_required_expertise(
        self,
        reasons: List[EscalationReason],
        context: EscalationContext
    ) -> Optional[ReviewerExpertise]:
        """Determine required reviewer expertise"""
        
        # Map escalation reasons to expertise requirements
        expertise_mapping = {
            EscalationReason.LEGAL_REVIEW: ReviewerExpertise.LEGAL_COMPLIANCE,
            EscalationReason.BRAND_COMPLIANCE: ReviewerExpertise.BRAND_VOICE,
            EscalationReason.DOMAIN_EXPERTISE: ReviewerExpertise.TECHNICAL_WRITING,
            EscalationReason.CREATIVE_JUDGMENT: ReviewerExpertise.CREATIVE_WRITING
        }
        
        for reason in reasons:
            if reason in expertise_mapping:
                return expertise_mapping[reason]
        
        # Check content type
        if "technical" in context.content_type.lower():
            return ReviewerExpertise.TECHNICAL_WRITING
        elif "marketing" in context.content_type.lower():
            return ReviewerExpertise.MARKETING_COPY
        
        # Default to general editing
        return ReviewerExpertise.GENERAL_EDITING
    
    def _calculate_deadline(self, priority: EscalationPriority, context: EscalationContext) -> datetime:
        """Calculate escalation deadline based on priority"""
        
        priority_hours = {
            EscalationPriority.URGENT: 2,
            EscalationPriority.HIGH: 8,
            EscalationPriority.MEDIUM: 24,
            EscalationPriority.LOW: 72
        }
        
        hours = priority_hours.get(priority, 24)
        
        # Consider user deadline if provided and earlier
        user_deadline = context.deadline
        calculated_deadline = datetime.now() + timedelta(hours=hours)
        
        if user_deadline and user_deadline < calculated_deadline:
            return user_deadline
        
        return calculated_deadline
    
    def get_escalation_status(self, escalation_id: str) -> Optional[Dict[str, Any]]:
        """Get escalation status"""
        
        if escalation_id in self.active_escalations:
            escalation = self.active_escalations[escalation_id]
            return {
                "escalation_id": escalation_id,
                "status": escalation.status.value,
                "priority": escalation.priority.value,
                "created_at": escalation.created_at.isoformat(),
                "assigned_reviewer": escalation.assigned_reviewer_id,
                "deadline": escalation.deadline.isoformat() if escalation.deadline else None,
                "has_feedback": bool(escalation.human_feedback)
            }
        
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get escalation performance statistics"""
        
        total_escalations = self.escalation_stats["total_escalations"]
        
        stats = {
            **self.escalation_stats,
            "active_escalations": len(self.active_escalations),
            "resolution_rate": (
                self.escalation_stats["resolved_escalations"] / max(1, total_escalations)
            ),
            "reviewer_pool_stats": self.reviewer_pool.get_reviewer_stats()
        }
        
        return stats


# Global escalation manager instance
_escalation_manager: Optional[EscalationManager] = None


def get_escalation_manager() -> EscalationManager:
    """Get the global escalation manager instance"""
    global _escalation_manager
    if _escalation_manager is None:
        _escalation_manager = EscalationManager()
    return _escalation_manager


# Export main classes
__all__ = [
    'EscalationReason',
    'EscalationPriority',
    'EscalationStatus',
    'ReviewerExpertise',
    'EscalationContext',
    'HumanReviewer',
    'EscalationRequest',
    'EscalationDecisionEngine',
    'HumanReviewerPool',
    'EscalationManager',
    'get_escalation_manager'
]


if __name__ == "__main__":
    # Example usage and testing
    async def test_human_escalation():
        """Test the human escalation system"""
        
        # Create mock quality assessment
        quality_assessment = QualityAssessmentResult()
        quality_assessment.overall_score = 55.0
        quality_assessment.all_issues = []
        quality_assessment.assessment_result = "fail"
        
        # Create escalation context
        context = EscalationContext(
            content_type="marketing_copy",
            content_length=1200,
            domain="marketing",
            quality_score=55.0,
            critical_issues_count=2,
            editing_iterations=3,
            importance_level="high"
        )
        
        try:
            manager = get_escalation_manager()
            
            # Test escalation creation
            escalation = await manager.create_escalation(
                content="Test content that needs human review",
                original_content="Original test content",
                quality_assessment=quality_assessment,
                context=context
            )
            
            print("Human Escalation Test Results:")
            print(f"Escalation ID: {escalation.escalation_id}")
            print(f"Priority: {escalation.priority.value}")
            print(f"Reasons: {[r.value for r in escalation.reasons]}")
            print(f"Required Expertise: {escalation.required_expertise.value if escalation.required_expertise else 'None'}")
            
            # Test assignment
            assigned = await manager.assign_escalation(escalation.escalation_id)
            print(f"Assignment Successful: {assigned}")
            
            if assigned:
                status = manager.get_escalation_status(escalation.escalation_id)
                if status:
                    print(f"Assigned to: {status['assigned_reviewer']}")
            
            # Test feedback processing
            feedback_processed = await manager.process_human_feedback(
                escalation.escalation_id,
                "Content reviewed and improved for marketing clarity",
                "Revised content with better marketing messaging"
            )
            print(f"Feedback Processing: {feedback_processed}")
            
            # Test resolution
            resolved = await manager.resolve_escalation(escalation.escalation_id)
            print(f"Resolution: {resolved}")
            
            # Get performance stats
            stats = manager.get_performance_stats()
            print(f"\nPerformance Stats:")
            print(f"- Total Escalations: {stats['total_escalations']}")
            print(f"- Resolution Rate: {stats['resolution_rate']:.2f}")
            print(f"- Available Reviewers: {stats['reviewer_pool_stats']['available_reviewers']}")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_human_escalation())