"""
Editor Agent Quality Assurance System

This module implements a comprehensive multi-dimensional quality assurance system
for the Editor Agent. It provides sophisticated quality assessment, pass/fail criteria,
quality gates, and detailed quality reporting capabilities.

Key Features:
- Multi-dimensional quality assessment framework
- Configurable pass/fail criteria and thresholds
- Quality gate patterns with escalation logic
- Comprehensive quality scoring and weighting
- Quality trend analysis and improvement tracking
- Detailed quality reporting and recommendations
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, src_path)

# Core imports with fallbacks
try:
    from core.errors import QualityAssuranceError, ValidationError
    from core.logging.logger import get_logger
    from core.monitoring.metrics import get_metrics_collector
except ImportError:
    # Mock implementations
    import logging
    class QualityAssuranceError(Exception): pass
    class ValidationError(Exception): pass
    def get_logger(name): return logging.getLogger(name)
    def get_metrics_collector(): return None

logger = get_logger(__name__)


class QualityDimension(str, Enum):
    """Quality assessment dimensions"""
    GRAMMAR = "grammar"                    # Grammar and spelling accuracy
    READABILITY = "readability"            # Content readability and clarity
    SEO = "seo"                           # Search engine optimization
    SENTIMENT = "sentiment"               # Emotional tone and brand alignment
    STRUCTURE = "structure"               # Content organization and flow
    ACCURACY = "accuracy"                 # Factual accuracy and correctness
    COMPLETENESS = "completeness"         # Content completeness and coverage
    CONSISTENCY = "consistency"           # Style and tone consistency
    ENGAGEMENT = "engagement"             # Reader engagement potential
    COMPLIANCE = "compliance"             # Brand and regulatory compliance


class QualityLevel(str, Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"     # 90-100: Exceptional quality
    GOOD = "good"              # 75-89: Good quality, minor issues
    ACCEPTABLE = "acceptable"   # 60-74: Acceptable, some improvements needed
    POOR = "poor"              # 40-59: Poor quality, major issues
    CRITICAL = "critical"       # 0-39: Critical issues, requires immediate attention


class AssessmentResult(str, Enum):
    """Quality assessment results"""
    PASS = "pass"              # Quality meets all criteria
    CONDITIONAL_PASS = "conditional_pass"  # Passes with minor conditions
    FAIL = "fail"              # Does not meet quality standards
    CRITICAL_FAIL = "critical_fail"  # Critical failure requiring escalation


class QualityIssueType(str, Enum):
    """Types of quality issues"""
    CRITICAL = "critical"       # Must be fixed before publication
    HIGH = "high"              # Should be fixed, significant impact
    MEDIUM = "medium"          # Should be fixed, moderate impact
    LOW = "low"               # Nice to fix, minor impact
    INFORMATIONAL = "informational"  # FYI only, no action required


@dataclass
class QualityIssue:
    """Individual quality issue"""
    
    dimension: QualityDimension
    issue_type: QualityIssueType
    category: str
    message: str
    
    # Issue details
    description: str = ""
    suggestion: str = ""
    confidence: float = 1.0
    
    # Location information
    position: Optional[int] = None
    line_number: Optional[int] = None
    context: Optional[str] = None
    
    # Impact assessment
    impact_score: float = 0.0  # 0-100, higher = more impact
    fix_difficulty: float = 0.0  # 0-100, higher = more difficult to fix
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dimension": self.dimension.value,
            "issue_type": self.issue_type.value,
            "category": self.category,
            "message": self.message,
            "description": self.description,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "position": self.position,
            "line_number": self.line_number,
            "context": self.context,
            "impact_score": self.impact_score,
            "fix_difficulty": self.fix_difficulty
        }


@dataclass
class QualityMetrics:
    """Quality metrics for a specific dimension"""
    
    dimension: QualityDimension
    score: float  # 0-100
    level: QualityLevel
    
    # Detailed metrics
    sub_scores: Dict[str, float] = field(default_factory=dict)
    issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Assessment metadata
    confidence: float = 1.0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "level": self.level.value,
            "sub_scores": self.sub_scores,
            "issues": [issue.to_dict() for issue in self.issues],
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "processing_time": self.processing_time
        }


@dataclass
class QualityCriteria:
    """Quality assessment criteria and thresholds"""
    
    # Dimension weights (must sum to 1.0)
    dimension_weights: Dict[QualityDimension, float] = field(default_factory=lambda: {
        QualityDimension.GRAMMAR: 0.25,
        QualityDimension.READABILITY: 0.20,
        QualityDimension.SEO: 0.20,
        QualityDimension.SENTIMENT: 0.15,
        QualityDimension.STRUCTURE: 0.10,
        QualityDimension.CONSISTENCY: 0.10
    })
    
    # Minimum thresholds for each dimension
    minimum_thresholds: Dict[QualityDimension, float] = field(default_factory=lambda: {
        QualityDimension.GRAMMAR: 85.0,
        QualityDimension.READABILITY: 75.0,
        QualityDimension.SEO: 70.0,
        QualityDimension.SENTIMENT: 60.0,
        QualityDimension.STRUCTURE: 70.0,
        QualityDimension.CONSISTENCY: 75.0
    })
    
    # Overall quality thresholds
    pass_threshold: float = 80.0           # Overall score needed to pass
    conditional_pass_threshold: float = 70.0  # Conditional pass threshold
    critical_fail_threshold: float = 50.0     # Below this is critical failure
    
    # Issue limits
    max_critical_issues: int = 0          # No critical issues allowed
    max_high_issues: int = 3              # Max high-priority issues
    max_total_issues: int = 10            # Max total issues
    
    # Special criteria
    require_all_dimensions_pass: bool = True   # All dimensions must meet minimums
    allow_dimension_compensation: bool = False  # Higher scores can compensate for lower ones
    
    def validate(self) -> bool:
        """Validate criteria configuration"""
        # Check weights sum to 1.0
        total_weight = sum(self.dimension_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValidationError(f"Dimension weights must sum to 1.0, got {total_weight}")
        
        # Check thresholds are valid
        if not (0 <= self.critical_fail_threshold <= self.conditional_pass_threshold <= self.pass_threshold <= 100):
            raise ValidationError("Quality thresholds must be in ascending order and within 0-100 range")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dimension_weights": {dim.value: weight for dim, weight in self.dimension_weights.items()},
            "minimum_thresholds": {dim.value: threshold for dim, threshold in self.minimum_thresholds.items()},
            "pass_threshold": self.pass_threshold,
            "conditional_pass_threshold": self.conditional_pass_threshold,
            "critical_fail_threshold": self.critical_fail_threshold,
            "max_critical_issues": self.max_critical_issues,
            "max_high_issues": self.max_high_issues,
            "max_total_issues": self.max_total_issues,
            "require_all_dimensions_pass": self.require_all_dimensions_pass,
            "allow_dimension_compensation": self.allow_dimension_compensation
        }


@dataclass
class QualityAssessmentResult:
    """Complete quality assessment result"""
    
    content: str
    original_content: str
    
    # Overall assessment
    overall_score: float
    quality_level: QualityLevel
    assessment_result: AssessmentResult
    
    # Dimensional assessments
    dimension_metrics: Dict[QualityDimension, QualityMetrics] = field(default_factory=dict)
    
    # Aggregated data
    all_issues: List[QualityIssue] = field(default_factory=list)
    all_recommendations: List[str] = field(default_factory=list)
    
    # Pass/fail details
    passed_dimensions: List[QualityDimension] = field(default_factory=list)
    failed_dimensions: List[QualityDimension] = field(default_factory=list)
    criteria_violations: List[str] = field(default_factory=list)
    
    # Assessment metadata
    criteria_used: Optional[QualityCriteria] = None
    processing_time: float = 0.0
    confidence: float = 1.0
    
    # Quality trends (if historical data available)
    improvement_score: Optional[float] = None  # Improvement since last assessment
    trend_direction: Optional[str] = None      # "improving", "declining", "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "original_content": self.original_content,
            "overall_score": self.overall_score,
            "quality_level": self.quality_level.value,
            "assessment_result": self.assessment_result.value,
            "dimension_metrics": {dim.value: metrics.to_dict() for dim, metrics in self.dimension_metrics.items()},
            "all_issues": [issue.to_dict() for issue in self.all_issues],
            "all_recommendations": self.all_recommendations,
            "passed_dimensions": [dim.value for dim in self.passed_dimensions],
            "failed_dimensions": [dim.value for dim in self.failed_dimensions],
            "criteria_violations": self.criteria_violations,
            "criteria_used": self.criteria_used.to_dict() if self.criteria_used else None,
            "processing_time": self.processing_time,
            "confidence": self.confidence,
            "improvement_score": self.improvement_score,
            "trend_direction": self.trend_direction
        }


class QualityAssessmentEngine:
    """Core quality assessment engine"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Assessment history for trend analysis
        self.assessment_history: List[QualityAssessmentResult] = []
        self.max_history_size = 100
    
    def assess_quality(
        self,
        content: str,
        original_content: str,
        dimension_metrics: Dict[QualityDimension, QualityMetrics],
        criteria: QualityCriteria
    ) -> QualityAssessmentResult:
        """
        Perform comprehensive quality assessment
        
        Args:
            content: Current content
            original_content: Original content before editing
            dimension_metrics: Quality metrics for each dimension
            criteria: Quality assessment criteria
            
        Returns:
            Complete quality assessment result
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting quality assessment for {len(content)} characters")
            
            # Validate criteria
            criteria.validate()
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(dimension_metrics, criteria)
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Evaluate pass/fail criteria
            assessment_result, passed_dims, failed_dims, violations = self._evaluate_criteria(
                dimension_metrics, criteria, overall_score
            )
            
            # Aggregate issues and recommendations
            all_issues, all_recommendations = self._aggregate_issues_and_recommendations(dimension_metrics)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(dimension_metrics)
            
            # Analyze trends if history available
            improvement_score, trend_direction = self._analyze_trends(overall_score)
            
            # Create result
            result = QualityAssessmentResult(
                content=content,
                original_content=original_content,
                overall_score=overall_score,
                quality_level=quality_level,
                assessment_result=assessment_result,
                dimension_metrics=dimension_metrics,
                all_issues=all_issues,
                all_recommendations=all_recommendations,
                passed_dimensions=passed_dims,
                failed_dimensions=failed_dims,
                criteria_violations=violations,
                criteria_used=criteria,
                processing_time=(datetime.now() - start_time).total_seconds(),
                confidence=confidence,
                improvement_score=improvement_score,
                trend_direction=trend_direction
            )
            
            # Store in history
            self._add_to_history(result)
            
            # Log metrics
            self._log_assessment_metrics(result)
            
            self.logger.info(f"Quality assessment completed: {assessment_result.value} ({overall_score:.1f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            raise QualityAssuranceError(f"Quality assessment failed: {str(e)}")
    
    def _calculate_overall_score(
        self,
        dimension_metrics: Dict[QualityDimension, QualityMetrics],
        criteria: QualityCriteria
    ) -> float:
        """Calculate weighted overall quality score"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in criteria.dimension_weights.items():
            if dimension in dimension_metrics:
                metrics = dimension_metrics[dimension]
                total_score += metrics.score * weight
                total_weight += weight
        
        # Normalize by actual weight (in case some dimensions are missing)
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Apply penalties for critical issues
        critical_issues = sum(
            len([issue for issue in metrics.issues if issue.issue_type == QualityIssueType.CRITICAL])
            for metrics in dimension_metrics.values()
        )
        
        if critical_issues > 0:
            penalty = min(20.0, critical_issues * 10.0)  # Up to 20 point penalty
            overall_score = max(0.0, overall_score - penalty)
        
        return overall_score
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level based on overall score"""
        
        if overall_score >= 90:
            return QualityLevel.EXCELLENT
        elif overall_score >= 75:
            return QualityLevel.GOOD
        elif overall_score >= 60:
            return QualityLevel.ACCEPTABLE
        elif overall_score >= 40:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _evaluate_criteria(
        self,
        dimension_metrics: Dict[QualityDimension, QualityMetrics],
        criteria: QualityCriteria,
        overall_score: float
    ) -> Tuple[AssessmentResult, List[QualityDimension], List[QualityDimension], List[str]]:
        """Evaluate pass/fail criteria"""
        
        passed_dimensions = []
        failed_dimensions = []
        violations = []
        
        # Check dimensional thresholds
        for dimension, metrics in dimension_metrics.items():
            threshold = criteria.minimum_thresholds.get(dimension, 60.0)
            
            if metrics.score >= threshold:
                passed_dimensions.append(dimension)
            else:
                failed_dimensions.append(dimension)
                violations.append(f"{dimension.value} score ({metrics.score:.1f}) below threshold ({threshold})")
        
        # Count issues by type
        issue_counts = {issue_type: 0 for issue_type in QualityIssueType}
        for metrics in dimension_metrics.values():
            for issue in metrics.issues:
                issue_counts[issue.issue_type] += 1
        
        # Check issue limits
        if issue_counts[QualityIssueType.CRITICAL] > criteria.max_critical_issues:
            violations.append(f"Too many critical issues: {issue_counts[QualityIssueType.CRITICAL]} > {criteria.max_critical_issues}")
        
        if issue_counts[QualityIssueType.HIGH] > criteria.max_high_issues:
            violations.append(f"Too many high-priority issues: {issue_counts[QualityIssueType.HIGH]} > {criteria.max_high_issues}")
        
        total_issues = sum(issue_counts.values())
        if total_issues > criteria.max_total_issues:
            violations.append(f"Too many total issues: {total_issues} > {criteria.max_total_issues}")
        
        # Determine final assessment result
        if overall_score < criteria.critical_fail_threshold:
            assessment_result = AssessmentResult.CRITICAL_FAIL
        elif issue_counts[QualityIssueType.CRITICAL] > criteria.max_critical_issues:
            assessment_result = AssessmentResult.CRITICAL_FAIL
        elif len(violations) > 0:
            assessment_result = AssessmentResult.FAIL
        elif overall_score >= criteria.pass_threshold and (not criteria.require_all_dimensions_pass or len(failed_dimensions) == 0):
            assessment_result = AssessmentResult.PASS
        elif overall_score >= criteria.conditional_pass_threshold:
            assessment_result = AssessmentResult.CONDITIONAL_PASS
        else:
            assessment_result = AssessmentResult.FAIL
        
        return assessment_result, passed_dimensions, failed_dimensions, violations
    
    def _aggregate_issues_and_recommendations(
        self,
        dimension_metrics: Dict[QualityDimension, QualityMetrics]
    ) -> Tuple[List[QualityIssue], List[str]]:
        """Aggregate all issues and recommendations"""
        
        all_issues = []
        all_recommendations = []
        
        for metrics in dimension_metrics.values():
            all_issues.extend(metrics.issues)
            all_recommendations.extend(metrics.recommendations)
        
        # Sort issues by priority and impact
        all_issues.sort(key=lambda issue: (
            ["critical", "high", "medium", "low", "informational"].index(issue.issue_type.value),
            -issue.impact_score,
            -issue.confidence
        ))
        
        # Remove duplicate recommendations
        seen_recommendations = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen_recommendations:
                unique_recommendations.append(rec)
                seen_recommendations.add(rec)
        
        return all_issues, unique_recommendations
    
    def _calculate_confidence(self, dimension_metrics: Dict[QualityDimension, QualityMetrics]) -> float:
        """Calculate overall confidence in assessment"""
        
        confidences = [metrics.confidence for metrics in dimension_metrics.values() if metrics.confidence > 0]
        
        if not confidences:
            return 0.0
        
        # Use weighted average based on dimension importance
        # For now, use simple average
        return sum(confidences) / len(confidences)
    
    def _analyze_trends(self, current_score: float) -> Tuple[Optional[float], Optional[str]]:
        """Analyze quality trends based on history"""
        
        if len(self.assessment_history) < 2:
            return None, None
        
        # Get recent scores
        recent_scores = [result.overall_score for result in self.assessment_history[-5:]]
        recent_scores.append(current_score)
        
        # Calculate improvement
        if len(recent_scores) >= 2:
            improvement = current_score - recent_scores[-2]
        else:
            improvement = 0.0
        
        # Determine trend direction
        if len(recent_scores) >= 3:
            # Calculate trend using simple linear regression
            trend_slope = self._calculate_trend_slope(recent_scores)
            
            if trend_slope > 2.0:
                trend_direction = "improving"
            elif trend_slope < -2.0:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"
        
        return improvement, trend_direction
    
    def _calculate_trend_slope(self, scores: List[float]) -> float:
        """Calculate trend slope using simple linear regression"""
        
        if len(scores) < 2:
            return 0.0
        
        n = len(scores)
        x_values = list(range(n))
        
        # Calculate means
        mean_x = sum(x_values) / n
        mean_y = sum(scores) / n
        
        # Calculate slope
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, scores))
        denominator = sum((x - mean_x) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _add_to_history(self, result: QualityAssessmentResult):
        """Add assessment result to history"""
        
        self.assessment_history.append(result)
        
        # Trim history if too large
        if len(self.assessment_history) > self.max_history_size:
            self.assessment_history = self.assessment_history[-self.max_history_size:]
    
    def _log_assessment_metrics(self, result: QualityAssessmentResult):
        """Log assessment metrics"""
        
        try:
            if self.metrics:
                self.metrics.record_counter("quality_assessments_total")
                self.metrics.record_histogram("quality_assessment_processing_time", result.processing_time)
                self.metrics.record_gauge("quality_overall_score", result.overall_score)
                self.metrics.record_counter(f"quality_assessment_result_{result.assessment_result.value}")
                
                # Log dimensional scores
                for dimension, metrics in result.dimension_metrics.items():
                    self.metrics.record_gauge(f"quality_dimension_score_{dimension.value}", metrics.score)
                
                # Log issue counts
                issue_counts = {}
                for issue in result.all_issues:
                    issue_type = issue.issue_type.value
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                
                for issue_type, count in issue_counts.items():
                    self.metrics.record_gauge(f"quality_issues_{issue_type}", count)
                    
        except Exception as e:
            self.logger.warning(f"Failed to log assessment metrics: {e}")
    
    def get_assessment_history(self) -> List[QualityAssessmentResult]:
        """Get assessment history"""
        return self.assessment_history.copy()
    
    def clear_history(self):
        """Clear assessment history"""
        self.assessment_history.clear()
        self.logger.info("Quality assessment history cleared")


class QualityGateManager:
    """Manages quality gates and decision-making"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.assessment_engine = QualityAssessmentEngine()
    
    def evaluate_quality_gate(
        self,
        content: str,
        original_content: str,
        dimension_metrics: Dict[QualityDimension, QualityMetrics],
        criteria: QualityCriteria
    ) -> Tuple[AssessmentResult, QualityAssessmentResult]:
        """
        Evaluate quality gate and return decision
        
        Returns:
            Tuple of (gate_decision, full_assessment_result)
        """
        
        self.logger.info("Evaluating quality gate")
        
        # Perform comprehensive assessment
        assessment = self.assessment_engine.assess_quality(
            content, original_content, dimension_metrics, criteria
        )
        
        # Gate decision is the same as assessment result
        gate_decision = assessment.assessment_result
        
        self.logger.info(f"Quality gate decision: {gate_decision.value} (score: {assessment.overall_score:.1f})")
        
        return gate_decision, assessment
    
    def should_escalate_to_human(self, assessment: QualityAssessmentResult) -> bool:
        """Determine if content should be escalated to human review"""
        
        # Escalation criteria
        escalation_reasons = []
        
        # Critical failures always escalate
        if assessment.assessment_result == AssessmentResult.CRITICAL_FAIL:
            escalation_reasons.append("Critical quality failure")
        
        # Multiple high-priority issues
        high_priority_issues = [
            issue for issue in assessment.all_issues 
            if issue.issue_type in [QualityIssueType.CRITICAL, QualityIssueType.HIGH]
        ]
        
        if len(high_priority_issues) >= 5:
            escalation_reasons.append(f"Too many high-priority issues ({len(high_priority_issues)})")
        
        # Low confidence in assessment
        if assessment.confidence < 0.6:
            escalation_reasons.append(f"Low assessment confidence ({assessment.confidence:.2f})")
        
        # Significant quality decline
        if assessment.improvement_score and assessment.improvement_score < -10:
            escalation_reasons.append(f"Quality decline ({assessment.improvement_score:.1f})")
        
        # Multiple failed dimensions
        if len(assessment.failed_dimensions) >= 3:
            escalation_reasons.append(f"Multiple failed dimensions ({len(assessment.failed_dimensions)})")
        
        should_escalate = len(escalation_reasons) > 0
        
        if should_escalate:
            self.logger.info(f"Escalating to human review: {'; '.join(escalation_reasons)}")
        
        return should_escalate
    
    def generate_quality_report(self, assessment: QualityAssessmentResult) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        
        # Executive summary
        summary = {
            "overall_quality": assessment.quality_level.value,
            "score": assessment.overall_score,
            "result": assessment.assessment_result.value,
            "total_issues": len(assessment.all_issues),
            "critical_issues": len([i for i in assessment.all_issues if i.issue_type == QualityIssueType.CRITICAL]),
            "processing_time": assessment.processing_time
        }
        
        # Dimensional breakdown
        dimensional_breakdown = {}
        for dimension, metrics in assessment.dimension_metrics.items():
            dimensional_breakdown[dimension.value] = {
                "score": metrics.score,
                "level": metrics.level.value,
                "issues": len(metrics.issues),
                "recommendations": len(metrics.recommendations)
            }
        
        # Issue summary
        issue_summary = {}
        for issue_type in QualityIssueType:
            count = len([i for i in assessment.all_issues if i.issue_type == issue_type])
            issue_summary[issue_type.value] = count
        
        # Quality trends
        trends = {}
        if assessment.improvement_score is not None:
            trends["improvement"] = assessment.improvement_score
        if assessment.trend_direction:
            trends["direction"] = assessment.trend_direction
        
        # Compliance status
        compliance = {
            "passed_dimensions": [dim.value for dim in assessment.passed_dimensions],
            "failed_dimensions": [dim.value for dim in assessment.failed_dimensions],
            "criteria_violations": assessment.criteria_violations
        }
        
        return {
            "summary": summary,
            "dimensional_breakdown": dimensional_breakdown,
            "issue_summary": issue_summary,
            "trends": trends,
            "compliance": compliance,
            "recommendations": assessment.all_recommendations[:10],  # Top 10 recommendations
            "generated_at": datetime.now().isoformat()
        }


# Global instances
_quality_assessment_engine: Optional[QualityAssessmentEngine] = None
_quality_gate_manager: Optional[QualityGateManager] = None


def get_quality_assessment_engine() -> QualityAssessmentEngine:
    """Get the global quality assessment engine instance"""
    global _quality_assessment_engine
    if _quality_assessment_engine is None:
        _quality_assessment_engine = QualityAssessmentEngine()
    return _quality_assessment_engine


def get_quality_gate_manager() -> QualityGateManager:
    """Get the global quality gate manager instance"""
    global _quality_gate_manager
    if _quality_gate_manager is None:
        _quality_gate_manager = QualityGateManager()
    return _quality_gate_manager


# Export main classes
__all__ = [
    'QualityDimension',
    'QualityLevel',
    'AssessmentResult',
    'QualityIssueType',
    'QualityIssue',
    'QualityMetrics',
    'QualityCriteria',
    'QualityAssessmentResult',
    'QualityAssessmentEngine',
    'QualityGateManager',
    'get_quality_assessment_engine',
    'get_quality_gate_manager'
]


if __name__ == "__main__":
    # Example usage and testing
    async def test_quality_assurance():
        """Test the quality assurance system"""
        
        # Create mock quality metrics
        grammar_metrics = QualityMetrics(
            dimension=QualityDimension.GRAMMAR,
            score=88.0,
            level=QualityLevel.GOOD,
            issues=[
                QualityIssue(
                    dimension=QualityDimension.GRAMMAR,
                    issue_type=QualityIssueType.MEDIUM,
                    category="spelling",
                    message="Possible spelling error: 'grammer' should be 'grammar'"
                )
            ],
            recommendations=["Review spelling accuracy"]
        )
        
        seo_metrics = QualityMetrics(
            dimension=QualityDimension.SEO,
            score=75.0,
            level=QualityLevel.GOOD,
            recommendations=["Add more target keywords"]
        )
        
        readability_metrics = QualityMetrics(
            dimension=QualityDimension.READABILITY,
            score=82.0,
            level=QualityLevel.GOOD,
            recommendations=["Simplify complex sentences"]
        )
        
        dimension_metrics = {
            QualityDimension.GRAMMAR: grammar_metrics,
            QualityDimension.SEO: seo_metrics,
            QualityDimension.READABILITY: readability_metrics
        }
        
        # Create quality criteria
        criteria = QualityCriteria()
        
        try:
            # Test quality assessment
            engine = get_quality_assessment_engine()
            gate_manager = get_quality_gate_manager()
            
            content = "Test content with improved quality."
            original_content = "Test content with grammer issues."
            
            # Perform assessment
            assessment = engine.assess_quality(content, original_content, dimension_metrics, criteria)
            
            # Evaluate quality gate
            gate_decision, _ = gate_manager.evaluate_quality_gate(content, original_content, dimension_metrics, criteria)
            
            # Generate report
            report = gate_manager.generate_quality_report(assessment)
            
            print("Quality Assurance Test Results:")
            print(f"Overall Score: {assessment.overall_score:.1f}")
            print(f"Quality Level: {assessment.quality_level.value}")
            print(f"Assessment Result: {assessment.assessment_result.value}")
            print(f"Gate Decision: {gate_decision.value}")
            print(f"Total Issues: {len(assessment.all_issues)}")
            print(f"Recommendations: {len(assessment.all_recommendations)}")
            print(f"Processing Time: {assessment.processing_time:.3f}s")
            
            if assessment.failed_dimensions:
                print(f"Failed Dimensions: {[dim.value for dim in assessment.failed_dimensions]}")
            
            if assessment.criteria_violations:
                print(f"Criteria Violations: {assessment.criteria_violations}")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_quality_assurance())