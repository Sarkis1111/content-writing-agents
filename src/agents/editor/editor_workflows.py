"""
Editor Agent LangGraph Workflows

This module provides specialized LangGraph workflow definitions and conditional logic patterns
specifically designed for content editing processes. It extends the base LangGraph framework
with editor-specific workflow templates and intelligent decision-making capabilities.

Key Features:
- Editor-specific workflow templates
- Advanced conditional logic for editing decisions
- Quality gate patterns and escalation logic
- Multi-stage editing orchestration
- Performance-optimized workflow paths
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, src_path)

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

# Framework imports with fallbacks
try:
    from frameworks.langgraph.state import ContentEditingState, QualityMetrics, StateStatus
    from frameworks.langgraph.workflows import WorkflowTemplate, WorkflowNode, WorkflowEdge, get_workflow_registry
    from core.logging.logger import get_logger
    from core.errors import WorkflowError, ValidationError
    from core.monitoring.metrics import get_metrics_collector
except ImportError:
    # Mock implementations for testing
    import logging
    
    class ContentEditingState(dict): pass
    class QualityMetrics: pass
    class StateStatus: pass
    class WorkflowTemplate: pass
    class WorkflowNode: pass
    class WorkflowEdge: pass
    class WorkflowError(Exception): pass
    class ValidationError(Exception): pass
    
    def get_workflow_registry(): return None
    def get_logger(name): return logging.getLogger(name)
    def get_metrics_collector(): return None

logger = get_logger(__name__)


class EditingWorkflowType(str, Enum):
    """Types of editing workflows available"""
    COMPREHENSIVE = "comprehensive"      # Full multi-stage editing
    RAPID = "rapid"                     # Fast editing for urgent content  
    GRAMMAR_ONLY = "grammar_only"       # Grammar and spelling only
    SEO_FOCUSED = "seo_focused"         # SEO optimization focused
    READABILITY = "readability"         # Readability improvement focused
    QUALITY_REVIEW = "quality_review"   # Quality assessment only


class EditingComplexity(str, Enum):
    """Content editing complexity levels"""
    SIMPLE = "simple"       # Basic content, minimal issues expected
    MEDIUM = "medium"       # Standard content, moderate editing needed
    COMPLEX = "complex"     # Advanced content, extensive editing required
    CRITICAL = "critical"   # High-stakes content, maximum quality required


class WorkflowDecision(str, Enum):
    """Workflow decision points"""
    CONTINUE = "continue"
    SKIP = "skip"
    ESCALATE = "escalate"
    RETRY = "retry"
    COMPLETE = "complete"
    FAIL = "fail"


@dataclass
class EditingWorkflowConfig:
    """Configuration for editing workflows"""
    
    workflow_type: EditingWorkflowType = EditingWorkflowType.COMPREHENSIVE
    complexity_level: EditingComplexity = EditingComplexity.MEDIUM
    
    # Quality thresholds
    grammar_threshold: float = 85.0
    seo_threshold: float = 75.0
    readability_threshold: float = 80.0
    overall_threshold: float = 80.0
    
    # Workflow control
    enable_parallel_processing: bool = False
    max_iterations_per_stage: int = 2
    auto_escalation_threshold: float = 60.0
    
    # Performance settings
    timeout_per_stage: int = 60
    enable_caching: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_type": self.workflow_type.value,
            "complexity_level": self.complexity_level.value,
            "grammar_threshold": self.grammar_threshold,
            "seo_threshold": self.seo_threshold,
            "readability_threshold": self.readability_threshold,
            "overall_threshold": self.overall_threshold,
            "enable_parallel_processing": self.enable_parallel_processing,
            "max_iterations_per_stage": self.max_iterations_per_stage,
            "auto_escalation_threshold": self.auto_escalation_threshold,
            "timeout_per_stage": self.timeout_per_stage,
            "enable_caching": self.enable_caching
        }


class EditorConditionalLogic:
    """Advanced conditional logic patterns for editing workflows"""
    
    def __init__(self, config: EditingWorkflowConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    # Primary Decision Functions
    
    def should_continue_editing(self, state: ContentEditingState) -> str:
        """Determine if editing should continue to next stage"""
        try:
            current_quality = state.get("current_quality", {})
            current_step = state.get("current_step", "")
            editing_iterations = state.get("editing_iterations", 0)
            max_iterations = state.get("max_iterations", 3)
            
            # Check iteration limits
            if editing_iterations >= max_iterations:
                self.logger.info(f"Max iterations reached ({editing_iterations}/{max_iterations})")
                return WorkflowDecision.COMPLETE.value
            
            # Check if current stage has critical failures
            if self._has_critical_failures(current_quality, current_step):
                return WorkflowDecision.ESCALATE.value
            
            # Check overall quality progression
            if self._quality_improving(state):
                return WorkflowDecision.CONTINUE.value
            
            # Check if minimum quality threshold met
            overall_score = current_quality.get("overall_score", 0.0)
            if overall_score >= self.config.overall_threshold:
                return WorkflowDecision.COMPLETE.value
            
            return WorkflowDecision.CONTINUE.value
            
        except Exception as e:
            self.logger.error(f"Decision logic error: {e}")
            return WorkflowDecision.FAIL.value
    
    def grammar_check_decision(self, state: ContentEditingState) -> str:
        """Conditional logic for grammar check stage"""
        try:
            current_quality = state.get("current_quality", {})
            grammar_score = current_quality.get("grammar_score", 0.0)
            
            if grammar_score >= self.config.grammar_threshold:
                self.logger.info(f"Grammar check passed: {grammar_score:.1f} >= {self.config.grammar_threshold}")
                return WorkflowDecision.CONTINUE.value
            
            elif grammar_score >= self.config.auto_escalation_threshold:
                self.logger.info(f"Grammar check needs improvement: {grammar_score:.1f}")
                return WorkflowDecision.RETRY.value
            
            else:
                self.logger.warning(f"Grammar check critical failure: {grammar_score:.1f}")
                return WorkflowDecision.ESCALATE.value
                
        except Exception as e:
            self.logger.error(f"Grammar decision error: {e}")
            return WorkflowDecision.FAIL.value
    
    def seo_optimization_decision(self, state: ContentEditingState) -> str:
        """Conditional logic for SEO optimization stage"""
        try:
            current_quality = state.get("current_quality", {})
            seo_score = current_quality.get("seo_score", 0.0)
            editing_requirements = state.get("editing_requirements", {})
            
            # Skip SEO if not required
            if not editing_requirements.get("require_seo", True):
                return WorkflowDecision.SKIP.value
            
            if seo_score >= self.config.seo_threshold:
                return WorkflowDecision.CONTINUE.value
            elif seo_score >= self.config.auto_escalation_threshold:
                return WorkflowDecision.RETRY.value
            else:
                return WorkflowDecision.ESCALATE.value
                
        except Exception as e:
            self.logger.error(f"SEO decision error: {e}")
            return WorkflowDecision.FAIL.value
    
    def readability_assessment_decision(self, state: ContentEditingState) -> str:
        """Conditional logic for readability assessment"""
        try:
            current_quality = state.get("current_quality", {})
            readability_score = current_quality.get("readability_score", 0.0)
            
            # Consider target audience complexity
            editing_requirements = state.get("editing_requirements", {})
            target_audience = editing_requirements.get("target_audience", "")
            
            # Adjust threshold based on audience
            threshold = self.config.readability_threshold
            if "professional" in target_audience.lower():
                threshold *= 0.9  # Allow slightly lower readability for professional content
            elif "general" in target_audience.lower():
                threshold *= 1.1  # Require higher readability for general audience
            
            if readability_score >= threshold:
                return WorkflowDecision.CONTINUE.value
            elif readability_score >= self.config.auto_escalation_threshold:
                return WorkflowDecision.RETRY.value
            else:
                return WorkflowDecision.ESCALATE.value
                
        except Exception as e:
            self.logger.error(f"Readability decision error: {e}")
            return WorkflowDecision.FAIL.value
    
    def quality_gate_decision(self, state: ContentEditingState) -> str:
        """Advanced quality gate decision logic"""
        try:
            current_quality = state.get("current_quality", {})
            editing_issues = state.get("editing_issues", [])
            editing_iterations = state.get("editing_iterations", 0)
            max_iterations = state.get("max_iterations", 3)
            
            # Calculate weighted quality score
            quality_weights = {
                "grammar_score": 0.3,
                "seo_score": 0.25,
                "readability_score": 0.25,
                "sentiment_score": 0.1,
                "style_score": 0.1
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, weight in quality_weights.items():
                score = current_quality.get(metric, 0.0)
                if score > 0:
                    weighted_score += score * weight
                    total_weight += weight
            
            final_score = weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Critical issues check
            critical_issues = [
                issue for issue in editing_issues 
                if issue.get("priority") == "critical"
            ]
            
            # Decision matrix
            if len(critical_issues) > 0:
                self.logger.warning(f"Quality gate: {len(critical_issues)} critical issues detected")
                return "escalate"
            
            elif final_score >= self.config.overall_threshold:
                self.logger.info(f"Quality gate passed: {final_score:.1f} >= {self.config.overall_threshold}")
                return "pass"
            
            elif editing_iterations >= max_iterations:
                self.logger.warning(f"Quality gate: max iterations reached ({editing_iterations})")
                return "escalate"
            
            elif final_score >= self.config.auto_escalation_threshold:
                self.logger.info(f"Quality gate: revision needed ({final_score:.1f})")
                return "revise"
            
            else:
                self.logger.warning(f"Quality gate: escalation required ({final_score:.1f})")
                return "escalate"
                
        except Exception as e:
            self.logger.error(f"Quality gate decision error: {e}")
            return "fail"
    
    def human_escalation_decision(self, state: ContentEditingState) -> str:
        """Determine if human escalation is necessary"""
        try:
            escalation_data = state.get("escalation_data", {})
            escalation_reasons = escalation_data.get("escalation_reasons", [])
            
            # Escalation priority matrix
            high_priority_reasons = [
                "critical quality failure",
                "multiple editing failures", 
                "compliance issues",
                "brand risk"
            ]
            
            has_high_priority = any(
                any(hp_reason in reason.lower() for hp_reason in high_priority_reasons)
                for reason in escalation_reasons
            )
            
            if has_high_priority:
                return "human_review"
            else:
                return "automated_review"
                
        except Exception as e:
            self.logger.error(f"Escalation decision error: {e}")
            return "automated_review"
    
    # Helper Methods
    
    def _has_critical_failures(self, current_quality: Dict[str, Any], current_step: str) -> bool:
        """Check for critical failures in current quality metrics"""
        try:
            # Define critical failure thresholds by stage
            critical_thresholds = {
                "grammar_check": {"grammar_score": 50.0},
                "seo_optimization": {"seo_score": 40.0},
                "readability_assessment": {"readability_score": 45.0}
            }
            
            if current_step in critical_thresholds:
                for metric, threshold in critical_thresholds[current_step].items():
                    if current_quality.get(metric, 0.0) < threshold:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _quality_improving(self, state: ContentEditingState) -> bool:
        """Check if quality metrics are improving over iterations"""
        try:
            quality_history = state.get("quality_history", [])
            
            if len(quality_history) < 2:
                return True  # Allow initial iterations
            
            latest_quality = quality_history[-1]
            previous_quality = quality_history[-2]
            
            latest_score = latest_quality.get("overall_score", 0.0)
            previous_score = previous_quality.get("overall_score", 0.0)
            
            # Consider improving if score increased by at least 5 points
            return latest_score >= previous_score + 5.0
            
        except Exception:
            return True  # Default to continue if error
    

class EditorWorkflowManager:
    """Manages editor-specific LangGraph workflows"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        self.workflow_registry = get_workflow_registry()
        
        # Workflow templates
        self.editor_templates: Dict[str, WorkflowTemplate] = {}
        self._initialize_editor_templates()
        
        # Active workflows
        self.active_workflows: Dict[str, Any] = {}
    
    def _initialize_editor_templates(self):
        """Initialize editor-specific workflow templates"""
        try:
            # Comprehensive Editing Workflow
            comprehensive_template = self._create_comprehensive_workflow()
            self.editor_templates["comprehensive"] = comprehensive_template
            
            # Rapid Editing Workflow
            rapid_template = self._create_rapid_workflow()
            self.editor_templates["rapid"] = rapid_template
            
            # Grammar-Only Workflow
            grammar_template = self._create_grammar_only_workflow()
            self.editor_templates["grammar_only"] = grammar_template
            
            # SEO-Focused Workflow
            seo_template = self._create_seo_focused_workflow()
            self.editor_templates["seo_focused"] = seo_template
            
            # Quality Review Workflow
            quality_template = self._create_quality_review_workflow()
            self.editor_templates["quality_review"] = quality_template
            
            self.logger.info(f"Initialized {len(self.editor_templates)} editor workflow templates")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize editor templates: {e}")
    
    def _create_comprehensive_workflow(self) -> WorkflowTemplate:
        """Create comprehensive editing workflow template"""
        return WorkflowTemplate(
            name="Comprehensive Content Editing",
            description="Full multi-stage content editing with quality gates and human escalation",
            state_type="content_editing",
            entry_point="initial_assessment",
            nodes=[
                WorkflowNode(
                    name="initial_assessment",
                    function="assess_content_complexity",
                    description="Assess content complexity and editing requirements",
                    tools=["content_analyzer"]
                ),
                WorkflowNode(
                    name="grammar_check",
                    function="comprehensive_grammar_check", 
                    description="Comprehensive grammar and spelling check",
                    tools=["grammar_checker"],
                    timeout=120
                ),
                WorkflowNode(
                    name="style_optimization",
                    function="optimize_writing_style",
                    description="Optimize writing style and tone consistency",
                    tools=["style_analyzer"]
                ),
                WorkflowNode(
                    name="seo_optimization", 
                    function="comprehensive_seo_optimization",
                    description="Comprehensive SEO optimization",
                    tools=["seo_analyzer"],
                    timeout=90
                ),
                WorkflowNode(
                    name="readability_enhancement",
                    function="enhance_readability",
                    description="Enhance content readability for target audience",
                    tools=["readability_scorer"]
                ),
                WorkflowNode(
                    name="sentiment_alignment",
                    function="align_content_sentiment",
                    description="Align content sentiment with brand voice",
                    tools=["sentiment_analyzer"]
                ),
                WorkflowNode(
                    name="comprehensive_review",
                    function="conduct_comprehensive_review",
                    description="Comprehensive quality review and scoring",
                    tools=["content_analyzer", "quality_assessor"]
                ),
                WorkflowNode(
                    name="quality_gate",
                    function="comprehensive_quality_gate",
                    description="Comprehensive quality gate with escalation logic",
                    interrupt=True
                ),
                WorkflowNode(
                    name="iterative_improvement",
                    function="apply_iterative_improvements",
                    description="Apply iterative improvements based on quality feedback"
                ),
                WorkflowNode(
                    name="expert_review",
                    function="escalate_to_expert_review",
                    description="Escalate to human expert for complex issues",
                    interrupt=True
                ),
                WorkflowNode(
                    name="final_optimization",
                    function="final_content_optimization",
                    description="Final content optimization and polishing"
                )
            ],
            edges=[
                WorkflowEdge("initial_assessment", "grammar_check"),
                WorkflowEdge("grammar_check", "style_optimization", "grammar_passed"),
                WorkflowEdge("grammar_check", "iterative_improvement", "grammar_needs_work"),
                WorkflowEdge("style_optimization", "seo_optimization"),
                WorkflowEdge("seo_optimization", "readability_enhancement"),
                WorkflowEdge("readability_enhancement", "sentiment_alignment"),
                WorkflowEdge("sentiment_alignment", "comprehensive_review"),
                WorkflowEdge("comprehensive_review", "quality_gate"),
                WorkflowEdge("quality_gate", "final_optimization", "quality_approved"),
                WorkflowEdge("quality_gate", "iterative_improvement", "needs_revision"),
                WorkflowEdge("quality_gate", "expert_review", "needs_escalation"),
                WorkflowEdge("iterative_improvement", "comprehensive_review"),
                WorkflowEdge("expert_review", "final_optimization")
            ],
            interrupt_before=["quality_gate", "expert_review"],
            interrupt_after=["final_optimization"]
        )
    
    def _create_rapid_workflow(self) -> WorkflowTemplate:
        """Create rapid editing workflow for urgent content"""
        return WorkflowTemplate(
            name="Rapid Content Editing",
            description="Streamlined editing workflow for urgent content needs",
            state_type="content_editing",
            entry_point="rapid_assessment",
            nodes=[
                WorkflowNode(
                    name="rapid_assessment",
                    function="rapid_content_assessment",
                    description="Quick content assessment for urgent editing",
                    timeout=30
                ),
                WorkflowNode(
                    name="essential_grammar",
                    function="essential_grammar_check",
                    description="Essential grammar and spelling corrections only",
                    tools=["grammar_checker"],
                    timeout=45
                ),
                WorkflowNode(
                    name="basic_readability",
                    function="basic_readability_check", 
                    description="Basic readability improvements",
                    tools=["readability_scorer"],
                    timeout=30
                ),
                WorkflowNode(
                    name="rapid_quality_check",
                    function="rapid_quality_assessment",
                    description="Rapid quality check with minimal thresholds"
                ),
                WorkflowNode(
                    name="rapid_finalization",
                    function="rapid_content_finalization",
                    description="Quick content finalization for delivery"
                )
            ],
            edges=[
                WorkflowEdge("rapid_assessment", "essential_grammar"),
                WorkflowEdge("essential_grammar", "basic_readability"),
                WorkflowEdge("basic_readability", "rapid_quality_check"),
                WorkflowEdge("rapid_quality_check", "rapid_finalization", "quality_acceptable"),
                WorkflowEdge("rapid_quality_check", "essential_grammar", "needs_quick_fix")
            ]
        )
    
    def _create_grammar_only_workflow(self) -> WorkflowTemplate:
        """Create grammar-only editing workflow"""
        return WorkflowTemplate(
            name="Grammar-Only Editing",
            description="Focused grammar and spelling correction workflow",
            state_type="content_editing",
            entry_point="deep_grammar_analysis",
            nodes=[
                WorkflowNode(
                    name="deep_grammar_analysis",
                    function="deep_grammar_analysis",
                    description="Comprehensive grammar analysis and detection",
                    tools=["grammar_checker"],
                    timeout=90
                ),
                WorkflowNode(
                    name="spelling_correction",
                    function="advanced_spelling_correction",
                    description="Advanced spelling correction and validation",
                    tools=["spell_checker"]
                ),
                WorkflowNode(
                    name="punctuation_review",
                    function="punctuation_consistency_review", 
                    description="Punctuation consistency and style review"
                ),
                WorkflowNode(
                    name="grammar_quality_gate",
                    function="grammar_focused_quality_gate",
                    description="Grammar-focused quality assessment"
                )
            ],
            edges=[
                WorkflowEdge("deep_grammar_analysis", "spelling_correction"),
                WorkflowEdge("spelling_correction", "punctuation_review"),
                WorkflowEdge("punctuation_review", "grammar_quality_gate"),
                WorkflowEdge("grammar_quality_gate", "deep_grammar_analysis", "needs_grammar_revision")
            ]
        )
    
    def _create_seo_focused_workflow(self) -> WorkflowTemplate:
        """Create SEO-focused editing workflow"""
        return WorkflowTemplate(
            name="SEO-Focused Editing",
            description="SEO optimization focused editing workflow",
            state_type="content_editing",
            entry_point="seo_analysis",
            nodes=[
                WorkflowNode(
                    name="seo_analysis",
                    function="comprehensive_seo_analysis",
                    description="Comprehensive SEO analysis and keyword research",
                    tools=["seo_analyzer", "keyword_analyzer"],
                    timeout=120
                ),
                WorkflowNode(
                    name="keyword_optimization",
                    function="optimize_keyword_usage",
                    description="Optimize keyword density and placement",
                    tools=["seo_optimizer"]
                ),
                WorkflowNode(
                    name="meta_optimization",
                    function="optimize_meta_elements",
                    description="Optimize meta descriptions, titles, and headers",
                    tools=["meta_optimizer"]
                ),
                WorkflowNode(
                    name="content_structure",
                    function="optimize_content_structure",
                    description="Optimize content structure for SEO",
                    tools=["structure_analyzer"]
                ),
                WorkflowNode(
                    name="seo_quality_gate",
                    function="seo_quality_assessment",
                    description="SEO-focused quality assessment and scoring"
                )
            ],
            edges=[
                WorkflowEdge("seo_analysis", "keyword_optimization"),
                WorkflowEdge("keyword_optimization", "meta_optimization"),
                WorkflowEdge("meta_optimization", "content_structure"),
                WorkflowEdge("content_structure", "seo_quality_gate"),
                WorkflowEdge("seo_quality_gate", "keyword_optimization", "needs_seo_revision")
            ]
        )
    
    def _create_quality_review_workflow(self) -> WorkflowTemplate:
        """Create quality review only workflow"""
        return WorkflowTemplate(
            name="Quality Review Only",
            description="Comprehensive quality assessment without modifications",
            state_type="content_editing", 
            entry_point="initial_quality_scan",
            nodes=[
                WorkflowNode(
                    name="initial_quality_scan",
                    function="comprehensive_quality_scan",
                    description="Comprehensive initial quality assessment",
                    tools=["content_analyzer", "quality_assessor"]
                ),
                WorkflowNode(
                    name="detailed_analysis",
                    function="detailed_content_analysis",
                    description="Detailed analysis across all quality dimensions",
                    tools=["grammar_checker", "seo_analyzer", "readability_scorer", "sentiment_analyzer"]
                ),
                WorkflowNode(
                    name="issue_categorization",
                    function="categorize_content_issues",
                    description="Categorize and prioritize identified issues"
                ),
                WorkflowNode(
                    name="recommendation_generation",
                    function="generate_improvement_recommendations",
                    description="Generate specific improvement recommendations"
                ),
                WorkflowNode(
                    name="quality_reporting",
                    function="generate_quality_report",
                    description="Generate comprehensive quality report"
                )
            ],
            edges=[
                WorkflowEdge("initial_quality_scan", "detailed_analysis"),
                WorkflowEdge("detailed_analysis", "issue_categorization"),
                WorkflowEdge("issue_categorization", "recommendation_generation"),
                WorkflowEdge("recommendation_generation", "quality_reporting")
            ]
        )
    
    async def create_workflow(
        self,
        workflow_type: EditingWorkflowType,
        config: EditingWorkflowConfig,
        node_functions: Dict[str, Callable],
        conditional_logic: EditorConditionalLogic
    ) -> Optional[Any]:
        """Create a LangGraph workflow for editing"""
        try:
            if not LANGGRAPH_AVAILABLE:
                self.logger.error("LangGraph not available")
                return None
            
            template_key = workflow_type.value
            if template_key not in self.editor_templates:
                raise WorkflowError(f"Unknown workflow type: {workflow_type}")
            
            template = self.editor_templates[template_key]
            
            # Create StateGraph
            workflow = StateGraph(ContentEditingState)
            
            # Add nodes with functions
            for node in template.nodes:
                if node.function not in node_functions:
                    raise WorkflowError(f"Node function not provided: {node.function}")
                
                workflow.add_node(node.name, node_functions[node.function])
            
            # Set entry point
            workflow.set_entry_point(template.entry_point)
            
            # Add edges with conditional logic
            for edge in template.edges:
                if edge.condition:
                    # Map conditions to conditional logic methods
                    condition_func = self._get_condition_function(edge.condition, conditional_logic)
                    
                    if condition_func:
                        # Create conditional edge mapping
                        condition_map = self._create_condition_map(edge, template)
                        workflow.add_conditional_edges(edge.from_node, condition_func, condition_map)
                    else:
                        self.logger.warning(f"Condition function not found: {edge.condition}")
                        workflow.add_edge(edge.from_node, edge.to_node)
                else:
                    workflow.add_edge(edge.from_node, edge.to_node)
            
            # Compile workflow
            compiled_workflow = workflow.compile()
            
            workflow_id = f"{workflow_type.value}_{hash(str(config.to_dict()))}_{int(datetime.now().timestamp())}"
            self.active_workflows[workflow_id] = compiled_workflow
            
            self.logger.info(f"Created editor workflow: {workflow_type.value}")
            
            if self.metrics:
                self.metrics.record_counter("editor_workflow_created", workflow_type=workflow_type.value)
            
            return compiled_workflow
            
        except Exception as e:
            self.logger.error(f"Failed to create editor workflow: {e}")
            return None
    
    def _get_condition_function(self, condition_name: str, logic: EditorConditionalLogic) -> Optional[Callable]:
        """Map condition names to conditional logic functions"""
        condition_map = {
            "grammar_passed": logic.grammar_check_decision,
            "grammar_needs_work": logic.grammar_check_decision,
            "seo_passed": logic.seo_optimization_decision,
            "seo_needs_work": logic.seo_optimization_decision,
            "quality_approved": logic.quality_gate_decision,
            "needs_revision": logic.quality_gate_decision,
            "needs_escalation": logic.quality_gate_decision,
            "quality_acceptable": logic.quality_gate_decision,
            "needs_quick_fix": logic.quality_gate_decision,
            "needs_grammar_revision": logic.grammar_check_decision,
            "needs_seo_revision": logic.seo_optimization_decision
        }
        
        return condition_map.get(condition_name)
    
    def _create_condition_map(self, edge: WorkflowEdge, template: WorkflowTemplate) -> Dict[str, str]:
        """Create condition mapping for conditional edges"""
        # Base condition mappings
        base_mappings = {
            "continue": edge.to_node,
            "pass": edge.to_node,
            "approved": edge.to_node,
            "acceptable": edge.to_node,
            "end": END,
            "complete": END
        }
        
        # Find alternative paths for "needs work" conditions
        if "needs" in edge.condition.lower():
            # Look for revision or improvement nodes
            revision_nodes = [node.name for node in template.nodes if "improvement" in node.name.lower() or "revision" in node.name.lower()]
            if revision_nodes:
                base_mappings["retry"] = revision_nodes[0]
                base_mappings["revise"] = revision_nodes[0]
        
        return base_mappings
    
    def get_workflow_templates(self) -> List[Dict[str, Any]]:
        """Get available workflow templates"""
        return [
            {
                "type": template_type,
                "name": template.name,
                "description": template.description,
                "nodes": len(template.nodes),
                "complexity": "high" if len(template.nodes) > 8 else "medium" if len(template.nodes) > 5 else "low"
            }
            for template_type, template in self.editor_templates.items()
        ]
    
    def cleanup_workflow(self, workflow_id: str):
        """Clean up completed workflow"""
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]
            self.logger.info(f"Cleaned up workflow: {workflow_id}")


# Global workflow manager instance
_editor_workflow_manager: Optional[EditorWorkflowManager] = None


def get_editor_workflow_manager() -> EditorWorkflowManager:
    """Get the global editor workflow manager instance"""
    global _editor_workflow_manager
    if _editor_workflow_manager is None:
        _editor_workflow_manager = EditorWorkflowManager()
    return _editor_workflow_manager


# Export main classes
__all__ = [
    'EditorConditionalLogic',
    'EditorWorkflowManager',
    'EditingWorkflowConfig',
    'EditingWorkflowType',
    'EditingComplexity',
    'WorkflowDecision',
    'get_editor_workflow_manager'
]


if __name__ == "__main__":
    # Test workflow creation
    async def test_editor_workflows():
        """Test editor workflow creation and logic"""
        
        config = EditingWorkflowConfig(
            workflow_type=EditingWorkflowType.COMPREHENSIVE,
            complexity_level=EditingComplexity.MEDIUM
        )
        
        conditional_logic = EditorConditionalLogic(config)
        workflow_manager = get_editor_workflow_manager()
        
        # Mock node functions
        mock_functions = {
            "assess_content_complexity": lambda state: state,
            "comprehensive_grammar_check": lambda state: state,
            "optimize_writing_style": lambda state: state,
            "comprehensive_seo_optimization": lambda state: state,
            # Add more mock functions as needed
        }
        
        try:
            workflow = await workflow_manager.create_workflow(
                EditingWorkflowType.COMPREHENSIVE,
                config,
                mock_functions,
                conditional_logic
            )
            
            if workflow:
                print("✓ Editor workflow created successfully")
                print(f"✓ Available templates: {len(workflow_manager.get_workflow_templates())}")
            else:
                print("✗ Failed to create editor workflow")
                
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    # Uncomment to run test
    # asyncio.run(test_editor_workflows())