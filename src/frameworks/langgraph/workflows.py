"""LangGraph workflow graph templates and conditional logic patterns."""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
import asyncio

from ...core.logging import get_framework_logger
from ...core.errors import LangGraphError, WorkflowError
from ...core.monitoring import get_metrics_collector
from .config import get_langgraph_framework
from .state import (
    ContentCreationState, ContentEditingState, ResearchState,
    StateStatus, ContentType, QualityMetrics, get_state_manager
)


@dataclass
class WorkflowNode:
    """Definition for a workflow node."""
    
    name: str
    function: str
    description: str
    tools: List[str] = field(default_factory=list)
    interrupt: bool = False
    timeout: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "function": self.function,
            "description": self.description,
            "tools": self.tools,
            "interrupt": self.interrupt,
            "timeout": self.timeout
        }


@dataclass
class WorkflowEdge:
    """Definition for a workflow edge."""
    
    from_node: str
    to_node: str
    condition: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "condition": self.condition,
            "description": self.description
        }


@dataclass
class WorkflowTemplate:
    """Template for a LangGraph workflow."""
    
    name: str
    description: str
    state_type: str
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]
    entry_point: str
    interrupt_before: List[str] = field(default_factory=list)
    interrupt_after: List[str] = field(default_factory=list)
    
    def validate(self):
        """Validate workflow template."""
        node_names = {node.name for node in self.nodes}
        
        # Validate entry point
        if self.entry_point not in node_names:
            raise LangGraphError(f"Entry point '{self.entry_point}' not in nodes")
        
        # Validate edges
        for edge in self.edges:
            if edge.from_node not in node_names:
                raise LangGraphError(f"Edge from_node '{edge.from_node}' not in nodes")
            if edge.to_node not in node_names:
                raise LangGraphError(f"Edge to_node '{edge.to_node}' not in nodes")


class WorkflowTemplateRegistry:
    """Registry for managing LangGraph workflow templates."""
    
    def __init__(self):
        self.logger = get_framework_logger("LangGraph")
        self.metrics = get_metrics_collector()
        self.state_manager = get_state_manager()
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default workflow templates."""
        
        # Content Creation Workflow
        content_creation_template = WorkflowTemplate(
            name="Content Creation Workflow",
            description="Complete content creation with research analysis, outline, writing, and review",
            state_type="content_creation",
            entry_point="analyze_research",
            nodes=[
                WorkflowNode(
                    name="analyze_research",
                    function="analyze_research_data",
                    description="Analyze research data and extract key insights",
                    tools=["content_analyzer", "topic_extractor"]
                ),
                WorkflowNode(
                    name="create_outline", 
                    function="generate_content_outline",
                    description="Create structured content outline",
                    tools=["content_analyzer"]
                ),
                WorkflowNode(
                    name="write_content",
                    function="generate_content",
                    description="Write main content based on outline and research",
                    tools=["content_generator", "headline_creator"]
                ),
                WorkflowNode(
                    name="self_review",
                    function="review_content_quality", 
                    description="Self-review content quality and identify improvements",
                    tools=["content_analyzer", "sentiment_analyzer"]
                ),
                WorkflowNode(
                    name="human_review",
                    function="escalate_to_human",
                    description="Escalate to human for review",
                    interrupt=True
                ),
                WorkflowNode(
                    name="finalize_content",
                    function="finalize_content_output",
                    description="Finalize content for delivery",
                    tools=["content_analyzer"]
                )
            ],
            edges=[
                WorkflowEdge(
                    from_node="analyze_research",
                    to_node="create_outline",
                    description="Move to outline creation"
                ),
                WorkflowEdge(
                    from_node="create_outline", 
                    to_node="write_content",
                    description="Move to content writing"
                ),
                WorkflowEdge(
                    from_node="write_content",
                    to_node="self_review",
                    description="Move to self-review"
                ),
                WorkflowEdge(
                    from_node="self_review",
                    to_node="write_content", 
                    condition="should_revise",
                    description="Revise content based on review"
                ),
                WorkflowEdge(
                    from_node="self_review",
                    to_node="human_review",
                    condition="needs_human_review",
                    description="Escalate for human review"
                ),
                WorkflowEdge(
                    from_node="self_review",
                    to_node="finalize_content",
                    condition="quality_approved",
                    description="Finalize approved content"
                ),
                WorkflowEdge(
                    from_node="human_review",
                    to_node="finalize_content",
                    description="Finalize after human review"
                )
            ],
            interrupt_before=["human_review"],
            interrupt_after=["finalize_content"]
        )
        self.register_template(content_creation_template)
        
        # Content Editing Workflow
        content_editing_template = WorkflowTemplate(
            name="Content Editing Workflow",
            description="Multi-stage content editing and optimization",
            state_type="content_editing",
            entry_point="grammar_check",
            nodes=[
                WorkflowNode(
                    name="grammar_check",
                    function="check_grammar_and_style",
                    description="Check grammar and style issues",
                    tools=["grammar_checker"]
                ),
                WorkflowNode(
                    name="seo_optimization",
                    function="optimize_for_seo",
                    description="Optimize content for search engines",
                    tools=["seo_optimizer"]
                ),
                WorkflowNode(
                    name="readability_check",
                    function="assess_readability",
                    description="Assess and improve readability",
                    tools=["readability_scorer"]
                ),
                WorkflowNode(
                    name="final_review",
                    function="conduct_final_review",
                    description="Conduct comprehensive final review",
                    tools=["content_analyzer", "sentiment_analyzer"]
                ),
                WorkflowNode(
                    name="quality_gate",
                    function="quality_gate_check",
                    description="Quality gate checkpoint",
                    interrupt=True
                ),
                WorkflowNode(
                    name="revision_loop",
                    function="apply_revisions",
                    description="Apply necessary revisions",
                    tools=["content_analyzer"]
                )
            ],
            edges=[
                WorkflowEdge(
                    from_node="grammar_check",
                    to_node="seo_optimization",
                    description="Move to SEO optimization"
                ),
                WorkflowEdge(
                    from_node="seo_optimization",
                    to_node="readability_check",
                    description="Move to readability check"
                ),
                WorkflowEdge(
                    from_node="readability_check",
                    to_node="final_review",
                    description="Move to final review"
                ),
                WorkflowEdge(
                    from_node="final_review",
                    to_node="quality_gate",
                    description="Move to quality gate"
                ),
                WorkflowEdge(
                    from_node="quality_gate",
                    to_node="revision_loop",
                    condition="needs_revision",
                    description="Apply revisions if needed"
                ),
                WorkflowEdge(
                    from_node="revision_loop",
                    to_node="final_review",
                    description="Re-review after revisions"
                )
            ],
            interrupt_after=["quality_gate"]
        )
        self.register_template(content_editing_template)
        
        # Research Workflow
        research_template = WorkflowTemplate(
            name="Research Workflow",
            description="Comprehensive research and analysis workflow",
            state_type="research",
            entry_point="web_research",
            nodes=[
                WorkflowNode(
                    name="web_research",
                    function="conduct_web_research",
                    description="Conduct comprehensive web research",
                    tools=["web_search", "content_retrieval"]
                ),
                WorkflowNode(
                    name="trend_analysis",
                    function="analyze_trends",
                    description="Analyze trends and patterns",
                    tools=["trend_analysis", "news_search"]
                ),
                WorkflowNode(
                    name="competitive_analysis",
                    function="analyze_competition",
                    description="Analyze competitive landscape",
                    tools=["web_search", "content_analyzer"]
                ),
                WorkflowNode(
                    name="fact_verification",
                    function="verify_facts",
                    description="Verify key facts and claims",
                    tools=["web_search", "content_analyzer"]
                ),
                WorkflowNode(
                    name="consolidate_findings",
                    function="consolidate_research",
                    description="Consolidate all research findings",
                    tools=["content_analyzer", "topic_extractor"]
                )
            ],
            edges=[
                WorkflowEdge(
                    from_node="web_research",
                    to_node="trend_analysis",
                    description="Move to trend analysis"
                ),
                WorkflowEdge(
                    from_node="trend_analysis",
                    to_node="competitive_analysis",
                    description="Move to competitive analysis"
                ),
                WorkflowEdge(
                    from_node="competitive_analysis", 
                    to_node="fact_verification",
                    description="Move to fact verification"
                ),
                WorkflowEdge(
                    from_node="fact_verification",
                    to_node="consolidate_findings",
                    description="Consolidate all findings"
                )
            ]
        )
        self.register_template(research_template)
        
        # Rapid Content Creation (Simplified)
        rapid_content_template = WorkflowTemplate(
            name="Rapid Content Creation",
            description="Streamlined content creation for urgent needs",
            state_type="content_creation",
            entry_point="quick_outline",
            nodes=[
                WorkflowNode(
                    name="quick_outline",
                    function="create_quick_outline",
                    description="Create quick content outline",
                    tools=["content_analyzer"]
                ),
                WorkflowNode(
                    name="rapid_write",
                    function="rapid_content_generation",
                    description="Generate content quickly",
                    tools=["content_generator"]
                ),
                WorkflowNode(
                    name="essential_review",
                    function="essential_quality_check",
                    description="Essential quality check only",
                    tools=["grammar_checker", "readability_scorer"]
                )
            ],
            edges=[
                WorkflowEdge(
                    from_node="quick_outline",
                    to_node="rapid_write",
                    description="Move to rapid writing"
                ),
                WorkflowEdge(
                    from_node="rapid_write",
                    to_node="essential_review",
                    description="Move to essential review"
                ),
                WorkflowEdge(
                    from_node="essential_review",
                    to_node="rapid_write",
                    condition="needs_quick_revision",
                    description="Quick revision if needed"
                )
            ]
        )
        self.register_template(rapid_content_template)
    
    def register_template(self, template: WorkflowTemplate):
        """Register a workflow template."""
        try:
            template.validate()
            template_id = template.name.lower().replace(" ", "_")
            self.templates[template_id] = template
            self.logger.info(f"Registered LangGraph workflow template: {template.name}")
            self.metrics.record_counter("workflow_template_registered", framework="langgraph")
        except Exception as e:
            raise LangGraphError(f"Failed to register template {template.name}: {e}")
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a workflow template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[WorkflowTemplate]:
        """List all registered templates."""
        return list(self.templates.values())
    
    async def create_graph_from_template(
        self,
        template_id: str,
        node_functions: Dict[str, Callable],
        condition_functions: Dict[str, Callable]
    ):
        """Create a LangGraph graph from a template."""
        template = self.get_template(template_id)
        if not template:
            raise LangGraphError(f"Template not found: {template_id}")
        
        try:
            from langgraph.graph import StateGraph, END
            
            # Get state schema
            if template.state_type == "content_creation":
                from .state import ContentCreationState
                state_schema = ContentCreationState
            elif template.state_type == "content_editing":
                from .state import ContentEditingState
                state_schema = ContentEditingState
            elif template.state_type == "research":
                from .state import ResearchState
                state_schema = ResearchState
            else:
                raise LangGraphError(f"Unknown state type: {template.state_type}")
            
            # Create graph
            workflow = StateGraph(state_schema)
            
            # Add nodes
            for node in template.nodes:
                if node.function not in node_functions:
                    raise LangGraphError(f"Node function not provided: {node.function}")
                
                workflow.add_node(node.name, node_functions[node.function])
            
            # Set entry point
            workflow.set_entry_point(template.entry_point)
            
            # Add edges
            for edge in template.edges:
                if edge.condition:
                    # Conditional edge
                    if edge.condition not in condition_functions:
                        raise LangGraphError(f"Condition function not provided: {edge.condition}")
                    
                    workflow.add_conditional_edges(
                        edge.from_node,
                        condition_functions[edge.condition],
                        {
                            "continue": edge.to_node,
                            "end": END
                        }
                    )
                else:
                    # Regular edge
                    workflow.add_edge(edge.from_node, edge.to_node)
            
            self.logger.info(f"Created LangGraph graph from template: {template.name}")
            self.metrics.record_counter("graph_created_from_template", framework="langgraph")
            
            return workflow
            
        except ImportError as e:
            raise LangGraphError(f"LangGraph not installed: {e}")
        except Exception as e:
            raise LangGraphError(f"Failed to create graph from template: {e}")


class ConditionalLogicPatterns:
    """Common conditional logic patterns for LangGraph workflows."""
    
    def __init__(self):
        self.logger = get_framework_logger("LangGraph")
    
    # Content Creation Conditions
    def should_revise(self, state: ContentCreationState) -> str:
        """Determine if content should be revised."""
        if not state.get("quality_metrics"):
            return "continue"
        
        quality = state["quality_metrics"]
        if isinstance(quality, dict):
            overall_score = quality.get("overall_score", 0.0)
        else:
            overall_score = getattr(quality, "overall_score", 0.0)
        
        if overall_score < 0.7:
            return "continue"  # Revise
        return "end"
    
    def needs_human_review(self, state: ContentCreationState) -> str:
        """Determine if human review is needed."""
        if state.get("review_required", False):
            return "continue"
        
        if not state.get("quality_metrics"):
            return "continue"
        
        quality = state["quality_metrics"]
        if isinstance(quality, dict):
            overall_score = quality.get("overall_score", 0.0)
            issues = quality.get("issues", [])
        else:
            overall_score = getattr(quality, "overall_score", 0.0)
            issues = getattr(quality, "issues", [])
        
        # Escalate if quality is borderline or there are critical issues
        if overall_score < 0.8 or any("critical" in issue.lower() for issue in issues):
            return "continue"
        
        return "end"
    
    def quality_approved(self, state: ContentCreationState) -> str:
        """Check if quality is approved for finalization."""
        if not state.get("quality_metrics"):
            return "end"
        
        quality = state["quality_metrics"]
        if isinstance(quality, dict):
            overall_score = quality.get("overall_score", 0.0)
        else:
            overall_score = getattr(quality, "overall_score", 0.0)
        
        return "continue" if overall_score >= 0.8 else "end"
    
    # Content Editing Conditions  
    def needs_revision(self, state: ContentEditingState) -> str:
        """Determine if content needs revision."""
        current_quality = state.get("current_quality")
        target_score = state.get("target_quality_score", 0.8)
        max_iterations = state.get("max_iterations", 3)
        current_iterations = state.get("editing_iterations", 0)
        
        if current_iterations >= max_iterations:
            return "end"  # Max iterations reached
        
        if not current_quality:
            return "continue"  # Need revision if no quality data
        
        if isinstance(current_quality, dict):
            overall_score = current_quality.get("overall_score", 0.0)
        else:
            overall_score = getattr(current_quality, "overall_score", 0.0)
        
        return "continue" if overall_score < target_score else "end"
    
    def quality_gate_passed(self, state: ContentEditingState) -> str:
        """Check if content passes quality gate."""
        current_quality = state.get("current_quality")
        target_score = state.get("target_quality_score", 0.8)
        
        if not current_quality:
            return "end"  # Fail if no quality data
        
        if isinstance(current_quality, dict):
            overall_score = current_quality.get("overall_score", 0.0)
        else:
            overall_score = getattr(current_quality, "overall_score", 0.0)
        
        return "continue" if overall_score >= target_score else "end"
    
    # Research Conditions
    def research_complete(self, state: ResearchState) -> str:
        """Check if research is complete."""
        completed_types = state.get("completed_research_types", [])
        required_types = ["web_research", "trend_analysis", "competitive_analysis"]
        
        if all(req_type in completed_types for req_type in required_types):
            return "continue"
        return "end"
    
    def needs_fact_verification(self, state: ResearchState) -> str:
        """Determine if additional fact verification is needed."""
        conflicting_info = state.get("conflicting_information", [])
        information_gaps = state.get("information_gaps", [])
        
        if conflicting_info or information_gaps:
            return "continue"
        return "end"
    
    # Rapid Content Conditions
    def needs_quick_revision(self, state: ContentCreationState) -> str:
        """Quick revision check for rapid workflows."""
        if not state.get("quality_metrics"):
            return "end"  # Skip if no quality data in rapid mode
        
        quality = state["quality_metrics"]
        if isinstance(quality, dict):
            grammar_score = quality.get("grammar_score", 1.0)
            readability_score = quality.get("readability_score", 1.0)
        else:
            grammar_score = getattr(quality, "grammar_score", 1.0)
            readability_score = getattr(quality, "readability_score", 1.0)
        
        # Only revise if grammar or readability are critically low
        if grammar_score < 0.6 or readability_score < 0.5:
            return "continue"
        return "end"


# Global workflow template registry instance
_workflow_registry: Optional[WorkflowTemplateRegistry] = None


def get_workflow_registry() -> WorkflowTemplateRegistry:
    """Get the global workflow template registry instance."""
    global _workflow_registry
    if _workflow_registry is None:
        _workflow_registry = WorkflowTemplateRegistry()
    return _workflow_registry


def get_conditional_logic() -> ConditionalLogicPatterns:
    """Get conditional logic patterns instance."""
    return ConditionalLogicPatterns()