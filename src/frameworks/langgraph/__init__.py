"""LangGraph framework integration module."""

from .config import (
    LangGraphConfig,
    LangGraphFramework,
    get_langgraph_framework,
    shutdown_langgraph_framework
)
from .state import (
    StateStatus,
    ContentType,
    ResearchData,
    ContentStrategy,
    QualityMetrics,
    ContentCreationState,
    ContentEditingState,
    ResearchState,
    StateManager,
    get_state_manager
)
from .workflows import (
    WorkflowNode,
    WorkflowEdge,
    WorkflowTemplate,
    WorkflowTemplateRegistry,
    ConditionalLogicPatterns,
    get_workflow_registry,
    get_conditional_logic
)

__all__ = [
    # Configuration
    "LangGraphConfig",
    "LangGraphFramework",
    "get_langgraph_framework",
    "shutdown_langgraph_framework",
    # State Management
    "StateStatus",
    "ContentType",
    "ResearchData",
    "ContentStrategy",
    "QualityMetrics",
    "ContentCreationState",
    "ContentEditingState",
    "ResearchState",
    "StateManager",
    "get_state_manager",
    # Workflows
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowTemplate",
    "WorkflowTemplateRegistry",
    "ConditionalLogicPatterns",
    "get_workflow_registry",
    "get_conditional_logic"
]