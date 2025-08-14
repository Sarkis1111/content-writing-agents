"""CrewAI framework integration module."""

from .config import (
    CrewAIConfig,
    CrewAIFramework,
    get_crewai_framework,
    shutdown_crewai_framework
)
from .agents import (
    AgentRole,
    AgentDefinition,
    AgentRegistry,
    get_agent_registry
)
from .crews import (
    ProcessType,
    TaskDefinition,
    CrewDefinition,
    CrewTemplateRegistry,
    get_crew_registry
)
from .delegation import (
    DelegationStrategy,
    TaskStatus,
    DelegatedTask,
    DelegationPlan,
    TaskDelegationManager,
    get_delegation_manager
)

__all__ = [
    # Configuration
    "CrewAIConfig",
    "CrewAIFramework",
    "get_crewai_framework",
    "shutdown_crewai_framework",
    # Agents
    "AgentRole",
    "AgentDefinition",
    "AgentRegistry",
    "get_agent_registry",
    # Crews
    "ProcessType",
    "TaskDefinition",
    "CrewDefinition", 
    "CrewTemplateRegistry",
    "get_crew_registry",
    # Delegation
    "DelegationStrategy",
    "TaskStatus",
    "DelegatedTask",
    "DelegationPlan",
    "TaskDelegationManager",
    "get_delegation_manager"
]