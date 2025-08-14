"""Agentic frameworks integration module."""

from .crewai import (
    # Configuration
    CrewAIConfig,
    CrewAIFramework,
    get_crewai_framework,
    shutdown_crewai_framework,
    # Agents
    AgentRole,
    AgentDefinition,
    AgentRegistry,
    get_agent_registry,
    # Crews
    ProcessType,
    TaskDefinition,
    CrewDefinition,
    CrewTemplateRegistry,
    get_crew_registry,
    # Delegation
    DelegationStrategy,
    TaskStatus,
    DelegatedTask,
    DelegationPlan,
    TaskDelegationManager,
    get_delegation_manager
)

from .langgraph import (
    # Configuration
    LangGraphConfig,
    LangGraphFramework,
    get_langgraph_framework,
    shutdown_langgraph_framework,
    # State Management
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

from .autogen import (
    # Configuration
    AutoGenConfig,
    AutoGenFramework,
    get_autogen_framework,
    shutdown_autogen_framework,
    # Conversations
    ConversationPattern,
    SpeakerSelection,
    ConversationTemplate,
    ConversationState,
    ConversationPatternRegistry,
    CommunicationProtocol,
    get_conversation_registry,
    get_communication_protocol,
    # Coordination
    CoordinationStrategy,
    GroupChatRole,
    GroupChatSession,
    CoordinationRule,
    GroupChatCoordinator,
    get_group_chat_coordinator
)

__all__ = [
    # CrewAI
    "CrewAIConfig",
    "CrewAIFramework", 
    "get_crewai_framework",
    "shutdown_crewai_framework",
    "AgentRole",
    "AgentDefinition",
    "AgentRegistry",
    "get_agent_registry",
    "ProcessType",
    "TaskDefinition",
    "CrewDefinition",
    "CrewTemplateRegistry",
    "get_crew_registry",
    "DelegationStrategy",
    "TaskStatus",
    "DelegatedTask",
    "DelegationPlan",
    "TaskDelegationManager",
    "get_delegation_manager",
    
    # LangGraph
    "LangGraphConfig",
    "LangGraphFramework",
    "get_langgraph_framework", 
    "shutdown_langgraph_framework",
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
    
    # AutoGen
    "AutoGenConfig",
    "AutoGenFramework",
    "get_autogen_framework",
    "shutdown_autogen_framework",
    "ConversationPattern",
    "SpeakerSelection",
    "ConversationTemplate",
    "ConversationState",
    "ConversationPatternRegistry",
    "CommunicationProtocol",
    "get_conversation_registry",
    "get_communication_protocol",
    "CoordinationStrategy",
    "GroupChatRole",
    "GroupChatSession",
    "CoordinationRule",
    "GroupChatCoordinator",
    "get_group_chat_coordinator"
]