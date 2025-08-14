"""AutoGen framework integration module."""

from .config import (
    AutoGenConfig,
    AutoGenFramework,
    get_autogen_framework,
    shutdown_autogen_framework
)
from .conversations import (
    ConversationPattern,
    SpeakerSelection,
    ConversationTemplate,
    ConversationState,
    ConversationPatternRegistry,
    CommunicationProtocol,
    get_conversation_registry,
    get_communication_protocol
)
from .coordination import (
    CoordinationStrategy,
    GroupChatRole,
    GroupChatSession,
    CoordinationRule,
    GroupChatCoordinator,
    get_group_chat_coordinator
)

__all__ = [
    # Configuration
    "AutoGenConfig",
    "AutoGenFramework",
    "get_autogen_framework",
    "shutdown_autogen_framework",
    # Conversations
    "ConversationPattern",
    "SpeakerSelection",
    "ConversationTemplate",
    "ConversationState",
    "ConversationPatternRegistry",
    "CommunicationProtocol",
    "get_conversation_registry",
    "get_communication_protocol",
    # Coordination
    "CoordinationStrategy",
    "GroupChatRole",
    "GroupChatSession",
    "CoordinationRule",
    "GroupChatCoordinator",
    "get_group_chat_coordinator"
]