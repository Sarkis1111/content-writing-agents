"""AutoGen conversation patterns and agent communication protocols."""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

from ...core.logging import get_framework_logger
from ...core.errors import AutoGenError, WorkflowError
from ...core.monitoring import get_metrics_collector
from .config import get_autogen_framework


class ConversationPattern(Enum):
    """AutoGen conversation patterns."""
    TWO_AGENT = "two_agent"
    GROUP_CHAT = "group_chat"
    HIERARCHICAL = "hierarchical"
    ROUND_ROBIN = "round_robin"
    BROADCAST = "broadcast"


class SpeakerSelection(Enum):
    """Speaker selection methods."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    MANUAL = "manual"
    AUTO = "auto"


@dataclass
class ConversationTemplate:
    """Template for AutoGen conversations."""
    
    name: str
    description: str
    pattern: ConversationPattern
    agents: List[Dict[str, Any]]
    initial_message: str
    max_rounds: int = 10
    speaker_selection: SpeakerSelection = SpeakerSelection.ROUND_ROBIN
    termination_conditions: List[str] = field(default_factory=list)
    conversation_flow: Optional[Dict[str, Any]] = None
    
    def validate(self):
        """Validate conversation template."""
        if not self.agents:
            raise AutoGenError(f"Conversation {self.name} must have at least one agent")
        
        if self.pattern == ConversationPattern.TWO_AGENT and len(self.agents) != 2:
            raise AutoGenError(f"Two-agent conversation {self.name} must have exactly 2 agents")
        
        if self.pattern == ConversationPattern.GROUP_CHAT and len(self.agents) < 3:
            raise AutoGenError(f"Group chat {self.name} must have at least 3 agents")


@dataclass 
class ConversationState:
    """State tracking for conversations."""
    
    conversation_id: str
    template_name: str
    status: str = "pending"  # pending, active, completed, failed
    current_round: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    current_speaker: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get conversation duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if conversation is active."""
        return self.status == "active"
    
    @property
    def is_complete(self) -> bool:
        """Check if conversation is complete."""
        return self.status in ["completed", "failed"]


class ConversationPatternRegistry:
    """Registry for AutoGen conversation patterns."""
    
    def __init__(self):
        self.logger = get_framework_logger("AutoGen")
        self.metrics = get_metrics_collector()
        self.templates: Dict[str, ConversationTemplate] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default conversation templates."""
        
        # Strategy Council Template
        strategy_council_template = ConversationTemplate(
            name="Strategy Council",
            description="Multi-agent strategic discussion for content planning",
            pattern=ConversationPattern.GROUP_CHAT,
            agents=[
                {
                    "id": "content_strategist",
                    "name": "Content Strategist", 
                    "type": "assistant",
                    "system_message": "You are a content strategy expert focused on overall content direction and alignment with business goals. Provide strategic insights and recommendations.",
                    "description": "Strategic planning specialist"
                },
                {
                    "id": "audience_analyst",
                    "name": "Audience Analyst",
                    "type": "assistant", 
                    "system_message": "You specialize in audience analysis and targeting, ensuring content resonates with intended audiences. Focus on demographic insights and engagement patterns.",
                    "description": "Audience research specialist"
                },
                {
                    "id": "competitive_analyst",
                    "name": "Competitive Analyst",
                    "type": "assistant",
                    "system_message": "You analyze competitive landscape and positioning opportunities for content. Identify market gaps and differentiation strategies.",
                    "description": "Market analysis specialist"
                },
                {
                    "id": "performance_optimizer",
                    "name": "Performance Optimizer",
                    "type": "assistant",
                    "system_message": "You focus on content performance optimization and measurable improvement recommendations. Emphasize data-driven decisions and KPIs.",
                    "description": "Performance optimization specialist"
                }
            ],
            initial_message="Let's develop a comprehensive content strategy for {topic}. Please share your perspectives on approach, audience targeting, competitive positioning, and performance optimization.",
            max_rounds=15,
            speaker_selection=SpeakerSelection.ROUND_ROBIN,
            termination_conditions=["consensus_reached", "max_rounds_reached"]
        )
        self.register_template(strategy_council_template)
        
        # Content Review Template
        content_review_template = ConversationTemplate(
            name="Content Review",
            description="Two-agent content review and improvement discussion",
            pattern=ConversationPattern.TWO_AGENT,
            agents=[
                {
                    "id": "content_reviewer",
                    "name": "Content Reviewer",
                    "type": "assistant",
                    "system_message": "You are an expert content reviewer. Analyze the provided content for quality, clarity, engagement, and alignment with objectives. Provide specific, actionable feedback.",
                    "description": "Content quality specialist"
                },
                {
                    "id": "content_improver",
                    "name": "Content Improver", 
                    "type": "assistant",
                    "system_message": "You specialize in improving content based on feedback. Take review comments and implement specific improvements to enhance quality, clarity, and impact.",
                    "description": "Content improvement specialist"
                }
            ],
            initial_message="Please review this content and provide improvement recommendations: {content}",
            max_rounds=6,
            speaker_selection=SpeakerSelection.AUTO,
            termination_conditions=["quality_approved", "max_iterations_reached"]
        )
        self.register_template(content_review_template)
        
        # Research Collaboration Template
        research_collaboration_template = ConversationTemplate(
            name="Research Collaboration",
            description="Multi-agent research discussion and synthesis",
            pattern=ConversationPattern.GROUP_CHAT,
            agents=[
                {
                    "id": "primary_researcher",
                    "name": "Primary Researcher",
                    "type": "assistant",
                    "system_message": "You are the lead researcher. Coordinate research efforts, synthesize findings, and ensure comprehensive coverage of the topic.",
                    "description": "Lead research coordinator"
                },
                {
                    "id": "fact_checker",
                    "name": "Fact Checker",
                    "type": "assistant",
                    "system_message": "You specialize in fact verification and source credibility assessment. Validate claims and ensure information accuracy.",
                    "description": "Fact verification specialist"
                },
                {
                    "id": "trend_analyst",
                    "name": "Trend Analyst", 
                    "type": "assistant",
                    "system_message": "You focus on identifying trends, patterns, and emerging developments related to the research topic. Provide forward-looking insights.",
                    "description": "Trend analysis specialist"
                }
            ],
            initial_message="Let's collaborate on researching {topic}. Primary Researcher will coordinate, Fact Checker will verify information, and Trend Analyst will identify patterns and trends.",
            max_rounds=12,
            speaker_selection=SpeakerSelection.ROUND_ROBIN,
            termination_conditions=["research_complete", "consensus_reached"]
        )
        self.register_template(research_collaboration_template)
        
        # Quick Decision Template
        quick_decision_template = ConversationTemplate(
            name="Quick Decision",
            description="Rapid decision making between two expert agents",
            pattern=ConversationPattern.TWO_AGENT,
            agents=[
                {
                    "id": "decision_analyst",
                    "name": "Decision Analyst",
                    "type": "assistant", 
                    "system_message": "You analyze decision options quickly and provide structured recommendations with clear pros and cons. Be concise and decisive.",
                    "description": "Decision analysis specialist"
                },
                {
                    "id": "implementation_expert",
                    "name": "Implementation Expert",
                    "type": "assistant",
                    "system_message": "You focus on practical implementation considerations. Assess feasibility, resource requirements, and execution challenges for proposed decisions.",
                    "description": "Implementation feasibility specialist"
                }
            ],
            initial_message="We need to make a quick decision about: {decision_topic}. Please analyze options and implementation considerations.",
            max_rounds=4,
            speaker_selection=SpeakerSelection.AUTO,
            termination_conditions=["decision_made", "consensus_reached"]
        )
        self.register_template(quick_decision_template)
        
        # Brainstorming Template
        brainstorming_template = ConversationTemplate(
            name="Creative Brainstorming",
            description="Multi-agent creative brainstorming session",
            pattern=ConversationPattern.BROADCAST,
            agents=[
                {
                    "id": "creative_thinker",
                    "name": "Creative Thinker",
                    "type": "assistant",
                    "system_message": "You are a creative ideation expert. Generate innovative, out-of-the-box ideas and creative solutions. Think beyond conventional approaches.",
                    "description": "Creative ideation specialist"
                },
                {
                    "id": "practical_evaluator",
                    "name": "Practical Evaluator",
                    "type": "assistant",
                    "system_message": "You evaluate ideas for practical feasibility and implementation. Build on creative ideas while ensuring they're actionable and realistic.",
                    "description": "Practical evaluation specialist"
                },
                {
                    "id": "innovation_synthesizer",
                    "name": "Innovation Synthesizer",
                    "type": "assistant",
                    "system_message": "You combine and refine ideas from different perspectives. Synthesize creative concepts with practical constraints to develop innovative solutions.",
                    "description": "Innovation synthesis specialist"
                }
            ],
            initial_message="Let's brainstorm creative approaches for {challenge}. Creative Thinker will generate ideas, Practical Evaluator will assess feasibility, and Innovation Synthesizer will combine perspectives.",
            max_rounds=10,
            speaker_selection=SpeakerSelection.ROUND_ROBIN,
            termination_conditions=["ideas_synthesized", "creative_consensus"]
        )
        self.register_template(brainstorming_template)
    
    def register_template(self, template: ConversationTemplate):
        """Register a conversation template."""
        try:
            template.validate()
            template_id = template.name.lower().replace(" ", "_")
            self.templates[template_id] = template
            self.logger.info(f"Registered AutoGen conversation template: {template.name}")
            self.metrics.record_counter("conversation_template_registered", framework="autogen")
        except Exception as e:
            raise AutoGenError(f"Failed to register conversation template {template.name}: {e}")
    
    def get_template(self, template_id: str) -> Optional[ConversationTemplate]:
        """Get a conversation template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[ConversationTemplate]:
        """List all registered conversation templates."""
        return list(self.templates.values())
    
    async def create_conversation_from_template(
        self,
        template_id: str,
        conversation_id: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> ConversationState:
        """Create a conversation from template."""
        template = self.get_template(template_id)
        if not template:
            raise AutoGenError(f"Conversation template not found: {template_id}")
        
        variables = variables or {}
        
        try:
            framework = await get_autogen_framework()
            
            # Create agents from template
            agent_ids = []
            for agent_def in template.agents:
                agent_id = f"{conversation_id}_{agent_def['id']}"
                
                # Replace variables in system message
                system_message = agent_def["system_message"]
                for key, value in variables.items():
                    system_message = system_message.replace(f"{{{key}}}", str(value))
                
                if agent_def["type"] == "assistant":
                    framework.create_assistant_agent(
                        agent_id=agent_id,
                        name=agent_def["name"],
                        system_message=system_message,
                        description=agent_def.get("description")
                    )
                elif agent_def["type"] == "conversable":
                    framework.create_conversable_agent(
                        agent_id=agent_id,
                        name=agent_def["name"],
                        system_message=system_message,
                        description=agent_def.get("description")
                    )
                elif agent_def["type"] == "user_proxy":
                    framework.create_user_proxy_agent(
                        agent_id=agent_id,
                        name=agent_def["name"],
                        system_message=system_message,
                        description=agent_def.get("description")
                    )
                
                agent_ids.append(agent_id)
            
            # Create group chat if needed
            if template.pattern in [ConversationPattern.GROUP_CHAT, ConversationPattern.ROUND_ROBIN, ConversationPattern.BROADCAST]:
                group_chat = framework.create_group_chat(
                    chat_id=f"{conversation_id}_group",
                    agents=agent_ids,
                    max_round=template.max_rounds,
                    speaker_selection_method=template.speaker_selection.value
                )
                
                manager = framework.create_group_chat_manager(
                    manager_id=f"{conversation_id}_manager",
                    group_chat_id=f"{conversation_id}_group",
                    name=f"{template.name} Manager"
                )
            
            # Create conversation state
            conversation_state = ConversationState(
                conversation_id=conversation_id,
                template_name=template.name,
                status="pending",
                participants=agent_ids
            )
            
            self.logger.info(f"Created conversation from template: {template.name} -> {conversation_id}")
            self.metrics.record_counter(
                "conversation_created",
                framework="autogen",
                template=template.name,
                agent_count=str(len(agent_ids))
            )
            
            return conversation_state
            
        except Exception as e:
            raise AutoGenError(f"Failed to create conversation from template {template.name}: {e}")


class CommunicationProtocol:
    """Agent communication protocols for AutoGen."""
    
    def __init__(self):
        self.logger = get_framework_logger("AutoGen")
        self.metrics = get_metrics_collector()
        self.active_conversations: Dict[str, ConversationState] = {}
    
    async def initiate_two_agent_conversation(
        self,
        conversation_id: str,
        initiator_id: str,
        recipient_id: str,
        initial_message: str,
        max_turns: int = 10
    ) -> ConversationState:
        """Initiate a two-agent conversation."""
        try:
            framework = await get_autogen_framework()
            
            initiator = framework.get_agent(initiator_id)
            recipient = framework.get_agent(recipient_id)
            
            conversation_state = ConversationState(
                conversation_id=conversation_id,
                template_name="two_agent_direct",
                status="active",
                participants=[initiator_id, recipient_id],
                current_speaker=initiator_id,
                start_time=datetime.now()
            )
            
            self.active_conversations[conversation_id] = conversation_state
            
            # Start conversation with monitoring
            with self.metrics.timer("two_agent_conversation", conversation_id=conversation_id):
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: initiator.initiate_chat(
                        recipient,
                        message=initial_message,
                        max_turns=max_turns
                    )
                )
            
            conversation_state.status = "completed"
            conversation_state.end_time = datetime.now()
            conversation_state.result = {"chat_result": result}
            
            self.logger.info(f"Two-agent conversation completed: {conversation_id}")
            self.metrics.record_counter(
                "two_agent_conversation_completed",
                framework="autogen",
                success="true"
            )
            
            return conversation_state
            
        except Exception as e:
            conversation_state.status = "failed"
            conversation_state.error = str(e)
            conversation_state.end_time = datetime.now()
            
            self.logger.error(f"Two-agent conversation failed: {conversation_id} - {e}")
            self.metrics.record_counter(
                "two_agent_conversation_completed",
                framework="autogen",
                success="false"
            )
            
            raise AutoGenError(f"Two-agent conversation failed: {e}")
    
    async def start_group_chat(
        self,
        conversation_id: str,
        group_chat_id: str,
        initial_message: str,
        initiator_id: Optional[str] = None
    ) -> ConversationState:
        """Start a group chat conversation."""
        try:
            framework = await get_autogen_framework()
            
            group_chat = framework.group_chats.get(group_chat_id)
            if not group_chat:
                raise AutoGenError(f"Group chat not found: {group_chat_id}")
            
            manager = framework.group_chats.get(f"{group_chat_id}_manager")
            if not manager:
                raise AutoGenError(f"Group chat manager not found for: {group_chat_id}")
            
            # Determine initiator
            if initiator_id:
                initiator = framework.get_agent(initiator_id)
            else:
                initiator = group_chat.agents[0] if group_chat.agents else None
            
            if not initiator:
                raise AutoGenError("No initiator available for group chat")
            
            conversation_state = ConversationState(
                conversation_id=conversation_id,
                template_name="group_chat",
                status="active",
                participants=[agent.name for agent in group_chat.agents],
                current_speaker=initiator.name,
                start_time=datetime.now()
            )
            
            self.active_conversations[conversation_id] = conversation_state
            
            # Start group chat with monitoring
            with self.metrics.timer("group_chat_conversation", conversation_id=conversation_id):
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: initiator.initiate_chat(
                        manager,
                        message=initial_message
                    )
                )
            
            conversation_state.status = "completed"
            conversation_state.end_time = datetime.now()
            conversation_state.result = {"chat_result": result}
            conversation_state.current_round = len(group_chat.messages)
            
            self.logger.info(f"Group chat conversation completed: {conversation_id}")
            self.metrics.record_counter(
                "group_chat_conversation_completed", 
                framework="autogen",
                success="true",
                rounds=str(conversation_state.current_round)
            )
            
            return conversation_state
            
        except Exception as e:
            conversation_state.status = "failed"
            conversation_state.error = str(e)
            conversation_state.end_time = datetime.now()
            
            self.logger.error(f"Group chat conversation failed: {conversation_id} - {e}")
            self.metrics.record_counter(
                "group_chat_conversation_completed",
                framework="autogen", 
                success="false"
            )
            
            raise AutoGenError(f"Group chat conversation failed: {e}")
    
    def get_conversation_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Get conversation state by ID."""
        return self.active_conversations.get(conversation_id)
    
    def list_active_conversations(self) -> List[str]:
        """List active conversation IDs."""
        return [
            conv_id for conv_id, state in self.active_conversations.items()
            if state.is_active
        ]
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation summary."""
        state = self.get_conversation_state(conversation_id)
        if not state:
            return None
        
        return {
            "conversation_id": state.conversation_id,
            "template_name": state.template_name,
            "status": state.status,
            "participants": state.participants,
            "current_round": state.current_round,
            "duration": state.duration,
            "start_time": state.start_time.isoformat() if state.start_time else None,
            "end_time": state.end_time.isoformat() if state.end_time else None,
            "error": state.error
        }


# Global conversation pattern registry instance
_conversation_registry: Optional[ConversationPatternRegistry] = None


def get_conversation_registry() -> ConversationPatternRegistry:
    """Get the global conversation pattern registry instance."""
    global _conversation_registry
    if _conversation_registry is None:
        _conversation_registry = ConversationPatternRegistry()
    return _conversation_registry


def get_communication_protocol() -> CommunicationProtocol:
    """Get communication protocol instance."""
    return CommunicationProtocol()