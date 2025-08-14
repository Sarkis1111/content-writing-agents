"""AutoGen group chat coordination and management."""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import json

from ...core.logging import get_framework_logger
from ...core.errors import AutoGenError, WorkflowError
from ...core.monitoring import get_metrics_collector
from .config import get_autogen_framework
from .conversations import ConversationState, get_communication_protocol


class CoordinationStrategy(Enum):
    """Group chat coordination strategies."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    MANUAL = "manual"
    AUTO_MULTIPLE = "auto_multiple"
    HIERARCHICAL = "hierarchical"


class GroupChatRole(Enum):
    """Roles in group chat coordination."""
    FACILITATOR = "facilitator"
    PARTICIPANT = "participant"
    OBSERVER = "observer"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"


@dataclass
class GroupChatSession:
    """Group chat session management."""
    
    session_id: str
    name: str
    description: str
    agents: List[str]
    coordination_strategy: CoordinationStrategy
    max_rounds: int = 10
    current_round: int = 0
    status: str = "pending"  # pending, active, paused, completed, failed
    messages: List[Dict[str, Any]] = field(default_factory=list)
    facilitator_id: Optional[str] = None
    session_context: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == "active"
    
    @property
    def completion_rate(self) -> float:
        """Get session completion rate."""
        return self.current_round / self.max_rounds if self.max_rounds > 0 else 0.0


@dataclass
class CoordinationRule:
    """Rules for group chat coordination."""
    
    name: str
    condition: str  # Python expression to evaluate
    action: str  # Action to take when condition is met
    priority: int = 1  # Higher priority rules are evaluated first
    enabled: bool = True
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the rule condition."""
        try:
            return eval(self.condition, {"__builtins__": {}}, context)
        except Exception:
            return False


class GroupChatCoordinator:
    """Advanced group chat coordination and management."""
    
    def __init__(self):
        self.logger = get_framework_logger("AutoGen")
        self.metrics = get_metrics_collector()
        self.communication_protocol = get_communication_protocol()
        self.active_sessions: Dict[str, GroupChatSession] = {}
        self.coordination_rules: List[CoordinationRule] = []
        self.session_templates: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_rules()
        self._initialize_session_templates()
    
    def _initialize_default_rules(self):
        """Initialize default coordination rules."""
        
        # Prevent domination by single speaker
        self.add_coordination_rule(CoordinationRule(
            name="prevent_speaker_domination",
            condition="last_speaker_count >= 3",
            action="select_different_speaker",
            priority=5
        ))
        
        # Ensure all participants contribute
        self.add_coordination_rule(CoordinationRule(
            name="encourage_participation",
            condition="silent_participants_count > 0 and current_round > 3",
            action="prompt_silent_participants",
            priority=3
        ))
        
        # Quality gate for decisions
        self.add_coordination_rule(CoordinationRule(
            name="decision_quality_gate", 
            condition="decision_proposed and consensus_score < 0.7",
            action="request_clarification",
            priority=4
        ))
        
        # Time management
        self.add_coordination_rule(CoordinationRule(
            name="time_management",
            condition="session_duration > max_duration * 0.8",
            action="wrap_up_discussion",
            priority=2
        ))
        
        # Conflict resolution
        self.add_coordination_rule(CoordinationRule(
            name="conflict_resolution",
            condition="conflict_detected",
            action="mediate_conflict",
            priority=5
        ))
    
    def _initialize_session_templates(self):
        """Initialize session templates."""
        
        # Strategy Development Session
        self.session_templates["strategy_development"] = {
            "name": "Strategy Development Session",
            "description": "Collaborative strategy development with multiple experts",
            "coordination_strategy": CoordinationStrategy.ROUND_ROBIN,
            "max_rounds": 15,
            "roles": {
                "facilitator": "content_strategist",
                "participants": ["audience_analyst", "competitive_analyst", "performance_optimizer"]
            },
            "phases": [
                {"name": "problem_definition", "rounds": 3},
                {"name": "idea_generation", "rounds": 5},
                {"name": "evaluation", "rounds": 4},
                {"name": "consensus_building", "rounds": 3}
            ]
        }
        
        # Research Synthesis Session
        self.session_templates["research_synthesis"] = {
            "name": "Research Synthesis Session",
            "description": "Collaborative research synthesis and analysis",
            "coordination_strategy": CoordinationStrategy.HIERARCHICAL,
            "max_rounds": 12,
            "roles": {
                "facilitator": "primary_researcher",
                "participants": ["fact_checker", "trend_analyst"]
            },
            "phases": [
                {"name": "data_presentation", "rounds": 4},
                {"name": "analysis", "rounds": 5},
                {"name": "synthesis", "rounds": 3}
            ]
        }
        
        # Creative Brainstorming Session
        self.session_templates["creative_brainstorming"] = {
            "name": "Creative Brainstorming Session",
            "description": "Open creative brainstorming with idea development",
            "coordination_strategy": CoordinationStrategy.AUTO_MULTIPLE,
            "max_rounds": 10,
            "roles": {
                "participants": ["creative_thinker", "practical_evaluator", "innovation_synthesizer"]
            },
            "phases": [
                {"name": "ideation", "rounds": 6},
                {"name": "refinement", "rounds": 4}
            ]
        }
        
        # Decision Making Session
        self.session_templates["decision_making"] = {
            "name": "Decision Making Session",
            "description": "Structured decision making with expert analysis",
            "coordination_strategy": CoordinationStrategy.MANUAL,
            "max_rounds": 8,
            "roles": {
                "facilitator": "decision_analyst",
                "participants": ["implementation_expert"]
            },
            "phases": [
                {"name": "option_analysis", "rounds": 3},
                {"name": "feasibility_assessment", "rounds": 3},
                {"name": "decision", "rounds": 2}
            ]
        }
    
    def add_coordination_rule(self, rule: CoordinationRule):
        """Add a coordination rule."""
        self.coordination_rules.append(rule)
        self.coordination_rules.sort(key=lambda r: r.priority, reverse=True)
        self.logger.info(f"Added coordination rule: {rule.name}")
    
    async def create_session_from_template(
        self,
        session_id: str,
        template_name: str,
        agent_assignments: Dict[str, str],
        context: Optional[Dict[str, Any]] = None
    ) -> GroupChatSession:
        """Create a group chat session from template."""
        
        if template_name not in self.session_templates:
            raise AutoGenError(f"Session template not found: {template_name}")
        
        template = self.session_templates[template_name]
        context = context or {}
        
        try:
            framework = await get_autogen_framework()
            
            # Validate agent assignments
            required_roles = set()
            if "facilitator" in template["roles"]:
                required_roles.add(template["roles"]["facilitator"])
            if "participants" in template["roles"]:
                required_roles.update(template["roles"]["participants"])
            
            assigned_roles = set(agent_assignments.keys())
            missing_roles = required_roles - assigned_roles
            if missing_roles:
                raise AutoGenError(f"Missing agent assignments for roles: {missing_roles}")
            
            # Create session
            session = GroupChatSession(
                session_id=session_id,
                name=template["name"],
                description=template["description"],
                agents=list(agent_assignments.values()),
                coordination_strategy=CoordinationStrategy(template["coordination_strategy"]),
                max_rounds=template["max_rounds"],
                facilitator_id=agent_assignments.get(template["roles"].get("facilitator")),
                session_context=context
            )
            
            # Create AutoGen group chat
            group_chat_id = f"{session_id}_group"
            framework.create_group_chat(
                chat_id=group_chat_id,
                agents=session.agents,
                max_round=session.max_rounds,
                speaker_selection_method=session.coordination_strategy.value
            )
            
            # Create group chat manager
            framework.create_group_chat_manager(
                manager_id=f"{session_id}_manager",
                group_chat_id=group_chat_id,
                name=f"{session.name} Manager"
            )
            
            self.active_sessions[session_id] = session
            
            self.logger.info(f"Created group chat session: {session.name} -> {session_id}")
            self.metrics.record_counter(
                "group_chat_session_created",
                framework="autogen",
                template=template_name,
                agents=str(len(session.agents))
            )
            
            return session
            
        except Exception as e:
            raise AutoGenError(f"Failed to create session from template {template_name}: {e}")
    
    async def start_session(
        self,
        session_id: str,
        initial_message: str,
        facilitator_override: Optional[str] = None
    ) -> GroupChatSession:
        """Start a group chat session."""
        
        if session_id not in self.active_sessions:
            raise AutoGenError(f"Session not found: {session_id}")
        
        session = self.active_sessions[session_id]
        
        try:
            framework = await get_autogen_framework()
            
            # Determine facilitator
            facilitator_id = facilitator_override or session.facilitator_id
            if not facilitator_id:
                facilitator_id = session.agents[0] if session.agents else None
            
            if not facilitator_id:
                raise AutoGenError("No facilitator available for session")
            
            facilitator = framework.get_agent(facilitator_id)
            manager = framework.group_chats.get(f"{session_id}_group_manager")
            
            if not manager:
                raise AutoGenError(f"Group chat manager not found for session: {session_id}")
            
            # Start session
            session.status = "active"
            session.start_time = datetime.now()
            
            self.logger.info(f"Starting group chat session: {session_id}")
            
            # Execute with coordination and monitoring
            with self.metrics.timer("group_chat_session", session_id=session_id):
                await self._execute_coordinated_session(session, facilitator, manager, initial_message)
            
            session.status = "completed"
            session.end_time = datetime.now()
            
            self.logger.info(f"Group chat session completed: {session_id}")
            self.metrics.record_counter(
                "group_chat_session_completed",
                framework="autogen",
                success="true",
                rounds=str(session.current_round)
            )
            
            return session
            
        except Exception as e:
            session.status = "failed"
            session.end_time = datetime.now()
            
            self.logger.error(f"Group chat session failed: {session_id} - {e}")
            self.metrics.record_counter(
                "group_chat_session_completed",
                framework="autogen", 
                success="false"
            )
            
            raise AutoGenError(f"Group chat session failed: {e}")
    
    async def _execute_coordinated_session(
        self,
        session: GroupChatSession,
        facilitator,
        manager,
        initial_message: str
    ):
        """Execute session with coordination rules."""
        
        # Create context for rule evaluation
        context = {
            "session": session,
            "current_round": session.current_round,
            "max_rounds": session.max_rounds,
            "message_count": len(session.messages),
            "agents_count": len(session.agents)
        }
        
        # Apply pre-session coordination rules
        self._apply_coordination_rules(context, "pre_session")
        
        # Execute the actual conversation
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: facilitator.initiate_chat(
                manager,
                message=initial_message
            )
        )
        
        session.result = {"chat_result": result}
        
        # Update session with final state
        group_chat = manager.groupchat
        session.current_round = len(group_chat.messages)
        session.messages = [
            {"speaker": msg.get("name", "unknown"), "content": msg.get("content", "")}
            for msg in group_chat.messages
        ]
        
        # Apply post-session coordination rules
        context.update({
            "current_round": session.current_round,
            "message_count": len(session.messages),
            "result": result
        })
        self._apply_coordination_rules(context, "post_session")
    
    def _apply_coordination_rules(self, context: Dict[str, Any], phase: str):
        """Apply coordination rules for given phase."""
        
        for rule in self.coordination_rules:
            if not rule.enabled:
                continue
            
            try:
                if rule.evaluate(context):
                    self._execute_coordination_action(rule.action, context)
                    self.logger.debug(f"Applied coordination rule: {rule.name} in phase {phase}")
            except Exception as e:
                self.logger.warning(f"Failed to apply coordination rule {rule.name}: {e}")
    
    def _execute_coordination_action(self, action: str, context: Dict[str, Any]):
        """Execute a coordination action."""
        
        if action == "select_different_speaker":
            self._handle_speaker_selection(context)
        elif action == "prompt_silent_participants":
            self._handle_silent_participants(context)
        elif action == "request_clarification":
            self._handle_clarification_request(context)
        elif action == "wrap_up_discussion":
            self._handle_wrap_up(context)
        elif action == "mediate_conflict":
            self._handle_conflict_mediation(context)
        else:
            self.logger.warning(f"Unknown coordination action: {action}")
    
    def _handle_speaker_selection(self, context: Dict[str, Any]):
        """Handle speaker selection coordination."""
        # Implementation would depend on specific AutoGen APIs
        self.logger.info("Coordinating speaker selection to prevent domination")
    
    def _handle_silent_participants(self, context: Dict[str, Any]):
        """Handle silent participant encouragement."""
        self.logger.info("Encouraging silent participants to contribute")
    
    def _handle_clarification_request(self, context: Dict[str, Any]):
        """Handle clarification requests."""
        self.logger.info("Requesting clarification for better consensus")
    
    def _handle_wrap_up(self, context: Dict[str, Any]):
        """Handle session wrap-up."""
        self.logger.info("Coordinating session wrap-up due to time constraints")
    
    def _handle_conflict_mediation(self, context: Dict[str, Any]):
        """Handle conflict mediation."""
        self.logger.info("Mediating detected conflict between participants")
    
    def pause_session(self, session_id: str) -> GroupChatSession:
        """Pause an active session."""
        if session_id not in self.active_sessions:
            raise AutoGenError(f"Session not found: {session_id}")
        
        session = self.active_sessions[session_id]
        if session.status == "active":
            session.status = "paused"
            self.logger.info(f"Paused session: {session_id}")
        
        return session
    
    def resume_session(self, session_id: str) -> GroupChatSession:
        """Resume a paused session."""
        if session_id not in self.active_sessions:
            raise AutoGenError(f"Session not found: {session_id}")
        
        session = self.active_sessions[session_id]
        if session.status == "paused":
            session.status = "active"
            self.logger.info(f"Resumed session: {session_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[GroupChatSession]:
        """Get session by ID."""
        return self.active_sessions.get(session_id)
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session status and metrics."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "name": session.name,
            "status": session.status,
            "current_round": session.current_round,
            "max_rounds": session.max_rounds,
            "completion_rate": session.completion_rate,
            "participants": session.agents,
            "message_count": len(session.messages),
            "duration": session.duration,
            "start_time": session.start_time.isoformat() if session.start_time else None,
            "end_time": session.end_time.isoformat() if session.end_time else None
        }
    
    def list_active_sessions(self) -> List[str]:
        """List active session IDs."""
        return [
            session_id for session_id, session in self.active_sessions.items()
            if session.is_active
        ]
    
    def get_session_templates(self) -> List[str]:
        """Get available session template names."""
        return list(self.session_templates.keys())


# Global group chat coordinator instance
_chat_coordinator: Optional[GroupChatCoordinator] = None


def get_group_chat_coordinator() -> GroupChatCoordinator:
    """Get the global group chat coordinator instance."""
    global _chat_coordinator
    if _chat_coordinator is None:
        _chat_coordinator = GroupChatCoordinator()
    return _chat_coordinator