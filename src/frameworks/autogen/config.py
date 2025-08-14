"""AutoGen framework configuration and initialization."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ...core.config import load_config
from ...core.logging import get_framework_logger
from ...core.errors import AutoGenError, ConfigurationError
from ...core.monitoring import get_metrics_collector


@dataclass
class AutoGenConfig:
    """Configuration for AutoGen framework."""
    
    cache_seed: int = 42
    temperature: float = 0.7
    max_tokens: int = 2000
    max_consecutive_auto_reply: int = 10
    human_input_mode: str = "NEVER"
    max_round: int = 15
    speaker_selection_method: str = "round_robin"
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    
    # Conversation Settings
    conversation_timeout: int = 300
    message_max_length: int = 4000
    enable_code_execution: bool = False
    work_dir: str = "./autogen_workdir"
    
    # Group Chat Settings
    admin_name: str = "Admin"
    allow_repeat_speaker: bool = True
    speaker_selection_auto_multiple_template: str = "round_robin"
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AutoGenConfig':
        """Create AutoGen config from application config."""
        autogen_config = config.get("frameworks", {}).get("autogen", {})
        
        return cls(
            cache_seed=autogen_config.get("cache_seed", 42),
            temperature=autogen_config.get("temperature", 0.7),
            max_tokens=autogen_config.get("max_tokens", 2000),
            max_consecutive_auto_reply=autogen_config.get("max_consecutive_auto_reply", 10),
            human_input_mode=autogen_config.get("human_input_mode", "NEVER"),
            max_round=autogen_config.get("max_round", 15),
            speaker_selection_method=autogen_config.get("speaker_selection_method", "round_robin"),
            llm_provider=config.get("apis", {}).get("openai", {}).get("provider", "openai"),
            llm_model=config.get("apis", {}).get("openai", {}).get("model", "gpt-4"),
            llm_api_key=config.get("openai_api_key"),
            conversation_timeout=autogen_config.get("conversation_timeout", 300),
            message_max_length=autogen_config.get("message_max_length", 4000),
            enable_code_execution=autogen_config.get("enable_code_execution", False),
            work_dir=autogen_config.get("work_dir", "./autogen_workdir"),
            admin_name=autogen_config.get("admin_name", "Admin"),
            allow_repeat_speaker=autogen_config.get("allow_repeat_speaker", True)
        )


class AutoGenFramework:
    """AutoGen framework integration and management."""
    
    def __init__(self, config: Optional[AutoGenConfig] = None):
        self.logger = get_framework_logger("AutoGen")
        self.metrics = get_metrics_collector()
        self.config = config or self._load_default_config()
        self.is_initialized = False
        
        # Framework components
        self.llm_config = None
        self.agents: Dict[str, Any] = {}
        self.group_chats: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []
    
    def _load_default_config(self) -> AutoGenConfig:
        """Load default configuration from application config."""
        try:
            app_config = load_config()
            return AutoGenConfig.from_config(app_config)
        except Exception as e:
            self.logger.error(f"Failed to load AutoGen config: {e}")
            return AutoGenConfig()  # Use defaults
    
    async def initialize(self):
        """Initialize the AutoGen framework."""
        if self.is_initialized:
            self.logger.warning("AutoGen framework already initialized")
            return
        
        try:
            self.logger.info("Initializing AutoGen framework...")
            
            # Initialize LLM configuration
            await self._initialize_llm_config()
            
            # Create work directory
            import os
            os.makedirs(self.config.work_dir, exist_ok=True)
            
            # Register health checks
            self._register_health_checks()
            
            self.is_initialized = True
            self.logger.info("AutoGen framework initialized successfully")
            self.metrics.record_counter("framework_initialized", framework="autogen")
            
        except Exception as e:
            error_msg = f"Failed to initialize AutoGen framework: {e}"
            self.logger.error(error_msg)
            self.metrics.record_counter("framework_initialization_failed", framework="autogen")
            raise AutoGenError(error_msg) from e
    
    async def _initialize_llm_config(self):
        """Initialize LLM configuration for AutoGen."""
        try:
            if self.config.llm_provider == "openai":
                self.llm_config = {
                    "model": self.config.llm_model,
                    "api_key": self.config.llm_api_key,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "cache_seed": self.config.cache_seed
                }
                
                if self.config.llm_base_url:
                    self.llm_config["base_url"] = self.config.llm_base_url
                    
            else:
                raise ConfigurationError(f"Unsupported LLM provider: {self.config.llm_provider}")
            
            self.logger.debug(f"LLM config initialized: {self.config.llm_provider}/{self.config.llm_model}")
            
        except Exception as e:
            raise AutoGenError(f"Failed to initialize LLM config: {e}")
    
    def _register_health_checks(self):
        """Register health checks for AutoGen framework."""
        from ...core.monitoring import get_health_monitor
        
        health_monitor = get_health_monitor()
        
        # Register framework import check
        health_monitor.register_check(
            "autogen_import",
            self._check_autogen_import,
            framework="autogen",
            timeout=5.0,
            interval=60.0
        )
        
        # Register LLM configuration check
        health_monitor.register_check(
            "autogen_llm_config",
            self._check_llm_config,
            framework="autogen",
            timeout=5.0,
            interval=30.0
        )
    
    def _check_autogen_import(self) -> bool:
        """Health check for AutoGen imports."""
        try:
            import autogen
            return True
        except ImportError:
            return False
    
    def _check_llm_config(self) -> bool:
        """Health check for LLM configuration."""
        return self.llm_config is not None
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get the configured LLM config."""
        if not self.is_initialized:
            raise AutoGenError("AutoGen framework not initialized")
        return self.llm_config.copy() if self.llm_config else {}
    
    def create_conversable_agent(
        self,
        agent_id: str,
        name: str,
        system_message: str,
        description: Optional[str] = None,
        **kwargs
    ):
        """Create a conversable agent."""
        if not self.is_initialized:
            raise AutoGenError("AutoGen framework not initialized")
        
        try:
            import autogen
            
            agent_config = {
                "name": name,
                "system_message": system_message,
                "llm_config": self.get_llm_config(),
                "human_input_mode": self.config.human_input_mode,
                "max_consecutive_auto_reply": self.config.max_consecutive_auto_reply,
                "description": description or f"Conversable agent: {name}"
            }
            
            # Override with any provided kwargs
            agent_config.update(kwargs)
            
            agent = autogen.ConversableAgent(**agent_config)
            self.agents[agent_id] = agent
            
            self.logger.info(f"Created AutoGen conversable agent: {name}")
            self.metrics.record_counter("agent_created", framework="autogen", agent_type="conversable")
            
            return agent
            
        except ImportError as e:
            raise AutoGenError(f"AutoGen not installed: {e}")
        except Exception as e:
            raise AutoGenError(f"Failed to create conversable agent {name}: {e}")
    
    def create_assistant_agent(
        self,
        agent_id: str,
        name: str,
        system_message: str,
        description: Optional[str] = None,
        **kwargs
    ):
        """Create an assistant agent."""
        if not self.is_initialized:
            raise AutoGenError("AutoGen framework not initialized")
        
        try:
            import autogen
            
            agent_config = {
                "name": name,
                "system_message": system_message,
                "llm_config": self.get_llm_config(),
                "description": description or f"Assistant agent: {name}"
            }
            
            # Override with any provided kwargs
            agent_config.update(kwargs)
            
            agent = autogen.AssistantAgent(**agent_config)
            self.agents[agent_id] = agent
            
            self.logger.info(f"Created AutoGen assistant agent: {name}")
            self.metrics.record_counter("agent_created", framework="autogen", agent_type="assistant")
            
            return agent
            
        except ImportError as e:
            raise AutoGenError(f"AutoGen not installed: {e}")
        except Exception as e:
            raise AutoGenError(f"Failed to create assistant agent {name}: {e}")
    
    def create_user_proxy_agent(
        self,
        agent_id: str,
        name: str,
        system_message: str = "A human admin.",
        description: Optional[str] = None,
        **kwargs
    ):
        """Create a user proxy agent."""
        if not self.is_initialized:
            raise AutoGenError("AutoGen framework not initialized")
        
        try:
            import autogen
            
            agent_config = {
                "name": name,
                "system_message": system_message,
                "human_input_mode": self.config.human_input_mode,
                "max_consecutive_auto_reply": self.config.max_consecutive_auto_reply,
                "code_execution_config": {"work_dir": self.config.work_dir} if self.config.enable_code_execution else False,
                "description": description or f"User proxy agent: {name}"
            }
            
            # Override with any provided kwargs
            agent_config.update(kwargs)
            
            agent = autogen.UserProxyAgent(**agent_config)
            self.agents[agent_id] = agent
            
            self.logger.info(f"Created AutoGen user proxy agent: {name}")
            self.metrics.record_counter("agent_created", framework="autogen", agent_type="user_proxy")
            
            return agent
            
        except ImportError as e:
            raise AutoGenError(f"AutoGen not installed: {e}")
        except Exception as e:
            raise AutoGenError(f"Failed to create user proxy agent {name}: {e}")
    
    def get_agent(self, agent_id: str):
        """Get an agent by ID."""
        if agent_id not in self.agents:
            raise AutoGenError(f"Agent not found: {agent_id}")
        return self.agents[agent_id]
    
    def list_agents(self) -> List[str]:
        """List all agent IDs."""
        return list(self.agents.keys())
    
    def create_group_chat(
        self,
        chat_id: str,
        agents: List[str],
        max_round: Optional[int] = None,
        admin_name: Optional[str] = None,
        speaker_selection_method: Optional[str] = None,
        allow_repeat_speaker: Optional[bool] = None
    ):
        """Create a group chat."""
        if not self.is_initialized:
            raise AutoGenError("AutoGen framework not initialized")
        
        try:
            import autogen
            
            # Get agent instances
            agent_instances = []
            for agent_id in agents:
                if agent_id not in self.agents:
                    raise AutoGenError(f"Agent not found: {agent_id}")
                agent_instances.append(self.agents[agent_id])
            
            group_chat = autogen.GroupChat(
                agents=agent_instances,
                messages=[],
                max_round=max_round or self.config.max_round,
                admin_name=admin_name or self.config.admin_name,
                speaker_selection_method=speaker_selection_method or self.config.speaker_selection_method,
                allow_repeat_speaker=allow_repeat_speaker if allow_repeat_speaker is not None else self.config.allow_repeat_speaker
            )
            
            self.group_chats[chat_id] = group_chat
            
            self.logger.info(f"Created AutoGen group chat: {chat_id} with {len(agents)} agents")
            self.metrics.record_counter(
                "group_chat_created",
                framework="autogen",
                agent_count=str(len(agents))
            )
            
            return group_chat
            
        except ImportError as e:
            raise AutoGenError(f"AutoGen not installed: {e}")
        except Exception as e:
            raise AutoGenError(f"Failed to create group chat {chat_id}: {e}")
    
    def create_group_chat_manager(
        self,
        manager_id: str,
        group_chat_id: str,
        name: str = "GroupChatManager",
        system_message: str = "Group chat manager.",
        **kwargs
    ):
        """Create a group chat manager."""
        if not self.is_initialized:
            raise AutoGenError("AutoGen framework not initialized")
        
        if group_chat_id not in self.group_chats:
            raise AutoGenError(f"Group chat not found: {group_chat_id}")
        
        try:
            import autogen
            
            group_chat = self.group_chats[group_chat_id]
            
            manager_config = {
                "name": name,
                "system_message": system_message,
                "groupchat": group_chat,
                "llm_config": self.get_llm_config()
            }
            
            # Override with any provided kwargs
            manager_config.update(kwargs)
            
            manager = autogen.GroupChatManager(**manager_config)
            
            # Store manager with the group chat
            self.group_chats[f"{group_chat_id}_manager"] = manager
            
            self.logger.info(f"Created AutoGen group chat manager: {name} for chat {group_chat_id}")
            self.metrics.record_counter("group_chat_manager_created", framework="autogen")
            
            return manager
            
        except ImportError as e:
            raise AutoGenError(f"AutoGen not installed: {e}")
        except Exception as e:
            raise AutoGenError(f"Failed to create group chat manager for {group_chat_id}: {e}")
    
    def get_framework_config(self) -> Dict[str, Any]:
        """Get AutoGen framework configuration as dictionary."""
        return {
            "cache_seed": self.config.cache_seed,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "max_round": self.config.max_round,
            "speaker_selection_method": self.config.speaker_selection_method,
            "human_input_mode": self.config.human_input_mode,
            "enable_code_execution": self.config.enable_code_execution,
            "llm": {
                "provider": self.config.llm_provider,
                "model": self.config.llm_model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
        }
    
    async def shutdown(self):
        """Shutdown the AutoGen framework."""
        if not self.is_initialized:
            return
        
        try:
            self.logger.info("Shutting down AutoGen framework...")
            
            # Clear agents and group chats
            self.agents.clear()
            self.group_chats.clear()
            self.conversation_history.clear()
            
            # Clear LLM config
            self.llm_config = None
            
            self.is_initialized = False
            self.logger.info("AutoGen framework shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during AutoGen shutdown: {e}")


# Global AutoGen framework instance
_autogen_framework: Optional[AutoGenFramework] = None


async def get_autogen_framework(config: Optional[AutoGenConfig] = None) -> AutoGenFramework:
    """Get the global AutoGen framework instance."""
    global _autogen_framework
    
    if _autogen_framework is None:
        _autogen_framework = AutoGenFramework(config)
        await _autogen_framework.initialize()
    
    return _autogen_framework


async def shutdown_autogen_framework():
    """Shutdown the global AutoGen framework instance."""
    global _autogen_framework
    
    if _autogen_framework:
        await _autogen_framework.shutdown()
        _autogen_framework = None