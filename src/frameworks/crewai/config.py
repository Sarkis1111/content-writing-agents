"""CrewAI framework configuration and initialization."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ...core.config import load_config
from ...core.logging import get_framework_logger
from ...core.errors import CrewAIError, ConfigurationError
from ...core.monitoring import get_metrics_collector


@dataclass
class CrewAIConfig:
    """Configuration for CrewAI framework."""
    
    memory: bool = True
    verbose: bool = True
    max_execution_time: int = 300
    max_agents_per_crew: int = 10
    embedder_provider: str = "openai"
    embedder_model: str = "text-embedding-3-small"
    process_type: str = "hierarchical"
    allow_delegation: bool = True
    max_iter: int = 5
    max_rpm: int = 10
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    
    # Tool Configuration
    tool_timeout: int = 30
    tool_retries: int = 3
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CrewAIConfig':
        """Create CrewAI config from application config."""
        crewai_config = config.get("frameworks", {}).get("crewai", {})
        
        return cls(
            memory=crewai_config.get("memory", True),
            verbose=crewai_config.get("verbose", True),
            max_execution_time=crewai_config.get("max_execution_time", 300),
            max_agents_per_crew=crewai_config.get("max_agents_per_crew", 10),
            embedder_provider=crewai_config.get("embedder", {}).get("provider", "openai"),
            embedder_model=crewai_config.get("embedder", {}).get("model", "text-embedding-3-small"),
            process_type=crewai_config.get("process_type", "hierarchical"),
            allow_delegation=crewai_config.get("allow_delegation", True),
            max_iter=crewai_config.get("max_iter", 5),
            max_rpm=crewai_config.get("max_rpm", 10),
            llm_provider=config.get("apis", {}).get("openai", {}).get("provider", "openai"),
            llm_model=config.get("apis", {}).get("openai", {}).get("model", "gpt-4"),
            llm_temperature=config.get("apis", {}).get("openai", {}).get("temperature", 0.7),
            llm_max_tokens=config.get("apis", {}).get("openai", {}).get("max_tokens", 4096),
            tool_timeout=crewai_config.get("tool_timeout", 30),
            tool_retries=crewai_config.get("tool_retries", 3)
        )


class CrewAIFramework:
    """CrewAI framework integration and management."""
    
    def __init__(self, config: Optional[CrewAIConfig] = None):
        self.logger = get_framework_logger("CrewAI")
        self.metrics = get_metrics_collector()
        self.config = config or self._load_default_config()
        self.is_initialized = False
        
        # Framework components
        self._llm = None
        self._embedder = None
        self.crews: Dict[str, Any] = {}
        self.agents: Dict[str, Any] = {}
        self.tasks: Dict[str, Any] = {}
    
    def _load_default_config(self) -> CrewAIConfig:
        """Load default configuration from application config."""
        try:
            app_config = load_config()
            return CrewAIConfig.from_config(app_config)
        except Exception as e:
            self.logger.error(f"Failed to load CrewAI config: {e}")
            return CrewAIConfig()  # Use defaults
    
    async def initialize(self):
        """Initialize the CrewAI framework."""
        if self.is_initialized:
            self.logger.warning("CrewAI framework already initialized")
            return
        
        try:
            self.logger.info("Initializing CrewAI framework...")
            
            # Initialize LLM
            await self._initialize_llm()
            
            # Initialize embedder if memory is enabled
            if self.config.memory:
                await self._initialize_embedder()
            
            # Register health checks
            self._register_health_checks()
            
            self.is_initialized = True
            self.logger.info("CrewAI framework initialized successfully")
            self.metrics.record_counter("framework_initialized", framework="crewai")
            
        except Exception as e:
            error_msg = f"Failed to initialize CrewAI framework: {e}"
            self.logger.error(error_msg)
            self.metrics.record_counter("framework_initialization_failed", framework="crewai")
            raise CrewAIError(error_msg) from e
    
    async def _initialize_llm(self):
        """Initialize the LLM for CrewAI."""
        try:
            if self.config.llm_provider == "openai":
                # Import here to avoid dependency issues
                from langchain_openai import ChatOpenAI
                
                self._llm = ChatOpenAI(
                    model=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                    max_retries=3,
                    request_timeout=60
                )
            else:
                raise ConfigurationError(f"Unsupported LLM provider: {self.config.llm_provider}")
            
            self.logger.debug(f"LLM initialized: {self.config.llm_provider}/{self.config.llm_model}")
            
        except ImportError as e:
            raise CrewAIError(f"Failed to import LLM dependencies: {e}")
        except Exception as e:
            raise CrewAIError(f"Failed to initialize LLM: {e}")
    
    async def _initialize_embedder(self):
        """Initialize embedder for memory functionality."""
        try:
            if self.config.embedder_provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                
                self._embedder = OpenAIEmbeddings(
                    model=self.config.embedder_model
                )
            else:
                raise ConfigurationError(f"Unsupported embedder provider: {self.config.embedder_provider}")
            
            self.logger.debug(f"Embedder initialized: {self.config.embedder_provider}/{self.config.embedder_model}")
            
        except ImportError as e:
            raise CrewAIError(f"Failed to import embedder dependencies: {e}")
        except Exception as e:
            raise CrewAIError(f"Failed to initialize embedder: {e}")
    
    def _register_health_checks(self):
        """Register health checks for CrewAI framework."""
        from ...core.monitoring import get_health_monitor
        
        health_monitor = get_health_monitor()
        
        # Register framework import check
        health_monitor.register_check(
            "crewai_import",
            self._check_crewai_import,
            framework="crewai",
            timeout=5.0,
            interval=60.0
        )
        
        # Register LLM connectivity check
        health_monitor.register_check(
            "crewai_llm_health",
            self._check_llm_health,
            framework="crewai", 
            timeout=10.0,
            interval=30.0
        )
    
    def _check_crewai_import(self) -> bool:
        """Health check for CrewAI imports."""
        try:
            import crewai
            return True
        except ImportError:
            return False
    
    async def _check_llm_health(self) -> bool:
        """Health check for LLM connectivity."""
        if not self._llm:
            return False
        
        try:
            # Simple test query
            response = await self._llm.apredict("Hello")
            return bool(response)
        except Exception:
            return False
    
    def get_llm(self):
        """Get the configured LLM instance."""
        if not self.is_initialized:
            raise CrewAIError("CrewAI framework not initialized")
        return self._llm
    
    def get_embedder(self):
        """Get the configured embedder instance."""
        if not self.is_initialized:
            raise CrewAIError("CrewAI framework not initialized")
        return self._embedder
    
    def get_framework_config(self) -> Dict[str, Any]:
        """Get CrewAI framework configuration as dictionary."""
        return {
            "memory": self.config.memory,
            "verbose": self.config.verbose,
            "max_execution_time": self.config.max_execution_time,
            "embedder": {
                "provider": self.config.embedder_provider,
                "config": {"model": self.config.embedder_model}
            } if self.config.memory else None,
            "llm": {
                "provider": self.config.llm_provider,
                "model": self.config.llm_model,
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens
            }
        }
    
    async def shutdown(self):
        """Shutdown the CrewAI framework."""
        if not self.is_initialized:
            return
        
        try:
            self.logger.info("Shutting down CrewAI framework...")
            
            # Clear crews, agents, and tasks
            self.crews.clear()
            self.agents.clear()
            self.tasks.clear()
            
            # Clear LLM and embedder references
            self._llm = None
            self._embedder = None
            
            self.is_initialized = False
            self.logger.info("CrewAI framework shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during CrewAI shutdown: {e}")


# Global CrewAI framework instance
_crewai_framework: Optional[CrewAIFramework] = None


async def get_crewai_framework(config: Optional[CrewAIConfig] = None) -> CrewAIFramework:
    """Get the global CrewAI framework instance."""
    global _crewai_framework
    
    if _crewai_framework is None:
        _crewai_framework = CrewAIFramework(config)
        await _crewai_framework.initialize()
    
    return _crewai_framework


async def shutdown_crewai_framework():
    """Shutdown the global CrewAI framework instance."""
    global _crewai_framework
    
    if _crewai_framework:
        await _crewai_framework.shutdown()
        _crewai_framework = None