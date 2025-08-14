"""LangGraph framework configuration and initialization."""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import os

from ...core.config import load_config
from ...core.logging import get_framework_logger
from ...core.errors import LangGraphError, ConfigurationError
from ...core.monitoring import get_metrics_collector


@dataclass
class LangGraphConfig:
    """Configuration for LangGraph framework."""
    
    checkpointer: str = "memory"
    interrupt_before: List[str] = field(default_factory=lambda: ["human_review"])
    interrupt_after: List[str] = field(default_factory=lambda: ["quality_gate"])
    max_iterations: int = 10
    max_workflow_steps: int = 50
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    
    # State Management
    state_storage: str = "memory"  # memory, redis, postgres
    state_ttl: int = 3600  # seconds
    
    # Execution Settings
    execution_timeout: int = 300
    step_timeout: int = 30
    parallel_execution: bool = True
    max_concurrent_nodes: int = 5
    
    # Monitoring
    enable_tracing: bool = True
    trace_sample_rate: float = 1.0
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LangGraphConfig':
        """Create LangGraph config from application config."""
        langgraph_config = config.get("frameworks", {}).get("langgraph", {})
        
        return cls(
            checkpointer=langgraph_config.get("checkpointer", "memory"),
            interrupt_before=langgraph_config.get("interrupt_before", ["human_review"]),
            interrupt_after=langgraph_config.get("interrupt_after", ["quality_gate"]),
            max_iterations=langgraph_config.get("max_iterations", 10),
            max_workflow_steps=langgraph_config.get("max_workflow_steps", 50),
            llm_provider=config.get("apis", {}).get("openai", {}).get("provider", "openai"),
            llm_model=config.get("apis", {}).get("openai", {}).get("model", "gpt-4"),
            llm_temperature=config.get("apis", {}).get("openai", {}).get("temperature", 0.7),
            llm_max_tokens=config.get("apis", {}).get("openai", {}).get("max_tokens", 4096),
            state_storage=langgraph_config.get("state_storage", "memory"),
            state_ttl=langgraph_config.get("state_ttl", 3600),
            execution_timeout=langgraph_config.get("execution_timeout", 300),
            step_timeout=langgraph_config.get("step_timeout", 30),
            parallel_execution=langgraph_config.get("parallel_execution", True),
            max_concurrent_nodes=langgraph_config.get("max_concurrent_nodes", 5),
            enable_tracing=langgraph_config.get("enable_tracing", True),
            trace_sample_rate=langgraph_config.get("trace_sample_rate", 1.0)
        )


class LangGraphFramework:
    """LangGraph framework integration and management."""
    
    def __init__(self, config: Optional[LangGraphConfig] = None):
        self.logger = get_framework_logger("LangGraph")
        self.metrics = get_metrics_collector()
        self.config = config or self._load_default_config()
        self.is_initialized = False
        
        # Framework components
        self._llm = None
        self._checkpointer = None
        self.graphs: Dict[str, Any] = {}
        self.compiled_graphs: Dict[str, Any] = {}
        self.active_executions: Dict[str, Any] = {}
    
    def _load_default_config(self) -> LangGraphConfig:
        """Load default configuration from application config."""
        try:
            app_config = load_config()
            return LangGraphConfig.from_config(app_config)
        except Exception as e:
            self.logger.error(f"Failed to load LangGraph config: {e}")
            return LangGraphConfig()  # Use defaults
    
    async def initialize(self):
        """Initialize the LangGraph framework."""
        if self.is_initialized:
            self.logger.warning("LangGraph framework already initialized")
            return
        
        try:
            self.logger.info("Initializing LangGraph framework...")
            
            # Initialize LLM
            await self._initialize_llm()
            
            # Initialize checkpointer
            await self._initialize_checkpointer()
            
            # Register health checks
            self._register_health_checks()
            
            self.is_initialized = True
            self.logger.info("LangGraph framework initialized successfully")
            self.metrics.record_counter("framework_initialized", framework="langgraph")
            
        except Exception as e:
            error_msg = f"Failed to initialize LangGraph framework: {e}"
            self.logger.error(error_msg)
            self.metrics.record_counter("framework_initialization_failed", framework="langgraph")
            raise LangGraphError(error_msg) from e
    
    async def _initialize_llm(self):
        """Initialize the LLM for LangGraph."""
        try:
            if self.config.llm_provider == "openai":
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
            raise LangGraphError(f"Failed to import LLM dependencies: {e}")
        except Exception as e:
            raise LangGraphError(f"Failed to initialize LLM: {e}")
    
    async def _initialize_checkpointer(self):
        """Initialize checkpointer for state persistence."""
        try:
            if self.config.checkpointer == "memory":
                from langgraph.checkpoint.memory import MemorySaver
                self._checkpointer = MemorySaver()
            elif self.config.checkpointer == "redis":
                # Redis checkpointer would be implemented here
                raise ConfigurationError("Redis checkpointer not yet implemented")
            elif self.config.checkpointer == "postgres":
                # Postgres checkpointer would be implemented here
                raise ConfigurationError("Postgres checkpointer not yet implemented")
            else:
                raise ConfigurationError(f"Unsupported checkpointer: {self.config.checkpointer}")
            
            self.logger.debug(f"Checkpointer initialized: {self.config.checkpointer}")
            
        except ImportError as e:
            raise LangGraphError(f"Failed to import checkpointer dependencies: {e}")
        except Exception as e:
            raise LangGraphError(f"Failed to initialize checkpointer: {e}")
    
    def _register_health_checks(self):
        """Register health checks for LangGraph framework."""
        from ...core.monitoring import get_health_monitor
        
        health_monitor = get_health_monitor()
        
        # Register framework import check
        health_monitor.register_check(
            "langgraph_import",
            self._check_langgraph_import,
            framework="langgraph",
            timeout=5.0,
            interval=60.0
        )
        
        # Register LLM connectivity check
        health_monitor.register_check(
            "langgraph_llm_health",
            self._check_llm_health,
            framework="langgraph",
            timeout=10.0,
            interval=30.0
        )
        
        # Register checkpointer health check
        health_monitor.register_check(
            "langgraph_checkpointer_health",
            self._check_checkpointer_health,
            framework="langgraph",
            timeout=5.0,
            interval=60.0
        )
    
    def _check_langgraph_import(self) -> bool:
        """Health check for LangGraph imports."""
        try:
            import langgraph
            from langgraph.graph import StateGraph
            return True
        except ImportError:
            return False
    
    async def _check_llm_health(self) -> bool:
        """Health check for LLM connectivity."""
        if not self._llm:
            return False
        
        try:
            response = await self._llm.apredict("Hello")
            return bool(response)
        except Exception:
            return False
    
    def _check_checkpointer_health(self) -> bool:
        """Health check for checkpointer."""
        return self._checkpointer is not None
    
    def get_llm(self):
        """Get the configured LLM instance."""
        if not self.is_initialized:
            raise LangGraphError("LangGraph framework not initialized")
        return self._llm
    
    def get_checkpointer(self):
        """Get the configured checkpointer instance."""
        if not self.is_initialized:
            raise LangGraphError("LangGraph framework not initialized")
        return self._checkpointer
    
    def register_graph(self, graph_id: str, graph):
        """Register a LangGraph graph."""
        if not self.is_initialized:
            raise LangGraphError("LangGraph framework not initialized")
        
        self.graphs[graph_id] = graph
        self.logger.info(f"Registered LangGraph graph: {graph_id}")
        self.metrics.record_counter("graph_registered", framework="langgraph")
    
    def compile_graph(self, graph_id: str, **compile_kwargs):
        """Compile a registered graph."""
        if graph_id not in self.graphs:
            raise LangGraphError(f"Graph not found: {graph_id}")
        
        try:
            graph = self.graphs[graph_id]
            
            # Add checkpointer to compile kwargs if not provided
            if "checkpointer" not in compile_kwargs and self._checkpointer:
                compile_kwargs["checkpointer"] = self._checkpointer
            
            # Add interrupt configuration
            if "interrupt_before" not in compile_kwargs:
                compile_kwargs["interrupt_before"] = self.config.interrupt_before
            
            if "interrupt_after" not in compile_kwargs:
                compile_kwargs["interrupt_after"] = self.config.interrupt_after
            
            compiled = graph.compile(**compile_kwargs)
            self.compiled_graphs[graph_id] = compiled
            
            self.logger.info(f"Compiled LangGraph graph: {graph_id}")
            self.metrics.record_counter("graph_compiled", framework="langgraph")
            
            return compiled
            
        except Exception as e:
            raise LangGraphError(f"Failed to compile graph {graph_id}: {e}")
    
    def get_compiled_graph(self, graph_id: str):
        """Get a compiled graph."""
        if graph_id not in self.compiled_graphs:
            # Try to compile if graph exists
            if graph_id in self.graphs:
                return self.compile_graph(graph_id)
            else:
                raise LangGraphError(f"Compiled graph not found: {graph_id}")
        
        return self.compiled_graphs[graph_id]
    
    async def execute_graph(
        self,
        graph_id: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ):
        """Execute a compiled graph."""
        compiled_graph = self.get_compiled_graph(graph_id)
        
        execution_id = f"{graph_id}_{id(inputs)}"
        
        try:
            self.active_executions[execution_id] = {
                "graph_id": graph_id,
                "status": "running",
                "started_at": self.metrics.start_workflow(execution_id, "langgraph")
            }
            
            self.logger.info(f"Executing LangGraph graph: {graph_id}")
            
            with self.metrics.timer("graph_execution", graph_id=graph_id):
                result = await compiled_graph.ainvoke(inputs, config=config)
            
            self.active_executions[execution_id]["status"] = "completed"
            self.metrics.complete_workflow(execution_id, "completed")
            
            self.logger.info(f"Graph {graph_id} executed successfully")
            self.metrics.record_counter("graph_executed", framework="langgraph", success="true")
            
            return result
            
        except Exception as e:
            self.active_executions[execution_id]["status"] = "failed"
            self.metrics.complete_workflow(execution_id, "failed")
            
            self.logger.error(f"Graph {graph_id} execution failed: {e}")
            self.metrics.record_counter("graph_executed", framework="langgraph", success="false")
            
            raise LangGraphError(f"Graph execution failed: {e}")
        finally:
            # Clean up execution tracking
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    def get_framework_config(self) -> Dict[str, Any]:
        """Get LangGraph framework configuration as dictionary."""
        return {
            "checkpointer": self.config.checkpointer,
            "interrupt_before": self.config.interrupt_before,
            "interrupt_after": self.config.interrupt_after,
            "max_iterations": self.config.max_iterations,
            "state_storage": self.config.state_storage,
            "execution_timeout": self.config.execution_timeout,
            "parallel_execution": self.config.parallel_execution,
            "llm": {
                "provider": self.config.llm_provider,
                "model": self.config.llm_model,
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens
            }
        }
    
    async def shutdown(self):
        """Shutdown the LangGraph framework."""
        if not self.is_initialized:
            return
        
        try:
            self.logger.info("Shutting down LangGraph framework...")
            
            # Cancel active executions
            for execution_id in list(self.active_executions.keys()):
                self.active_executions[execution_id]["status"] = "cancelled"
                del self.active_executions[execution_id]
            
            # Clear graphs
            self.graphs.clear()
            self.compiled_graphs.clear()
            
            # Clear components
            self._llm = None
            self._checkpointer = None
            
            self.is_initialized = False
            self.logger.info("LangGraph framework shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during LangGraph shutdown: {e}")


# Global LangGraph framework instance
_langgraph_framework: Optional[LangGraphFramework] = None


async def get_langgraph_framework(config: Optional[LangGraphConfig] = None) -> LangGraphFramework:
    """Get the global LangGraph framework instance."""
    global _langgraph_framework
    
    if _langgraph_framework is None:
        _langgraph_framework = LangGraphFramework(config)
        await _langgraph_framework.initialize()
    
    return _langgraph_framework


async def shutdown_langgraph_framework():
    """Shutdown the global LangGraph framework instance."""
    global _langgraph_framework
    
    if _langgraph_framework:
        await _langgraph_framework.shutdown()
        _langgraph_framework = None