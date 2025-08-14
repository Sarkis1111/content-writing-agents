"""
Tool Registration System Compatible with Frameworks

This module provides a unified tool registration and execution system that works
seamlessly across CrewAI, LangGraph, and AutoGen frameworks. It implements the
MCP protocol for tool discovery, registration, and execution coordination.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Protocol, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import inspect
import uuid
import json
import logging
from datetime import datetime

from pydantic import BaseModel, ValidationError

from ..core.config.loader import ConfigLoader  
from ..core.monitoring.metrics import MetricsCollector
from ..core.errors.exceptions import ToolError, ValidationError as ToolValidationError
from ..core.errors.handlers import handle_async_errors
from .server import FrameworkType


class ToolType(str, Enum):
    """Types of tools available in the system"""
    RESEARCH = "research"
    ANALYSIS = "analysis" 
    WRITING = "writing"
    EDITING = "editing"
    COORDINATION = "coordination"
    UTILITY = "utility"


class ToolCompatibility(str, Enum):
    """Framework compatibility levels"""
    ALL_FRAMEWORKS = "all"
    CREWAI_ONLY = "crewai_only"
    LANGGRAPH_ONLY = "langgraph_only"
    AUTOGEN_ONLY = "autogen_only"
    CREWAI_LANGGRAPH = "crewai_langgraph"
    CREWAI_AUTOGEN = "crewai_autogen"
    LANGGRAPH_AUTOGEN = "langgraph_autogen"


@dataclass
class ToolParameter:
    """Tool parameter definition"""
    name: str
    param_type: str
    description: str
    required: bool = True
    default: Any = None
    validation_rules: Optional[Dict[str, Any]] = None
    examples: Optional[List[str]] = None


@dataclass  
class ToolDefinition:
    """Complete tool definition with metadata"""
    name: str
    description: str
    tool_type: ToolType
    parameters: List[ToolParameter]
    compatibility: ToolCompatibility = ToolCompatibility.ALL_FRAMEWORKS
    version: str = "1.0.0"
    mcp_compatible: bool = True
    async_execution: bool = True
    timeout: Optional[int] = None
    rate_limit: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for tool parameters"""
        properties = {}
        required = []
        
        for param in self.parameters:
            param_schema = {
                "type": param.param_type,
                "description": param.description
            }
            
            if param.default is not None:
                param_schema["default"] = param.default
                
            if param.examples:
                param_schema["examples"] = param.examples
                
            if param.validation_rules:
                param_schema.update(param.validation_rules)
                
            properties[param.name] = param_schema
            
            if param.required:
                required.append(param.name)
                
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
        
    def is_compatible_with_framework(self, framework: FrameworkType) -> bool:
        """Check if tool is compatible with a specific framework"""
        if self.compatibility == ToolCompatibility.ALL_FRAMEWORKS:
            return True
            
        compatibility_map = {
            ToolCompatibility.CREWAI_ONLY: [FrameworkType.CREWAI],
            ToolCompatibility.LANGGRAPH_ONLY: [FrameworkType.LANGGRAPH], 
            ToolCompatibility.AUTOGEN_ONLY: [FrameworkType.AUTOGEN],
            ToolCompatibility.CREWAI_LANGGRAPH: [FrameworkType.CREWAI, FrameworkType.LANGGRAPH],
            ToolCompatibility.CREWAI_AUTOGEN: [FrameworkType.CREWAI, FrameworkType.AUTOGEN],
            ToolCompatibility.LANGGRAPH_AUTOGEN: [FrameworkType.LANGGRAPH, FrameworkType.AUTOGEN]
        }
        
        return framework in compatibility_map.get(self.compatibility, [])


class BaseTool(ABC):
    """Abstract base class for all tools"""
    
    def __init__(self, definition: ToolDefinition, logger: logging.Logger, metrics: MetricsCollector):
        self.definition = definition
        self.logger = logger
        self.metrics = metrics
        self.execution_count = 0
        self.last_executed = None
        
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
        
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters against schema"""
        schema = self.definition.get_parameter_schema()
        
        # Basic validation - can be extended with jsonschema library
        validated = {}
        
        # Check required parameters
        for required_param in schema.get("required", []):
            if required_param not in parameters:
                raise ToolValidationError(f"Required parameter '{required_param}' missing")
                
        # Process parameters
        for param_name, param_schema in schema.get("properties", {}).items():
            if param_name in parameters:
                validated[param_name] = parameters[param_name]
            elif "default" in param_schema:
                validated[param_name] = param_schema["default"]
                
        return validated
        
    async def pre_execute_hook(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Hook called before tool execution"""
        return parameters
        
    async def post_execute_hook(self, result: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called after tool execution"""
        return result
        
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics"""
        return {
            "execution_count": self.execution_count,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "name": self.definition.name,
            "version": self.definition.version
        }


class ToolRegistry:
    """Central registry for all tools in the system"""
    
    def __init__(self, config: ConfigLoader, logger: logging.Logger, metrics: MetricsCollector):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.tools: Dict[str, BaseTool] = {}
        self.tool_definitions: Dict[str, ToolDefinition] = {}
        self.framework_tool_mapping: Dict[FrameworkType, List[str]] = {
            framework: [] for framework in FrameworkType
        }
        
    def register_tool(self, tool: BaseTool):
        """Register a tool in the system"""
        tool_name = tool.definition.name
        
        if tool_name in self.tools:
            self.logger.warning(f"Tool {tool_name} is already registered. Overwriting.")
            
        self.tools[tool_name] = tool
        self.tool_definitions[tool_name] = tool.definition
        
        # Update framework mappings
        for framework in FrameworkType:
            if tool.definition.is_compatible_with_framework(framework):
                if tool_name not in self.framework_tool_mapping[framework]:
                    self.framework_tool_mapping[framework].append(tool_name)
                    
        self.logger.info(f"Registered tool: {tool_name}")
        self.metrics.increment_counter(f"tool_registry.registered.{tool.definition.tool_type.value}")
        
    def unregister_tool(self, tool_name: str):
        """Unregister a tool from the system"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            del self.tool_definitions[tool_name]
            
            # Remove from framework mappings
            for framework_tools in self.framework_tool_mapping.values():
                if tool_name in framework_tools:
                    framework_tools.remove(tool_name)
                    
            self.logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False
        
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a registered tool by name"""
        return self.tools.get(tool_name)
        
    def list_tools(self, framework: Optional[FrameworkType] = None, 
                  tool_type: Optional[ToolType] = None) -> List[ToolDefinition]:
        """List available tools with optional filtering"""
        tools = list(self.tool_definitions.values())
        
        if framework:
            tools = [tool for tool in tools if tool.is_compatible_with_framework(framework)]
            
        if tool_type:
            tools = [tool for tool in tools if tool.tool_type == tool_type]
            
        return tools
        
    def get_framework_tools(self, framework: FrameworkType) -> List[str]:
        """Get list of tool names compatible with a framework"""
        return self.framework_tool_mapping[framework].copy()
        
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a tool with parameters"""
        if tool_name not in self.tools:
            raise ToolError(f"Tool '{tool_name}' not found", tool_name=tool_name)
            
        tool = self.tools[tool_name]
        
        try:
            # Validate parameters
            validated_params = await tool.validate_parameters(parameters)
            
            # Pre-execute hook
            processed_params = await tool.pre_execute_hook(validated_params, context)
            
            # Record execution metrics
            with self.metrics.timer(f"tool_execution.{tool_name}.duration"):
                # Execute tool
                result = await tool.execute(processed_params, context)
                
                # Post-execute hook
                final_result = await tool.post_execute_hook(result, processed_params)
                
                # Update tool stats
                tool.execution_count += 1
                tool.last_executed = datetime.utcnow()
                
                self.metrics.increment_counter(f"tool_execution.{tool_name}.success")
                return final_result
                
        except Exception as e:
            self.metrics.increment_counter(f"tool_execution.{tool_name}.error")
            self.logger.error(f"Tool execution failed for {tool_name}: {e}")
            raise ToolError(f"Tool execution failed: {e}", tool_name=tool_name)


class FrameworkToolAdapter:
    """Adapts tools for use with specific frameworks"""
    
    def __init__(self, registry: ToolRegistry, logger: logging.Logger):
        self.registry = registry
        self.logger = logger
        self.adapters: Dict[FrameworkType, Callable] = {}
        
    def register_adapter(self, framework: FrameworkType, adapter_func: Callable):
        """Register a tool adapter for a specific framework"""
        self.adapters[framework] = adapter_func
        self.logger.info(f"Registered tool adapter for {framework.value}")
        
    async def adapt_tool_for_framework(self, tool_name: str, framework: FrameworkType) -> Any:
        """Adapt a tool for use with a specific framework"""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            raise ToolError(f"Tool '{tool_name}' not found", tool_name=tool_name)
            
        if not tool.definition.is_compatible_with_framework(framework):
            raise ToolError(f"Tool '{tool_name}' not compatible with {framework.value}", 
                          tool_name=tool_name, framework=framework.value)
            
        if framework in self.adapters:
            adapter = self.adapters[framework]
            return await adapter(tool)
        else:
            # Return generic wrapper
            return self._create_generic_wrapper(tool, framework)
            
    def _create_generic_wrapper(self, tool: BaseTool, framework: FrameworkType) -> Dict[str, Any]:
        """Create a generic framework-agnostic tool wrapper"""
        return {
            "name": tool.definition.name,
            "description": tool.definition.description,
            "parameters": tool.definition.get_parameter_schema(),
            "execute": lambda params, context=None: self.registry.execute_tool(
                tool.definition.name, params, context
            )
        }


class ToolExecutor:
    """Handles tool execution with retry logic and error handling"""
    
    def __init__(self, registry: ToolRegistry, logger: logging.Logger, metrics: MetricsCollector):
        self.registry = registry
        self.logger = logger
        self.metrics = metrics
        self.execution_queue = asyncio.Queue()
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
    async def execute_tool_async(self, tool_name: str, parameters: Dict[str, Any], 
                                context: Optional[Dict[str, Any]] = None,
                                execution_id: Optional[str] = None) -> str:
        """Execute tool asynchronously and return execution ID"""
        if execution_id is None:
            execution_id = str(uuid.uuid4())
            
        execution_info = {
            "id": execution_id,
            "tool_name": tool_name,
            "parameters": parameters,
            "context": context,
            "status": "queued",
            "created_at": datetime.utcnow(),
            "result": None,
            "error": None
        }
        
        self.active_executions[execution_id] = execution_info
        await self.execution_queue.put(execution_info)
        
        # Start background execution
        asyncio.create_task(self._process_execution_queue())
        
        return execution_id
        
    async def _process_execution_queue(self):
        """Process queued tool executions"""
        while not self.execution_queue.empty():
            try:
                execution_info = await self.execution_queue.get()
                execution_id = execution_info["id"]
                
                if execution_id in self.active_executions:
                    self.active_executions[execution_id]["status"] = "running"
                    
                    try:
                        result = await self.registry.execute_tool(
                            execution_info["tool_name"],
                            execution_info["parameters"], 
                            execution_info["context"]
                        )
                        
                        self.active_executions[execution_id].update({
                            "status": "completed",
                            "result": result,
                            "completed_at": datetime.utcnow()
                        })
                        
                    except Exception as e:
                        self.active_executions[execution_id].update({
                            "status": "failed",
                            "error": str(e),
                            "completed_at": datetime.utcnow()
                        })
                        
            except Exception as e:
                self.logger.error(f"Error processing execution queue: {e}")
                
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a tool execution"""
        return self.active_executions.get(execution_id)
        
    def cleanup_completed_executions(self, max_age_hours: int = 24):
        """Clean up old completed executions"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for execution_id, info in self.active_executions.items():
            if (info.get("completed_at") and 
                info["completed_at"] < cutoff_time and 
                info["status"] in ["completed", "failed"]):
                to_remove.append(execution_id)
                
        for execution_id in to_remove:
            del self.active_executions[execution_id]
            
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old tool executions")


# Example tool implementations

class WebSearchTool(BaseTool):
    """Example web search tool implementation"""
    
    def __init__(self, logger: logging.Logger, metrics: MetricsCollector):
        definition = ToolDefinition(
            name="web_search",
            description="Search the web for relevant information",
            tool_type=ToolType.RESEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    param_type="string", 
                    description="Search query string",
                    required=True,
                    examples=["machine learning trends 2024", "content writing best practices"]
                ),
                ToolParameter(
                    name="max_results",
                    param_type="integer",
                    description="Maximum number of results to return",
                    required=False,
                    default=10,
                    validation_rules={"minimum": 1, "maximum": 100}
                )
            ],
            compatibility=ToolCompatibility.ALL_FRAMEWORKS,
            async_execution=True,
            timeout=30,
            rate_limit=100  # requests per minute
        )
        super().__init__(definition, logger, metrics)
        
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute web search"""
        query = parameters["query"]
        max_results = parameters.get("max_results", 10)
        
        self.logger.info(f"Executing web search for query: {query}")
        
        # Simulate web search - replace with actual implementation
        await asyncio.sleep(1)  # Simulate API call
        
        results = [
            {
                "title": f"Search result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a sample snippet for result {i+1} about {query}",
                "relevance_score": 1.0 - (i * 0.1)
            }
            for i in range(min(max_results, 5))  # Return up to 5 mock results
        ]
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "search_time_ms": 1000,
            "source": "web_search_tool"
        }


class ContentAnalyzerTool(BaseTool):
    """Example content analysis tool"""
    
    def __init__(self, logger: logging.Logger, metrics: MetricsCollector):
        definition = ToolDefinition(
            name="analyze_content", 
            description="Analyze content structure and quality",
            tool_type=ToolType.ANALYSIS,
            parameters=[
                ToolParameter(
                    name="content",
                    param_type="string",
                    description="Content to analyze",
                    required=True
                ),
                ToolParameter(
                    name="analysis_type",
                    param_type="string", 
                    description="Type of analysis to perform",
                    required=False,
                    default="comprehensive",
                    validation_rules={"enum": ["basic", "comprehensive", "seo", "readability"]}
                )
            ],
            compatibility=ToolCompatibility.ALL_FRAMEWORKS
        )
        super().__init__(definition, logger, metrics)
        
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute content analysis"""
        content = parameters["content"]
        analysis_type = parameters.get("analysis_type", "comprehensive")
        
        self.logger.info(f"Analyzing content ({len(content)} chars) with {analysis_type} analysis")
        
        # Simulate analysis - replace with actual implementation
        await asyncio.sleep(0.5)
        
        return {
            "content_length": len(content),
            "word_count": len(content.split()),
            "readability_score": 0.85,
            "sentiment_score": 0.65, 
            "key_topics": ["example", "content", "analysis"],
            "quality_score": 0.78,
            "analysis_type": analysis_type,
            "timestamp": datetime.utcnow().isoformat()
        }


# Tool discovery and auto-registration

class ToolDiscovery:
    """Discovers and auto-registers tools"""
    
    def __init__(self, registry: ToolRegistry, logger: logging.Logger):
        self.registry = registry
        self.logger = logger
        
    async def discover_tools_from_config(self, config: ConfigLoader):
        """Discover tools from configuration"""
        tool_configs = config.get_tool_configs()
        
        for tool_name, tool_config in tool_configs.items():
            await self._register_tool_from_config(tool_name, tool_config)
            
    async def _register_tool_from_config(self, tool_name: str, config: Dict[str, Any]):
        """Register a tool from configuration"""
        try:
            # Create tool definition from config
            definition = self._create_definition_from_config(tool_name, config)
            
            # Create tool instance (simplified - would need factory pattern for real implementation)
            if tool_name == "web_search":
                tool = WebSearchTool(self.logger, self.registry.metrics)
            elif tool_name == "analyze_content":
                tool = ContentAnalyzerTool(self.logger, self.registry.metrics)
            else:
                self.logger.warning(f"Unknown tool type for {tool_name}, skipping")
                return
                
            self.registry.register_tool(tool)
            self.logger.info(f"Auto-registered tool: {tool_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register tool {tool_name}: {e}")
            
    def _create_definition_from_config(self, tool_name: str, config: Dict[str, Any]) -> ToolDefinition:
        """Create tool definition from config"""
        # This is simplified - real implementation would parse full config
        return ToolDefinition(
            name=tool_name,
            description=config.get("description", "Tool description"),
            tool_type=ToolType(config.get("type", "utility")),
            parameters=[],  # Would parse from config
            compatibility=ToolCompatibility.ALL_FRAMEWORKS
        )