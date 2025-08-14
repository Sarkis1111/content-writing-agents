"""
MCP (Model Context Protocol) Server Module

This module provides the complete MCP server implementation with framework abstraction
for coordinating multi-framework agentic operations across CrewAI, LangGraph, and AutoGen.

The MCP server serves as the central coordination layer that enables:
- Unified communication between different agentic frameworks
- Tool registration and execution across frameworks
- Workflow orchestration and management
- Security and authentication
- Health monitoring and diagnostics
- Message routing and translation

Key Components:
- MCPServer: Main server with WebSocket and HTTP endpoints
- MessageRouter: Routes messages between frameworks and components
- ToolRegistry: Centralized tool registration and execution
- CommunicationManager: Framework-agnostic communication layer
- SecurityManager: Authentication, authorization, and audit logging
- HealthCheckManager: Comprehensive health monitoring

Usage:
    from src.mcp import MCPServer, start_mcp_server
    
    # Start server with default configuration
    await start_mcp_server()
    
    # Or create server instance with custom config
    server = MCPServer("config/custom.yaml")
    await server.start_server()
"""

from .server import (
    MCPServer,
    MCPMessage,
    MessageType,
    FrameworkType,
    ConnectionManager,
    FrameworkAbstractionLayer,
    start_mcp_server
)

from .message_handlers import (
    MessageRouter,
    WorkflowOrchestrator,
    FrameworkBridge,
    MessageQueue,
    WorkflowContext,
    WorkflowStatus,
    MessagePriority,
    InitializeHandler,
    WorkflowHandler
)

from .tools import (
    ToolRegistry,
    ToolDefinition,
    ToolParameter,
    BaseTool,
    ToolType,
    ToolCompatibility,
    FrameworkToolAdapter,
    ToolExecutor,
    WebSearchTool,
    ContentAnalyzerTool,
    ToolDiscovery
)

from .communication import (
    CommunicationManager,
    CommunicationEndpoint,
    CommunicationChannel,
    MessagePersistence,
    RedisBroker,
    MemoryBroker,
    CrewAICommunicationAdapter,
    LangGraphCommunicationAdapter,
    AutoGenCommunicationAdapter
)

from .security import (
    SecurityManager,
    User,
    UserRole,
    Permission,
    Session,
    SecurityConfig,
    PasswordManager,
    TokenManager,
    EncryptionManager,
    RateLimiter,
    AuditLogger
)

from .health import (
    HealthCheckManager,
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    SystemHealth,
    ComponentType,
    SystemResourcesHealthCheck,
    FrameworkHealthCheck,
    DatabaseHealthCheck,
    RedisHealthCheck,
    ExternalAPIHealthCheck,
    create_health_router
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Agentic Content Writing System"
__description__ = "MCP server for multi-framework agentic operations"

# Default configuration
DEFAULT_SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "INFO",
    "enable_cors": True,
    "enable_authentication": True,
    "enable_health_checks": True
}

# Export commonly used items at package level
__all__ = [
    # Core server components
    "MCPServer",
    "start_mcp_server",
    "MCPMessage",
    "MessageType", 
    "FrameworkType",
    
    # Message handling
    "MessageRouter",
    "WorkflowOrchestrator", 
    "WorkflowContext",
    "WorkflowStatus",
    
    # Tool system
    "ToolRegistry",
    "ToolDefinition",
    "BaseTool",
    "ToolType",
    "ToolCompatibility",
    
    # Communication
    "CommunicationManager",
    "CommunicationChannel",
    "MessagePersistence",
    
    # Security
    "SecurityManager",
    "User",
    "UserRole", 
    "Permission",
    "SecurityConfig",
    
    # Health monitoring
    "HealthCheckManager",
    "HealthCheck",
    "HealthStatus",
    "SystemHealth",
    "create_health_router",
    
    # Package info
    "__version__",
    "DEFAULT_SERVER_CONFIG"
]


# Convenience functions for quick setup

async def create_mcp_server_with_all_components(config_path: str = None) -> MCPServer:
    """
    Create a fully configured MCP server with all components initialized.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured MCPServer instance
    """
    # Create server
    server = MCPServer(config_path)
    
    # Initialize all components would happen in server startup
    # This is just a convenience function for external usage
    
    return server


def get_default_workflow_templates() -> dict:
    """
    Get default workflow templates for common content creation patterns.
    
    Returns:
        Dictionary of workflow templates
    """
    return {
        "comprehensive_content_creation": {
            "name": "Comprehensive Content Creation Pipeline",
            "description": "Full multi-framework content creation process",
            "frameworks_used": ["crewai", "autogen", "langgraph"],
            "steps": [
                {
                    "framework": "crewai",
                    "agent": "research_crew",
                    "action": "comprehensive_research",
                    "inputs": ["topic", "requirements"],
                    "outputs": ["research_data", "trend_insights", "competitive_analysis"],
                    "timeout": 600
                },
                {
                    "framework": "autogen", 
                    "agent": "strategy_council",
                    "action": "develop_strategy",
                    "inputs": ["research_data", "business_objectives"],
                    "outputs": ["content_strategy", "messaging_framework", "targeting_approach"],
                    "timeout": 400
                },
                {
                    "framework": "langgraph",
                    "agent": "content_writer", 
                    "action": "create_content",
                    "inputs": ["content_strategy", "research_data"],
                    "outputs": ["draft_content", "content_outline", "supporting_materials"],
                    "timeout": 500
                },
                {
                    "framework": "langgraph",
                    "agent": "content_editor",
                    "action": "edit_and_optimize", 
                    "inputs": ["draft_content", "content_strategy"],
                    "outputs": ["final_content", "quality_report", "optimization_recommendations"],
                    "timeout": 300
                }
            ]
        },
        
        "rapid_content_generation": {
            "name": "Rapid Content Generation",
            "description": "Fast-track content creation for urgent needs",
            "frameworks_used": ["crewai", "langgraph"],
            "steps": [
                {
                    "framework": "crewai",
                    "agent": "research_crew",
                    "action": "quick_research",
                    "inputs": ["topic", "urgency_level"],
                    "outputs": ["key_insights", "essential_facts"],
                    "timeout": 300,
                    "parallel_execution": True
                },
                {
                    "framework": "langgraph",
                    "agent": "content_writer", 
                    "action": "rapid_content_creation",
                    "inputs": ["key_insights", "content_type"],
                    "outputs": ["content_draft"],
                    "timeout": 200
                },
                {
                    "framework": "langgraph",
                    "agent": "content_editor",
                    "action": "essential_review",
                    "inputs": ["content_draft"],
                    "outputs": ["final_content", "basic_quality_report"],
                    "timeout": 150
                }
            ]
        }
    }


# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"MCP module initialized - version {__version__}")
logger.info("Available components: Server, MessageRouter, ToolRegistry, CommunicationManager, SecurityManager, HealthCheckManager")