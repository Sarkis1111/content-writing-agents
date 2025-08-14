"""
MCP Server Foundation with Framework Abstraction

This module implements the core MCP (Model Context Protocol) server that provides
a unified interface for multi-framework agentic operations across CrewAI, LangGraph,
and AutoGen. It serves as the central coordination layer for all agent interactions.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, ValidationError

from ..core.config.loader import ConfigLoader
from ..core.logging.logger import LoggingManager
from ..core.monitoring.metrics import MetricsCollector
from ..core.monitoring.health import HealthMonitor
from ..core.errors.exceptions import MCPServerError, FrameworkError
from ..core.errors.handlers import ErrorHandler


class MessageType(str, Enum):
    """MCP Message Types for protocol communication"""
    INITIALIZE = "initialize"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    TOOL_CALL = "tool_call"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_UPDATE = "workflow_update"
    WORKFLOW_COMPLETE = "workflow_complete"


class FrameworkType(str, Enum):
    """Supported agentic frameworks"""
    CREWAI = "crewai"
    LANGGRAPH = "langgraph"
    AUTOGEN = "autogen"


@dataclass
class MCPMessage:
    """Base MCP protocol message structure"""
    message_id: str
    message_type: MessageType
    framework: Optional[FrameworkType] = None
    timestamp: Optional[datetime] = None
    payload: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON serialization"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "framework": self.framework.value if self.framework else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "payload": self.payload or {},
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """Create MCPMessage from dictionary"""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            framework=FrameworkType(data["framework"]) if data.get("framework") else None,
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {})
        )


class ConnectionManager:
    """Manages WebSocket connections for MCP clients"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.framework_connections: Dict[FrameworkType, List[str]] = {
            framework: [] for framework in FrameworkType
        }
        self.logger = LoggingManager().get_framework_logger("mcp", "connection_manager")

    async def connect(self, websocket: WebSocket, client_id: str, framework: Optional[FrameworkType] = None):
        """Accept new WebSocket connection and register client"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        if framework:
            if client_id not in self.framework_connections[framework]:
                self.framework_connections[framework].append(client_id)
        
        self.logger.info(f"Client {client_id} connected", extra={
            "client_id": client_id,
            "framework": framework.value if framework else None,
            "total_connections": len(self.active_connections)
        })

    def disconnect(self, client_id: str):
        """Remove client connection and clean up"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # Remove from framework connections
            for framework, clients in self.framework_connections.items():
                if client_id in clients:
                    clients.remove(client_id)
        
        self.logger.info(f"Client {client_id} disconnected", extra={
            "client_id": client_id,
            "total_connections": len(self.active_connections)
        })

    async def send_to_client(self, client_id: str, message: MCPMessage):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_text(json.dumps(message.to_dict()))
                self.logger.debug(f"Message sent to client {client_id}", extra={
                    "client_id": client_id,
                    "message_type": message.message_type.value,
                    "message_id": message.message_id
                })
            except Exception as e:
                self.logger.error(f"Failed to send message to client {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast_to_framework(self, framework: FrameworkType, message: MCPMessage):
        """Broadcast message to all clients of a specific framework"""
        client_ids = self.framework_connections[framework].copy()
        for client_id in client_ids:
            await self.send_to_client(client_id, message)

    async def broadcast_to_all(self, message: MCPMessage):
        """Broadcast message to all connected clients"""
        client_ids = list(self.active_connections.keys())
        for client_id in client_ids:
            await self.send_to_client(client_id, message)


class FrameworkAbstractionLayer:
    """
    Provides a unified interface for interacting with different agentic frameworks.
    Abstracts framework-specific implementations behind a common API.
    """

    def __init__(self, config: ConfigLoader, logger: logging.Logger, metrics: MetricsCollector):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.framework_handlers: Dict[FrameworkType, Any] = {}
        self.initialized_frameworks: set = set()

    async def initialize_framework(self, framework: FrameworkType) -> bool:
        """Initialize a specific agentic framework"""
        try:
            if framework == FrameworkType.CREWAI:
                from ..frameworks.crewai.config import CrewAIFramework
                handler = CrewAIFramework(self.config, self.logger, self.metrics)
                await handler.initialize()
                self.framework_handlers[framework] = handler
                
            elif framework == FrameworkType.LANGGRAPH:
                from ..frameworks.langgraph.config import LangGraphFramework
                handler = LangGraphFramework(self.config, self.logger, self.metrics)
                await handler.initialize()
                self.framework_handlers[framework] = handler
                
            elif framework == FrameworkType.AUTOGEN:
                from ..frameworks.autogen.config import AutoGenFramework
                handler = AutoGenFramework(self.config, self.logger, self.metrics)
                await handler.initialize()
                self.framework_handlers[framework] = handler
            
            self.initialized_frameworks.add(framework)
            self.logger.info(f"Framework {framework.value} initialized successfully")
            self.metrics.increment_counter(f"framework.{framework.value}.initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize framework {framework.value}: {e}")
            self.metrics.increment_counter(f"framework.{framework.value}.initialization_failed")
            raise FrameworkError(f"Framework initialization failed: {framework.value}", 
                               framework=framework.value, details={"error": str(e)})

    async def initialize_all_frameworks(self) -> Dict[FrameworkType, bool]:
        """Initialize all supported frameworks"""
        results = {}
        for framework in FrameworkType:
            try:
                results[framework] = await self.initialize_framework(framework)
            except Exception as e:
                self.logger.error(f"Framework {framework.value} initialization failed: {e}")
                results[framework] = False
        
        successful_count = sum(1 for success in results.values() if success)
        self.logger.info(f"Framework initialization complete: {successful_count}/{len(FrameworkType)} successful")
        return results

    async def execute_framework_action(self, framework: FrameworkType, action: str, 
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action on a specific framework"""
        if framework not in self.initialized_frameworks:
            raise FrameworkError(f"Framework {framework.value} not initialized", 
                               framework=framework.value)
        
        handler = self.framework_handlers[framework]
        
        # Record execution metrics
        with self.metrics.timer(f"framework.{framework.value}.{action}.duration"):
            try:
                result = await handler.execute_action(action, parameters)
                self.metrics.increment_counter(f"framework.{framework.value}.{action}.success")
                return result
            except Exception as e:
                self.metrics.increment_counter(f"framework.{framework.value}.{action}.error")
                raise

    def get_framework_status(self, framework: FrameworkType) -> Dict[str, Any]:
        """Get current status of a specific framework"""
        if framework not in self.initialized_frameworks:
            return {"status": "not_initialized", "initialized": False}
        
        handler = self.framework_handlers[framework]
        return handler.get_status()

    def list_available_actions(self, framework: FrameworkType) -> List[str]:
        """Get list of available actions for a framework"""
        if framework not in self.initialized_frameworks:
            return []
        
        handler = self.framework_handlers[framework]
        return handler.list_actions()


class MCPServer:
    """
    Main MCP Server class that orchestrates multi-framework agentic operations.
    Provides unified API for client interactions with CrewAI, LangGraph, and AutoGen.
    """

    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = ConfigLoader(config_path)
        
        # Initialize core systems
        self.logging_manager = LoggingManager(self.config)
        self.logger = self.logging_manager.get_framework_logger("mcp", "server")
        
        self.metrics = MetricsCollector(self.config)
        self.health_monitor = HealthMonitor(self.config, self.metrics)
        self.error_handler = ErrorHandler(self.config, self.logger, self.metrics)
        
        # Initialize MCP components
        self.connection_manager = ConnectionManager()
        self.framework_layer = FrameworkAbstractionLayer(self.config, self.logger, self.metrics)
        
        # FastAPI app
        self.app = FastAPI(
            title="Agentic Content Writing MCP Server",
            description="Multi-framework agentic content writing system with MCP protocol",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self.logger.info("MCP Server initialized")

    def _setup_routes(self):
        """Set up FastAPI routes for MCP server"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize frameworks on server startup"""
            self.logger.info("Starting MCP server initialization")
            try:
                await self.framework_layer.initialize_all_frameworks()
                await self.health_monitor.start()
                self.logger.info("MCP server startup completed successfully")
            except Exception as e:
                self.logger.error(f"MCP server startup failed: {e}")
                raise

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on server shutdown"""
            self.logger.info("Shutting down MCP server")
            await self.health_monitor.stop()

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                health_status = await self.health_monitor.get_system_health()
                return {"status": "healthy", "details": health_status}
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail="Health check failed")

        @self.app.get("/frameworks/status")
        async def get_frameworks_status():
            """Get status of all frameworks"""
            try:
                status = {}
                for framework in FrameworkType:
                    status[framework.value] = self.framework_layer.get_framework_status(framework)
                return {"frameworks": status}
            except Exception as e:
                self.logger.error(f"Failed to get framework status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get framework status")

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """Main WebSocket endpoint for MCP protocol communication"""
            await self.connection_manager.connect(websocket, client_id)
            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    await self._handle_websocket_message(client_id, data)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(client_id)
            except Exception as e:
                self.logger.error(f"WebSocket error for client {client_id}: {e}")
                self.connection_manager.disconnect(client_id)

    async def _handle_websocket_message(self, client_id: str, raw_message: str):
        """Handle incoming WebSocket messages"""
        try:
            # Parse message
            message_data = json.loads(raw_message)
            message = MCPMessage.from_dict(message_data)
            
            self.logger.debug(f"Received message from client {client_id}", extra={
                "client_id": client_id,
                "message_type": message.message_type.value,
                "message_id": message.message_id
            })
            
            # Route message based on type
            response = await self._route_message(message)
            
            # Send response back to client
            if response:
                await self.connection_manager.send_to_client(client_id, response)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON from client {client_id}: {e}")
            error_response = MCPMessage(
                message_id="error",
                message_type=MessageType.ERROR,
                payload={"error": "Invalid JSON format", "details": str(e)}
            )
            await self.connection_manager.send_to_client(client_id, error_response)
            
        except Exception as e:
            self.logger.error(f"Error handling message from client {client_id}: {e}")
            error_response = MCPMessage(
                message_id="error", 
                message_type=MessageType.ERROR,
                payload={"error": "Internal server error", "details": str(e)}
            )
            await self.connection_manager.send_to_client(client_id, error_response)

    async def _route_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Route messages to appropriate handlers based on message type"""
        try:
            if message.message_type == MessageType.INITIALIZE:
                return await self._handle_initialize(message)
            elif message.message_type == MessageType.REQUEST:
                return await self._handle_request(message)
            elif message.message_type == MessageType.TOOL_CALL:
                return await self._handle_tool_call(message)
            elif message.message_type == MessageType.WORKFLOW_START:
                return await self._handle_workflow_start(message)
            else:
                self.logger.warning(f"Unhandled message type: {message.message_type}")
                return MCPMessage(
                    message_id=f"response_{message.message_id}",
                    message_type=MessageType.ERROR,
                    payload={"error": f"Unsupported message type: {message.message_type}"}
                )
                
        except Exception as e:
            self.logger.error(f"Error routing message {message.message_id}: {e}")
            return MCPMessage(
                message_id=f"response_{message.message_id}",
                message_type=MessageType.ERROR,
                payload={"error": "Message routing failed", "details": str(e)}
            )

    async def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """Handle initialization requests"""
        framework = message.framework
        if framework and framework in FrameworkType:
            try:
                success = await self.framework_layer.initialize_framework(framework)
                return MCPMessage(
                    message_id=f"response_{message.message_id}",
                    message_type=MessageType.RESPONSE,
                    framework=framework,
                    payload={"initialized": success, "framework": framework.value}
                )
            except Exception as e:
                return MCPMessage(
                    message_id=f"response_{message.message_id}",
                    message_type=MessageType.ERROR,
                    framework=framework,
                    payload={"error": f"Framework initialization failed: {e}"}
                )
        else:
            return MCPMessage(
                message_id=f"response_{message.message_id}",
                message_type=MessageType.ERROR,
                payload={"error": "Invalid or missing framework specification"}
            )

    async def _handle_request(self, message: MCPMessage) -> MCPMessage:
        """Handle general requests"""
        # Implementation will be expanded in subsequent tasks
        return MCPMessage(
            message_id=f"response_{message.message_id}",
            message_type=MessageType.RESPONSE,
            payload={"status": "request_received", "message": "Request handling not yet implemented"}
        )

    async def _handle_tool_call(self, message: MCPMessage) -> MCPMessage:
        """Handle tool execution requests"""
        # Implementation will be expanded when tool registration is complete
        return MCPMessage(
            message_id=f"response_{message.message_id}",
            message_type=MessageType.RESPONSE,
            payload={"status": "tool_call_received", "message": "Tool execution not yet implemented"}
        )

    async def _handle_workflow_start(self, message: MCPMessage) -> MCPMessage:
        """Handle workflow execution requests"""
        # Implementation will be expanded with workflow orchestration
        return MCPMessage(
            message_id=f"response_{message.message_id}",
            message_type=MessageType.RESPONSE,
            payload={"status": "workflow_start_received", "message": "Workflow execution not yet implemented"}
        )

    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the MCP server"""
        self.logger.info(f"Starting MCP server on {host}:{port}")
        
        # Get server configuration from config
        server_config = self.config.get_mcp_config()
        actual_host = server_config.host or host
        actual_port = server_config.port or port
        
        config = uvicorn.Config(
            app=self.app,
            host=actual_host,
            port=actual_port,
            log_config=None  # Use our custom logging
        )
        
        server = uvicorn.Server(config)
        await server.serve()


# Convenience function for starting the server
async def start_mcp_server(config_path: Optional[str] = None, host: str = "0.0.0.0", port: int = 8000):
    """Start the MCP server with configuration"""
    server = MCPServer(config_path)
    await server.start_server(host, port)


if __name__ == "__main__":
    # For direct execution
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(start_mcp_server(config_path))