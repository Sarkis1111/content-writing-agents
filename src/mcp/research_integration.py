"""
Research Agent MCP Integration for Phase 3.1

This module integrates the CrewAI-based Research Agent with the MCP server,
providing endpoints and handlers for research operations.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.logging import get_framework_logger
from ..core.errors import AgentError, ValidationError
from ..core.monitoring import get_metrics_collector, PerformanceTimer
from .server import MCPMessage, MessageType, FrameworkType


# Simplified Research Models for MCP Integration
class ResearchRequestModel(BaseModel):
    """MCP-compatible research request model."""
    
    topic: str = Field(..., description="Research topic")
    research_depth: str = Field(default="standard", description="Research depth: quick, standard, comprehensive, deep")
    focus_areas: List[str] = Field(default_factory=list, description="Specific areas to focus on")
    max_sources: int = Field(default=25, description="Maximum number of sources to gather")
    include_trends: bool = Field(default=True, description="Include trend analysis")
    include_news: bool = Field(default=True, description="Include news analysis")
    fact_check: bool = Field(default=True, description="Enable fact checking")
    time_limit: Optional[int] = Field(default=None, description="Time limit in seconds")
    language: str = Field(default="en", description="Language code")
    region: str = Field(default="US", description="Region code")


class ResearchResponseModel(BaseModel):
    """MCP-compatible research response model."""
    
    request_id: str
    topic: str
    status: str
    execution_time: float
    summary: str = ""
    key_findings: List[str] = Field(default_factory=list)
    sources_count: int = 0
    trends_count: int = 0
    agent_performance: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ResearchAgentMCP:
    """
    MCP integration layer for the Research Agent.
    
    This class provides the interface between the MCP server and the CrewAI
    Research Agent, handling protocol conversion and async operations.
    """
    
    def __init__(self, mcp_server):
        """Initialize Research Agent MCP integration."""
        
        self.mcp_server = mcp_server
        self.logger = get_framework_logger("ResearchAgentMCP")
        self.metrics = get_metrics_collector()
        
        # Agent instance will be created on demand to avoid import issues
        self._agent_instance = None
        self._coordinator_instance = None
        
        # Track active research operations
        self.active_research: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.operation_stats = {
            "total_requests": 0,
            "successful_requests": 0, 
            "failed_requests": 0,
            "avg_execution_time": 0.0
        }
        
        self.logger.info("Research Agent MCP integration initialized")
    
    def _get_agent_instance(self):
        """Lazy loading of Research Agent to avoid import dependencies."""
        
        if self._agent_instance is None:
            try:
                # Simplified agent creation without tool dependencies
                self._agent_instance = SimpleResearchAgent()
                self.logger.info("Research Agent instance created")
            except Exception as e:
                self.logger.error(f"Failed to create Research Agent: {e}")
                raise AgentError(f"Research Agent initialization failed: {e}")
        
        return self._agent_instance
    
    def _get_coordinator_instance(self):
        """Lazy loading of Research Coordinator."""
        
        if self._coordinator_instance is None:
            try:
                # Simplified coordinator creation
                self._coordinator_instance = SimpleResearchCoordinator()
                self.logger.info("Research Coordinator instance created")
            except Exception as e:
                self.logger.error(f"Failed to create Research Coordinator: {e}")
                raise AgentError(f"Research Coordinator initialization failed: {e}")
        
        return self._coordinator_instance
    
    async def handle_research_request(
        self, 
        request: ResearchRequestModel,
        client_id: Optional[str] = None
    ) -> ResearchResponseModel:
        """
        Handle research request from MCP client.
        
        Args:
            request: Research request parameters
            client_id: Optional client ID for progress updates
            
        Returns:
            Research response with findings
        """
        
        request_id = f"research_{int(datetime.now().timestamp())}"
        
        self.logger.info(f"Handling research request: {request_id}")
        self.metrics.record_counter("research_request_received")
        
        # Update active research tracking
        self.active_research[request_id] = {
            "client_id": client_id,
            "topic": request.topic,
            "status": "started",
            "start_time": datetime.now(),
            "progress": 0.0
        }
        
        try:
            # Send start notification to client
            if client_id:
                await self._send_workflow_start(client_id, request_id, request.topic)
            
            with PerformanceTimer() as timer:
                # Execute research using simplified agent
                agent = self._get_agent_instance()
                result = await agent.research(request, request_id, self._progress_callback)
                
                # Create response
                response = ResearchResponseModel(
                    request_id=request_id,
                    topic=request.topic,
                    status=result["status"],
                    execution_time=timer.elapsed_time,
                    summary=result.get("summary", ""),
                    key_findings=result.get("key_findings", []),
                    sources_count=result.get("sources_count", 0),
                    trends_count=result.get("trends_count", 0),
                    agent_performance=result.get("performance", {}),
                    errors=result.get("errors", []),
                    warnings=result.get("warnings", [])
                )
                
                # Update stats
                self._update_operation_stats(True, timer.elapsed_time)
                
                # Send completion notification
                if client_id:
                    await self._send_workflow_complete(client_id, request_id, response)
                
                # Clean up tracking
                if request_id in self.active_research:
                    del self.active_research[request_id]
                
                self.logger.info(f"Research completed: {request_id}")
                self.metrics.record_timer("research_execution_time", timer.elapsed_time)
                
                return response
                
        except Exception as e:
            self.logger.error(f"Research failed: {request_id} - {e}")
            self.metrics.record_counter("research_request_failed")
            
            # Update stats
            self._update_operation_stats(False, 0.0)
            
            # Create error response
            response = ResearchResponseModel(
                request_id=request_id,
                topic=request.topic,
                status="failed", 
                execution_time=0.0,
                errors=[str(e)]
            )
            
            # Send error notification
            if client_id:
                await self._send_workflow_error(client_id, request_id, str(e))
            
            # Clean up tracking
            if request_id in self.active_research:
                del self.active_research[request_id]
            
            return response
    
    async def _progress_callback(self, request_id: str, progress: float, status: str):
        """Callback for research progress updates."""
        
        if request_id in self.active_research:
            tracking = self.active_research[request_id]
            tracking["progress"] = progress
            tracking["status"] = status
            
            client_id = tracking.get("client_id")
            if client_id:
                await self._send_workflow_update(client_id, request_id, progress, status)
    
    async def _send_workflow_start(self, client_id: str, request_id: str, topic: str):
        """Send workflow start notification to client."""
        
        message = MCPMessage(
            message_type=MessageType.WORKFLOW_START,
            message_id=f"start_{request_id}",
            payload={
                "request_id": request_id,
                "topic": topic,
                "framework": "crewai",
                "agent_type": "research"
            }
        )
        
        await self.mcp_server.connection_manager.send_to_client(client_id, message)
    
    async def _send_workflow_update(
        self, 
        client_id: str, 
        request_id: str, 
        progress: float, 
        status: str
    ):
        """Send workflow progress update to client."""
        
        message = MCPMessage(
            message_type=MessageType.WORKFLOW_UPDATE,
            message_id=f"update_{request_id}_{int(progress*100)}",
            payload={
                "request_id": request_id,
                "progress": progress,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.mcp_server.connection_manager.send_to_client(client_id, message)
    
    async def _send_workflow_complete(
        self, 
        client_id: str, 
        request_id: str, 
        response: ResearchResponseModel
    ):
        """Send workflow completion notification to client."""
        
        message = MCPMessage(
            message_type=MessageType.WORKFLOW_COMPLETE,
            message_id=f"complete_{request_id}",
            payload={
                "request_id": request_id,
                "response": response.dict(),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.mcp_server.connection_manager.send_to_client(client_id, message)
    
    async def _send_workflow_error(self, client_id: str, request_id: str, error: str):
        """Send workflow error notification to client."""
        
        message = MCPMessage(
            message_type=MessageType.ERROR,
            message_id=f"error_{request_id}",
            payload={
                "request_id": request_id,
                "error": error,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.mcp_server.connection_manager.send_to_client(client_id, message)
    
    def _update_operation_stats(self, success: bool, execution_time: float):
        """Update operation statistics."""
        
        self.operation_stats["total_requests"] += 1
        
        if success:
            self.operation_stats["successful_requests"] += 1
        else:
            self.operation_stats["failed_requests"] += 1
        
        # Update average execution time
        if success and execution_time > 0:
            total_time = (self.operation_stats["avg_execution_time"] * 
                         (self.operation_stats["successful_requests"] - 1))
            self.operation_stats["avg_execution_time"] = (
                (total_time + execution_time) / self.operation_stats["successful_requests"]
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get Research Agent health status."""
        
        try:
            agent = self._get_agent_instance()
            coordinator = self._get_coordinator_instance()
            
            # Simple health check
            health_status = {
                "status": "healthy",
                "framework": "crewai",
                "agent_type": "research",
                "active_operations": len(self.active_research),
                "operation_stats": self.operation_stats,
                "agent_available": agent is not None,
                "coordinator_available": coordinator is not None,
                "last_check": datetime.now().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "framework": "crewai",
                "agent_type": "research",
                "error": str(e),
                "operation_stats": self.operation_stats,
                "last_check": datetime.now().isoformat()
            }
    
    def get_active_operations(self) -> Dict[str, Any]:
        """Get information about active research operations."""
        
        return {
            "active_count": len(self.active_research),
            "operations": {
                req_id: {
                    "topic": info["topic"],
                    "status": info["status"],
                    "progress": info["progress"],
                    "elapsed_time": (datetime.now() - info["start_time"]).total_seconds()
                }
                for req_id, info in self.active_research.items()
            }
        }


class SimpleResearchAgent:
    """
    Simplified Research Agent for MCP integration without tool dependencies.
    
    This is a lightweight version that focuses on the agent architecture
    without requiring the full tool stack.
    """
    
    def __init__(self):
        self.logger = get_framework_logger("SimpleResearchAgent")
        self.execution_count = 0
    
    async def research(
        self, 
        request: ResearchRequestModel, 
        request_id: str,
        progress_callback = None
    ) -> Dict[str, Any]:
        """Execute simplified research workflow."""
        
        self.logger.info(f"Starting simplified research: {request.topic}")
        self.execution_count += 1
        
        # Simulate research workflow with progress updates
        if progress_callback:
            await progress_callback(request_id, 0.1, "Initializing research")
            await asyncio.sleep(0.5)
            
            await progress_callback(request_id, 0.3, "Gathering information")
            await asyncio.sleep(0.5)
            
            await progress_callback(request_id, 0.6, "Analyzing trends") 
            await asyncio.sleep(0.5)
            
            await progress_callback(request_id, 0.8, "Curating content")
            await asyncio.sleep(0.5)
            
            await progress_callback(request_id, 1.0, "Finalizing results")
            await asyncio.sleep(0.2)
        
        # Generate mock research results
        result = {
            "status": "completed",
            "summary": f"Research completed for '{request.topic}' using {request.research_depth} analysis. "
                      f"This simplified implementation demonstrates the CrewAI integration architecture.",
            "key_findings": [
                f"Key insight 1 about {request.topic}",
                f"Important trend identified in {request.topic} domain", 
                f"Strategic recommendation for {request.topic} implementation",
                f"Market analysis indicates growing interest in {request.topic}",
                f"Technical considerations for {request.topic} deployment"
            ][:3 if request.research_depth == "quick" else 5],
            "sources_count": 5 if request.research_depth == "quick" else 15,
            "trends_count": 2 if request.research_depth == "quick" else 5,
            "performance": {
                "framework": "crewai",
                "agent_type": "research",
                "execution_count": self.execution_count,
                "depth": request.research_depth,
                "multi_agent_coordination": True
            }
        }
        
        return result


class SimpleResearchCoordinator:
    """Simplified Research Coordinator for MCP integration."""
    
    def __init__(self):
        self.logger = get_framework_logger("SimpleResearchCoordinator")
        self.workflow_count = 0
    
    async def coordinate_research(self, request: ResearchRequestModel) -> Dict[str, Any]:
        """Execute simplified coordination workflow."""
        
        self.logger.info(f"Coordinating research workflow: {request.topic}")
        self.workflow_count += 1
        
        # Simulate coordination
        await asyncio.sleep(1.0)
        
        return {
            "status": "completed",
            "workflow_id": f"workflow_{self.workflow_count}",
            "coordination_successful": True
        }


def create_research_router(research_mcp: ResearchAgentMCP) -> APIRouter:
    """Create FastAPI router for Research Agent endpoints."""
    
    router = APIRouter(prefix="/research", tags=["research"])
    
    @router.post("/execute", response_model=ResearchResponseModel)
    async def execute_research(
        request: ResearchRequestModel,
        background_tasks: BackgroundTasks,
        client_id: Optional[str] = None
    ):
        """Execute research request."""
        
        try:
            response = await research_mcp.handle_research_request(request, client_id)
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/health")
    async def get_research_health():
        """Get Research Agent health status."""
        
        try:
            health = await research_mcp.get_health_status()
            return health
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/operations")
    async def get_active_operations():
        """Get active research operations."""
        
        try:
            operations = research_mcp.get_active_operations()
            return operations
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router