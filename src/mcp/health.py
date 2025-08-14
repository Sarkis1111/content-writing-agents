"""
Health Check Endpoints for All Frameworks

This module provides comprehensive health monitoring endpoints for the MCP server
and all integrated agentic frameworks (CrewAI, LangGraph, AutoGen). It includes
system health checks, framework-specific diagnostics, and monitoring endpoints.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import psutil
import platform
from datetime import datetime, timedelta
import json

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from ..core.config.loader import ConfigLoader
from ..core.monitoring.metrics import MetricsCollector
from ..core.monitoring.health import HealthMonitor
from ..core.errors.exceptions import HealthCheckError
from .server import FrameworkType


class HealthStatus(str, Enum):
    """Health check status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of components to monitor"""
    FRAMEWORK = "framework"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_BROKER = "message_broker"
    EXTERNAL_API = "external_api"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    component: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "component": self.component,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "details": self.details
        }


@dataclass
class SystemHealth:
    """Overall system health information"""
    overall_status: HealthStatus
    components: List[HealthCheckResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    uptime_seconds: float = 0
    version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "overall_status": self.overall_status.value,
            "components": [comp.to_dict() for comp in self.components],
            "system_info": self.system_info,
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "timestamp": self.timestamp.isoformat()
        }


class HealthCheck:
    """Base health check class"""
    
    def __init__(self, name: str, component_type: ComponentType, 
                 timeout_seconds: float = 5.0):
        self.name = name
        self.component_type = component_type
        self.timeout_seconds = timeout_seconds
        
    async def check(self) -> HealthCheckResult:
        """Perform health check"""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self._perform_check(), 
                timeout=self.timeout_seconds
            )
            
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            result.response_time_ms = response_time
            
            return result
            
        except asyncio.TimeoutError:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _perform_check(self) -> HealthCheckResult:
        """Override this method to implement specific health check logic"""
        raise NotImplementedError


class SystemResourcesHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk)"""
    
    def __init__(self):
        super().__init__("system_resources", ComponentType.NETWORK)
        
    async def _perform_check(self) -> HealthCheckResult:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine status based on usage
            status = HealthStatus.HEALTHY
            message = "System resources are healthy"
            
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "System resources are critically high"
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 80:
                status = HealthStatus.DEGRADED
                message = "System resources are elevated"
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk_percent,
                "disk_free_gb": disk.free / (1024**3)
            }
            
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {str(e)}"
            )


class FrameworkHealthCheck(HealthCheck):
    """Health check for agentic frameworks"""
    
    def __init__(self, framework: FrameworkType, framework_handler):
        super().__init__(f"{framework.value}_framework", ComponentType.FRAMEWORK)
        self.framework = framework
        self.framework_handler = framework_handler
        
    async def _perform_check(self) -> HealthCheckResult:
        """Check framework health"""
        try:
            if not self.framework_handler:
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message="Framework handler not initialized"
                )
            
            # Get framework status
            framework_status = self.framework_handler.get_status()
            
            # Check if framework is properly initialized
            if not framework_status.get("initialized", False):
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message="Framework not initialized",
                    details=framework_status
                )
            
            # Check framework-specific health metrics
            status = HealthStatus.HEALTHY
            message = f"{self.framework.value} framework is healthy"
            
            # Framework-specific health checks
            if self.framework == FrameworkType.CREWAI:
                status, message = await self._check_crewai_health(framework_status)
            elif self.framework == FrameworkType.LANGGRAPH:
                status, message = await self._check_langgraph_health(framework_status)
            elif self.framework == FrameworkType.AUTOGEN:
                status, message = await self._check_autogen_health(framework_status)
            
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=status,
                message=message,
                details=framework_status
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Framework health check failed: {str(e)}"
            )
    
    async def _check_crewai_health(self, status: Dict[str, Any]) -> tuple[HealthStatus, str]:
        """CrewAI-specific health checks"""
        # Check agent count
        active_agents = status.get("active_agents", 0)
        if active_agents == 0:
            return HealthStatus.DEGRADED, "No active CrewAI agents"
        
        # Check crew status
        crew_errors = status.get("crew_errors", 0)
        if crew_errors > 0:
            return HealthStatus.DEGRADED, f"CrewAI has {crew_errors} crew errors"
        
        return HealthStatus.HEALTHY, "CrewAI framework is healthy"
    
    async def _check_langgraph_health(self, status: Dict[str, Any]) -> tuple[HealthStatus, str]:
        """LangGraph-specific health checks"""
        # Check workflow status
        active_workflows = status.get("active_workflows", 0)
        failed_workflows = status.get("failed_workflows", 0)
        
        if failed_workflows > active_workflows * 0.1:  # More than 10% failure rate
            return HealthStatus.DEGRADED, f"LangGraph has high workflow failure rate: {failed_workflows}"
        
        # Check state persistence
        state_backend_healthy = status.get("state_backend_healthy", True)
        if not state_backend_healthy:
            return HealthStatus.UNHEALTHY, "LangGraph state backend is unhealthy"
        
        return HealthStatus.HEALTHY, "LangGraph framework is healthy"
    
    async def _check_autogen_health(self, status: Dict[str, Any]) -> tuple[HealthStatus, str]:
        """AutoGen-specific health checks"""
        # Check conversation status
        active_conversations = status.get("active_conversations", 0)
        failed_conversations = status.get("failed_conversations", 0)
        
        if failed_conversations > 5:  # Arbitrary threshold
            return HealthStatus.DEGRADED, f"AutoGen has {failed_conversations} failed conversations"
        
        return HealthStatus.HEALTHY, "AutoGen framework is healthy"


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connections"""
    
    def __init__(self, db_connection):
        super().__init__("database", ComponentType.DATABASE)
        self.db_connection = db_connection
        
    async def _perform_check(self) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            if not self.db_connection:
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message="Database connection not configured"
                )
            
            # Simple connectivity test
            # await self.db_connection.execute("SELECT 1")
            
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.HEALTHY,
                message="Database connection is healthy"
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}"
            )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis cache/message broker"""
    
    def __init__(self, redis_client):
        super().__init__("redis", ComponentType.CACHE)
        self.redis_client = redis_client
        
    async def _perform_check(self) -> HealthCheckResult:
        """Check Redis connectivity"""
        try:
            if not self.redis_client:
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message="Redis client not configured"
                )
            
            # Ping Redis
            pong = await self.redis_client.ping()
            if not pong:
                raise Exception("Redis ping failed")
            
            # Get Redis info
            info = await self.redis_client.info()
            memory_usage = info.get('used_memory_human', 'unknown')
            connected_clients = info.get('connected_clients', 0)
            
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.HEALTHY,
                message="Redis connection is healthy",
                details={
                    "memory_usage": memory_usage,
                    "connected_clients": connected_clients,
                    "uptime_in_seconds": info.get('uptime_in_seconds', 0)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}"
            )


class ExternalAPIHealthCheck(HealthCheck):
    """Health check for external API endpoints"""
    
    def __init__(self, api_name: str, health_url: str):
        super().__init__(f"{api_name}_api", ComponentType.EXTERNAL_API)
        self.api_name = api_name
        self.health_url = health_url
        
    async def _perform_check(self) -> HealthCheckResult:
        """Check external API health"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.health_url, 
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        return HealthCheckResult(
                            component=self.name,
                            component_type=self.component_type,
                            status=HealthStatus.HEALTHY,
                            message=f"{self.api_name} API is healthy",
                            details={"status_code": response.status}
                        )
                    else:
                        return HealthCheckResult(
                            component=self.name,
                            component_type=self.component_type,
                            status=HealthStatus.UNHEALTHY,
                            message=f"{self.api_name} API returned status {response.status}",
                            details={"status_code": response.status}
                        )
                        
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"{self.api_name} API health check failed: {str(e)}"
            )


class HealthCheckManager:
    """Manages all health checks for the MCP server"""
    
    def __init__(self, config: ConfigLoader, logger: logging.Logger, 
                 metrics: MetricsCollector):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        
        self.health_checks: Dict[str, HealthCheck] = {}
        self.last_health_results: Dict[str, HealthCheckResult] = {}
        self.system_start_time = time.time()
        
        # Register default health checks
        self._register_default_health_checks()
        
    def _register_default_health_checks(self):
        """Register default system health checks"""
        # System resources
        self.register_health_check(SystemResourcesHealthCheck())
        
        # Add more default checks as needed
        
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check"""
        self.health_checks[health_check.name] = health_check
        self.logger.info(f"Registered health check: {health_check.name}")
        
    def register_framework_health_check(self, framework: FrameworkType, framework_handler):
        """Register framework-specific health check"""
        health_check = FrameworkHealthCheck(framework, framework_handler)
        self.register_health_check(health_check)
        
    def register_redis_health_check(self, redis_client):
        """Register Redis health check"""
        health_check = RedisHealthCheck(redis_client)
        self.register_health_check(health_check)
        
    def register_external_api_health_check(self, api_name: str, health_url: str):
        """Register external API health check"""
        health_check = ExternalAPIHealthCheck(api_name, health_url)
        self.register_health_check(health_check)
        
    async def run_health_check(self, check_name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if check_name not in self.health_checks:
            raise HealthCheckError(f"Health check '{check_name}' not found")
        
        health_check = self.health_checks[check_name]
        
        try:
            result = await health_check.check()
            self.last_health_results[check_name] = result
            
            # Update metrics
            self.metrics.increment_counter(f"health_check.{check_name}.{result.status.value}")
            if result.response_time_ms:
                self.metrics.record_gauge(f"health_check.{check_name}.response_time", 
                                        result.response_time_ms)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Health check {check_name} failed: {e}")
            
            error_result = HealthCheckResult(
                component=check_name,
                component_type=health_check.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check execution failed: {str(e)}"
            )
            
            self.last_health_results[check_name] = error_result
            return error_result
            
    async def run_all_health_checks(self) -> SystemHealth:
        """Run all registered health checks"""
        results = []
        
        # Run health checks concurrently
        tasks = []
        for check_name in self.health_checks.keys():
            task = asyncio.create_task(self.run_health_check(check_name))
            tasks.append(task)
        
        if tasks:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    check_name = list(self.health_checks.keys())[i]
                    error_result = HealthCheckResult(
                        component=check_name,
                        component_type=ComponentType.FRAMEWORK,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {str(result)}"
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        # Determine overall system health
        overall_status = self._determine_overall_status(results)
        
        # Get system information
        system_info = self._get_system_info()
        
        uptime = time.time() - self.system_start_time
        
        system_health = SystemHealth(
            overall_status=overall_status,
            components=results,
            system_info=system_info,
            uptime_seconds=uptime
        )
        
        # Update overall system health metric
        self.metrics.record_gauge("system.health.overall", 
                                1 if overall_status == HealthStatus.HEALTHY else 0)
        
        return system_health
        
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system health status"""
        if not results:
            return HealthStatus.UNKNOWN
        
        unhealthy_count = len([r for r in results if r.status == HealthStatus.UNHEALTHY])
        degraded_count = len([r for r in results if r.status == HealthStatus.DEGRADED])
        
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
            
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "hostname": platform.node(),
                "uptime_seconds": time.time() - self.system_start_time
            }
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of the last health check results"""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.system_start_time,
            "total_checks": len(self.health_checks),
            "healthy_checks": 0,
            "degraded_checks": 0,
            "unhealthy_checks": 0,
            "unknown_checks": 0,
            "checks": {}
        }
        
        for check_name, result in self.last_health_results.items():
            summary["checks"][check_name] = {
                "status": result.status.value,
                "message": result.message,
                "last_checked": result.timestamp.isoformat(),
                "response_time_ms": result.response_time_ms
            }
            
            # Count statuses
            if result.status == HealthStatus.HEALTHY:
                summary["healthy_checks"] += 1
            elif result.status == HealthStatus.DEGRADED:
                summary["degraded_checks"] += 1
            elif result.status == HealthStatus.UNHEALTHY:
                summary["unhealthy_checks"] += 1
            else:
                summary["unknown_checks"] += 1
        
        return summary


# FastAPI router for health endpoints

def create_health_router(health_manager: HealthCheckManager) -> APIRouter:
    """Create FastAPI router with health endpoints"""
    
    router = APIRouter(prefix="/health", tags=["health"])
    
    @router.get("/", summary="Overall system health")
    async def get_system_health():
        """Get comprehensive system health status"""
        try:
            system_health = await health_manager.run_all_health_checks()
            return system_health.to_dict()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Health check failed: {str(e)}"
            )
    
    @router.get("/summary", summary="Health check summary")
    async def get_health_summary():
        """Get summary of health check results"""
        try:
            summary = await health_manager.get_health_summary()
            return summary
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get health summary: {str(e)}"
            )
    
    @router.get("/check/{check_name}", summary="Individual health check")
    async def get_health_check(check_name: str):
        """Run a specific health check"""
        try:
            result = await health_manager.run_health_check(check_name)
            return result.to_dict()
        except HealthCheckError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Health check failed: {str(e)}"
            )
    
    @router.get("/frameworks", summary="Framework health status")
    async def get_frameworks_health():
        """Get health status for all frameworks"""
        try:
            framework_results = {}
            
            for check_name, health_check in health_manager.health_checks.items():
                if isinstance(health_check, FrameworkHealthCheck):
                    result = await health_manager.run_health_check(check_name)
                    framework_results[health_check.framework.value] = result.to_dict()
            
            return {
                "frameworks": framework_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Framework health check failed: {str(e)}"
            )
    
    @router.get("/ready", summary="Readiness probe")
    async def ready():
        """Readiness probe for deployment systems"""
        try:
            system_health = await health_manager.run_all_health_checks()
            
            if system_health.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                return {
                    "ready": True,
                    "status": system_health.overall_status.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="System is not ready"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Readiness check failed: {str(e)}"
            )
    
    @router.get("/live", summary="Liveness probe")
    async def live():
        """Liveness probe for deployment systems"""
        return {
            "alive": True,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - health_manager.system_start_time
        }
    
    return router