"""Health check system for monitoring framework and component status."""

import asyncio
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from ..logging import get_component_logger


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


class HealthCheck:
    """Individual health check definition."""
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], Any],
        timeout: float = 10.0,
        interval: float = 30.0,
        is_critical: bool = True
    ):
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.interval = interval
        self.is_critical = is_critical
        self.last_result: Optional[HealthCheckResult] = None
        self.last_check_time: Optional[datetime] = None
    
    async def execute(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Execute check with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.check_func),
                timeout=self.timeout
            )
            
            duration = time.time() - start_time
            
            if result is True or (isinstance(result, dict) and result.get("healthy", False)):
                status = HealthStatus.HEALTHY
                message = "Check passed"
                metadata = result if isinstance(result, dict) else {}
            elif result is False:
                status = HealthStatus.UNHEALTHY
                message = "Check failed"
                metadata = {}
            else:
                status = HealthStatus.DEGRADED
                message = str(result)
                metadata = result if isinstance(result, dict) else {}
                
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            status = HealthStatus.UNHEALTHY
            message = f"Check timed out after {self.timeout}s"
            metadata = {"timeout": self.timeout}
            
        except Exception as e:
            duration = time.time() - start_time
            status = HealthStatus.UNHEALTHY
            message = f"Check failed with error: {str(e)}"
            metadata = {"error": str(e), "error_type": type(e).__name__}
        
        result = HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            duration=duration,
            timestamp=timestamp,
            metadata=metadata
        )
        
        self.last_result = result
        self.last_check_time = timestamp
        
        return result


class HealthMonitor:
    """Health monitoring system for all framework components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.framework_checks: Dict[str, List[str]] = {}
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.logger = get_component_logger("health_monitor")
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], Any],
        framework: Optional[str] = None,
        timeout: float = 10.0,
        interval: float = 30.0,
        is_critical: bool = True
    ):
        """Register a new health check."""
        health_check = HealthCheck(
            name=name,
            check_func=check_func,
            timeout=timeout,
            interval=interval,
            is_critical=is_critical
        )
        
        self.health_checks[name] = health_check
        
        # Associate with framework if specified
        if framework:
            if framework not in self.framework_checks:
                self.framework_checks[framework] = []
            self.framework_checks[framework].append(name)
        
        self.logger.info(f"Registered health check: {name} (framework: {framework})")
    
    def register_framework_checks(self, framework: str):
        """Register default health checks for a framework."""
        
        if framework == "crewai":
            self.register_check(
                f"{framework}_import",
                lambda: self._check_import("crewai"),
                framework=framework,
                interval=60.0
            )
        elif framework == "langgraph":
            self.register_check(
                f"{framework}_import",
                lambda: self._check_import("langgraph"),
                framework=framework,
                interval=60.0
            )
        elif framework == "autogen":
            self.register_check(
                f"{framework}_import", 
                lambda: self._check_import("autogen"),
                framework=framework,
                interval=60.0
            )
    
    def _check_import(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    async def execute_check(self, check_name: str) -> HealthCheckResult:
        """Execute a specific health check."""
        if check_name not in self.health_checks:
            return HealthCheckResult(
                name=check_name,
                status=HealthStatus.UNKNOWN,
                message="Health check not found",
                duration=0.0,
                timestamp=datetime.now()
            )
        
        return await self.health_checks[check_name].execute()
    
    async def execute_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Execute all registered health checks."""
        results = {}
        
        # Execute all checks concurrently
        check_tasks = []
        for name, check in self.health_checks.items():
            task = asyncio.create_task(check.execute())
            check_tasks.append((name, task))
        
        # Wait for all checks to complete
        for name, task in check_tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Unexpected error: {str(e)}",
                    duration=0.0,
                    timestamp=datetime.now()
                )
        
        return results
    
    async def get_framework_health(self, framework: str) -> Dict[str, Any]:
        """Get health status for a specific framework."""
        if framework not in self.framework_checks:
            return {
                "framework": framework,
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks registered for this framework",
                "checks": []
            }
        
        check_names = self.framework_checks[framework]
        results = []
        overall_status = HealthStatus.HEALTHY
        
        for check_name in check_names:
            result = await self.execute_check(check_name)
            results.append(result.to_dict())
            
            # Determine overall status
            if result.status == HealthStatus.UNHEALTHY:
                if self.health_checks[check_name].is_critical:
                    overall_status = HealthStatus.UNHEALTHY
                elif overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return {
            "framework": framework,
            "status": overall_status.value,
            "message": f"Framework health based on {len(results)} checks",
            "checks": results
        }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        all_results = await self.execute_all_checks()
        
        # Calculate overall system health
        healthy_count = sum(1 for r in all_results.values() if r.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for r in all_results.values() if r.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for r in all_results.values() if r.status == HealthStatus.UNHEALTHY)
        
        total_checks = len(all_results)
        critical_unhealthy = sum(
            1 for name, result in all_results.items()
            if result.status == HealthStatus.UNHEALTHY and self.health_checks[name].is_critical
        )
        
        # Determine overall status
        if critical_unhealthy > 0:
            overall_status = HealthStatus.UNHEALTHY
            message = f"System unhealthy: {critical_unhealthy} critical checks failed"
        elif unhealthy_count > 0 or degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
            message = f"System degraded: {unhealthy_count} unhealthy, {degraded_count} degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            message = f"System healthy: all {total_checks} checks passed"
        
        # Framework breakdown
        framework_health = {}
        for framework in self.framework_checks:
            framework_health[framework] = await self.get_framework_health(framework)
        
        return {
            "status": overall_status.value,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": total_checks,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "critical_failures": critical_unhealthy
            },
            "frameworks": framework_health,
            "checks": {name: result.to_dict() for name, result in all_results.items()}
        }
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.is_running:
            try:
                # Execute all health checks
                results = await self.execute_all_checks()
                
                # Log any unhealthy checks
                for name, result in results.items():
                    if result.status == HealthStatus.UNHEALTHY:
                        self.logger.warning(f"Health check failed: {name} - {result.message}")
                    elif result.status == HealthStatus.DEGRADED:
                        self.logger.info(f"Health check degraded: {name} - {result.message}")
                
                # Wait before next check cycle
                monitoring_config = self.config.get("monitoring", {})
                interval = monitoring_config.get("interval", 30)
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor(config: Optional[Dict[str, Any]] = None) -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor(config)
    return _health_monitor