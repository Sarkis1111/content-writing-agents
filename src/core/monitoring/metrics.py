"""Monitoring and metrics collection for agentic workflows."""

import time
import threading
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from ..logging import get_component_logger


@dataclass
class MetricData:
    """Container for metric data points."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    framework: Optional[str] = None
    agent: Optional[str] = None
    tool: Optional[str] = None


@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution."""
    workflow_id: str
    framework: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    agent_count: int = 0
    tool_calls: int = 0
    errors: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    execution_steps: list = field(default_factory=list)


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics_collector: 'MetricsCollector', metric_name: str, **tags):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timing(
                self.metric_name, 
                duration, 
                **self.tags
            )


class MetricsCollector:
    """Central metrics collection system."""
    
    def __init__(self, max_metrics_history: int = 10000):
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.workflow_metrics: Dict[str, WorkflowMetrics] = {}
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
        self.logger = get_component_logger("metrics")
    
    def record_counter(self, name: str, value: float = 1.0, **tags):
        """Record a counter metric."""
        with self.lock:
            key = f"{name}:{json.dumps(tags, sort_keys=True)}"
            self.counters[key] += value
            
            metric = MetricData(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags
            )
            self.metrics_history.append(metric)
            self.logger.debug(f"Counter metric recorded: {name}={value}, tags={tags}")
    
    def record_gauge(self, name: str, value: float, **tags):
        """Record a gauge metric."""
        with self.lock:
            key = f"{name}:{json.dumps(tags, sort_keys=True)}"
            self.gauges[key] = value
            
            metric = MetricData(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags
            )
            self.metrics_history.append(metric)
            self.logger.debug(f"Gauge metric recorded: {name}={value}, tags={tags}")
    
    def record_timing(self, name: str, duration: float, **tags):
        """Record a timing metric."""
        with self.lock:
            key = f"{name}:{json.dumps(tags, sort_keys=True)}"
            self.histograms[key].append(duration)
            
            metric = MetricData(
                name=name,
                value=duration,
                timestamp=datetime.now(),
                tags=tags
            )
            self.metrics_history.append(metric)
            self.logger.debug(f"Timing metric recorded: {name}={duration:.3f}s, tags={tags}")
    
    def timer(self, metric_name: str, **tags) -> PerformanceTimer:
        """Create a performance timer context manager."""
        return PerformanceTimer(self, metric_name, **tags)
    
    def start_workflow(self, workflow_id: str, framework: str) -> WorkflowMetrics:
        """Start tracking a workflow."""
        with self.lock:
            workflow = WorkflowMetrics(
                workflow_id=workflow_id,
                framework=framework,
                start_time=datetime.now()
            )
            self.workflow_metrics[workflow_id] = workflow
            self.record_counter("workflow_started", framework=framework)
            self.logger.info(f"Workflow started: {workflow_id} ({framework})")
            return workflow
    
    def complete_workflow(self, workflow_id: str, status: str = "completed"):
        """Complete workflow tracking."""
        with self.lock:
            if workflow_id in self.workflow_metrics:
                workflow = self.workflow_metrics[workflow_id]
                workflow.end_time = datetime.now()
                workflow.status = status
                
                if workflow.start_time:
                    duration = (workflow.end_time - workflow.start_time).total_seconds()
                    self.record_timing(
                        "workflow_duration",
                        duration,
                        framework=workflow.framework,
                        status=status
                    )
                
                self.record_counter(
                    f"workflow_{status}",
                    framework=workflow.framework
                )
                
                self.logger.info(f"Workflow {status}: {workflow_id} ({workflow.framework})")
    
    def record_agent_action(self, workflow_id: str, agent_name: str, action: str, **metadata):
        """Record an agent action within a workflow."""
        with self.lock:
            if workflow_id in self.workflow_metrics:
                workflow = self.workflow_metrics[workflow_id]
                workflow.execution_steps.append({
                    "timestamp": datetime.now().isoformat(),
                    "agent": agent_name,
                    "action": action,
                    "metadata": metadata
                })
                
                self.record_counter(
                    "agent_action",
                    framework=workflow.framework,
                    agent=agent_name,
                    action=action
                )
    
    def record_tool_usage(self, workflow_id: str, tool_name: str, success: bool = True, **metadata):
        """Record tool usage within a workflow."""
        with self.lock:
            if workflow_id in self.workflow_metrics:
                workflow = self.workflow_metrics[workflow_id]
                workflow.tool_calls += 1
                if not success:
                    workflow.errors += 1
                
                self.record_counter(
                    "tool_usage",
                    framework=workflow.framework,
                    tool=tool_name,
                    success=str(success)
                )
    
    def record_token_usage(self, workflow_id: str, tokens: int, cost: float = 0.0):
        """Record token usage and cost."""
        with self.lock:
            if workflow_id in self.workflow_metrics:
                workflow = self.workflow_metrics[workflow_id]
                workflow.total_tokens += tokens
                workflow.total_cost += cost
                
                self.record_counter("tokens_used", value=tokens)
                self.record_counter("api_cost", value=cost)
    
    def get_workflow_stats(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific workflow."""
        with self.lock:
            if workflow_id not in self.workflow_metrics:
                return None
            
            workflow = self.workflow_metrics[workflow_id]
            stats = {
                "workflow_id": workflow.workflow_id,
                "framework": workflow.framework,
                "status": workflow.status,
                "start_time": workflow.start_time.isoformat(),
                "agent_count": workflow.agent_count,
                "tool_calls": workflow.tool_calls,
                "errors": workflow.errors,
                "total_tokens": workflow.total_tokens,
                "total_cost": workflow.total_cost,
                "execution_steps": len(workflow.execution_steps)
            }
            
            if workflow.end_time:
                stats["end_time"] = workflow.end_time.isoformat()
                stats["duration"] = (workflow.end_time - workflow.start_time).total_seconds()
            
            return stats
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        with self.lock:
            active_workflows = sum(1 for w in self.workflow_metrics.values() if w.status == "running")
            total_workflows = len(self.workflow_metrics)
            
            # Calculate average response times per framework
            framework_stats = defaultdict(list)
            for key, times in self.histograms.items():
                if "workflow_duration" in key and times:
                    try:
                        tags = json.loads(key.split(":", 1)[1])
                        framework = tags.get("framework", "unknown")
                        framework_stats[framework].extend(times)
                    except (json.JSONDecodeError, IndexError):
                        continue
            
            avg_response_times = {}
            for framework, times in framework_stats.items():
                if times:
                    avg_response_times[framework] = sum(times) / len(times)
            
            return {
                "active_workflows": active_workflows,
                "total_workflows": total_workflows,
                "total_metrics_collected": len(self.metrics_history),
                "average_response_times": avg_response_times,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges)
            }
    
    def get_framework_stats(self, framework: str) -> Dict[str, Any]:
        """Get statistics for a specific framework."""
        with self.lock:
            framework_workflows = [
                w for w in self.workflow_metrics.values() 
                if w.framework == framework
            ]
            
            total_workflows = len(framework_workflows)
            active_workflows = sum(1 for w in framework_workflows if w.status == "running")
            completed_workflows = sum(1 for w in framework_workflows if w.status == "completed")
            failed_workflows = sum(1 for w in framework_workflows if w.status == "failed")
            
            total_tokens = sum(w.total_tokens for w in framework_workflows)
            total_cost = sum(w.total_cost for w in framework_workflows)
            total_errors = sum(w.errors for w in framework_workflows)
            
            return {
                "framework": framework,
                "total_workflows": total_workflows,
                "active_workflows": active_workflows,
                "completed_workflows": completed_workflows,
                "failed_workflows": failed_workflows,
                "success_rate": completed_workflows / total_workflows if total_workflows > 0 else 0,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "total_errors": total_errors
            }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector