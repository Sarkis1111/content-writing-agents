"""
Research Crew Coordinator - Advanced task delegation and workflow management.

This module implements sophisticated coordination patterns for the Research Agent,
managing the interaction between specialized sub-agents and optimizing workflow execution.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from ...core.logging import get_framework_logger
from ...core.errors import AgentError, WorkflowError
from ...core.monitoring import get_metrics_collector, PerformanceTimer
from ...frameworks.crewai import get_crew_registry, get_agent_registry
from .research_agent import ResearchRequest, ResearchResponse


class TaskPriority(Enum):
    """Task priority levels for delegation."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned" 
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResearchTask:
    """Individual research task for delegation."""
    
    task_id: str
    agent_role: str
    description: str
    parameters: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    estimated_duration: Optional[int] = None  # seconds
    actual_duration: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time if task is completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return self.actual_duration
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies completed)."""
        return self.status == TaskStatus.PENDING and len(self.dependencies) == 0
    
    def mark_started(self):
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def mark_completed(self, result: Any):
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result
        if self.started_at:
            self.actual_duration = (self.completed_at - self.started_at).total_seconds()
    
    def mark_failed(self, error: str):
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error
        if self.started_at:
            self.actual_duration = (self.completed_at - self.started_at).total_seconds()


@dataclass
class WorkflowPlan:
    """Research workflow execution plan."""
    
    workflow_id: str
    request: ResearchRequest
    tasks: List[ResearchTask]
    execution_strategy: str  # "parallel", "sequential", "hybrid"
    created_at: datetime = field(default_factory=datetime.now)
    estimated_total_time: Optional[int] = None
    
    @property
    def ready_tasks(self) -> List[ResearchTask]:
        """Get tasks that are ready to execute."""
        return [task for task in self.tasks if task.is_ready]
    
    @property
    def active_tasks(self) -> List[ResearchTask]:
        """Get tasks currently in progress."""
        return [task for task in self.tasks if task.status == TaskStatus.IN_PROGRESS]
    
    @property
    def completed_tasks(self) -> List[ResearchTask]:
        """Get completed tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.COMPLETED]
    
    @property
    def failed_tasks(self) -> List[ResearchTask]:
        """Get failed tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.FAILED]
    
    @property
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] 
                  for task in self.tasks)
    
    @property
    def success_rate(self) -> float:
        """Calculate workflow success rate."""
        if not self.tasks:
            return 0.0
        completed = len(self.completed_tasks)
        total = len(self.tasks)
        return completed / total


class ResearchCoordinator:
    """
    Advanced coordinator for Research Agent task delegation and workflow management.
    
    This coordinator implements sophisticated patterns for managing multi-agent
    research workflows with the following capabilities:
    
    - Dynamic task decomposition based on research requirements
    - Intelligent agent assignment based on specialization and load
    - Parallel and sequential execution strategies
    - Dependency management and task ordering
    - Real-time workflow monitoring and adaptation
    - Performance optimization and resource balancing
    """
    
    def __init__(self):
        """Initialize the research coordinator."""
        
        self.logger = get_framework_logger("ResearchCoordinator")
        self.metrics = get_metrics_collector()
        
        # Framework components
        self.crew_registry = get_crew_registry()
        self.agent_registry = get_agent_registry()
        
        # Workflow management
        self.active_workflows: Dict[str, WorkflowPlan] = {}
        self.workflow_history: List[WorkflowPlan] = []
        
        # Agent performance tracking
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_concurrent_tasks = 5
        self.task_timeout = 300  # 5 minutes default
        self.workflow_timeout = 1800  # 30 minutes default
        
        self.logger.info("Research Coordinator initialized")
    
    async def coordinate_research(self, request: ResearchRequest) -> ResearchResponse:
        """
        Coordinate comprehensive research workflow.
        
        Args:
            request: Research request to coordinate
            
        Returns:
            Aggregated research response from all agents
        """
        
        workflow_id = f"workflow_{int(datetime.now().timestamp())}"
        
        self.logger.info(f"Starting research coordination: {workflow_id}")
        
        with PerformanceTimer() as timer:
            try:
                # Create workflow plan
                plan = self._create_workflow_plan(workflow_id, request)
                self.active_workflows[workflow_id] = plan
                
                # Execute workflow
                results = await self._execute_workflow(plan)
                
                # Aggregate results
                response = self._aggregate_results(request, workflow_id, results, timer.elapsed_time)
                
                # Archive workflow
                self._archive_workflow(plan)
                
                self.logger.info(f"Research coordination completed: {workflow_id}")
                self.metrics.record_timer("workflow_execution_time", timer.elapsed_time)
                
                return response
                
            except Exception as e:
                self.logger.error(f"Workflow coordination failed: {workflow_id} - {e}")
                self.metrics.record_counter("workflow_failed", error=str(e))
                
                # Clean up
                if workflow_id in self.active_workflows:
                    del self.active_workflows[workflow_id]
                
                raise WorkflowError(f"Research coordination failed: {e}")
    
    def _create_workflow_plan(self, workflow_id: str, request: ResearchRequest) -> WorkflowPlan:
        """Create detailed workflow execution plan."""
        
        tasks = []
        
        # Decompose research request into specialized tasks
        if request.research_depth in ["comprehensive", "deep"]:
            tasks.extend(self._create_comprehensive_tasks(request))
        elif request.research_depth == "standard":
            tasks.extend(self._create_standard_tasks(request))
        else:  # quick
            tasks.extend(self._create_quick_tasks(request))
        
        # Determine execution strategy
        strategy = self._determine_execution_strategy(request, tasks)
        
        # Set task dependencies
        self._set_task_dependencies(tasks, strategy)
        
        # Estimate execution time
        estimated_time = self._estimate_workflow_time(tasks, strategy)
        
        plan = WorkflowPlan(
            workflow_id=workflow_id,
            request=request,
            tasks=tasks,
            execution_strategy=strategy,
            estimated_total_time=estimated_time
        )
        
        self.logger.info(
            f"Created workflow plan: {workflow_id} with {len(tasks)} tasks, "
            f"strategy: {strategy}, estimated time: {estimated_time}s"
        )
        
        return plan
    
    def _create_comprehensive_tasks(self, request: ResearchRequest) -> List[ResearchTask]:
        """Create tasks for comprehensive research."""
        
        tasks = [
            # Primary web research
            ResearchTask(
                task_id="web_research_primary",
                agent_role="web_researcher",
                description=f"Conduct comprehensive web research on '{request.topic}'",
                parameters={
                    "topic": request.topic,
                    "max_sources": max(30, request.max_sources // 2),
                    "focus_areas": request.focus_areas,
                    "language": request.language,
                    "region": request.region
                },
                priority=TaskPriority.HIGH,
                estimated_duration=180
            ),
            
            # Trend analysis
            ResearchTask(
                task_id="trend_analysis",
                agent_role="trend_analyst",
                description=f"Analyze trends and patterns for '{request.topic}'",
                parameters={
                    "topic": request.topic,
                    "include_related": True,
                    "time_range": "12m",
                    "region": request.region
                },
                priority=TaskPriority.HIGH,
                estimated_duration=120
            ),
            
            # News analysis
            ResearchTask(
                task_id="news_analysis",
                agent_role="web_researcher",
                description=f"Gather recent news and updates on '{request.topic}'",
                parameters={
                    "topic": request.topic,
                    "time_range": "30d",
                    "source_types": ["news", "press_release"],
                    "max_articles": 20
                },
                priority=TaskPriority.NORMAL,
                estimated_duration=90
            ),
            
            # Content curation
            ResearchTask(
                task_id="content_curation",
                agent_role="content_curator",
                description=f"Curate and organize research findings for '{request.topic}'",
                parameters={
                    "topic": request.topic,
                    "organization_scheme": "thematic",
                    "include_summaries": True
                },
                priority=TaskPriority.NORMAL,
                estimated_duration=100,
                dependencies=["web_research_primary", "news_analysis"]
            )
        ]
        
        # Add fact-checking if requested
        if request.fact_check:
            tasks.append(
                ResearchTask(
                    task_id="fact_verification",
                    agent_role="fact_checker",
                    description=f"Verify key facts and claims about '{request.topic}'",
                    parameters={
                        "topic": request.topic,
                        "verification_level": "high",
                        "cross_reference": True
                    },
                    priority=TaskPriority.HIGH,
                    estimated_duration=150,
                    dependencies=["content_curation"]
                )
            )
        
        # Deep research additional tasks for "deep" mode
        if request.research_depth == "deep":
            tasks.extend([
                ResearchTask(
                    task_id="academic_research",
                    agent_role="web_researcher",
                    description=f"Find academic and scholarly sources on '{request.topic}'",
                    parameters={
                        "topic": request.topic,
                        "source_types": ["academic", "scholarly"],
                        "max_sources": 15
                    },
                    priority=TaskPriority.NORMAL,
                    estimated_duration=200
                ),
                
                ResearchTask(
                    task_id="competitive_analysis",
                    agent_role="web_researcher", 
                    description=f"Analyze competitive landscape for '{request.topic}'",
                    parameters={
                        "topic": request.topic,
                        "analysis_depth": "comprehensive",
                        "include_market_data": True
                    },
                    priority=TaskPriority.NORMAL,
                    estimated_duration=180
                )
            ])
        
        return tasks
    
    def _create_standard_tasks(self, request: ResearchRequest) -> List[ResearchTask]:
        """Create tasks for standard research."""
        
        tasks = [
            ResearchTask(
                task_id="web_research",
                agent_role="web_researcher",
                description=f"Research '{request.topic}' using web sources",
                parameters={
                    "topic": request.topic,
                    "max_sources": request.max_sources,
                    "language": request.language,
                    "region": request.region
                },
                priority=TaskPriority.HIGH,
                estimated_duration=120
            ),
            
            ResearchTask(
                task_id="content_organization",
                agent_role="content_curator",
                description=f"Organize research findings for '{request.topic}'",
                parameters={
                    "topic": request.topic,
                    "organization_scheme": "simple"
                },
                priority=TaskPriority.NORMAL,
                estimated_duration=60,
                dependencies=["web_research"]
            )
        ]
        
        if request.include_trends:
            tasks.append(
                ResearchTask(
                    task_id="basic_trends",
                    agent_role="trend_analyst",
                    description=f"Basic trend analysis for '{request.topic}'",
                    parameters={
                        "topic": request.topic,
                        "time_range": "6m"
                    },
                    priority=TaskPriority.NORMAL,
                    estimated_duration=60
                )
            )
        
        return tasks
    
    def _create_quick_tasks(self, request: ResearchRequest) -> List[ResearchTask]:
        """Create tasks for quick research."""
        
        return [
            ResearchTask(
                task_id="quick_research",
                agent_role="web_researcher", 
                description=f"Quick research overview of '{request.topic}'",
                parameters={
                    "topic": request.topic,
                    "max_sources": min(10, request.max_sources),
                    "speed_mode": True
                },
                priority=TaskPriority.CRITICAL,
                estimated_duration=45
            )
        ]
    
    def _determine_execution_strategy(self, request: ResearchRequest, tasks: List[ResearchTask]) -> str:
        """Determine optimal execution strategy based on request and tasks."""
        
        # Quick research always sequential due to speed requirements
        if request.research_depth == "quick":
            return "sequential"
        
        # If time is critical, prefer parallel
        if request.time_limit and request.time_limit < 300:  # Less than 5 minutes
            return "parallel"
        
        # Check for dependencies
        has_dependencies = any(task.dependencies for task in tasks)
        
        if has_dependencies:
            return "hybrid"  # Mix of parallel and sequential based on dependencies
        else:
            return "parallel" if len(tasks) > 2 else "sequential"
    
    def _set_task_dependencies(self, tasks: List[ResearchTask], strategy: str):
        """Set task dependencies based on execution strategy."""
        
        if strategy == "sequential":
            # Chain all tasks sequentially
            for i in range(1, len(tasks)):
                tasks[i].dependencies = [tasks[i-1].task_id]
        
        elif strategy == "hybrid":
            # Dependencies are already set in task creation
            # Just validate they make sense
            task_ids = {task.task_id for task in tasks}
            for task in tasks:
                invalid_deps = [dep for dep in task.dependencies if dep not in task_ids]
                if invalid_deps:
                    self.logger.warning(f"Invalid dependencies for {task.task_id}: {invalid_deps}")
                    task.dependencies = [dep for dep in task.dependencies if dep in task_ids]
        
        # Parallel strategy keeps dependencies as defined in task creation
    
    def _estimate_workflow_time(self, tasks: List[ResearchTask], strategy: str) -> int:
        """Estimate total workflow execution time."""
        
        if strategy == "sequential":
            return sum(task.estimated_duration or 60 for task in tasks)
        
        elif strategy == "parallel":
            return max(task.estimated_duration or 60 for task in tasks)
        
        else:  # hybrid
            # Simplified estimation - assumes optimal parallelization
            total_time = sum(task.estimated_duration or 60 for task in tasks)
            parallel_factor = min(len(tasks), self.max_concurrent_tasks)
            return int(total_time / parallel_factor * 1.3)  # 30% overhead
    
    async def _execute_workflow(self, plan: WorkflowPlan) -> Dict[str, Any]:
        """Execute workflow according to plan."""
        
        self.logger.info(f"Executing workflow: {plan.workflow_id} with strategy: {plan.execution_strategy}")
        
        if plan.execution_strategy == "sequential":
            return await self._execute_sequential(plan)
        elif plan.execution_strategy == "parallel": 
            return await self._execute_parallel(plan)
        else:  # hybrid
            return await self._execute_hybrid(plan)
    
    async def _execute_sequential(self, plan: WorkflowPlan) -> Dict[str, Any]:
        """Execute tasks sequentially."""
        
        results = {}
        
        for task in plan.tasks:
            try:
                self.logger.info(f"Starting sequential task: {task.task_id}")
                task.mark_started()
                
                result = await self._execute_single_task(task)
                task.mark_completed(result)
                results[task.task_id] = result
                
                # Update dependencies for remaining tasks
                self._resolve_dependencies(plan.tasks, task.task_id)
                
            except Exception as e:
                task.mark_failed(str(e))
                results[task.task_id] = {"error": str(e)}
                
                # Continue with remaining tasks in sequential mode
                self.logger.warning(f"Task failed but continuing: {task.task_id} - {e}")
        
        return results
    
    async def _execute_parallel(self, plan: WorkflowPlan) -> Dict[str, Any]:
        """Execute all tasks in parallel."""
        
        tasks_coroutines = []
        
        for task in plan.tasks:
            task.mark_started()
            coroutine = self._execute_task_with_error_handling(task)
            tasks_coroutines.append(coroutine)
        
        # Execute all tasks concurrently
        task_results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
        
        # Process results
        results = {}
        for task, result in zip(plan.tasks, task_results):
            if isinstance(result, Exception):
                task.mark_failed(str(result))
                results[task.task_id] = {"error": str(result)}
            else:
                task.mark_completed(result)
                results[task.task_id] = result
        
        return results
    
    async def _execute_hybrid(self, plan: WorkflowPlan) -> Dict[str, Any]:
        """Execute tasks with dependency-aware parallelization."""
        
        results = {}
        completed_tasks = set()
        
        while not plan.is_complete:
            # Get ready tasks (no pending dependencies)
            ready_tasks = [
                task for task in plan.tasks 
                if task.status == TaskStatus.PENDING and 
                all(dep in completed_tasks for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                # Check if we're stuck
                pending_tasks = [task for task in plan.tasks if task.status == TaskStatus.PENDING]
                if pending_tasks:
                    self.logger.error(f"Workflow stuck - pending tasks with unresolved dependencies")
                    # Force execute remaining tasks
                    ready_tasks = pending_tasks
                else:
                    break  # No more tasks to execute
            
            # Limit concurrent execution
            batch_size = min(len(ready_tasks), self.max_concurrent_tasks)
            batch_tasks = ready_tasks[:batch_size]
            
            # Execute batch
            batch_coroutines = []
            for task in batch_tasks:
                task.mark_started()
                coroutine = self._execute_task_with_error_handling(task)
                batch_coroutines.append(coroutine)
            
            batch_results = await asyncio.gather(*batch_coroutines, return_exceptions=True)
            
            # Process batch results
            for task, result in zip(batch_tasks, batch_results):
                if isinstance(result, Exception):
                    task.mark_failed(str(result))
                    results[task.task_id] = {"error": str(result)}
                else:
                    task.mark_completed(result)
                    results[task.task_id] = result
                
                completed_tasks.add(task.task_id)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        return results
    
    async def _execute_task_with_error_handling(self, task: ResearchTask):
        """Execute single task with comprehensive error handling."""
        
        try:
            return await self._execute_single_task(task)
        except Exception as e:
            self.logger.error(f"Task execution failed: {task.task_id} - {e}")
            raise e
    
    async def _execute_single_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute a single research task."""
        
        # This is a simplified task execution - in practice, this would
        # delegate to the appropriate agent/tool based on the task type
        
        self.logger.info(f"Executing task: {task.task_id} with agent: {task.agent_role}")
        
        # Simulate task execution with delay
        execution_time = task.estimated_duration or 30
        await asyncio.sleep(min(execution_time / 10, 3))  # Scaled down for demo
        
        # Mock result based on task type
        if "web_research" in task.task_id:
            return {
                "type": "web_research",
                "sources": [
                    f"https://example.com/source1_{task.task_id}",
                    f"https://example.com/source2_{task.task_id}"
                ],
                "findings": [
                    f"Key finding 1 for {task.parameters.get('topic', 'unknown topic')}",
                    f"Key finding 2 for {task.parameters.get('topic', 'unknown topic')}"
                ],
                "summary": f"Research summary for {task.parameters.get('topic', 'unknown topic')}"
            }
        
        elif "trend" in task.task_id:
            return {
                "type": "trend_analysis",
                "trends": [
                    {"keyword": task.parameters.get('topic', 'unknown'), "direction": "rising", "score": 0.8}
                ],
                "insights": [
                    f"Trend insight for {task.parameters.get('topic', 'unknown topic')}"
                ]
            }
        
        elif "content_curation" in task.task_id or "organization" in task.task_id:
            return {
                "type": "content_curation",
                "categories": {
                    "general": [f"Organized content for {task.parameters.get('topic', 'unknown topic')}"],
                    "key_points": ["Point 1", "Point 2"]
                },
                "summary": f"Curated content summary for {task.parameters.get('topic', 'unknown topic')}"
            }
        
        elif "fact" in task.task_id:
            return {
                "type": "fact_checking",
                "verified_facts": [
                    {"claim": f"Claim about {task.parameters.get('topic', 'unknown')}", "verified": True, "confidence": 0.9}
                ],
                "credibility_score": 0.85
            }
        
        else:
            return {
                "type": "generic_research",
                "data": f"Generic research result for {task.parameters.get('topic', 'unknown topic')}",
                "task_id": task.task_id
            }
    
    def _resolve_dependencies(self, tasks: List[ResearchTask], completed_task_id: str):
        """Remove completed task from other tasks' dependencies."""
        
        for task in tasks:
            if completed_task_id in task.dependencies:
                task.dependencies.remove(completed_task_id)
    
    def _aggregate_results(
        self, 
        request: ResearchRequest, 
        workflow_id: str,
        results: Dict[str, Any],
        execution_time: float
    ) -> ResearchResponse:
        """Aggregate task results into final research response."""
        
        # Combine all findings
        all_sources = []
        all_findings = []
        all_trends = []
        summaries = []
        
        for task_id, result in results.items():
            if "error" not in result:
                if result.get("type") == "web_research":
                    all_sources.extend(result.get("sources", []))
                    all_findings.extend(result.get("findings", []))
                    if result.get("summary"):
                        summaries.append(result["summary"])
                
                elif result.get("type") == "trend_analysis":
                    all_trends.extend(result.get("trends", []))
                
                elif result.get("type") == "content_curation":
                    if result.get("summary"):
                        summaries.append(result["summary"])
        
        # Create mock source objects
        from .research_agent import SourceInfo, TrendData
        
        sources = [
            SourceInfo(
                url=url,
                title=f"Research Source",
                credibility_score=0.8,
                relevance_score=0.9,
                content_type="article"
            ) for url in all_sources[:request.max_sources]
        ]
        
        trends = [
            TrendData(
                keyword=trend.get("keyword", "unknown"),
                trend_direction=trend.get("direction", "stable"),
                growth_rate=trend.get("score", 0.5),
                interest_score=trend.get("score", 0.5)
            ) for trend in all_trends
        ]
        
        # Create response
        response = ResearchResponse(
            topic=request.topic,
            request_id=workflow_id,
            execution_time=execution_time,
            status="completed",
            summary=" ".join(summaries) if summaries else f"Research completed for {request.topic}",
            key_findings=all_findings,
            sources=sources,
            trends=trends,
            agent_performance={
                "workflow_id": workflow_id,
                "total_tasks": len(results),
                "successful_tasks": len([r for r in results.values() if "error" not in r]),
                "execution_time": execution_time,
                "coordination_overhead": 0.1  # 10% overhead for coordination
            }
        )
        
        return response
    
    def _archive_workflow(self, plan: WorkflowPlan):
        """Archive completed workflow."""
        
        # Remove from active workflows
        if plan.workflow_id in self.active_workflows:
            del self.active_workflows[plan.workflow_id]
        
        # Add to history
        self.workflow_history.append(plan)
        
        # Keep only recent history (last 100 workflows)
        if len(self.workflow_history) > 100:
            self.workflow_history = self.workflow_history[-100:]
        
        # Update agent performance metrics
        self._update_agent_performance(plan)
        
        self.logger.info(f"Archived workflow: {plan.workflow_id}")
    
    def _update_agent_performance(self, plan: WorkflowPlan):
        """Update agent performance metrics from completed workflow."""
        
        for task in plan.tasks:
            agent_role = task.agent_role
            
            if agent_role not in self.agent_performance:
                self.agent_performance[agent_role] = {
                    "total_tasks": 0,
                    "successful_tasks": 0,
                    "total_execution_time": 0.0,
                    "avg_execution_time": 0.0
                }
            
            perf = self.agent_performance[agent_role]
            perf["total_tasks"] += 1
            
            if task.status == TaskStatus.COMPLETED:
                perf["successful_tasks"] += 1
                
                if task.execution_time:
                    perf["total_execution_time"] += task.execution_time
                    perf["avg_execution_time"] = (
                        perf["total_execution_time"] / perf["successful_tasks"]
                    )
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination performance statistics."""
        
        active_count = len(self.active_workflows)
        total_processed = len(self.workflow_history)
        
        return {
            "active_workflows": active_count,
            "total_processed": total_processed,
            "agent_performance": self.agent_performance,
            "avg_tasks_per_workflow": (
                sum(len(plan.tasks) for plan in self.workflow_history) / 
                max(total_processed, 1)
            ),
            "success_rate": (
                sum(plan.success_rate for plan in self.workflow_history) /
                max(total_processed, 1) if total_processed > 0 else 0.0
            )
        }