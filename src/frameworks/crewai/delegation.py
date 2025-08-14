"""CrewAI task delegation patterns and workflow management."""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

from ...core.logging import get_framework_logger
from ...core.errors import CrewAIError, WorkflowError
from ...core.monitoring import get_metrics_collector, get_health_monitor
from ...utils.retry import retry_async
from .config import get_crewai_framework
from .crews import get_crew_registry


class DelegationStrategy(Enum):
    """Task delegation strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    CONDITIONAL = "conditional"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class DelegatedTask:
    """Represents a delegated task."""
    
    task_id: str
    crew_id: str
    description: str
    inputs: Dict[str, Any]
    expected_output: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
    
    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries


@dataclass
class DelegationPlan:
    """Represents a task delegation plan."""
    
    plan_id: str
    strategy: DelegationStrategy
    tasks: List[DelegatedTask]
    workflow_variables: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_ready_tasks(self) -> List[DelegatedTask]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            if task.dependencies:
                dependencies_met = all(
                    self.get_task(dep_id).status == TaskStatus.COMPLETED 
                    for dep_id in task.dependencies
                )
                if not dependencies_met:
                    continue
            
            ready_tasks.append(task)
        
        return ready_tasks
    
    def get_task(self, task_id: str) -> Optional[DelegatedTask]:
        """Get a task by ID."""
        return next((task for task in self.tasks if task.task_id == task_id), None)
    
    def get_completed_tasks(self) -> List[DelegatedTask]:
        """Get all completed tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.COMPLETED]
    
    def get_failed_tasks(self) -> List[DelegatedTask]:
        """Get all failed tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.FAILED]
    
    @property
    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        return all(task.is_complete for task in self.tasks)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of completed tasks."""
        completed_tasks = [task for task in self.tasks if task.is_complete]
        if not completed_tasks:
            return 0.0
        
        successful_tasks = [task for task in completed_tasks if task.status == TaskStatus.COMPLETED]
        return len(successful_tasks) / len(completed_tasks)


class TaskDelegationManager:
    """Manages task delegation and execution for CrewAI."""
    
    def __init__(self):
        self.logger = get_framework_logger("CrewAI")
        self.metrics = get_metrics_collector()
        self.crew_registry = get_crew_registry()
        self.active_plans: Dict[str, DelegationPlan] = {}
        self.execution_history: List[DelegationPlan] = []
        self._stop_event = asyncio.Event()
    
    async def create_delegation_plan(
        self,
        plan_id: str,
        strategy: DelegationStrategy,
        workflow_config: Dict[str, Any]
    ) -> DelegationPlan:
        """Create a new task delegation plan."""
        
        try:
            tasks = []
            
            if strategy == DelegationStrategy.SEQUENTIAL:
                tasks = self._create_sequential_plan(workflow_config)
            elif strategy == DelegationStrategy.PARALLEL:
                tasks = self._create_parallel_plan(workflow_config)
            elif strategy == DelegationStrategy.HIERARCHICAL:
                tasks = self._create_hierarchical_plan(workflow_config)
            elif strategy == DelegationStrategy.CONDITIONAL:
                tasks = self._create_conditional_plan(workflow_config)
            else:
                raise CrewAIError(f"Unsupported delegation strategy: {strategy}")
            
            plan = DelegationPlan(
                plan_id=plan_id,
                strategy=strategy,
                tasks=tasks,
                workflow_variables=workflow_config.get("variables", {}),
                timeout=workflow_config.get("timeout")
            )
            
            self.active_plans[plan_id] = plan
            self.logger.info(f"Created delegation plan: {plan_id} with {len(tasks)} tasks")
            self.metrics.record_counter(
                "delegation_plan_created",
                framework="crewai",
                strategy=strategy.value,
                task_count=str(len(tasks))
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to create delegation plan {plan_id}: {e}")
            raise CrewAIError(f"Failed to create delegation plan: {e}")
    
    def _create_sequential_plan(self, config: Dict[str, Any]) -> List[DelegatedTask]:
        """Create tasks for sequential execution."""
        tasks = []
        workflows = config.get("workflows", [])
        
        prev_task_id = None
        for i, workflow in enumerate(workflows):
            task_id = f"task_{i}"
            task = DelegatedTask(
                task_id=task_id,
                crew_id=workflow["crew"],
                description=workflow.get("description", f"Execute workflow step {i}"),
                inputs=workflow.get("inputs", {}),
                expected_output=workflow.get("expected_output", "Workflow step completed"),
                dependencies=[prev_task_id] if prev_task_id else [],
                metadata=workflow.get("metadata", {})
            )
            tasks.append(task)
            prev_task_id = task_id
        
        return tasks
    
    def _create_parallel_plan(self, config: Dict[str, Any]) -> List[DelegatedTask]:
        """Create tasks for parallel execution."""
        tasks = []
        workflows = config.get("workflows", [])
        
        for i, workflow in enumerate(workflows):
            task_id = f"task_{i}"
            task = DelegatedTask(
                task_id=task_id,
                crew_id=workflow["crew"],
                description=workflow.get("description", f"Execute workflow step {i}"),
                inputs=workflow.get("inputs", {}),
                expected_output=workflow.get("expected_output", "Workflow step completed"),
                metadata=workflow.get("metadata", {})
            )
            tasks.append(task)
        
        return tasks
    
    def _create_hierarchical_plan(self, config: Dict[str, Any]) -> List[DelegatedTask]:
        """Create tasks for hierarchical execution."""
        tasks = []
        
        # Create coordination task first
        coord_task = DelegatedTask(
            task_id="coordination",
            crew_id="meta_coordination_crew",
            description="Coordinate hierarchical workflow execution",
            inputs=config.get("coordination_inputs", {}),
            expected_output="Coordination plan and task assignments"
        )
        tasks.append(coord_task)
        
        # Create worker tasks that depend on coordination
        workflows = config.get("workflows", [])
        for i, workflow in enumerate(workflows):
            task_id = f"worker_task_{i}"
            task = DelegatedTask(
                task_id=task_id,
                crew_id=workflow["crew"],
                description=workflow.get("description", f"Execute worker task {i}"),
                inputs=workflow.get("inputs", {}),
                expected_output=workflow.get("expected_output", "Worker task completed"),
                dependencies=["coordination"],
                metadata=workflow.get("metadata", {})
            )
            tasks.append(task)
        
        return tasks
    
    def _create_conditional_plan(self, config: Dict[str, Any]) -> List[DelegatedTask]:
        """Create tasks for conditional execution."""
        tasks = []
        
        # Create initial task
        initial_task = DelegatedTask(
            task_id="initial",
            crew_id=config["initial_crew"],
            description=config.get("initial_description", "Execute initial task"),
            inputs=config.get("initial_inputs", {}),
            expected_output=config.get("initial_output", "Initial task completed")
        )
        tasks.append(initial_task)
        
        # Create conditional branches
        conditions = config.get("conditions", {})
        for condition_name, condition_config in conditions.items():
            task_id = f"conditional_{condition_name}"
            task = DelegatedTask(
                task_id=task_id,
                crew_id=condition_config["crew"],
                description=condition_config.get("description", f"Execute {condition_name} branch"),
                inputs=condition_config.get("inputs", {}),
                expected_output=condition_config.get("expected_output", f"{condition_name} completed"),
                dependencies=["initial"],
                metadata={"condition": condition_name}
            )
            tasks.append(task)
        
        return tasks
    
    @retry_async(max_attempts=3, initial_delay=1.0)
    async def execute_task(self, task: DelegatedTask) -> Any:
        """Execute a single delegated task."""
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            # Get CrewAI framework
            framework = await get_crewai_framework()
            
            # Get crew instance
            crew = self.crew_registry.create_crew_instance(
                task.crew_id,
                framework.get_llm(),
                variables=task.inputs
            )
            
            # Execute crew
            self.logger.info(f"Executing task {task.task_id} with crew {task.crew_id}")
            
            with self.metrics.timer("task_execution", task_id=task.task_id, crew=task.crew_id):
                result = await asyncio.get_event_loop().run_in_executor(
                    None, crew.kickoff
                )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            self.logger.info(f"Task {task.task_id} completed successfully")
            self.metrics.record_counter(
                "task_completed",
                framework="crewai",
                crew=task.crew_id,
                success="true"
            )
            
            return result
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
            self.metrics.record_counter(
                "task_completed",
                framework="crewai",
                crew=task.crew_id,
                success="false"
            )
            
            raise CrewAIError(f"Task execution failed: {e}")
    
    async def execute_plan(self, plan_id: str) -> DelegationPlan:
        """Execute a delegation plan."""
        
        plan = self.active_plans.get(plan_id)
        if not plan:
            raise CrewAIError(f"Delegation plan not found: {plan_id}")
        
        try:
            self.logger.info(f"Starting execution of delegation plan: {plan_id}")
            
            workflow_id = self.metrics.start_workflow(plan_id, "crewai")
            
            if plan.strategy == DelegationStrategy.PARALLEL:
                await self._execute_parallel(plan)
            else:
                await self._execute_sequential(plan)
            
            self.metrics.complete_workflow(plan_id, "completed")
            
            # Move to history
            self.execution_history.append(plan)
            del self.active_plans[plan_id]
            
            self.logger.info(f"Delegation plan {plan_id} completed with {plan.success_rate:.2%} success rate")
            
            return plan
            
        except Exception as e:
            self.metrics.complete_workflow(plan_id, "failed")
            self.logger.error(f"Delegation plan {plan_id} failed: {e}")
            raise CrewAIError(f"Plan execution failed: {e}")
    
    async def _execute_sequential(self, plan: DelegationPlan):
        """Execute plan sequentially."""
        while not plan.is_complete:
            ready_tasks = plan.get_ready_tasks()
            
            if not ready_tasks:
                # Check if we're stuck (no ready tasks but not complete)
                if not plan.is_complete:
                    failed_tasks = plan.get_failed_tasks()
                    if failed_tasks:
                        # Try to retry failed tasks
                        retry_tasks = [task for task in failed_tasks if task.can_retry]
                        if retry_tasks:
                            for task in retry_tasks:
                                task.retry_count += 1
                                task.status = TaskStatus.PENDING
                                task.error = None
                        else:
                            raise CrewAIError("Workflow stuck: no ready tasks and no retryable failures")
                    else:
                        raise CrewAIError("Workflow stuck: no ready tasks available")
                break
            
            # Execute first ready task
            task = ready_tasks[0]
            task.assigned_to = f"crew_{task.crew_id}"
            
            await self.execute_task(task)
    
    async def _execute_parallel(self, plan: DelegationPlan):
        """Execute plan in parallel."""
        while not plan.is_complete:
            ready_tasks = plan.get_ready_tasks()
            
            if not ready_tasks:
                break
            
            # Execute all ready tasks in parallel
            execution_tasks = []
            for task in ready_tasks:
                task.assigned_to = f"crew_{task.crew_id}"
                execution_tasks.append(self.execute_task(task))
            
            # Wait for all tasks to complete
            await asyncio.gather(*execution_tasks, return_exceptions=True)
    
    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a delegation plan."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            # Check history
            historical_plan = next(
                (p for p in self.execution_history if p.plan_id == plan_id), 
                None
            )
            if historical_plan:
                plan = historical_plan
            else:
                return None
        
        return {
            "plan_id": plan.plan_id,
            "strategy": plan.strategy.value,
            "total_tasks": len(plan.tasks),
            "completed_tasks": len(plan.get_completed_tasks()),
            "failed_tasks": len(plan.get_failed_tasks()),
            "success_rate": plan.success_rate,
            "is_complete": plan.is_complete,
            "created_at": plan.created_at.isoformat(),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "duration": task.duration,
                    "retry_count": task.retry_count
                }
                for task in plan.tasks
            ]
        }


# Global task delegation manager instance
_delegation_manager: Optional[TaskDelegationManager] = None


def get_delegation_manager() -> TaskDelegationManager:
    """Get the global task delegation manager instance."""
    global _delegation_manager
    if _delegation_manager is None:
        _delegation_manager = TaskDelegationManager()
    return _delegation_manager