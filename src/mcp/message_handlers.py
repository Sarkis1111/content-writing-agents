"""
Message Handling System for Agentic Workflows

This module implements sophisticated message handling patterns for coordinating
multi-framework agentic workflows. It provides message routing, transformation,
and orchestration capabilities for CrewAI, LangGraph, and AutoGen interactions.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from datetime import datetime, timedelta
import logging

from ..core.monitoring.metrics import MetricsCollector
from ..core.errors.exceptions import MessageHandlingError, WorkflowError
from ..core.errors.handlers import handle_async_errors
from .server import MCPMessage, MessageType, FrameworkType


class WorkflowStatus(str, Enum):
    """Workflow execution status states"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessagePriority(str, Enum):
    """Message priority levels for queue management"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class WorkflowContext:
    """Context information for workflow execution"""
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    frameworks_involved: List[FrameworkType] = field(default_factory=list)
    current_step: Optional[str] = None
    step_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    timeout: Optional[timedelta] = None
    
    def update_status(self, status: WorkflowStatus, step: Optional[str] = None):
        """Update workflow status and current step"""
        self.status = status
        if step:
            self.current_step = step
        self.updated_at = datetime.utcnow()


@dataclass
class MessageRoute:
    """Message routing information"""
    message_id: str
    source_framework: Optional[FrameworkType]
    target_framework: FrameworkType
    action: str
    parameters: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timeout: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None


class MessageHandler(Protocol):
    """Protocol for message handlers"""
    
    async def handle(self, message: MCPMessage, context: Optional[WorkflowContext] = None) -> MCPMessage:
        """Handle a message and return response"""
        ...


class MessageRouter:
    """Routes messages to appropriate handlers based on content and context"""
    
    def __init__(self, logger: logging.Logger, metrics: MetricsCollector):
        self.logger = logger
        self.metrics = metrics
        self.handlers: Dict[str, MessageHandler] = {}
        self.route_patterns: Dict[str, str] = {}
        
    def register_handler(self, message_type: str, handler: MessageHandler):
        """Register a handler for specific message types"""
        self.handlers[message_type] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")
        
    def register_route_pattern(self, pattern: str, handler_name: str):
        """Register routing patterns for message classification"""
        self.route_patterns[pattern] = handler_name
        
    async def route_message(self, message: MCPMessage, context: Optional[WorkflowContext] = None) -> MCPMessage:
        """Route message to appropriate handler"""
        handler_key = self._determine_handler(message)
        
        if handler_key not in self.handlers:
            error_msg = f"No handler found for message type: {handler_key}"
            self.logger.error(error_msg)
            self.metrics.increment_counter("message_routing.handler_not_found")
            raise MessageHandlingError(error_msg, message_id=message.message_id)
        
        handler = self.handlers[handler_key]
        
        # Record routing metrics
        self.metrics.increment_counter(f"message_routing.{handler_key}")
        
        with self.metrics.timer(f"message_handling.{handler_key}.duration"):
            try:
                response = await handler.handle(message, context)
                self.metrics.increment_counter(f"message_handling.{handler_key}.success")
                return response
            except Exception as e:
                self.metrics.increment_counter(f"message_handling.{handler_key}.error")
                raise
    
    def _determine_handler(self, message: MCPMessage) -> str:
        """Determine which handler should process the message"""
        # Primary routing by message type
        if message.message_type:
            return message.message_type.value
            
        # Fallback to payload analysis
        if message.payload:
            for pattern, handler_name in self.route_patterns.items():
                if pattern in str(message.payload):
                    return handler_name
                    
        return "default"


class WorkflowOrchestrator:
    """Orchestrates complex multi-framework workflows"""
    
    def __init__(self, logger: logging.Logger, metrics: MetricsCollector):
        self.logger = logger
        self.metrics = metrics
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.framework_layer = None  # Will be injected
        
    def register_workflow_template(self, name: str, template: Dict[str, Any]):
        """Register a workflow template for reuse"""
        self.workflow_templates[name] = template
        self.logger.info(f"Registered workflow template: {name}")
        
    async def start_workflow(self, workflow_name: str, parameters: Dict[str, Any], 
                           workflow_id: Optional[str] = None) -> str:
        """Start a new workflow execution"""
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())
            
        if workflow_name not in self.workflow_templates:
            raise WorkflowError(f"Unknown workflow template: {workflow_name}")
            
        template = self.workflow_templates[workflow_name]
        
        # Create workflow context
        context = WorkflowContext(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            frameworks_involved=[FrameworkType(fw) for fw in template.get("frameworks_used", [])],
            metadata=parameters,
            timeout=timedelta(seconds=template.get("timeout", 3600))  # Default 1 hour
        )
        
        self.active_workflows[workflow_id] = context
        self.message_queues[workflow_id] = asyncio.Queue()
        
        self.logger.info(f"Started workflow {workflow_name} with ID {workflow_id}")
        self.metrics.increment_counter(f"workflow.{workflow_name}.started")
        
        # Execute workflow asynchronously
        asyncio.create_task(self._execute_workflow(context, template))
        
        return workflow_id
        
    async def _execute_workflow(self, context: WorkflowContext, template: Dict[str, Any]):
        """Execute a workflow according to its template"""
        try:
            context.update_status(WorkflowStatus.RUNNING)
            steps = template.get("steps", [])
            
            for i, step in enumerate(steps):
                await self._execute_workflow_step(context, step, i)
                
                # Check for workflow cancellation or timeout
                if context.status == WorkflowStatus.CANCELLED:
                    self.logger.info(f"Workflow {context.workflow_id} was cancelled")
                    break
                    
                if self._is_workflow_timeout(context):
                    context.update_status(WorkflowStatus.FAILED)
                    self.logger.error(f"Workflow {context.workflow_id} timed out")
                    break
                    
            # Mark workflow as completed if all steps succeeded
            if context.status == WorkflowStatus.RUNNING:
                context.update_status(WorkflowStatus.COMPLETED)
                self.logger.info(f"Workflow {context.workflow_id} completed successfully")
                self.metrics.increment_counter(f"workflow.{context.workflow_name}.completed")
                
        except Exception as e:
            context.update_status(WorkflowStatus.FAILED)
            self.logger.error(f"Workflow {context.workflow_id} failed: {e}")
            self.metrics.increment_counter(f"workflow.{context.workflow_name}.failed")
            
        finally:
            # Cleanup
            if context.workflow_id in self.message_queues:
                del self.message_queues[context.workflow_id]
                
    async def _execute_workflow_step(self, context: WorkflowContext, step: Dict[str, Any], step_index: int):
        """Execute a single workflow step"""
        step_name = step.get("action", f"step_{step_index}")
        framework = FrameworkType(step["framework"])
        
        self.logger.info(f"Executing step {step_name} in workflow {context.workflow_id}")
        context.update_status(WorkflowStatus.RUNNING, step_name)
        
        try:
            # Prepare step parameters
            parameters = step.get("inputs", {})
            # Inject data from previous steps
            for key, value in context.step_data.items():
                if isinstance(parameters, dict) and f"{{{key}}}" in str(parameters):
                    # Simple template substitution
                    parameters = json.loads(json.dumps(parameters).replace(f"{{{key}}}", str(value)))
                    
            # Execute the step via framework layer
            if self.framework_layer:
                result = await self.framework_layer.execute_framework_action(
                    framework, step_name, parameters
                )
                
                # Store step results for next steps
                outputs = step.get("outputs", [])
                for output_key in outputs:
                    if output_key in result:
                        context.step_data[output_key] = result[output_key]
                        
                self.logger.info(f"Step {step_name} completed successfully")
                self.metrics.increment_counter(f"workflow_step.{step_name}.success")
                
        except Exception as e:
            self.logger.error(f"Step {step_name} failed in workflow {context.workflow_id}: {e}")
            self.metrics.increment_counter(f"workflow_step.{step_name}.error")
            context.update_status(WorkflowStatus.FAILED)
            raise
            
    def _is_workflow_timeout(self, context: WorkflowContext) -> bool:
        """Check if workflow has exceeded timeout"""
        if not context.timeout:
            return False
        return datetime.utcnow() - context.created_at > context.timeout
        
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Get current status of a workflow"""
        return self.active_workflows.get(workflow_id)
        
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id in self.active_workflows:
            context = self.active_workflows[workflow_id]
            context.update_status(WorkflowStatus.CANCELLED)
            self.logger.info(f"Cancelled workflow {workflow_id}")
            return True
        return False


class FrameworkBridge:
    """Bridges communication between different agentic frameworks"""
    
    def __init__(self, logger: logging.Logger, metrics: MetricsCollector):
        self.logger = logger
        self.metrics = metrics
        self.translation_rules: Dict[str, Dict[str, Any]] = {}
        
    def register_translation_rule(self, source_framework: FrameworkType, 
                                target_framework: FrameworkType, 
                                rule: Dict[str, Any]):
        """Register message translation rules between frameworks"""
        key = f"{source_framework.value}_to_{target_framework.value}"
        self.translation_rules[key] = rule
        self.logger.debug(f"Registered translation rule: {key}")
        
    async def translate_message(self, message: MCPMessage, 
                              target_framework: FrameworkType) -> MCPMessage:
        """Translate message format between frameworks"""
        if not message.framework:
            # No translation needed if source framework not specified
            message.framework = target_framework
            return message
            
        if message.framework == target_framework:
            # No translation needed for same framework
            return message
            
        rule_key = f"{message.framework.value}_to_{target_framework.value}"
        
        if rule_key not in self.translation_rules:
            self.logger.warning(f"No translation rule found for {rule_key}")
            # Return message with updated framework
            message.framework = target_framework
            return message
            
        rule = self.translation_rules[rule_key]
        translated_payload = await self._apply_translation_rule(message.payload, rule)
        
        translated_message = MCPMessage(
            message_id=message.message_id,
            message_type=message.message_type,
            framework=target_framework,
            payload=translated_payload,
            metadata=message.metadata
        )
        
        self.metrics.increment_counter(f"message_translation.{rule_key}")
        return translated_message
        
    async def _apply_translation_rule(self, payload: Dict[str, Any], 
                                    rule: Dict[str, Any]) -> Dict[str, Any]:
        """Apply translation rule to message payload"""
        # Simple field mapping - can be extended for complex transformations
        field_mapping = rule.get("field_mapping", {})
        translated = {}
        
        for source_field, target_field in field_mapping.items():
            if source_field in payload:
                translated[target_field] = payload[source_field]
                
        # Copy unmapped fields
        for key, value in payload.items():
            if key not in field_mapping and key not in translated:
                translated[key] = value
                
        return translated


class MessageQueue:
    """Priority-based message queue for workflow coordination"""
    
    def __init__(self, name: str, logger: logging.Logger, metrics: MetricsCollector):
        self.name = name
        self.logger = logger
        self.metrics = metrics
        self.queues = {
            MessagePriority.CRITICAL: asyncio.Queue(),
            MessagePriority.HIGH: asyncio.Queue(),
            MessagePriority.NORMAL: asyncio.Queue(),
            MessagePriority.LOW: asyncio.Queue()
        }
        self.active = True
        
    async def enqueue(self, route: MessageRoute):
        """Add message route to appropriate priority queue"""
        queue = self.queues[route.priority]
        await queue.put(route)
        
        self.logger.debug(f"Enqueued message {route.message_id} with priority {route.priority.value}")
        self.metrics.increment_counter(f"message_queue.{self.name}.enqueued.{route.priority.value}")
        
    async def dequeue(self) -> Optional[MessageRoute]:
        """Get next message route based on priority"""
        if not self.active:
            return None
            
        # Check queues in priority order
        for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                        MessagePriority.NORMAL, MessagePriority.LOW]:
            queue = self.queues[priority]
            if not queue.empty():
                try:
                    route = await asyncio.wait_for(queue.get(), timeout=0.1)
                    self.metrics.increment_counter(f"message_queue.{self.name}.dequeued.{priority.value}")
                    return route
                except asyncio.TimeoutError:
                    continue
                    
        return None
        
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current size of each priority queue"""
        return {
            priority.value: queue.qsize() 
            for priority, queue in self.queues.items()
        }
        
    async def shutdown(self):
        """Shutdown the message queue"""
        self.active = False


# Concrete message handlers

class InitializeHandler:
    """Handles framework initialization messages"""
    
    def __init__(self, framework_layer, logger: logging.Logger):
        self.framework_layer = framework_layer
        self.logger = logger
        
    async def handle(self, message: MCPMessage, context: Optional[WorkflowContext] = None) -> MCPMessage:
        """Handle initialization request"""
        framework = message.framework
        if not framework:
            return MCPMessage(
                message_id=f"response_{message.message_id}",
                message_type=MessageType.ERROR,
                payload={"error": "Framework not specified"}
            )
            
        try:
            success = await self.framework_layer.initialize_framework(framework)
            return MCPMessage(
                message_id=f"response_{message.message_id}",
                message_type=MessageType.RESPONSE,
                framework=framework,
                payload={
                    "initialized": success,
                    "framework": framework.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            self.logger.error(f"Framework initialization failed: {e}")
            return MCPMessage(
                message_id=f"response_{message.message_id}",
                message_type=MessageType.ERROR,
                framework=framework,
                payload={"error": str(e)}
            )


class WorkflowHandler:
    """Handles workflow execution messages"""
    
    def __init__(self, orchestrator: WorkflowOrchestrator, logger: logging.Logger):
        self.orchestrator = orchestrator
        self.logger = logger
        
    @handle_async_errors
    async def handle(self, message: MCPMessage, context: Optional[WorkflowContext] = None) -> MCPMessage:
        """Handle workflow-related messages"""
        if message.message_type == MessageType.WORKFLOW_START:
            return await self._handle_workflow_start(message)
        else:
            return MCPMessage(
                message_id=f"response_{message.message_id}",
                message_type=MessageType.ERROR,
                payload={"error": f"Unsupported workflow message type: {message.message_type}"}
            )
            
    async def _handle_workflow_start(self, message: MCPMessage) -> MCPMessage:
        """Handle workflow start request"""
        payload = message.payload or {}
        workflow_name = payload.get("workflow_name")
        parameters = payload.get("parameters", {})
        
        if not workflow_name:
            return MCPMessage(
                message_id=f"response_{message.message_id}",
                message_type=MessageType.ERROR,
                payload={"error": "Workflow name not specified"}
            )
            
        try:
            workflow_id = await self.orchestrator.start_workflow(workflow_name, parameters)
            return MCPMessage(
                message_id=f"response_{message.message_id}",
                message_type=MessageType.RESPONSE,
                payload={
                    "workflow_id": workflow_id,
                    "status": "started",
                    "workflow_name": workflow_name
                }
            )
        except Exception as e:
            return MCPMessage(
                message_id=f"response_{message.message_id}",
                message_type=MessageType.ERROR,
                payload={"error": f"Failed to start workflow: {e}"}
            )