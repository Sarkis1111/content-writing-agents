"""Custom exception classes for the agentic system."""

from typing import Optional, Dict, Any
from datetime import datetime


class AgenticSystemError(Exception):
    """Base exception class for all agentic system errors."""
    
    def __init__(
        self,
        message: str,
        framework: Optional[str] = None,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.framework = framework
        self.agent = agent
        self.tool = tool
        self.workflow_id = workflow_id
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "framework": self.framework,
            "agent": self.agent,
            "tool": self.tool,
            "workflow_id": self.workflow_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class FrameworkError(AgenticSystemError):
    """Base class for framework-specific errors."""
    pass


class CrewAIError(FrameworkError):
    """CrewAI-specific errors."""
    
    def __init__(
        self,
        message: str,
        crew_name: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, framework="crewai", **kwargs)
        self.crew_name = crew_name
        self.task_name = task_name
        if crew_name:
            self.metadata["crew_name"] = crew_name
        if task_name:
            self.metadata["task_name"] = task_name


class LangGraphError(FrameworkError):
    """LangGraph-specific errors."""
    
    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        node_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, framework="langgraph", **kwargs)
        self.workflow_name = workflow_name
        self.node_name = node_name
        if workflow_name:
            self.metadata["workflow_name"] = workflow_name
        if node_name:
            self.metadata["node_name"] = node_name


class AutoGenError(FrameworkError):
    """AutoGen-specific errors."""
    
    def __init__(
        self,
        message: str,
        group_chat_name: Optional[str] = None,
        round_number: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, framework="autogen", **kwargs)
        self.group_chat_name = group_chat_name
        self.round_number = round_number
        if group_chat_name:
            self.metadata["group_chat_name"] = group_chat_name
        if round_number:
            self.metadata["round_number"] = round_number


class AgentError(AgenticSystemError):
    """Agent-specific errors."""
    
    def __init__(
        self,
        message: str,
        agent_type: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.agent_type = agent_type
        self.action = action
        if agent_type:
            self.metadata["agent_type"] = agent_type
        if action:
            self.metadata["action"] = action


class ToolError(AgenticSystemError):
    """Tool-specific errors."""
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, tool=tool_name, **kwargs)
        self.operation = operation
        if operation:
            self.metadata["operation"] = operation


class ConfigurationError(AgenticSystemError):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_file = config_file
        if config_key:
            self.metadata["config_key"] = config_key
        if config_file:
            self.metadata["config_file"] = config_file


class APIError(AgenticSystemError):
    """External API-related errors."""
    
    def __init__(
        self,
        message: str,
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.api_name = api_name
        self.status_code = status_code
        self.response_body = response_body
        if api_name:
            self.metadata["api_name"] = api_name
        if status_code:
            self.metadata["status_code"] = status_code
        if response_body:
            self.metadata["response_body"] = response_body


class RateLimitError(APIError):
    """API rate limiting errors."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.metadata["retry_after"] = retry_after


class AuthenticationError(APIError):
    """API authentication errors."""
    pass


class ValidationError(AgenticSystemError):
    """Data validation errors."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        if field_name:
            self.metadata["field_name"] = field_name
        if field_value is not None:
            self.metadata["field_value"] = str(field_value)


class WorkflowError(AgenticSystemError):
    """Workflow execution errors."""
    
    def __init__(
        self,
        message: str,
        step_name: Optional[str] = None,
        step_index: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.step_name = step_name
        self.step_index = step_index
        if step_name:
            self.metadata["step_name"] = step_name
        if step_index is not None:
            self.metadata["step_index"] = step_index


class ResourceError(AgenticSystemError):
    """Resource-related errors (memory, disk, network)."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        if resource_type:
            self.metadata["resource_type"] = resource_type
        if current_usage is not None:
            self.metadata["current_usage"] = current_usage
        if limit is not None:
            self.metadata["limit"] = limit


class TimeoutError(AgenticSystemError):
    """Operation timeout errors."""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration
        self.operation = operation
        if timeout_duration is not None:
            self.metadata["timeout_duration"] = timeout_duration
        if operation:
            self.metadata["operation"] = operation


class ConcurrencyError(AgenticSystemError):
    """Concurrency-related errors."""
    
    def __init__(
        self,
        message: str,
        resource_name: Optional[str] = None,
        lock_timeout: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.resource_name = resource_name
        self.lock_timeout = lock_timeout
        if resource_name:
            self.metadata["resource_name"] = resource_name
        if lock_timeout is not None:
            self.metadata["lock_timeout"] = lock_timeout


class MCPServerError(AgenticSystemError):
    """MCP server-specific errors."""
    pass


class MessageHandlingError(AgenticSystemError):
    """Message handling and routing errors."""
    
    def __init__(
        self,
        message: str,
        message_id: Optional[str] = None,
        message_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.message_id = message_id
        self.message_type = message_type
        if message_id:
            self.metadata["message_id"] = message_id
        if message_type:
            self.metadata["message_type"] = message_type


class CommunicationError(AgenticSystemError):
    """Communication and connection errors."""
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.endpoint = endpoint
        self.status_code = status_code
        if endpoint:
            self.metadata["endpoint"] = endpoint
        if status_code:
            self.metadata["status_code"] = status_code


# Aliases for backward compatibility with tools
ToolExecutionError = ToolError