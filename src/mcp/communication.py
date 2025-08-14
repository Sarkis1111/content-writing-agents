"""
Framework-Agnostic Communication Layer

This module provides a unified communication interface that abstracts away
the differences between CrewAI, LangGraph, and AutoGen frameworks, enabling
seamless inter-framework communication and coordination through the MCP protocol.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Protocol, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import redis
from concurrent.futures import ThreadPoolExecutor

from ..core.config.loader import ConfigLoader
from ..core.monitoring.metrics import MetricsCollector
from ..core.errors.exceptions import CommunicationError, FrameworkError
from ..core.errors.handlers import handle_async_errors
from .server import FrameworkType, MCPMessage, MessageType


class CommunicationChannel(str, Enum):
    """Available communication channels"""
    DIRECT = "direct"          # Direct framework-to-framework
    BROADCAST = "broadcast"    # One-to-many messaging
    PUBSUB = "pubsub"         # Publish-subscribe pattern
    QUEUE = "queue"           # Message queue
    EVENT_STREAM = "event_stream"  # Event streaming


class MessagePersistence(str, Enum):
    """Message persistence options"""
    NONE = "none"             # No persistence
    MEMORY = "memory"         # In-memory storage
    REDIS = "redis"           # Redis persistence
    DATABASE = "database"     # Database storage


@dataclass
class CommunicationConfig:
    """Communication layer configuration"""
    default_channel: CommunicationChannel = CommunicationChannel.DIRECT
    persistence: MessagePersistence = MessagePersistence.REDIS
    message_ttl: int = 3600  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_compression: bool = False
    enable_encryption: bool = False
    batch_size: int = 100
    connection_timeout: int = 30


@dataclass
class CommunicationEndpoint:
    """Represents a communication endpoint for a framework or component"""
    endpoint_id: str
    framework: FrameworkType
    component_name: str
    capabilities: List[str] = field(default_factory=list)
    status: str = "active"
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_status(self, status: str):
        """Update endpoint status"""
        self.status = status
        self.last_seen = datetime.utcnow()


class MessageBroker(ABC):
    """Abstract message broker interface"""
    
    @abstractmethod
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish message to channel"""
        pass
    
    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable) -> str:
        """Subscribe to channel with callback"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from channel"""
        pass
    
    @abstractmethod
    async def send_direct(self, target: str, message: Dict[str, Any]) -> bool:
        """Send direct message to target"""
        pass
    
    @abstractmethod
    async def get_messages(self, queue_name: str, count: int = 1) -> List[Dict[str, Any]]:
        """Get messages from queue"""
        pass


class RedisBroker(MessageBroker):
    """Redis-based message broker implementation"""
    
    def __init__(self, config: ConfigLoader, logger: logging.Logger, metrics: MetricsCollector):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.redis_client = None
        self.pubsub = None
        self.subscriptions: Dict[str, str] = {}  # subscription_id -> channel
        self.active_channels: Set[str] = set()
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            redis_config = self.config.get_redis_config()
            self.redis_client = redis.asyncio.from_url(
                redis_config.get("url", "redis://localhost:6379"),
                decode_responses=True,
                socket_timeout=30,
                socket_connect_timeout=10
            )
            
            # Test connection
            await self.redis_client.ping()
            self.pubsub = self.redis_client.pubsub()
            
            self.logger.info("Redis broker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis broker: {e}")
            raise CommunicationError(f"Redis initialization failed: {e}")
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish message to Redis channel"""
        try:
            message_json = json.dumps(message)
            result = await self.redis_client.publish(channel, message_json)
            
            self.metrics.increment_counter(f"message_broker.redis.publish.{channel}")
            self.logger.debug(f"Published message to channel {channel}")
            
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Failed to publish to channel {channel}: {e}")
            self.metrics.increment_counter("message_broker.redis.publish.error")
            return False
    
    async def subscribe(self, channel: str, callback: Callable) -> str:
        """Subscribe to Redis channel"""
        try:
            subscription_id = str(uuid.uuid4())
            
            await self.pubsub.subscribe(channel)
            self.subscriptions[subscription_id] = channel
            self.active_channels.add(channel)
            
            # Start message handler task
            asyncio.create_task(self._handle_subscription_messages(callback))
            
            self.logger.info(f"Subscribed to channel {channel} with ID {subscription_id}")
            return subscription_id
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to channel {channel}: {e}")
            raise CommunicationError(f"Subscription failed: {e}")
    
    async def _handle_subscription_messages(self, callback: Callable):
        """Handle incoming subscription messages"""
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        await callback(message["channel"], data)
                        self.metrics.increment_counter("message_broker.redis.received")
                    except Exception as e:
                        self.logger.error(f"Error processing subscription message: {e}")
                        self.metrics.increment_counter("message_broker.redis.process_error")
                        
        except Exception as e:
            self.logger.error(f"Subscription message handler error: {e}")
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from Redis channel"""
        try:
            if subscription_id in self.subscriptions:
                channel = self.subscriptions[subscription_id]
                await self.pubsub.unsubscribe(channel)
                
                del self.subscriptions[subscription_id]
                if channel not in self.subscriptions.values():
                    self.active_channels.discard(channel)
                
                self.logger.info(f"Unsubscribed from {channel}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe {subscription_id}: {e}")
            return False
    
    async def send_direct(self, target: str, message: Dict[str, Any]) -> bool:
        """Send direct message via Redis list"""
        try:
            queue_name = f"direct:{target}"
            message_json = json.dumps(message)
            
            await self.redis_client.lpush(queue_name, message_json)
            await self.redis_client.expire(queue_name, 3600)  # 1 hour TTL
            
            self.metrics.increment_counter(f"message_broker.redis.direct.{target}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send direct message to {target}: {e}")
            return False
    
    async def get_messages(self, queue_name: str, count: int = 1) -> List[Dict[str, Any]]:
        """Get messages from Redis queue"""
        try:
            messages = []
            for _ in range(count):
                result = await self.redis_client.rpop(queue_name)
                if result:
                    messages.append(json.loads(result))
                else:
                    break
                    
            self.metrics.increment_counter(f"message_broker.redis.get.{queue_name}", len(messages))
            return messages
            
        except Exception as e:
            self.logger.error(f"Failed to get messages from {queue_name}: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup Redis connections"""
        try:
            if self.pubsub:
                await self.pubsub.close()
            if self.redis_client:
                await self.redis_client.close()
                
        except Exception as e:
            self.logger.error(f"Error during Redis cleanup: {e}")


class MemoryBroker(MessageBroker):
    """In-memory message broker for testing/development"""
    
    def __init__(self, logger: logging.Logger, metrics: MetricsCollector):
        self.logger = logger
        self.metrics = metrics
        self.channels: Dict[str, List[Callable]] = {}
        self.queues: Dict[str, List[Dict[str, Any]]] = {}
        self.message_history: List[Dict[str, Any]] = []
        
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish to memory channel"""
        self.message_history.append({
            "channel": channel,
            "message": message,
            "timestamp": datetime.utcnow()
        })
        
        if channel in self.channels:
            for callback in self.channels[channel]:
                try:
                    await callback(channel, message)
                except Exception as e:
                    self.logger.error(f"Callback error for channel {channel}: {e}")
        
        self.metrics.increment_counter(f"message_broker.memory.publish.{channel}")
        return True
    
    async def subscribe(self, channel: str, callback: Callable) -> str:
        """Subscribe to memory channel"""
        if channel not in self.channels:
            self.channels[channel] = []
        
        subscription_id = str(uuid.uuid4())
        self.channels[channel].append(callback)
        
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from memory channel"""
        # Simplified - would need better tracking in production
        return True
    
    async def send_direct(self, target: str, message: Dict[str, Any]) -> bool:
        """Send direct message to memory queue"""
        queue_name = f"direct:{target}"
        if queue_name not in self.queues:
            self.queues[queue_name] = []
        
        self.queues[queue_name].append(message)
        return True
    
    async def get_messages(self, queue_name: str, count: int = 1) -> List[Dict[str, Any]]:
        """Get messages from memory queue"""
        if queue_name not in self.queues:
            return []
        
        messages = []
        for _ in range(min(count, len(self.queues[queue_name]))):
            if self.queues[queue_name]:
                messages.append(self.queues[queue_name].pop(0))
                
        return messages


class CommunicationManager:
    """Central communication manager for framework coordination"""
    
    def __init__(self, config: ConfigLoader, logger: logging.Logger, metrics: MetricsCollector):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.comm_config = CommunicationConfig()  # Would load from config
        
        # Initialize broker based on configuration
        if self.comm_config.persistence == MessagePersistence.REDIS:
            self.broker = RedisBroker(config, logger, metrics)
        else:
            self.broker = MemoryBroker(logger, metrics)
        
        self.endpoints: Dict[str, CommunicationEndpoint] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.active_subscriptions: Dict[str, str] = {}
        
        # Background tasks
        self.cleanup_task = None
        
    async def initialize(self):
        """Initialize communication manager"""
        await self.broker.initialize()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_task())
        
        self.logger.info("Communication manager initialized")
    
    async def register_endpoint(self, framework: FrameworkType, component_name: str, 
                              capabilities: List[str] = None) -> str:
        """Register a communication endpoint"""
        endpoint_id = f"{framework.value}:{component_name}:{uuid.uuid4().hex[:8]}"
        
        endpoint = CommunicationEndpoint(
            endpoint_id=endpoint_id,
            framework=framework,
            component_name=component_name,
            capabilities=capabilities or [],
            last_seen=datetime.utcnow()
        )
        
        self.endpoints[endpoint_id] = endpoint
        
        # Subscribe to endpoint-specific channel
        channel = f"endpoint:{endpoint_id}"
        subscription_id = await self.broker.subscribe(channel, self._handle_endpoint_message)
        self.active_subscriptions[endpoint_id] = subscription_id
        
        self.logger.info(f"Registered endpoint {endpoint_id}")
        self.metrics.increment_counter(f"communication.endpoint.registered.{framework.value}")
        
        return endpoint_id
    
    async def unregister_endpoint(self, endpoint_id: str):
        """Unregister a communication endpoint"""
        if endpoint_id in self.endpoints:
            # Unsubscribe from endpoint channel
            if endpoint_id in self.active_subscriptions:
                subscription_id = self.active_subscriptions[endpoint_id]
                await self.broker.unsubscribe(subscription_id)
                del self.active_subscriptions[endpoint_id]
            
            framework = self.endpoints[endpoint_id].framework
            del self.endpoints[endpoint_id]
            
            self.logger.info(f"Unregistered endpoint {endpoint_id}")
            self.metrics.increment_counter(f"communication.endpoint.unregistered.{framework.value}")
    
    async def send_direct_message(self, source_endpoint: str, target_endpoint: str, 
                                message: Dict[str, Any]) -> bool:
        """Send direct message between endpoints"""
        try:
            # Validate endpoints exist
            if source_endpoint not in self.endpoints:
                raise CommunicationError(f"Source endpoint {source_endpoint} not found")
            if target_endpoint not in self.endpoints:
                raise CommunicationError(f"Target endpoint {target_endpoint} not found")
            
            # Create communication message
            comm_message = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "source": source_endpoint,
                "target": target_endpoint,
                "type": "direct_message",
                "payload": message
            }
            
            # Send via broker
            success = await self.broker.send_direct(target_endpoint, comm_message)
            
            if success:
                self.metrics.increment_counter("communication.direct_message.sent")
                self.logger.debug(f"Sent direct message from {source_endpoint} to {target_endpoint}")
            else:
                self.metrics.increment_counter("communication.direct_message.failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send direct message: {e}")
            self.metrics.increment_counter("communication.direct_message.error")
            return False
    
    async def broadcast_message(self, source_endpoint: str, target_framework: Optional[FrameworkType], 
                              message: Dict[str, Any]) -> int:
        """Broadcast message to framework or all endpoints"""
        try:
            if source_endpoint not in self.endpoints:
                raise CommunicationError(f"Source endpoint {source_endpoint} not found")
            
            # Determine target channel
            if target_framework:
                channel = f"framework:{target_framework.value}"
            else:
                channel = "broadcast:all"
            
            # Create broadcast message
            broadcast_message = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "source": source_endpoint,
                "type": "broadcast",
                "target_framework": target_framework.value if target_framework else "all",
                "payload": message
            }
            
            # Publish to channel
            success = await self.broker.publish(channel, broadcast_message)
            
            if success:
                # Count potential recipients
                recipients = 0
                if target_framework:
                    recipients = len([ep for ep in self.endpoints.values() 
                                    if ep.framework == target_framework])
                else:
                    recipients = len(self.endpoints)
                
                self.metrics.increment_counter("communication.broadcast.sent")
                self.logger.info(f"Broadcast message from {source_endpoint} to {recipients} recipients")
                return recipients
            else:
                self.metrics.increment_counter("communication.broadcast.failed")
                return 0
                
        except Exception as e:
            self.logger.error(f"Failed to broadcast message: {e}")
            self.metrics.increment_counter("communication.broadcast.error")
            return 0
    
    async def subscribe_to_events(self, endpoint_id: str, event_types: List[str], 
                                callback: Callable) -> List[str]:
        """Subscribe endpoint to specific event types"""
        subscription_ids = []
        
        for event_type in event_types:
            try:
                channel = f"events:{event_type}"
                subscription_id = await self.broker.subscribe(channel, callback)
                subscription_ids.append(subscription_id)
                
                self.logger.info(f"Endpoint {endpoint_id} subscribed to events: {event_type}")
                
            except Exception as e:
                self.logger.error(f"Failed to subscribe {endpoint_id} to {event_type}: {e}")
        
        return subscription_ids
    
    async def publish_event(self, source_endpoint: str, event_type: str, 
                          event_data: Dict[str, Any]) -> bool:
        """Publish an event"""
        try:
            event_message = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "source": source_endpoint,
                "event_type": event_type,
                "data": event_data
            }
            
            channel = f"events:{event_type}"
            success = await self.broker.publish(channel, event_message)
            
            if success:
                self.metrics.increment_counter(f"communication.events.published.{event_type}")
                self.logger.debug(f"Published event {event_type} from {source_endpoint}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to publish event {event_type}: {e}")
            return False
    
    async def get_endpoint_messages(self, endpoint_id: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get pending messages for an endpoint"""
        try:
            messages = await self.broker.get_messages(endpoint_id, count)
            
            self.metrics.increment_counter(f"communication.messages.retrieved.{endpoint_id}")
            return messages
            
        except Exception as e:
            self.logger.error(f"Failed to get messages for {endpoint_id}: {e}")
            return []
    
    async def _handle_endpoint_message(self, channel: str, message: Dict[str, Any]):
        """Handle messages received on endpoint channels"""
        try:
            # Extract endpoint ID from channel name
            endpoint_id = channel.split(":")[-1] if ":" in channel else channel
            
            if endpoint_id in self.endpoints:
                endpoint = self.endpoints[endpoint_id]
                endpoint.update_status("active")
                
                # Process message based on type
                message_type = message.get("type", "unknown")
                
                if message_type in self.message_handlers:
                    handler = self.message_handlers[message_type]
                    await handler(endpoint_id, message)
                else:
                    self.logger.debug(f"No handler for message type: {message_type}")
                    
                self.metrics.increment_counter(f"communication.messages.handled.{message_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling endpoint message: {e}")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a handler for specific message types"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered message handler for type: {message_type}")
    
    async def get_endpoint_status(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for an endpoint"""
        if endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            return {
                "endpoint_id": endpoint.endpoint_id,
                "framework": endpoint.framework.value,
                "component": endpoint.component_name,
                "status": endpoint.status,
                "capabilities": endpoint.capabilities,
                "last_seen": endpoint.last_seen.isoformat() if endpoint.last_seen else None,
                "metadata": endpoint.metadata
            }
        return None
    
    def list_endpoints(self, framework: Optional[FrameworkType] = None) -> List[Dict[str, Any]]:
        """List all registered endpoints, optionally filtered by framework"""
        endpoints = list(self.endpoints.values())
        
        if framework:
            endpoints = [ep for ep in endpoints if ep.framework == framework]
        
        return [
            {
                "endpoint_id": ep.endpoint_id,
                "framework": ep.framework.value,
                "component": ep.component_name,
                "status": ep.status,
                "capabilities": ep.capabilities,
                "last_seen": ep.last_seen.isoformat() if ep.last_seen else None
            }
            for ep in endpoints
        ]
    
    async def _cleanup_task(self):
        """Background task for cleanup operations"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup stale endpoints
                cutoff_time = datetime.utcnow() - timedelta(minutes=10)
                stale_endpoints = []
                
                for endpoint_id, endpoint in self.endpoints.items():
                    if endpoint.last_seen and endpoint.last_seen < cutoff_time:
                        stale_endpoints.append(endpoint_id)
                
                for endpoint_id in stale_endpoints:
                    await self.unregister_endpoint(endpoint_id)
                    self.logger.info(f"Cleaned up stale endpoint: {endpoint_id}")
                
                if stale_endpoints:
                    self.metrics.increment_counter("communication.cleanup.stale_endpoints", 
                                                 len(stale_endpoints))
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
    
    async def shutdown(self):
        """Shutdown communication manager"""
        try:
            # Cancel cleanup task
            if self.cleanup_task:
                self.cleanup_task.cancel()
            
            # Unsubscribe all
            for subscription_id in self.active_subscriptions.values():
                await self.broker.unsubscribe(subscription_id)
            
            # Cleanup broker
            if hasattr(self.broker, 'cleanup'):
                await self.broker.cleanup()
            
            self.logger.info("Communication manager shut down")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Framework-specific communication adapters

class CrewAICommunicationAdapter:
    """Communication adapter for CrewAI framework"""
    
    def __init__(self, comm_manager: CommunicationManager, logger: logging.Logger):
        self.comm_manager = comm_manager
        self.logger = logger
        self.endpoint_id = None
    
    async def initialize(self, component_name: str = "crewai_adapter") -> str:
        """Initialize CrewAI communication"""
        self.endpoint_id = await self.comm_manager.register_endpoint(
            FrameworkType.CREWAI, 
            component_name,
            ["task_delegation", "crew_coordination", "agent_communication"]
        )
        
        # Subscribe to CrewAI-specific events
        await self.comm_manager.subscribe_to_events(
            self.endpoint_id,
            ["task_created", "task_completed", "crew_status_change"],
            self._handle_crewai_event
        )
        
        return self.endpoint_id
    
    async def _handle_crewai_event(self, channel: str, event: Dict[str, Any]):
        """Handle CrewAI-specific events"""
        event_type = event.get("event_type")
        self.logger.info(f"Handling CrewAI event: {event_type}")
        
        # Process CrewAI-specific event logic here
        
    async def send_task_to_framework(self, target_framework: FrameworkType, task_data: Dict[str, Any]):
        """Send task data to another framework"""
        if self.endpoint_id:
            await self.comm_manager.broadcast_message(
                self.endpoint_id,
                target_framework,
                {"type": "task_data", "data": task_data}
            )


class LangGraphCommunicationAdapter:
    """Communication adapter for LangGraph framework"""
    
    def __init__(self, comm_manager: CommunicationManager, logger: logging.Logger):
        self.comm_manager = comm_manager
        self.logger = logger
        self.endpoint_id = None
    
    async def initialize(self, component_name: str = "langgraph_adapter") -> str:
        """Initialize LangGraph communication"""
        self.endpoint_id = await self.comm_manager.register_endpoint(
            FrameworkType.LANGGRAPH,
            component_name,
            ["workflow_execution", "state_management", "conditional_logic"]
        )
        
        await self.comm_manager.subscribe_to_events(
            self.endpoint_id,
            ["workflow_started", "workflow_completed", "state_changed"],
            self._handle_langgraph_event
        )
        
        return self.endpoint_id
    
    async def _handle_langgraph_event(self, channel: str, event: Dict[str, Any]):
        """Handle LangGraph-specific events"""
        event_type = event.get("event_type")
        self.logger.info(f"Handling LangGraph event: {event_type}")


class AutoGenCommunicationAdapter:
    """Communication adapter for AutoGen framework"""
    
    def __init__(self, comm_manager: CommunicationManager, logger: logging.Logger):
        self.comm_manager = comm_manager
        self.logger = logger
        self.endpoint_id = None
    
    async def initialize(self, component_name: str = "autogen_adapter") -> str:
        """Initialize AutoGen communication"""
        self.endpoint_id = await self.comm_manager.register_endpoint(
            FrameworkType.AUTOGEN,
            component_name, 
            ["group_chat", "conversation_management", "consensus_building"]
        )
        
        await self.comm_manager.subscribe_to_events(
            self.endpoint_id,
            ["conversation_started", "consensus_reached", "agent_response"],
            self._handle_autogen_event
        )
        
        return self.endpoint_id
    
    async def _handle_autogen_event(self, channel: str, event: Dict[str, Any]):
        """Handle AutoGen-specific events"""
        event_type = event.get("event_type")
        self.logger.info(f"Handling AutoGen event: {event_type}")