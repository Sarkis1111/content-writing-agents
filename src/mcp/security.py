"""
Security and Authentication System for MCP Protocol

This module provides comprehensive security features for the MCP server including
authentication, authorization, encryption, rate limiting, and audit logging to
ensure secure multi-framework agentic operations.
"""

import asyncio
import hashlib
import hmac
import secrets
import jwt
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..core.config.loader import ConfigLoader
from ..core.monitoring.metrics import MetricsCollector
from ..core.errors.exceptions import SecurityError, AuthenticationError, AuthorizationError
from ..core.errors.handlers import handle_async_errors
from .server import FrameworkType


class UserRole(str, Enum):
    """User roles for authorization"""
    ADMIN = "admin"
    OPERATOR = "operator"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    SERVICE = "service"


class Permission(str, Enum):
    """System permissions"""
    # System administration
    SYSTEM_ADMIN = "system.admin"
    SYSTEM_CONFIG = "system.config"
    SYSTEM_MONITORING = "system.monitoring"
    
    # Framework management
    FRAMEWORK_MANAGE = "framework.manage"
    FRAMEWORK_INITIALIZE = "framework.initialize"
    FRAMEWORK_EXECUTE = "framework.execute"
    
    # Tool operations
    TOOL_REGISTER = "tool.register"
    TOOL_EXECUTE = "tool.execute"
    TOOL_MANAGE = "tool.manage"
    
    # Workflow operations
    WORKFLOW_CREATE = "workflow.create"
    WORKFLOW_EXECUTE = "workflow.execute"
    WORKFLOW_MANAGE = "workflow.manage"
    
    # Communication
    MESSAGE_SEND = "message.send"
    MESSAGE_BROADCAST = "message.broadcast"
    
    # Monitoring and logs
    METRICS_VIEW = "metrics.view"
    LOGS_VIEW = "logs.view"
    AUDIT_VIEW = "audit.view"


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_encryption: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True
    
    # JWT settings
    jwt_secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    jwt_refresh_expiration_days: int = 7
    
    # Encryption settings
    encryption_key: Optional[str] = None
    password_hash_iterations: int = 100000
    
    # Rate limiting
    default_rate_limit: int = 100  # requests per minute
    burst_limit: int = 200
    rate_limit_window: int = 60  # seconds
    
    # Session settings
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 10
    
    # Security headers
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_csrf_protection: bool = True


@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    password_hash: Optional[str] = None
    api_key: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session information"""
    session_id: str
    user_id: str
    framework: Optional[FrameworkType] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True


class PasswordManager:
    """Secure password hashing and verification"""
    
    def __init__(self, iterations: int = 100000):
        self.iterations = iterations
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> str:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.iterations,
        )
        
        key = kdf.derive(password.encode('utf-8'))
        
        # Combine salt and key for storage
        combined = salt + key
        return base64.b64encode(combined).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            combined = base64.b64decode(password_hash)
            salt = combined[:32]
            stored_key = combined[32:]
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.iterations,
            )
            
            key = kdf.derive(password.encode('utf-8'))
            return hmac.compare_digest(stored_key, key)
            
        except Exception:
            return False


class TokenManager:
    """JWT token management for authentication"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        if expires_delta is None:
            expires_delta = timedelta(hours=24)
        
        expire = datetime.utcnow() + expires_delta
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [perm.value for perm in user.permissions],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT refresh token"""
        if expires_delta is None:
            expires_delta = timedelta(days=7)
        
        expire = datetime.utcnow() + expires_delta
        
        payload = {
            "user_id": user.user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")


class EncryptionManager:
    """Data encryption/decryption for sensitive information"""
    
    def __init__(self, key: Optional[str] = None):
        if key:
            self.fernet = Fernet(key.encode() if isinstance(key, str) else key)
        else:
            # Generate new key
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
            self.key = key.decode()
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary as JSON"""
        json_data = json.dumps(data)
        return self.encrypt(json_data)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt and parse as dictionary"""
        json_data = self.decrypt(encrypted_data)
        return json.loads(json_data)


class RateLimiter:
    """Rate limiting for API requests"""
    
    def __init__(self, logger: logging.Logger, metrics: MetricsCollector):
        self.logger = logger
        self.metrics = metrics
        self.request_counts: Dict[str, Dict[str, Any]] = {}
        self.cleanup_task = None
        
    async def start(self):
        """Start rate limiter with cleanup task"""
        self.cleanup_task = asyncio.create_task(self._cleanup_task())
    
    async def check_rate_limit(self, identifier: str, limit: int = 100, 
                             window: int = 60) -> bool:
        """Check if request is within rate limit"""
        now = datetime.utcnow()
        
        if identifier not in self.request_counts:
            self.request_counts[identifier] = {
                "count": 0,
                "window_start": now,
                "last_request": now
            }
        
        record = self.request_counts[identifier]
        
        # Reset window if expired
        if (now - record["window_start"]).seconds >= window:
            record["count"] = 0
            record["window_start"] = now
        
        # Check limit
        if record["count"] >= limit:
            self.metrics.increment_counter("rate_limiting.exceeded")
            self.logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Update counts
        record["count"] += 1
        record["last_request"] = now
        
        self.metrics.increment_counter("rate_limiting.allowed")
        return True
    
    async def _cleanup_task(self):
        """Clean up old rate limit records"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                now = datetime.utcnow()
                cutoff = now - timedelta(hours=1)
                
                to_remove = []
                for identifier, record in self.request_counts.items():
                    if record["last_request"] < cutoff:
                        to_remove.append(identifier)
                
                for identifier in to_remove:
                    del self.request_counts[identifier]
                
                if to_remove:
                    self.logger.debug(f"Cleaned up {len(to_remove)} rate limit records")
                    
            except Exception as e:
                self.logger.error(f"Error in rate limiter cleanup: {e}")
    
    async def stop(self):
        """Stop rate limiter"""
        if self.cleanup_task:
            self.cleanup_task.cancel()


class AuditLogger:
    """Security audit logging"""
    
    def __init__(self, logger: logging.Logger, metrics: MetricsCollector, 
                 encryption_manager: Optional[EncryptionManager] = None):
        self.logger = logger
        self.metrics = metrics
        self.encryption_manager = encryption_manager
        self.audit_records: List[Dict[str, Any]] = []
        
    async def log_authentication_event(self, user_id: str, event_type: str, 
                                     success: bool, details: Optional[Dict[str, Any]] = None):
        """Log authentication events"""
        await self._log_audit_event(
            category="authentication",
            user_id=user_id,
            event_type=event_type,
            success=success,
            details=details
        )
        
    async def log_authorization_event(self, user_id: str, resource: str, 
                                    permission: str, granted: bool):
        """Log authorization events"""
        await self._log_audit_event(
            category="authorization",
            user_id=user_id,
            event_type="permission_check",
            success=granted,
            details={
                "resource": resource,
                "permission": permission
            }
        )
        
    async def log_framework_event(self, user_id: str, framework: str, 
                                action: str, success: bool, details: Optional[Dict[str, Any]] = None):
        """Log framework operations"""
        await self._log_audit_event(
            category="framework",
            user_id=user_id,
            event_type=action,
            success=success,
            details={
                "framework": framework,
                **(details or {})
            }
        )
        
    async def log_security_event(self, event_type: str, severity: str, 
                               details: Dict[str, Any]):
        """Log security events"""
        await self._log_audit_event(
            category="security",
            event_type=event_type,
            success=severity != "critical",
            details={
                "severity": severity,
                **details
            }
        )
        
    async def _log_audit_event(self, category: str, event_type: str, 
                             success: bool, user_id: Optional[str] = None,
                             details: Optional[Dict[str, Any]] = None):
        """Internal audit event logging"""
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "event_type": event_type,
            "user_id": user_id,
            "success": success,
            "details": details or {}
        }
        
        # Encrypt sensitive audit data if encryption is available
        if self.encryption_manager and category in ["authentication", "security"]:
            audit_record["encrypted_details"] = self.encryption_manager.encrypt_dict(
                audit_record["details"]
            )
            audit_record["details"] = {"encrypted": True}
        
        self.audit_records.append(audit_record)
        
        # Log to standard logging
        log_level = logging.WARNING if not success else logging.INFO
        self.logger.log(
            log_level,
            f"AUDIT: {category}.{event_type} - {'SUCCESS' if success else 'FAILURE'}",
            extra=audit_record
        )
        
        # Update metrics
        self.metrics.increment_counter(f"audit.{category}.{event_type}")
        if not success:
            self.metrics.increment_counter(f"audit.{category}.failures")


class SecurityManager:
    """Central security management for MCP server"""
    
    def __init__(self, config: ConfigLoader, logger: logging.Logger, metrics: MetricsCollector):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.security_config = SecurityConfig()  # Would load from config
        
        # Initialize security components
        self.password_manager = PasswordManager(self.security_config.password_hash_iterations)
        self.token_manager = TokenManager(self.security_config.jwt_secret_key)
        self.encryption_manager = EncryptionManager(self.security_config.encryption_key)
        self.rate_limiter = RateLimiter(logger, metrics)
        self.audit_logger = AuditLogger(logger, metrics, self.encryption_manager)
        
        # In-memory storage (would use database in production)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        
        # Role-permission mappings
        self.role_permissions = {
            UserRole.ADMIN: set(Permission),  # All permissions
            UserRole.OPERATOR: {
                Permission.FRAMEWORK_EXECUTE, Permission.TOOL_EXECUTE,
                Permission.WORKFLOW_EXECUTE, Permission.MESSAGE_SEND,
                Permission.METRICS_VIEW, Permission.LOGS_VIEW
            },
            UserRole.DEVELOPER: {
                Permission.FRAMEWORK_EXECUTE, Permission.TOOL_REGISTER,
                Permission.TOOL_EXECUTE, Permission.WORKFLOW_CREATE,
                Permission.WORKFLOW_EXECUTE, Permission.MESSAGE_SEND
            },
            UserRole.VIEWER: {
                Permission.METRICS_VIEW, Permission.LOGS_VIEW
            },
            UserRole.SERVICE: {
                Permission.FRAMEWORK_EXECUTE, Permission.TOOL_EXECUTE,
                Permission.WORKFLOW_EXECUTE, Permission.MESSAGE_SEND
            }
        }
        
    async def initialize(self):
        """Initialize security manager"""
        await self.rate_limiter.start()
        
        # Create default admin user if none exists
        if not self.users:
            await self.create_default_admin()
        
        self.logger.info("Security manager initialized")
    
    async def create_default_admin(self):
        """Create default admin user"""
        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@localhost",
            role=UserRole.ADMIN,
            permissions=self.role_permissions[UserRole.ADMIN]
        )
        
        # Set default password (should be changed on first login)
        default_password = "admin123"  # In production, generate random password
        admin_user.password_hash = self.password_manager.hash_password(default_password)
        admin_user.api_key = secrets.token_urlsafe(32)
        
        self.users[admin_user.user_id] = admin_user
        self.api_keys[admin_user.api_key] = admin_user.user_id
        
        self.logger.warning("Created default admin user - CHANGE DEFAULT PASSWORD!")
    
    async def create_user(self, username: str, email: str, role: UserRole, 
                        password: str, creator_user_id: str) -> User:
        """Create new user account"""
        # Check permissions
        if not await self.check_permission(creator_user_id, Permission.SYSTEM_ADMIN):
            raise AuthorizationError("Insufficient permissions to create user")
        
        # Generate user ID
        user_id = f"user_{secrets.token_urlsafe(8)}"
        
        # Create user
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=self.role_permissions[role]
        )
        
        user.password_hash = self.password_manager.hash_password(password)
        user.api_key = secrets.token_urlsafe(32)
        
        self.users[user_id] = user
        self.api_keys[user.api_key] = user_id
        
        await self.audit_logger.log_authentication_event(
            creator_user_id, "user_created", True, 
            {"created_user": user_id, "role": role.value}
        )
        
        self.logger.info(f"Created user {username} with role {role.value}")
        return user
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: Optional[str] = None) -> Optional[str]:
        """Authenticate user and create session"""
        try:
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username and u.is_active:
                    user = u
                    break
            
            if not user:
                await self.audit_logger.log_authentication_event(
                    username, "login_attempt", False, {"reason": "user_not_found"}
                )
                return None
            
            # Check rate limit
            rate_limit_key = f"auth:{username}:{ip_address or 'unknown'}"
            if not await self.rate_limiter.check_rate_limit(rate_limit_key, limit=5, window=300):
                await self.audit_logger.log_authentication_event(
                    user.user_id, "login_attempt", False, {"reason": "rate_limited"}
                )
                return None
            
            # Verify password
            if not self.password_manager.verify_password(password, user.password_hash):
                await self.audit_logger.log_authentication_event(
                    user.user_id, "login_attempt", False, {"reason": "invalid_password"}
                )
                return None
            
            # Create session
            session_id = await self.create_session(user, ip_address)
            
            # Update user login time
            user.last_login = datetime.utcnow()
            
            await self.audit_logger.log_authentication_event(
                user.user_id, "login_success", True, {"session_id": session_id}
            )
            
            self.logger.info(f"User {username} authenticated successfully")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Authentication error for {username}: {e}")
            await self.audit_logger.log_authentication_event(
                username, "login_error", False, {"error": str(e)}
            )
            return None
    
    async def authenticate_api_key(self, api_key: str) -> Optional[str]:
        """Authenticate using API key"""
        try:
            if api_key not in self.api_keys:
                await self.audit_logger.log_authentication_event(
                    "unknown", "api_key_attempt", False, {"reason": "invalid_key"}
                )
                return None
            
            user_id = self.api_keys[api_key]
            user = self.users[user_id]
            
            if not user.is_active:
                await self.audit_logger.log_authentication_event(
                    user_id, "api_key_attempt", False, {"reason": "user_inactive"}
                )
                return None
            
            # Create session for API key authentication
            session_id = await self.create_session(user)
            
            await self.audit_logger.log_authentication_event(
                user_id, "api_key_success", True, {"session_id": session_id}
            )
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"API key authentication error: {e}")
            return None
    
    async def create_session(self, user: User, ip_address: Optional[str] = None) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            ip_address=ip_address,
            permissions=user.permissions
        )
        
        self.sessions[session_id] = session
        
        # Clean up old sessions if user has too many
        user_sessions = [s for s in self.sessions.values() if s.user_id == user.user_id]
        if len(user_sessions) > self.security_config.max_concurrent_sessions:
            # Remove oldest session
            oldest_session = min(user_sessions, key=lambda s: s.created_at)
            await self.invalidate_session(oldest_session.session_id)
        
        return session_id
    
    async def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate and update session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session is expired
        timeout = timedelta(minutes=self.security_config.session_timeout_minutes)
        if datetime.utcnow() - session.last_activity > timeout:
            await self.invalidate_session(session_id)
            return None
        
        # Update activity
        session.last_activity = datetime.utcnow()
        return session
    
    async def invalidate_session(self, session_id: str):
        """Invalidate user session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            del self.sessions[session_id]
            
            await self.audit_logger.log_authentication_event(
                session.user_id, "session_invalidated", True, {"session_id": session_id}
            )
    
    async def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        
        has_permission = permission in user.permissions
        
        await self.audit_logger.log_authorization_event(
            user_id, "permission_check", permission.value, has_permission
        )
        
        return has_permission
    
    @handle_async_errors
    async def authorize_framework_operation(self, session_id: str, framework: FrameworkType, 
                                          operation: str) -> bool:
        """Authorize framework operation"""
        session = await self.validate_session(session_id)
        if not session:
            return False
        
        # Determine required permission based on operation
        permission_map = {
            "initialize": Permission.FRAMEWORK_INITIALIZE,
            "execute": Permission.FRAMEWORK_EXECUTE,
            "manage": Permission.FRAMEWORK_MANAGE
        }
        
        required_permission = permission_map.get(operation, Permission.FRAMEWORK_EXECUTE)
        
        has_permission = required_permission in session.permissions
        
        await self.audit_logger.log_framework_event(
            session.user_id, framework.value, operation, has_permission
        )
        
        return has_permission
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        session = await self.validate_session(session_id)
        if not session:
            return None
        
        user = self.users.get(session.user_id)
        if not user:
            return None
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [perm.value for perm in session.permissions],
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        }
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        timeout = timedelta(minutes=self.security_config.session_timeout_minutes)
        cutoff = datetime.utcnow() - timeout
        
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.last_activity < cutoff
        ]
        
        for session_id in expired_sessions:
            await self.invalidate_session(session_id)
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def shutdown(self):
        """Shutdown security manager"""
        await self.rate_limiter.stop()
        self.logger.info("Security manager shut down")