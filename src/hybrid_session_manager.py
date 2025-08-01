"""
Hybrid session manager that supports both Redis and PostgreSQL
"""

import logging
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

# Add path to pyairtable-common
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../pyairtable-common'))

try:
    from pyairtable_common.database.session_repository import PostgreSQLSessionRepository
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from main import RedisSessionManager, InMemorySessionManager

logger = logging.getLogger(__name__)


class HybridSessionManager:
    """
    Hybrid session manager that can use both Redis and PostgreSQL
    
    This allows for gradual migration from Redis to PostgreSQL while maintaining
    backward compatibility and providing fallback options.
    
    Strategy:
    1. Try PostgreSQL first (if available and enabled)
    2. Fall back to Redis if PostgreSQL fails
    3. Fall back to in-memory if both fail
    """
    
    def __init__(
        self,
        redis_url: str,
        redis_password: Optional[str] = None,
        session_timeout: int = 3600,
        use_postgres: bool = True,
        use_redis_fallback: bool = True
    ):
        self.redis_url = redis_url
        self.redis_password = redis_password
        self.session_timeout = session_timeout
        self.use_postgres = use_postgres and POSTGRES_AVAILABLE
        self.use_redis_fallback = use_redis_fallback
        
        # Initialize managers
        self.postgres_manager: Optional[PostgreSQLSessionRepository] = None
        self.redis_manager: Optional[RedisSessionManager] = None
        self.memory_manager: Optional[InMemorySessionManager] = None
        
        if self.use_postgres:
            self.postgres_manager = PostgreSQLSessionRepository(session_timeout)
        
        if self.use_redis_fallback:
            self.redis_manager = RedisSessionManager(redis_url, redis_password)
        
        # Always have in-memory as final fallback
        self.memory_manager = InMemorySessionManager()
        
        # Track which manager is being used
        self.primary_manager = None
        self.active_manager_name = "uninitialized"
    
    async def initialize(self):
        """Initialize the hybrid session manager"""
        errors = []
        
        # Try PostgreSQL first
        if self.postgres_manager:
            try:
                await self.postgres_manager.initialize()
                self.primary_manager = self.postgres_manager
                self.active_manager_name = "postgresql"
                logger.info("✅ Using PostgreSQL session manager (primary)")
                return
            except Exception as e:
                error_msg = f"PostgreSQL session manager failed: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        # Fall back to Redis
        if self.redis_manager:
            try:
                await self.redis_manager.initialize()
                self.primary_manager = self.redis_manager
                self.active_manager_name = "redis"
                logger.info("⚠️ Using Redis session manager (fallback)")
                return
            except Exception as e:
                error_msg = f"Redis session manager failed: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        # Final fallback to in-memory
        await self.memory_manager.initialize()
        self.primary_manager = self.memory_manager
        self.active_manager_name = "memory"
        logger.warning("⚠️ Using in-memory session manager (final fallback)")
        logger.warning(f"Previous errors: {'; '.join(errors)}")
    
    async def close(self):
        """Close all session managers"""
        if self.postgres_manager:
            try:
                await self.postgres_manager.close()
            except Exception as e:
                logger.error(f"Error closing PostgreSQL manager: {e}")
        
        if self.redis_manager:
            try:
                await self.redis_manager.close()
            except Exception as e:
                logger.error(f"Error closing Redis manager: {e}")
        
        if self.memory_manager:
            try:
                await self.memory_manager.close()
            except Exception as e:
                logger.error(f"Error closing memory manager: {e}")
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session with automatic fallback"""
        if not self.primary_manager:
            raise RuntimeError("Session manager not initialized")
        
        try:
            session = await self.primary_manager.get_session(session_id)
            
            # Add manager info to session metadata
            if "metadata" not in session:
                session["metadata"] = {}
            session["metadata"]["manager"] = self.active_manager_name
            
            return session
            
        except Exception as e:
            logger.error(f"Error getting session {session_id} from {self.active_manager_name}: {e}")
            
            # Try fallback if not already using memory manager
            if self.active_manager_name != "memory" and self.memory_manager:
                logger.warning(f"Falling back to memory manager for session {session_id}")
                return await self.memory_manager.get_session(session_id)
            
            raise
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        message: str,
        tools_used: List[str] = None,
        **kwargs
    ):
        """Add message with automatic fallback"""
        if not self.primary_manager:
            raise RuntimeError("Session manager not initialized")
        
        try:
            await self.primary_manager.add_message(
                session_id, role, message, tools_used, **kwargs
            )
            
            # If using PostgreSQL, also try to save to Redis as backup (if available)
            if (self.active_manager_name == "postgresql" and 
                self.redis_manager and 
                self.use_redis_fallback):
                try:
                    await self.redis_manager.add_message(
                        session_id, role, message, tools_used, **kwargs
                    )
                except Exception as e:
                    logger.debug(f"Redis backup failed for message: {e}")
            
        except Exception as e:
            logger.error(f"Error adding message to {self.active_manager_name}: {e}")
            
            # Try fallback
            if self.active_manager_name != "memory" and self.memory_manager:
                logger.warning(f"Falling back to memory manager for adding message")
                await self.memory_manager.add_message(
                    session_id, role, message, tools_used, **kwargs
                )
            else:
                raise
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear session with automatic fallback"""
        if not self.primary_manager:
            raise RuntimeError("Session manager not initialized")
        
        try:
            result = await self.primary_manager.clear_session(session_id)
            
            # Clear from backup systems too
            if self.redis_manager and self.active_manager_name != "redis":
                try:
                    await self.redis_manager.clear_session(session_id)
                except Exception as e:
                    logger.debug(f"Redis backup clear failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error clearing session {session_id} from {self.active_manager_name}: {e}")
            
            # Try fallback
            if self.active_manager_name != "memory" and self.memory_manager:
                return await self.memory_manager.clear_session(session_id)
            
            raise
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get session history with automatic fallback"""
        if not self.primary_manager:
            raise RuntimeError("Session manager not initialized")
        
        try:
            return await self.primary_manager.get_session_history(session_id)
        except Exception as e:
            logger.error(f"Error getting history for session {session_id}: {e}")
            
            # Try fallback
            if self.active_manager_name != "memory" and self.memory_manager:
                return await self.memory_manager.get_session_history(session_id)
            
            raise
    
    async def cleanup_expired_sessions(self):
        """Cleanup expired sessions from all managers"""
        if not self.primary_manager:
            return
        
        try:
            await self.primary_manager.cleanup_expired_sessions()
        except Exception as e:
            logger.error(f"Error cleaning up sessions from {self.active_manager_name}: {e}")
        
        # Also cleanup fallback managers
        if self.redis_manager and self.active_manager_name != "redis":
            try:
                await self.redis_manager.cleanup_expired_sessions()
            except Exception as e:
                logger.debug(f"Redis cleanup failed: {e}")
        
        if self.memory_manager and self.active_manager_name != "memory":
            try:
                await self.memory_manager.cleanup_expired_sessions()
            except Exception as e:
                logger.debug(f"Memory cleanup failed: {e}")
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get status of all session managers"""
        return {
            "active_manager": self.active_manager_name,
            "postgres_available": self.postgres_manager is not None,
            "redis_available": self.redis_manager is not None,
            "memory_available": self.memory_manager is not None,
            "use_postgres": self.use_postgres,
            "use_redis_fallback": self.use_redis_fallback,
            "session_timeout": self.session_timeout
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for all session managers"""
        status = self.get_manager_status()
        
        # Test primary manager
        if self.primary_manager:
            try:
                test_session = await self.primary_manager.get_session("health-check-test")
                status[f"{self.active_manager_name}_healthy"] = True
            except Exception as e:
                status[f"{self.active_manager_name}_healthy"] = False
                status[f"{self.active_manager_name}_error"] = str(e)
        
        return status