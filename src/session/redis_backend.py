"""
Redis-backed session manager for persistent conversation storage
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import redis.asyncio as redis

from .manager import SessionManagerInterface

logger = logging.getLogger(__name__)


class RedisSessionManager(SessionManagerInterface):
    """Redis-backed session manager for persistent conversation storage"""
    
    def __init__(self, redis_url: str, password: Optional[str] = None, session_timeout: int = 3600):
        # Build connection URL with password if provided
        if password and "://" in redis_url:
            protocol, rest = redis_url.split("://", 1)
            redis_url = f"{protocol}://:{password}@{rest}"
        
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.session_timeout = session_timeout
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            await self.redis_client.ping()
            logger.info(f"✅ Connected to Redis for session storage")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis session manager closed")
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session from Redis or create new one"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        try:
            # Get session data from Redis
            session_data = await self.redis_client.get(f"session:{session_id}")
            
            if session_data:
                session = json.loads(session_data)
                # Update last activity timestamp
                session["last_activity"] = datetime.now().isoformat()
                await self._save_session(session_id, session)
                logger.debug(f"Retrieved session {session_id} from Redis")
                return session
            else:
                # Create new session
                return await self._create_new_session(session_id)
                
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            # Return new session as fallback
            return await self._create_new_session(session_id)
    
    async def _create_new_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new session"""
        session = {
            "history": [],
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        await self._save_session(session_id, session)
        logger.info(f"Created new session {session_id}")
        return session
    
    async def _save_session(self, session_id: str, session: Dict[str, Any]):
        """Save session to Redis with TTL"""
        if not self.redis_client:
            return
        
        try:
            session_json = json.dumps(session, default=str)  # Handle datetime serialization
            await self.redis_client.setex(
                f"session:{session_id}", 
                self.session_timeout, 
                session_json
            )
            logger.debug(f"Saved session {session_id} to Redis")
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
    
    async def add_message(
        self, 
        session_id: str, 
        role: str, 
        message: str, 
        tools_used: List[str] = None,
        **kwargs
    ):
        """Add a message to the session history"""
        session = await self.get_session(session_id)
        
        message_entry = {
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if tools_used:
            message_entry["tools_used"] = tools_used
        
        # Add any additional kwargs to the message entry
        for key, value in kwargs.items():
            if key not in message_entry:  # Don't override existing keys
                message_entry[key] = value
        
        session["history"].append(message_entry)
        session["last_activity"] = datetime.now().isoformat()
        
        await self._save_session(session_id, session)
        logger.debug(f"Added {role} message to session {session_id}")
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear a session"""
        if not self.redis_client:
            return False
        
        try:
            deleted = await self.redis_client.delete(f"session:{session_id}")
            logger.info(f"Cleared session {session_id}")
            return deleted > 0
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
            return False
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get only the history of a session"""
        session = await self.get_session(session_id)
        return session.get("history", [])
    
    async def cleanup_expired_sessions(self):
        """Cleanup is handled by Redis TTL, but we can check for manual cleanup"""
        if not self.redis_client:
            return
        
        try:
            # Get all session keys
            session_keys = await self.redis_client.keys("session:*")
            logger.info(f"Found {len(session_keys)} active sessions in Redis")
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")