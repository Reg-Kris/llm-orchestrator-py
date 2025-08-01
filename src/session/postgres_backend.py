"""
PostgreSQL-backed session manager using pyairtable-common database infrastructure
"""

import logging
import os
import sys
from typing import Dict, Any, List, Optional

from .manager import SessionManagerInterface

logger = logging.getLogger(__name__)


class PostgreSQLSessionManager(SessionManagerInterface):
    """
    PostgreSQL-backed session manager using pyairtable-common infrastructure
    
    This is a wrapper around the PostgreSQLSessionRepository from pyairtable-common
    to maintain compatibility with the SessionManagerInterface
    """
    
    def __init__(self, session_timeout: int = 3600):
        self.session_timeout = session_timeout
        self.repository = None
        
        # Try to import the PostgreSQL repository
        try:
            # Add path to pyairtable-common
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../pyairtable-common'))
            from pyairtable_common.database.session_repository import PostgreSQLSessionRepository
            self.repository = PostgreSQLSessionRepository(session_timeout)
        except ImportError as e:
            logger.error(f"Failed to import PostgreSQL session repository: {e}")
            raise
    
    async def initialize(self):
        """Initialize the PostgreSQL session repository"""
        if not self.repository:
            raise RuntimeError("PostgreSQL repository not available")
        
        await self.repository.initialize()
        logger.info("✅ PostgreSQL session manager initialized")
    
    async def close(self):
        """Close the PostgreSQL session repository"""
        if self.repository:
            await self.repository.close()
            logger.info("PostgreSQL session manager closed")
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session from PostgreSQL or create new one"""
        if not self.repository:
            raise RuntimeError("PostgreSQL repository not initialized")
        
        return await self.repository.get_session(session_id)
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        message: str,
        tools_used: List[str] = None,
        **kwargs
    ):
        """Add a message to the session history"""
        if not self.repository:
            raise RuntimeError("PostgreSQL repository not initialized")
        
        await self.repository.add_message(
            session_id, role, message, tools_used, **kwargs
        )
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear a session"""
        if not self.repository:
            raise RuntimeError("PostgreSQL repository not initialized")
        
        return await self.repository.clear_session(session_id)
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get session history"""
        if not self.repository:
            raise RuntimeError("PostgreSQL repository not initialized")
        
        return await self.repository.get_session_history(session_id)
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        if not self.repository:
            return
        
        await self.repository.cleanup_expired_sessions()