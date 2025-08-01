"""
In-memory session manager for development and fallback scenarios
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .manager import SessionManagerInterface

logger = logging.getLogger(__name__)


class InMemorySessionManager(SessionManagerInterface):
    """Fallback in-memory session manager (for development/testing)"""
    
    def __init__(self, session_timeout: int = 3600):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = session_timeout
    
    async def initialize(self):
        """Initialize in-memory session storage"""
        logger.info("Using in-memory session storage (development mode)")
    
    async def close(self):
        """Close in-memory session storage (no-op)"""
        pass
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create session in memory"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "session_id": session_id
            }
        
        self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
        return self.sessions[session_id]
    
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
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear session from memory"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get session history from memory"""
        session = await self.get_session(session_id)
        return session.get("history", [])
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions from memory"""
        now = datetime.now()
        expired = []
        
        for session_id, session in self.sessions.items():
            try:
                last_activity = datetime.fromisoformat(session["last_activity"])
                if (now - last_activity).seconds > self.session_timeout:
                    expired.append(session_id)
            except (ValueError, KeyError):
                # Handle malformed timestamps or missing last_activity
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")