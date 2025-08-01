"""
Database-backed session management for persistent conversation storage.
"""
import uuid
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, and_, func
from sqlalchemy.exc import IntegrityError, NoResultFound
import logging

# Import conversation models (in production, these would come from pyairtable-common)
from .models import ConversationSession, Message, ToolExecution, SessionStatus, MessageRole

logger = logging.getLogger(__name__)


class DatabaseSessionManager:
    """Database-backed session manager for persistent conversation storage."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.session_factory = None
        self.initialized = False
    
    def _get_database_url(self) -> str:
        """Get database URL from environment."""
        url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        
        if url:
            # Convert postgres:// to postgresql+asyncpg://
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif not url.startswith("postgresql+asyncpg://"):
                if url.startswith("postgresql://"):
                    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        else:
            # Default local development URL
            url = "postgresql+asyncpg://postgres:postgres@localhost:5432/pyairtable_platform"
        
        return url
    
    async def initialize(self):
        """Initialize database connection."""
        if self.initialized:
            return
        
        try:
            self.engine = create_async_engine(
                self.database_url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                echo=os.getenv("SQL_ECHO", "false").lower() == "true"
            )
            
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.session_factory() as session:
                await session.execute(select(1))
            
            self.initialized = True
            logger.info("Database session manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database session manager: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database session manager closed")
    
    @asynccontextmanager
    async def get_db_session(self):
        """Get database session with automatic cleanup."""
        if not self.initialized:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def get_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        base_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> ConversationSession:
        """Get existing session or create new one."""
        async with self.get_db_session() as db_session:
            # Try to find existing session
            stmt = select(ConversationSession).where(
                ConversationSession.session_key == session_id
            )
            result = await db_session.execute(stmt)
            conversation_session = result.scalar_one_or_none()
            
            if conversation_session:
                # Update last activity
                conversation_session.update_activity()
                await db_session.flush()
                logger.debug(f"Retrieved existing session: {session_id}")
                return conversation_session
            
            # Create new session
            try:
                conversation_session = ConversationSession(
                    session_key=session_id,
                    user_id=user_id,
                    base_id=base_id,
                    title=title or f"Chat Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                    status=SessionStatus.ACTIVE,
                    last_activity=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(hours=24),  # 24 hour expiry
                    model_config={
                        "model": "gemini-2.5-flash",
                        "temperature": 0.1,
                        "max_tokens": 4000
                    }
                )
                
                db_session.add(conversation_session)
                await db_session.flush()
                await db_session.refresh(conversation_session)
                
                logger.info(f"Created new session: {session_id}")
                return conversation_session
                
            except IntegrityError as e:
                await db_session.rollback()
                logger.error(f"Failed to create session {session_id}: {e}")
                raise
    
    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        thinking_process: Optional[str] = None,
        tools_used: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        token_count: int = 0,
        processing_time_ms: Optional[int] = None,
        model_used: Optional[str] = None,
        is_error: bool = False,
        error_details: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add a message to the conversation."""
        async with self.get_db_session() as db_session:
            # Get the session
            stmt = select(ConversationSession).where(
                ConversationSession.session_key == session_id
            )
            result = await db_session.execute(stmt)
            conversation_session = result.scalar_one_or_none()
            
            if not conversation_session:
                raise ValueError(f"Session {session_id} not found")
            
            # Get next sequence number
            stmt = select(func.max(Message.sequence_number)).where(
                Message.session_id == conversation_session.id
            )
            result = await db_session.execute(stmt)
            max_seq = result.scalar() or 0
            sequence_number = max_seq + 1
            
            # Create message
            message = Message(
                session_id=conversation_session.id,
                sequence_number=sequence_number,
                role=role,
                content=content,
                thinking_process=thinking_process,
                tools_used=tools_used,
                tool_results=tool_results,
                token_count=token_count,
                processing_time_ms=processing_time_ms,
                model_used=model_used,
                is_error=is_error,
                error_details=error_details
            )
            
            db_session.add(message)
            
            # Update session statistics
            conversation_session.add_message_stats(
                token_count=token_count,
                tool_executions=len(tools_used) if tools_used else 0
            )
            
            await db_session.flush()
            await db_session.refresh(message)
            
            logger.debug(f"Added message to session {session_id}: {role.value}")
            return message
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Message]:
        """Get conversation history for a session."""
        async with self.get_db_session() as db_session:
            # Get session ID
            stmt = select(ConversationSession.id).where(
                ConversationSession.session_key == session_id
            )
            result = await db_session.execute(stmt)
            db_session_id = result.scalar_one_or_none()
            
            if not db_session_id:
                return []
            
            # Get messages
            stmt = (
                select(Message)
                .where(Message.session_id == db_session_id)
                .order_by(Message.sequence_number.desc())
                .limit(limit)
                .offset(offset)
            )
            
            result = await db_session.execute(stmt)
            messages = list(result.scalars().all())
            
            # Return in chronological order (oldest first)
            messages.reverse()
            
            logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
    
    async def get_session_info(self, session_id: str) -> Optional[ConversationSession]:
        """Get session information."""
        async with self.get_db_session() as db_session:
            stmt = select(ConversationSession).where(
                ConversationSession.session_key == session_id
            )
            result = await db_session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        async with self.get_db_session() as db_session:
            # Get session
            stmt = select(ConversationSession).where(
                ConversationSession.session_key == session_id
            )
            result = await db_session.execute(stmt)
            conversation_session = result.scalar_one_or_none()
            
            if not conversation_session:
                return False
            
            # Delete session (messages will be cascade deleted)
            await db_session.delete(conversation_session)
            await db_session.flush()
            
            logger.info(f"Deleted session: {session_id}")
            return True
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        async with self.get_db_session() as db_session:
            now = datetime.utcnow()
            
            # Find expired sessions
            stmt = select(ConversationSession).where(
                and_(
                    ConversationSession.expires_at < now,
                    ConversationSession.status == SessionStatus.ACTIVE
                )
            )
            
            result = await db_session.execute(stmt)
            expired_sessions = list(result.scalars().all())
            
            # Mark as expired (soft delete)
            for session in expired_sessions:
                session.status = SessionStatus.EXPIRED
            
            await db_session.flush()
            
            count = len(expired_sessions)
            if count > 0:
                logger.info(f"Marked {count} sessions as expired")
            
            return count
    
    async def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        async with self.get_db_session() as db_session:
            stmt = select(func.count(ConversationSession.id)).where(
                ConversationSession.status == SessionStatus.ACTIVE
            )
            result = await db_session.execute(stmt)
            return result.scalar() or 0
    
    async def record_tool_execution(
        self,
        message_id: uuid.UUID,
        tool_name: str,
        input_parameters: Dict[str, Any],
        output_result: Optional[Dict[str, Any]],
        execution_time_ms: int,
        success: bool,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None
    ) -> ToolExecution:
        """Record detailed tool execution information."""
        async with self.get_db_session() as db_session:
            tool_execution = ToolExecution(
                message_id=message_id,
                tool_name=tool_name,
                input_parameters=input_parameters,
                output_result=output_result,
                execution_time_ms=execution_time_ms,
                success=success,
                error_message=error_message,
                error_type=error_type
            )
            
            db_session.add(tool_execution)
            await db_session.flush()
            await db_session.refresh(tool_execution)
            
            logger.debug(f"Recorded tool execution: {tool_name} ({'success' if success else 'failure'})")
            return tool_execution
    
    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        async with self.get_db_session() as db_session:
            # Get session
            stmt = select(ConversationSession).where(
                ConversationSession.session_key == session_id
            )
            result = await db_session.execute(stmt)
            conversation_session = result.scalar_one_or_none()
            
            if not conversation_session:
                return {}
            
            return {
                "session_id": session_id,
                "created_at": conversation_session.created_at.isoformat(),
                "last_activity": conversation_session.last_activity.isoformat(),
                "status": conversation_session.status.value,
                "message_count": conversation_session.message_count,
                "total_tokens": conversation_session.total_tokens,
                "tool_executions": conversation_session.tool_executions,
                "model_config": conversation_session.model_config,
                "base_id": conversation_session.base_id,
                "title": conversation_session.title
            }


# Global session manager instance
_session_manager: Optional[DatabaseSessionManager] = None


def get_session_manager() -> DatabaseSessionManager:
    """Get global session manager instance."""
    global _session_manager
    
    if _session_manager is None:
        _session_manager = DatabaseSessionManager()
    
    return _session_manager