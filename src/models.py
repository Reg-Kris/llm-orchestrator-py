"""
Local database models for LLM Orchestrator service.
These mirror the models from pyairtable-common for conversation persistence.
"""
import enum
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, ForeignKey, Boolean, Enum as SQLEnum, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped

Base = declarative_base()


class SessionStatus(enum.Enum):
    """Conversation session status enum."""
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    ERROR = "error"


class MessageRole(enum.Enum):
    """Message role enum."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationSession(Base):
    """Conversation session model for persistent chat management."""
    
    __tablename__ = 'conversation_sessions'
    __table_args__ = {'schema': 'conversations'}
    
    # Primary key
    id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Timestamps
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    updated_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Session identification
    session_key: Mapped[str] = Column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )
    
    # User and context information
    user_id: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        index=True
    )
    
    base_id: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        index=True
    )
    
    table_id: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True
    )
    
    # Session metadata
    title: Mapped[Optional[str]] = Column(
        String(500),
        nullable=True
    )
    
    description: Mapped[Optional[str]] = Column(
        Text,
        nullable=True
    )
    
    # Session state
    status: Mapped[SessionStatus] = Column(
        SQLEnum(SessionStatus, name="session_status"),
        default=SessionStatus.ACTIVE,
        nullable=False,
        index=True
    )
    
    last_activity: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )
    
    expires_at: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True
    )
    
    # Configuration and settings
    model_config: Mapped[Optional[Dict[str, Any]]] = Column(
        JSONB,
        nullable=True
    )
    
    system_prompt: Mapped[Optional[str]] = Column(
        Text,
        nullable=True
    )
    
    context_data: Mapped[Optional[Dict[str, Any]]] = Column(
        JSONB,
        nullable=True
    )
    
    # Statistics
    message_count: Mapped[int] = Column(
        Integer,
        default=0,
        nullable=False
    )
    
    total_tokens: Mapped[int] = Column(
        Integer,
        default=0,
        nullable=False
    )
    
    tool_executions: Mapped[int] = Column(
        Integer,
        default=0,
        nullable=False
    )
    
    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def add_message_stats(self, token_count: int = 0, tool_executions: int = 0):
        """Update session statistics when a message is added."""
        self.message_count += 1
        self.total_tokens += token_count
        self.tool_executions += tool_executions
        self.update_activity()


class Message(Base):
    """Individual message model for chat conversations."""
    
    __tablename__ = 'messages'
    __table_args__ = {'schema': 'conversations'}
    
    # Primary key
    id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Timestamps
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    updated_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Session relationship
    session_id: Mapped[UUID] = Column(
        UUID(as_uuid=True),
        ForeignKey('conversations.conversation_sessions.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    session: Mapped[ConversationSession] = relationship(
        "ConversationSession",
        back_populates="messages"
    )
    
    # Message identification
    sequence_number: Mapped[int] = Column(
        Integer,
        nullable=False
    )
    
    role: Mapped[MessageRole] = Column(
        SQLEnum(MessageRole, name="message_role"),
        nullable=False,
        index=True
    )
    
    # Message content
    content: Mapped[str] = Column(
        Text,
        nullable=False
    )
    
    thinking_process: Mapped[Optional[str]] = Column(
        Text,
        nullable=True
    )
    
    # Tool execution details
    tools_used: Mapped[Optional[List[Dict[str, Any]]]] = Column(
        JSONB,
        nullable=True
    )
    
    tool_results: Mapped[Optional[List[Dict[str, Any]]]] = Column(
        JSONB,
        nullable=True
    )
    
    # Metadata
    token_count: Mapped[int] = Column(
        Integer,
        default=0,
        nullable=False
    )
    
    processing_time_ms: Mapped[Optional[int]] = Column(
        Integer,
        nullable=True
    )
    
    model_used: Mapped[Optional[str]] = Column(
        String(100),
        nullable=True
    )
    
    # Status and error handling
    is_error: Mapped[bool] = Column(
        Boolean,
        default=False,
        nullable=False
    )
    
    error_details: Mapped[Optional[Dict[str, Any]]] = Column(
        JSONB,
        nullable=True
    )
    
    # Additional metadata
    metadata: Mapped[Optional[Dict[str, Any]]] = Column(
        JSONB,
        nullable=True
    )


class ToolExecution(Base):
    """Detailed tool execution tracking."""
    
    __tablename__ = 'tool_executions'
    __table_args__ = {'schema': 'conversations'}
    
    # Primary key
    id: Mapped[uuid.UUID] = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Timestamps
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Message relationship
    message_id: Mapped[UUID] = Column(
        UUID(as_uuid=True),
        ForeignKey('conversations.messages.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    message: Mapped[Message] = relationship("Message")
    
    # Tool details
    tool_name: Mapped[str] = Column(
        String(100),
        nullable=False,
        index=True
    )
    
    tool_version: Mapped[Optional[str]] = Column(
        String(50),
        nullable=True
    )
    
    # Execution details
    input_parameters: Mapped[Dict[str, Any]] = Column(
        JSONB,
        nullable=False
    )
    
    output_result: Mapped[Optional[Dict[str, Any]]] = Column(
        JSONB,
        nullable=True
    )
    
    # Performance metrics
    execution_time_ms: Mapped[int] = Column(
        Integer,
        nullable=False
    )
    
    memory_usage_mb: Mapped[Optional[int]] = Column(
        Integer,
        nullable=True
    )
    
    # Status
    success: Mapped[bool] = Column(
        Boolean,
        nullable=False
    )
    
    error_message: Mapped[Optional[str]] = Column(
        Text,
        nullable=True
    )
    
    error_type: Mapped[Optional[str]] = Column(
        String(100),
        nullable=True
    )
    
    # Context
    correlation_context: Mapped[Optional[Dict[str, Any]]] = Column(
        JSONB,
        nullable=True
    )