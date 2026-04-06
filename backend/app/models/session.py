"""Session model for managing agent sessions."""
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, Text, DateTime, JSON
from sqlalchemy.orm import relationship
from app.models.base import Base, generate_uuid


class Session(Base):
    """
    Represents a user session with persistent memory.
    
    A session can have multiple executions, and memory persists
    across executions within the same session.
    """
    __tablename__ = "sessions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=True)
    
    # Context storage
    # Large contexts are stored here instead of in the prompt
    stored_context = Column(Text, nullable=True)
    context_metadata = Column(JSON, nullable=True)  # {size, hash, type, etc.}
    
    # Session metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    executions = relationship("Execution", back_populates="session", cascade="all, delete-orphan")
    memories = relationship("AgentMemory", back_populates="session", cascade="all, delete-orphan")
    
    def to_dict(self, include_executions: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "context_metadata": self.context_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "memory_count": len(self.memories) if self.memories else 0,
        }
        if include_executions:
            result["executions"] = [e.to_dict() for e in self.executions]
        return result
