"""Execution models for tracking agent runs."""
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
from app.models.base import Base, generate_uuid
import enum


class ExecutionStatus(str, enum.Enum):
    """Status of an execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(str, enum.Enum):
    """Type of execution node."""
    ROOT = "root"
    CHILD = "child"


class Execution(Base):
    """
    Represents a complete execution run.
    
    An execution contains a tree of ExecutionNodes, where each node
    represents either the root agent or a child agent spawned via llm_query().
    """
    __tablename__ = "executions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=True)
    
    # Input
    user_query = Column(Text, nullable=False)
    context_size = Column(Integer, nullable=False)  # Size of context in characters
    context_hash = Column(String(64), nullable=True)  # Hash of context for deduplication
    
    # Status
    status = Column(SQLEnum(ExecutionStatus), default=ExecutionStatus.PENDING)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Cost tracking
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    
    # Result
    final_result = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    nodes = relationship("ExecutionNode", back_populates="execution", cascade="all, delete-orphan")
    session = relationship("Session", back_populates="executions")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_query": self.user_query,
            "context_size": self.context_size,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "final_result": self.final_result,
            "error_message": self.error_message,
        }


class ExecutionNode(Base):
    """
    Represents a single agent execution within the tree.
    
    The root node is the main agent that receives the user query.
    Child nodes are spawned when the agent calls llm_query().
    """
    __tablename__ = "execution_nodes"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    execution_id = Column(String(36), ForeignKey("executions.id"), nullable=False)
    parent_node_id = Column(String(36), ForeignKey("execution_nodes.id"), nullable=True)
    
    # Node info
    node_type = Column(SQLEnum(NodeType), default=NodeType.ROOT)
    depth = Column(Integer, default=0)  # 0 for root, 1+ for children
    sequence_number = Column(Integer, default=0)  # Order among siblings
    
    # Input to this node
    prompt = Column(Text, nullable=True)  # The prompt/query given to this agent
    chunk_start = Column(Integer, nullable=True)  # If processing a chunk, start index
    chunk_end = Column(Integer, nullable=True)  # If processing a chunk, end index
    
    # Agent's generated code
    generated_code = Column(Text, nullable=True)
    
    # Status
    status = Column(SQLEnum(ExecutionStatus), default=ExecutionStatus.PENDING)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # LLM call details
    model_used = Column(String(100), nullable=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    
    # Output
    output = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Memory snapshot before and after
    memory_before = Column(JSON, nullable=True)
    memory_after = Column(JSON, nullable=True)
    
    # Relationships
    execution = relationship("Execution", back_populates="nodes")
    parent = relationship("ExecutionNode", remote_side=[id], backref="children")
    
    def to_dict(self, include_children: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "execution_id": self.execution_id,
            "parent_node_id": self.parent_node_id,
            "node_type": self.node_type.value,
            "depth": self.depth,
            "sequence_number": self.sequence_number,
            "prompt": self.prompt,
            "chunk_start": self.chunk_start,
            "chunk_end": self.chunk_end,
            "generated_code": self.generated_code,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "model_used": self.model_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "output": self.output,
            "error_message": self.error_message,
            "memory_before": self.memory_before,
            "memory_after": self.memory_after,
        }
        if include_children:
            result["children"] = [child.to_dict(include_children=True) for child in self.children]
        return result


class AgentMemory(Base):
    """
    Persistent memory for agents across executions.
    
    Stores learned facts, preferences, and other information
    that should persist between agent runs.
    """
    __tablename__ = "agent_memories"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    
    # Memory content
    key = Column(String(255), nullable=False)
    value = Column(JSON, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    source_execution_id = Column(String(36), ForeignKey("executions.id"), nullable=True)
    source_node_id = Column(String(36), ForeignKey("execution_nodes.id"), nullable=True)
    
    # Relationships
    session = relationship("Session", back_populates="memories")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
