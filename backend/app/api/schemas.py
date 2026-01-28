"""Pydantic schemas for API requests and responses."""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


# ============ Session Schemas ============

class SessionCreate(BaseModel):
    """Request to create a new session."""
    name: Optional[str] = None
    context: Optional[str] = Field(None, description="Large context to store")
    context_metadata: Optional[Dict[str, Any]] = None


class SessionUpdate(BaseModel):
    """Request to update a session."""
    name: Optional[str] = None
    context: Optional[str] = None
    context_metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Session response."""
    id: str
    name: Optional[str]
    context_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    memory_count: int = 0

    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    """List of sessions."""
    sessions: List[SessionResponse]
    total: int


# ============ Execution Schemas ============

class ExecutionCreate(BaseModel):
    """Request to create and run an execution."""
    user_query: str = Field(..., description="The question or task")
    context: Optional[str] = Field(None, description="Context to process (if not using session context)")
    session_id: Optional[str] = Field(None, description="Session ID to use stored context and memory")
    model: Optional[str] = Field(None, description="Model to use (defaults to gpt-4-turbo-preview)")


class ExecutionResponse(BaseModel):
    """Execution response."""
    id: str
    session_id: Optional[str]
    user_query: str
    context_size: int
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    final_result: Optional[str]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class ExecutionDetailResponse(ExecutionResponse):
    """Detailed execution response with tree."""
    tree: Optional[Dict[str, Any]] = None
    generated_code: Optional[str] = None


class ExecutionListResponse(BaseModel):
    """List of executions."""
    executions: List[ExecutionResponse]
    total: int


# ============ Execution Node Schemas ============

class ExecutionNodeResponse(BaseModel):
    """Execution node response."""
    id: str
    execution_id: str
    parent_node_id: Optional[str]
    node_type: str
    depth: int
    sequence_number: int
    prompt: Optional[str]
    generated_code: Optional[str]
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    model_used: Optional[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    output: Optional[str]
    error_message: Optional[str]
    memory_before: Optional[Dict[str, Any]]
    memory_after: Optional[Dict[str, Any]]
    children: List["ExecutionNodeResponse"] = []

    class Config:
        from_attributes = True


# ============ Memory Schemas ============

class MemorySetRequest(BaseModel):
    """Request to set a memory value."""
    key: str
    value: Any


class MemoryResponse(BaseModel):
    """Memory response."""
    key: str
    value: Any
    created_at: datetime
    updated_at: datetime


class MemoryDictResponse(BaseModel):
    """All memory as a dictionary."""
    memory: Dict[str, Any]


# ============ Stats Schemas ============

class UsageStats(BaseModel):
    """Usage statistics."""
    total_executions: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    average_cost_per_execution: float
    executions_by_status: Dict[str, int]


# Enable forward references
ExecutionNodeResponse.model_rebuild()
