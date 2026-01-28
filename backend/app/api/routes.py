"""API routes for RLM Engine."""
import hashlib
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import json
import asyncio

from app.database import get_db
from app.repositories.execution import ExecutionRepository
from app.repositories.session import SessionRepository
from app.engine.agent import LettaAgent, AgentConfig
from app.engine.llm import LLMClient
from app.models.execution import ExecutionStatus
from app.api.schemas import (
    SessionCreate, SessionUpdate, SessionResponse, SessionListResponse,
    ExecutionCreate, ExecutionResponse, ExecutionDetailResponse, ExecutionListResponse,
    MemorySetRequest, MemoryDictResponse,
    UsageStats,
)

router = APIRouter()


# ============ Health Check ============

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rlm-engine"}


# ============ Session Endpoints ============

@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new session."""
    repo = SessionRepository(db)
    
    context_metadata = request.context_metadata or {}
    if request.context:
        context_metadata.update({
            "size": len(request.context),
            "hash": hashlib.sha256(request.context.encode()).hexdigest(),
        })
    
    session = await repo.create_session(
        name=request.name,
        context=request.context,
        context_metadata=context_metadata,
    )
    
    return SessionResponse(
        id=session.id,
        name=session.name,
        context_metadata=session.context_metadata,
        created_at=session.created_at,
        updated_at=session.updated_at,
        memory_count=0,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List all sessions."""
    repo = SessionRepository(db)
    sessions = await repo.list_sessions(limit=limit, offset=offset)
    
    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=s.id,
                name=s.name,
                context_metadata=s.context_metadata,
                created_at=s.created_at,
                updated_at=s.updated_at,
                memory_count=len(s.memories) if s.memories else 0,
            )
            for s in sessions
        ],
        total=len(sessions),
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a session by ID."""
    repo = SessionRepository(db)
    session = await repo.get_session(session_id, include_memories=True)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        id=session.id,
        name=session.name,
        context_metadata=session.context_metadata,
        created_at=session.created_at,
        updated_at=session.updated_at,
        memory_count=len(session.memories) if session.memories else 0,
    )


@router.put("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    request: SessionUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a session."""
    repo = SessionRepository(db)
    
    context_metadata = request.context_metadata
    if request.context:
        context_metadata = context_metadata or {}
        context_metadata.update({
            "size": len(request.context),
            "hash": hashlib.sha256(request.context.encode()).hexdigest(),
        })
    
    session = await repo.update_session(
        session_id=session_id,
        name=request.name,
        context=request.context,
        context_metadata=context_metadata,
    )
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        id=session.id,
        name=session.name,
        context_metadata=session.context_metadata,
        created_at=session.created_at,
        updated_at=session.updated_at,
        memory_count=len(session.memories) if session.memories else 0,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a session."""
    repo = SessionRepository(db)
    deleted = await repo.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"status": "deleted", "session_id": session_id}


# ============ Memory Endpoints ============

@router.get("/sessions/{session_id}/memory", response_model=MemoryDictResponse)
async def get_session_memory(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get all memory for a session."""
    repo = SessionRepository(db)
    session = await repo.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    memory = await repo.get_session_memory(session_id)
    return MemoryDictResponse(memory=memory)


@router.post("/sessions/{session_id}/memory")
async def set_memory(
    session_id: str,
    request: MemorySetRequest,
    db: AsyncSession = Depends(get_db),
):
    """Set a memory value."""
    repo = SessionRepository(db)
    session = await repo.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    memory = await repo.set_memory(session_id, request.key, request.value)
    
    return {
        "key": memory.key,
        "value": memory.value,
        "updated_at": memory.updated_at.isoformat(),
    }


@router.delete("/sessions/{session_id}/memory/{key}")
async def delete_memory(
    session_id: str,
    key: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a memory value."""
    repo = SessionRepository(db)
    deleted = await repo.delete_memory(session_id, key)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory key not found")
    
    return {"status": "deleted", "key": key}


# ============ Execution Endpoints ============

@router.post("/execute", response_model=ExecutionResponse)
async def create_execution(
    request: ExecutionCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create and run an execution.
    
    This is a synchronous endpoint that waits for the execution to complete.
    For long-running executions, use the streaming endpoint instead.
    """
    session_repo = SessionRepository(db)
    execution_repo = ExecutionRepository(db)
    
    # Get context
    context = request.context
    memory = {}
    
    if request.session_id:
        session = await session_repo.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.stored_context:
            context = session.stored_context
        memory = await session_repo.get_session_memory(request.session_id)
    
    if not context:
        raise HTTPException(
            status_code=400,
            detail="Either context or session_id with stored context is required"
        )
    
    # Create LLM client and agent
    llm_client = LLMClient()
    config = AgentConfig(model=request.model) if request.model else AgentConfig()
    agent = LettaAgent(llm_client=llm_client, config=config)
    
    # Run the agent
    trace = await agent.run(
        user_query=request.user_query,
        context=context,
        memory=memory,
    )
    
    # Save to database
    execution = await execution_repo.save_execution_trace(trace)
    if request.session_id:
        execution.session_id = request.session_id
        await db.flush()
    
    return ExecutionResponse(
        id=execution.id,
        session_id=execution.session_id,
        user_query=execution.user_query,
        context_size=execution.context_size,
        status=execution.status.value,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        total_input_tokens=execution.total_input_tokens,
        total_output_tokens=execution.total_output_tokens,
        total_cost_usd=execution.total_cost_usd,
        final_result=execution.final_result,
        error_message=execution.error_message,
    )


@router.post("/execute/stream")
async def create_execution_stream(
    request: ExecutionCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create and run an execution with streaming updates.
    
    Returns a server-sent events stream with real-time updates.
    """
    session_repo = SessionRepository(db)
    
    # Get context
    context = request.context
    memory = {}
    
    if request.session_id:
        session = await session_repo.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.stored_context:
            context = session.stored_context
        memory = await session_repo.get_session_memory(request.session_id)
    
    if not context:
        raise HTTPException(
            status_code=400,
            detail="Either context or session_id with stored context is required"
        )
    
    async def event_generator():
        """Generate server-sent events."""
        updates_queue = asyncio.Queue()
        
        def on_update(update: Dict[str, Any]):
            asyncio.get_event_loop().call_soon_threadsafe(
                updates_queue.put_nowait, update
            )
        
        # Create agent with update callback
        llm_client = LLMClient()
        config = AgentConfig(model=request.model) if request.model else AgentConfig()
        agent = LettaAgent(
            llm_client=llm_client,
            config=config,
            on_node_update=on_update,
        )
        
        # Start execution in background
        async def run_agent():
            try:
                trace = await agent.run(
                    user_query=request.user_query,
                    context=context,
                    memory=memory,
                )
                await updates_queue.put({"type": "complete", "data": trace.to_dict()})
            except Exception as e:
                await updates_queue.put({"type": "error", "data": {"error": str(e)}})
        
        task = asyncio.create_task(run_agent())
        
        try:
            while True:
                try:
                    update = await asyncio.wait_for(updates_queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(update)}\n\n"
                    
                    if update["type"] in ["complete", "error"]:
                        break
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            if not task.done():
                task.cancel()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/executions", response_model=ExecutionListResponse)
async def list_executions(
    session_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List executions."""
    repo = ExecutionRepository(db)
    executions = await repo.list_executions(
        session_id=session_id,
        limit=limit,
        offset=offset,
    )
    
    return ExecutionListResponse(
        executions=[
            ExecutionResponse(
                id=e.id,
                session_id=e.session_id,
                user_query=e.user_query,
                context_size=e.context_size,
                status=e.status.value,
                started_at=e.started_at,
                completed_at=e.completed_at,
                total_input_tokens=e.total_input_tokens,
                total_output_tokens=e.total_output_tokens,
                total_cost_usd=e.total_cost_usd,
                final_result=e.final_result,
                error_message=e.error_message,
            )
            for e in executions
        ],
        total=len(executions),
    )


@router.get("/executions/{execution_id}", response_model=ExecutionDetailResponse)
async def get_execution(
    execution_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get execution details including the execution tree."""
    repo = ExecutionRepository(db)
    execution = await repo.get_execution(execution_id, include_nodes=True)
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    # Get tree structure
    tree_data = await repo.get_execution_tree(execution_id)
    
    # Get generated code from root node
    root_node = next((n for n in execution.nodes if n.depth == 0), None)
    
    return ExecutionDetailResponse(
        id=execution.id,
        session_id=execution.session_id,
        user_query=execution.user_query,
        context_size=execution.context_size,
        status=execution.status.value,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        total_input_tokens=execution.total_input_tokens,
        total_output_tokens=execution.total_output_tokens,
        total_cost_usd=execution.total_cost_usd,
        final_result=execution.final_result,
        error_message=execution.error_message,
        tree=tree_data,
        generated_code=root_node.generated_code if root_node else None,
    )


@router.get("/executions/{execution_id}/tree")
async def get_execution_tree(
    execution_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get the execution tree structure for visualization."""
    repo = ExecutionRepository(db)
    execution = await repo.get_execution(execution_id)
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    tree = await repo.get_execution_tree(execution_id)
    return tree


# ============ Stats Endpoints ============

@router.get("/stats", response_model=UsageStats)
async def get_stats(
    session_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get usage statistics."""
    repo = ExecutionRepository(db)
    executions = await repo.list_executions(session_id=session_id, limit=1000)
    
    total_input_tokens = sum(e.total_input_tokens for e in executions)
    total_output_tokens = sum(e.total_output_tokens for e in executions)
    total_cost = sum(e.total_cost_usd for e in executions)
    
    status_counts = {}
    for e in executions:
        status = e.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return UsageStats(
        total_executions=len(executions),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=total_cost,
        average_cost_per_execution=total_cost / len(executions) if executions else 0,
        executions_by_status=status_counts,
    )
