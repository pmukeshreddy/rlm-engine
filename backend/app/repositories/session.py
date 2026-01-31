"""Repository for session-related database operations."""
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload

from app.models.session import Session
from app.models.execution import AgentMemory


class SessionRepository:
    """Repository for managing sessions and agent memory."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_session(
        self,
        name: Optional[str] = None,
        context: Optional[str] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new session."""
        sess = Session(
            name=name,
            stored_context=context,
            context_metadata=context_metadata,
        )
        self.session.add(sess)
        await self.session.flush()
        return sess
    
    async def get_session(
        self,
        session_id: str,
        include_executions: bool = False,
        include_memories: bool = False,
    ) -> Optional[Session]:
        """Get a session by ID."""
        query = select(Session).where(Session.id == session_id)
        
        if include_executions:
            query = query.options(selectinload(Session.executions))
        if include_memories:
            query = query.options(selectinload(Session.memories))
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def update_session(
        self,
        session_id: str,
        name: Optional[str] = None,
        context: Optional[str] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Session]:
        """Update a session."""
        sess = await self.get_session(session_id)
        
        if sess:
            if name is not None:
                sess.name = name
            if context is not None:
                sess.stored_context = context
            if context_metadata is not None:
                sess.context_metadata = context_metadata
            sess.updated_at = datetime.utcnow()
            await self.session.flush()
        
        return sess
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data."""
        sess = await self.get_session(session_id)
        if sess:
            await self.session.delete(sess)
            await self.session.flush()
            return True
        return False
    
    async def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Session]:
        """List all sessions."""
        result = await self.session.execute(
            select(Session)
            .options(selectinload(Session.memories))
            .order_by(desc(Session.updated_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())
    
    async def get_session_memory(self, session_id: str) -> Dict[str, Any]:
        """Get all memory for a session as a dictionary."""
        result = await self.session.execute(
            select(AgentMemory).where(AgentMemory.session_id == session_id)
        )
        memories = result.scalars().all()
        
        return {mem.key: mem.value for mem in memories}
    
    async def set_memory(
        self,
        session_id: str,
        key: str,
        value: Any,
        source_execution_id: Optional[str] = None,
        source_node_id: Optional[str] = None,
    ) -> AgentMemory:
        """Set a memory value for a session."""
        # Check if key exists
        result = await self.session.execute(
            select(AgentMemory)
            .where(AgentMemory.session_id == session_id)
            .where(AgentMemory.key == key)
        )
        memory = result.scalar_one_or_none()
        
        if memory:
            memory.value = value
            memory.updated_at = datetime.utcnow()
            if source_execution_id:
                memory.source_execution_id = source_execution_id
            if source_node_id:
                memory.source_node_id = source_node_id
        else:
            memory = AgentMemory(
                session_id=session_id,
                key=key,
                value=value,
                source_execution_id=source_execution_id,
                source_node_id=source_node_id,
            )
            self.session.add(memory)
        
        await self.session.flush()
        return memory
    
    async def delete_memory(self, session_id: str, key: str) -> bool:
        """Delete a specific memory key."""
        result = await self.session.execute(
            select(AgentMemory)
            .where(AgentMemory.session_id == session_id)
            .where(AgentMemory.key == key)
        )
        memory = result.scalar_one_or_none()
        
        if memory:
            await self.session.delete(memory)
            await self.session.flush()
            return True
        return False
    
    async def clear_memory(self, session_id: str) -> int:
        """Clear all memory for a session."""
        result = await self.session.execute(
            select(AgentMemory).where(AgentMemory.session_id == session_id)
        )
        memories = list(result.scalars().all())
        
        for memory in memories:
            await self.session.delete(memory)
        
        await self.session.flush()
        return len(memories)
