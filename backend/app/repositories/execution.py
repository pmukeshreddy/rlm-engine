"""Repository for execution-related database operations."""
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload

from app.models.execution import Execution, ExecutionNode, ExecutionStatus, NodeType
from app.engine.agent import ExecutionTrace


class ExecutionRepository:
    """Repository for managing executions and execution nodes."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_execution(
        self,
        user_query: str,
        context_size: int,
        session_id: Optional[str] = None,
        context_hash: Optional[str] = None,
    ) -> Execution:
        """Create a new execution record."""
        execution = Execution(
            user_query=user_query,
            context_size=context_size,
            session_id=session_id,
            context_hash=context_hash,
            status=ExecutionStatus.PENDING,
        )
        self.session.add(execution)
        await self.session.flush()
        return execution
    
    async def update_execution(
        self,
        execution_id: str,
        status: Optional[ExecutionStatus] = None,
        final_result: Optional[str] = None,
        error_message: Optional[str] = None,
        total_input_tokens: Optional[int] = None,
        total_output_tokens: Optional[int] = None,
        total_cost_usd: Optional[float] = None,
    ) -> Optional[Execution]:
        """Update an execution record."""
        result = await self.session.execute(
            select(Execution).where(Execution.id == execution_id)
        )
        execution = result.scalar_one_or_none()
        
        if execution:
            if status:
                execution.status = status
                if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
                    execution.completed_at = datetime.utcnow()
            if final_result is not None:
                execution.final_result = final_result
            if error_message is not None:
                execution.error_message = error_message
            if total_input_tokens is not None:
                execution.total_input_tokens = total_input_tokens
            if total_output_tokens is not None:
                execution.total_output_tokens = total_output_tokens
            if total_cost_usd is not None:
                execution.total_cost_usd = total_cost_usd
            
            await self.session.flush()
        
        return execution
    
    async def get_execution(self, execution_id: str, include_nodes: bool = False) -> Optional[Execution]:
        """Get an execution by ID."""
        query = select(Execution).where(Execution.id == execution_id)
        if include_nodes:
            query = query.options(selectinload(Execution.nodes))
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def list_executions(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Execution]:
        """List executions, optionally filtered by session."""
        query = select(Execution).order_by(desc(Execution.started_at))
        
        if session_id:
            query = query.where(Execution.session_id == session_id)
        
        query = query.limit(limit).offset(offset)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def create_node(
        self,
        execution_id: str,
        node_type: NodeType,
        prompt: Optional[str] = None,
        parent_node_id: Optional[str] = None,
        depth: int = 0,
        sequence_number: int = 0,
    ) -> ExecutionNode:
        """Create a new execution node."""
        node = ExecutionNode(
            execution_id=execution_id,
            node_type=node_type,
            prompt=prompt,
            parent_node_id=parent_node_id,
            depth=depth,
            sequence_number=sequence_number,
            status=ExecutionStatus.PENDING,
        )
        self.session.add(node)
        await self.session.flush()
        return node
    
    async def update_node(
        self,
        node_id: str,
        status: Optional[ExecutionStatus] = None,
        generated_code: Optional[str] = None,
        output: Optional[str] = None,
        error_message: Optional[str] = None,
        model_used: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        memory_before: Optional[Dict[str, Any]] = None,
        memory_after: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExecutionNode]:
        """Update an execution node."""
        result = await self.session.execute(
            select(ExecutionNode).where(ExecutionNode.id == node_id)
        )
        node = result.scalar_one_or_none()
        
        if node:
            if status:
                node.status = status
                if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
                    node.completed_at = datetime.utcnow()
            if generated_code is not None:
                node.generated_code = generated_code
            if output is not None:
                node.output = output
            if error_message is not None:
                node.error_message = error_message
            if model_used:
                node.model_used = model_used
            if input_tokens is not None:
                node.input_tokens = input_tokens
            if output_tokens is not None:
                node.output_tokens = output_tokens
            if cost_usd is not None:
                node.cost_usd = cost_usd
            if memory_before is not None:
                node.memory_before = memory_before
            if memory_after is not None:
                node.memory_after = memory_after
            
            await self.session.flush()
        
        return node
    
    async def get_node(self, node_id: str) -> Optional[ExecutionNode]:
        """Get a node by ID."""
        result = await self.session.execute(
            select(ExecutionNode).where(ExecutionNode.id == node_id)
        )
        return result.scalar_one_or_none()
    
    async def get_execution_tree(self, execution_id: str) -> Dict[str, Any]:
        """Get the full execution tree for visualization."""
        result = await self.session.execute(
            select(ExecutionNode)
            .where(ExecutionNode.execution_id == execution_id)
            .order_by(ExecutionNode.depth, ExecutionNode.sequence_number)
        )
        nodes = list(result.scalars().all())
        
        # Build tree structure
        node_map = {node.id: node.to_dict() for node in nodes}
        root_nodes = []
        
        for node in nodes:
            node_dict = node_map[node.id]
            node_dict['children'] = []
            
            if node.parent_node_id and node.parent_node_id in node_map:
                node_map[node.parent_node_id]['children'].append(node_dict)
            else:
                root_nodes.append(node_dict)
        
        return {
            "execution_id": execution_id,
            "tree": root_nodes[0] if root_nodes else None,
            "total_nodes": len(nodes),
        }
    
    async def save_execution_trace(self, trace: ExecutionTrace) -> Execution:
        """Save a complete execution trace to the database."""
        # Create or update execution
        execution = await self.get_execution(trace.execution_id)
        
        if not execution:
            execution = Execution(
                id=trace.execution_id,
                user_query=trace.user_query,
                context_size=trace.context_size,
                context_hash=trace.context_hash,
                status=ExecutionStatus.COMPLETED if trace.execution_result and trace.execution_result.success else ExecutionStatus.FAILED,
                started_at=trace.started_at,
                completed_at=trace.completed_at,
                total_input_tokens=trace.total_input_tokens,
                total_output_tokens=trace.total_output_tokens,
                total_cost_usd=trace.total_cost_usd,
                final_result=trace.execution_result.final_result if trace.execution_result else None,
                error_message=trace.execution_result.error if trace.execution_result else None,
            )
            self.session.add(execution)
        else:
            execution.status = ExecutionStatus.COMPLETED if trace.execution_result and trace.execution_result.success else ExecutionStatus.FAILED
            execution.completed_at = trace.completed_at
            execution.total_input_tokens = trace.total_input_tokens
            execution.total_output_tokens = trace.total_output_tokens
            execution.total_cost_usd = trace.total_cost_usd
            execution.final_result = trace.execution_result.final_result if trace.execution_result else None
            execution.error_message = trace.execution_result.error if trace.execution_result else None
        
        # Create root node
        root_node = ExecutionNode(
            id=trace.root_node_id,
            execution_id=trace.execution_id,
            node_type=NodeType.ROOT,
            depth=0,
            sequence_number=0,
            prompt=trace.user_query,
            generated_code=trace.generated_code,
            status=ExecutionStatus.COMPLETED if trace.execution_result and trace.execution_result.success else ExecutionStatus.FAILED,
            started_at=trace.started_at,
            completed_at=trace.completed_at,
            model_used=trace.code_generation_response.model if trace.code_generation_response else None,
            input_tokens=trace.code_generation_response.input_tokens if trace.code_generation_response else 0,
            output_tokens=trace.code_generation_response.output_tokens if trace.code_generation_response else 0,
            cost_usd=trace.code_generation_response.cost_usd if trace.code_generation_response else 0,
            output=trace.execution_result.final_result if trace.execution_result else None,
            error_message=trace.execution_result.error if trace.execution_result else None,
        )
        self.session.add(root_node)
        
        # Create child nodes from trace
        for i, child_trace in enumerate(trace.child_traces):
            child_node = ExecutionNode(
                execution_id=trace.execution_id,
                parent_node_id=trace.root_node_id,
                node_type=NodeType.CHILD,
                depth=child_trace.get('depth', 1),
                sequence_number=i,
                prompt=child_trace.get('prompt_preview', ''),
                status=ExecutionStatus.COMPLETED,
                model_used=child_trace.get('model'),
                input_tokens=child_trace.get('input_tokens', 0),
                output_tokens=child_trace.get('output_tokens', 0),
                cost_usd=child_trace.get('cost_usd', 0),
                output=child_trace.get('response_preview', ''),
            )
            self.session.add(child_node)
        
        await self.session.flush()
        return execution
