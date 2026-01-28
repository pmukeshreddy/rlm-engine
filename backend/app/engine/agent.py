"""Letta Agent - The core agent orchestrator."""
import hashlib
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from dataclasses import dataclass, field

from app.engine.llm import LLMClient, LLMResponse
from app.engine.repl import REPLExecutor, ExecutionResult, ChildCall
from app.config import settings


@dataclass
class AgentConfig:
    """Configuration for a Letta agent."""
    model: str = settings.default_model
    max_chunk_size: int = settings.default_chunk_size
    max_recursion_depth: int = settings.max_recursion_depth
    execution_timeout: int = settings.execution_timeout


@dataclass
class ExecutionTrace:
    """Complete trace of an agent execution."""
    execution_id: str
    root_node_id: str
    user_query: str
    context_size: int
    context_hash: str
    
    # Root agent
    generated_code: str
    code_generation_response: Optional[LLMResponse] = None
    
    # Execution
    execution_result: Optional[ExecutionResult] = None
    
    # Aggregates
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    
    # Child nodes (for tree visualization)
    child_traces: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "execution_id": self.execution_id,
            "root_node_id": self.root_node_id,
            "user_query": self.user_query,
            "context_size": self.context_size,
            "context_hash": self.context_hash,
            "generated_code": self.generated_code,
            "execution_result": {
                "success": self.execution_result.success if self.execution_result else None,
                "final_result": self.execution_result.final_result if self.execution_result else None,
                "error": self.execution_result.error if self.execution_result else None,
                "output_log": self.execution_result.output_log if self.execution_result else [],
                "child_calls": self.execution_result.child_calls if self.execution_result else [],
                "execution_time_ms": self.execution_result.execution_time_ms if self.execution_result else 0,
            } if self.execution_result else None,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "child_traces": self.child_traces,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class LettaAgent:
    """
    The main Letta agent that processes large contexts.
    
    This agent:
    1. Receives a user query and large context
    2. Generates Python code to process the context
    3. Executes the code in a REPL environment
    4. Spawns child agents via llm_query() calls
    5. Returns the final result
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[AgentConfig] = None,
        on_node_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the Letta agent.
        
        Args:
            llm_client: LLM client for API calls
            config: Agent configuration
            on_node_update: Callback for real-time node updates (for streaming to UI)
        """
        self.llm_client = llm_client or LLMClient()
        self.config = config or AgentConfig()
        self.on_node_update = on_node_update
        
        self._current_trace: Optional[ExecutionTrace] = None
        self._child_sequence = 0
        self._current_depth = 0
    
    def _hash_context(self, context: str) -> str:
        """Generate a hash of the context for caching/deduplication."""
        return hashlib.sha256(context.encode()).hexdigest()
    
    def _get_context_info(self, context: str) -> Dict[str, Any]:
        """Get metadata about the context (without the content)."""
        return {
            "size": len(context),
            "hash": self._hash_context(context),
            "type": "text",
            "preview": context[:200] + "..." if len(context) > 200 else context,
        }
    
    async def _child_agent_query(self, prompt: str, memory: Dict[str, Any]) -> LLMResponse:
        """
        Execute a child agent query.
        
        This is called when the agent code uses llm_query().
        """
        self._child_sequence += 1
        
        response = await self.llm_client.child_agent_query(
            prompt=prompt,
            parent_memory=memory,
            model=self.config.model,
        )
        
        # Track the child call
        child_trace = {
            "sequence": self._child_sequence,
            "depth": self._current_depth + 1,
            "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "response_preview": response.content[:500] + "..." if len(response.content) > 500 else response.content,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "model": response.model,
        }
        
        if self._current_trace:
            self._current_trace.child_traces.append(child_trace)
            self._current_trace.total_input_tokens += response.input_tokens
            self._current_trace.total_output_tokens += response.output_tokens
            self._current_trace.total_cost_usd += response.cost_usd
        
        # Notify listeners
        if self.on_node_update:
            self.on_node_update({
                "type": "child_complete",
                "data": child_trace,
            })
        
        return response
    
    async def run(
        self,
        user_query: str,
        context: str,
        memory: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> ExecutionTrace:
        """
        Run the agent on a user query with the given context.
        
        Args:
            user_query: The user's question/task
            context: The full context string (can be very large)
            memory: Optional persistent memory from previous runs
            execution_id: Optional ID for tracking (generated if not provided)
        
        Returns:
            ExecutionTrace with complete execution details
        """
        from uuid import uuid4
        
        execution_id = execution_id or str(uuid4())
        root_node_id = str(uuid4())
        memory = memory or {}
        
        # Initialize trace
        self._current_trace = ExecutionTrace(
            execution_id=execution_id,
            root_node_id=root_node_id,
            user_query=user_query,
            context_size=len(context),
            context_hash=self._hash_context(context),
            generated_code="",
            started_at=datetime.utcnow(),
        )
        
        self._child_sequence = 0
        self._current_depth = 0
        
        # Notify start
        if self.on_node_update:
            self.on_node_update({
                "type": "execution_start",
                "data": {
                    "execution_id": execution_id,
                    "root_node_id": root_node_id,
                    "user_query": user_query,
                    "context_size": len(context),
                }
            })
        
        try:
            # Step 1: Generate agent code
            context_info = self._get_context_info(context)
            
            if self.on_node_update:
                self.on_node_update({
                    "type": "generating_code",
                    "data": {"context_info": context_info}
                })
            
            code_response = await self.llm_client.generate_agent_code(
                user_query=user_query,
                context_info=context_info,
                memory=memory,
                model=self.config.model,
            )
            
            self._current_trace.generated_code = code_response.content
            self._current_trace.code_generation_response = code_response
            self._current_trace.total_input_tokens += code_response.input_tokens
            self._current_trace.total_output_tokens += code_response.output_tokens
            self._current_trace.total_cost_usd += code_response.cost_usd
            
            if self.on_node_update:
                self.on_node_update({
                    "type": "code_generated",
                    "data": {
                        "code": code_response.content,
                        "tokens": code_response.input_tokens + code_response.output_tokens,
                        "cost": code_response.cost_usd,
                    }
                })
            
            # Step 2: Execute the code in REPL
            async def child_query(prompt: str) -> LLMResponse:
                return await self._child_agent_query(prompt, memory)
            
            repl = REPLExecutor(
                context=context,
                memory=memory,
                llm_query_fn=child_query,
            )
            
            if self.on_node_update:
                self.on_node_update({
                    "type": "executing_code",
                    "data": {}
                })
            
            result = await repl.execute(
                code=code_response.content,
                timeout=self.config.execution_timeout,
            )
            
            self._current_trace.execution_result = result
            self._current_trace.completed_at = datetime.utcnow()
            
            # Final notification
            if self.on_node_update:
                self.on_node_update({
                    "type": "execution_complete",
                    "data": {
                        "success": result.success,
                        "final_result": result.final_result,
                        "error": result.error,
                        "total_cost": self._current_trace.total_cost_usd,
                    }
                })
            
            return self._current_trace
            
        except Exception as e:
            self._current_trace.completed_at = datetime.utcnow()
            self._current_trace.execution_result = ExecutionResult(
                success=False,
                error=str(e),
            )
            
            if self.on_node_update:
                self.on_node_update({
                    "type": "execution_error",
                    "data": {"error": str(e)}
                })
            
            return self._current_trace


class RecursiveLettaAgent(LettaAgent):
    """
    Extended Letta agent that supports recursive child agents.
    
    Child agents can also spawn their own children (up to max depth),
    enabling more complex processing patterns.
    """
    
    async def _child_agent_query(self, prompt: str, memory: Dict[str, Any]) -> LLMResponse:
        """Execute a child query that may recursively spawn more children."""
        self._child_sequence += 1
        new_depth = self._current_depth + 1
        
        if new_depth >= self.config.max_recursion_depth:
            # At max depth, just do a simple completion
            return await self.llm_client.child_agent_query(
                prompt=prompt,
                parent_memory=memory,
                model=self.config.model,
            )
        
        # Check if prompt is large enough to warrant recursive processing
        if len(prompt) > self.config.max_chunk_size:
            # Create a child agent for recursive processing
            child_agent = RecursiveLettaAgent(
                llm_client=self.llm_client,
                config=self.config,
                on_node_update=self.on_node_update,
            )
            child_agent._current_depth = new_depth
            
            # Run child agent with the prompt as context
            child_trace = await child_agent.run(
                user_query="Process and respond to this request",
                context=prompt,
                memory=memory,
            )
            
            # Track in parent trace
            if self._current_trace:
                self._current_trace.child_traces.append(child_trace.to_dict())
                self._current_trace.total_input_tokens += child_trace.total_input_tokens
                self._current_trace.total_output_tokens += child_trace.total_output_tokens
                self._current_trace.total_cost_usd += child_trace.total_cost_usd
            
            # Return the result as an LLMResponse
            return LLMResponse(
                content=child_trace.execution_result.final_result if child_trace.execution_result and child_trace.execution_result.success else f"Error: {child_trace.execution_result.error if child_trace.execution_result else 'Unknown'}",
                model=self.config.model,
                input_tokens=child_trace.total_input_tokens,
                output_tokens=child_trace.total_output_tokens,
                cost_usd=child_trace.total_cost_usd,
            )
        else:
            # Small enough, just do a direct call
            return await super()._child_agent_query(prompt, memory)
