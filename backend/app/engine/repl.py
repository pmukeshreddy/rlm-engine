"""REPL Executor for running agent-generated code."""
import asyncio
import traceback
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import re


@dataclass
class ExecutionResult:
    """Result of executing code in the REPL."""
    success: bool
    final_result: Optional[str] = None
    error: Optional[str] = None
    output_log: List[str] = field(default_factory=list)
    child_calls: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0
    memory_changes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildCall:
    """Record of a child agent call."""
    prompt: str
    result: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    execution_time_ms: float


class FinalResultException(Exception):
    """Raised when FINAL() is called to stop execution with a result."""
    def __init__(self, result: str):
        self.result = result
        super().__init__(f"FINAL called with result")


class REPLExecutor:
    """
    Executes agent-generated Python code in a sandboxed environment.
    
    Provides:
    - context: The large context string
    - llm_query(prompt): Function to spawn child agents
    - FINAL(result): Function to return the final result
    - memory: Read-only dict of persistent memory
    """
    
    def __init__(
        self,
        context: str,
        memory: Dict[str, Any],
        llm_query_fn: Callable[[str], Any],
        on_child_call: Optional[Callable[[ChildCall], None]] = None,
    ):
        """
        Initialize the REPL executor.
        
        Args:
            context: The full context string
            memory: Dictionary of persistent memory
            llm_query_fn: Async function to call child agents
            on_child_call: Callback when a child call completes
        """
        self.context = context
        self.memory = memory.copy()  # Working copy of memory
        self._initial_memory = memory.copy()  # Original state for diff
        self._llm_query_fn = llm_query_fn
        self._on_child_call = on_child_call
        
        self._final_result: Optional[str] = None
        self._child_calls: List[ChildCall] = []
        self._output_log: List[str] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._memory_changes: Dict[str, Any] = {}  # Track what changed
    
    def _create_final_fn(self):
        """Create the FINAL function for the REPL environment."""
        def final(result: Any) -> None:
            """Signal completion with a final result."""
            result_str = str(result) if not isinstance(result, str) else result
            raise FinalResultException(result_str)
        return final
    
    def _create_set_memory_fn(self):
        """Create the set_memory function for persisting data."""
        def set_memory(key: str, value: Any) -> None:
            """
            Store a value in persistent memory.
            
            Args:
                key: The key to store under
                value: The value to store (must be JSON-serializable)
            """
            self.memory[key] = value
            self._memory_changes[key] = value
            self._output_log.append(f"[set_memory] {key} = {str(value)[:100]}")
        return set_memory
    
    def _create_get_memory_fn(self):
        """Create the get_memory function for reading data."""
        def get_memory(key: str, default: Any = None) -> Any:
            """
            Get a value from persistent memory.
            
            Args:
                key: The key to retrieve
                default: Default value if key doesn't exist
            
            Returns:
                The stored value or default
            """
            return self.memory.get(key, default)
        return get_memory
    
    def get_memory_changes(self) -> Dict[str, Any]:
        """Get all memory changes made during execution."""
        return self._memory_changes.copy()
    
    def _create_llm_query_fn(self):
        """Create the llm_query function for the REPL environment."""
        def llm_query(prompt: str) -> str:
            """
            Spawn a child agent to answer a question.
            
            This runs synchronously from the agent's perspective,
            but internally uses async to call the LLM.
            """
            if self._loop is None:
                raise RuntimeError("No event loop available for llm_query")
            
            start_time = datetime.utcnow()
            
            # Run the async function in the event loop
            future = asyncio.run_coroutine_threadsafe(
                self._llm_query_fn(prompt),
                self._loop
            )
            
            try:
                # Wait for the result (with timeout)
                response = future.result(timeout=120)  # 2 minute timeout per call
                
                end_time = datetime.utcnow()
                execution_time_ms = (end_time - start_time).total_seconds() * 1000
                
                # Record the child call
                child_call = ChildCall(
                    prompt=prompt[:1000] + "..." if len(prompt) > 1000 else prompt,
                    result=response.content[:1000] + "..." if len(response.content) > 1000 else response.content,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                    execution_time_ms=execution_time_ms,
                )
                self._child_calls.append(child_call)
                
                if self._on_child_call:
                    self._on_child_call(child_call)
                
                self._output_log.append(f"[llm_query] Tokens: {response.input_tokens}+{response.output_tokens}, Cost: ${response.cost_usd:.4f}")
                
                return response.content
                
            except asyncio.TimeoutError:
                self._output_log.append(f"[llm_query] TIMEOUT after 120s")
                raise TimeoutError("llm_query timed out after 120 seconds")
            except Exception as e:
                self._output_log.append(f"[llm_query] ERROR: {str(e)}")
                raise
        
        return llm_query
    
    def _create_print_fn(self):
        """Create a print function that logs output."""
        def custom_print(*args, **kwargs):
            message = " ".join(str(arg) for arg in args)
            self._output_log.append(f"[print] {message}")
        return custom_print
    
    def _sanitize_code(self, code: str) -> str:
        """
        Sanitize the code for execution.
        
        Removes markdown code blocks and dangerous constructs.
        """
        # Remove markdown code blocks
        code = re.sub(r'^```python\s*\n?', '', code.strip())
        code = re.sub(r'^```\s*\n?', '', code.strip())
        code = re.sub(r'\n?```\s*$', '', code.strip())
        
        # Basic safety checks (not comprehensive, just catches obvious issues)
        dangerous_patterns = [
            r'\bimport\s+os\b',
            r'\bimport\s+subprocess\b',
            r'\bimport\s+sys\b',
            r'\b__import__\b',
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\bopen\s*\(',
            r'\bfile\s*\(',
            r'\bcompile\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                raise ValueError(f"Potentially dangerous code pattern detected: {pattern}")
        
        return code
    
    async def execute(self, code: str, timeout: int = 300) -> ExecutionResult:
        """
        Execute the agent-generated code.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
        
        Returns:
            ExecutionResult with success/failure and results
        """
        start_time = datetime.utcnow()
        self._loop = asyncio.get_event_loop()
        
        try:
            # Sanitize code
            code = self._sanitize_code(code)
            
            # Create the execution environment
            env = {
                'context': self.context,
                'memory': self.memory,
                'llm_query': self._create_llm_query_fn(),
                'FINAL': self._create_final_fn(),
                'set_memory': self._create_set_memory_fn(),
                'get_memory': self._create_get_memory_fn(),
                'print': self._create_print_fn(),
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'reversed': reversed,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'isinstance': isinstance,
                'type': type,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                '__builtins__': {},  # Restrict builtins
            }
            
            # Execute in a thread to allow async calls within sync exec
            def run_code():
                exec(code, env)
            
            # Run with timeout
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, run_code),
                timeout=timeout
            )
            
            # If we get here without FINAL being called, that's an error
            end_time = datetime.utcnow()
            return ExecutionResult(
                success=False,
                error="Code completed without calling FINAL(result)",
                output_log=self._output_log,
                child_calls=[vars(c) for c in self._child_calls],
                execution_time_ms=(end_time - start_time).total_seconds() * 1000,
                memory_changes=self._memory_changes,
            )
            
        except FinalResultException as e:
            # FINAL was called - this is success!
            end_time = datetime.utcnow()
            return ExecutionResult(
                success=True,
                final_result=e.result,
                output_log=self._output_log,
                child_calls=[vars(c) for c in self._child_calls],
                execution_time_ms=(end_time - start_time).total_seconds() * 1000,
                memory_changes=self._memory_changes,
            )
            
        except asyncio.TimeoutError:
            end_time = datetime.utcnow()
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout} seconds",
                output_log=self._output_log,
                child_calls=[vars(c) for c in self._child_calls],
                execution_time_ms=(end_time - start_time).total_seconds() * 1000,
                memory_changes=self._memory_changes,
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                output_log=self._output_log,
                child_calls=[vars(c) for c in self._child_calls],
                execution_time_ms=(end_time - start_time).total_seconds() * 1000,
                memory_changes=self._memory_changes,
            )


class AsyncREPLExecutor(REPLExecutor):
    """
    Async-native REPL executor that uses native async for llm_query.
    
    This version allows true concurrent child agent calls when the
    agent code uses asyncio patterns.
    """
    
    def _create_llm_query_fn(self):
        """Create an async llm_query function."""
        async def llm_query_async(prompt: str) -> str:
            """Spawn a child agent asynchronously."""
            start_time = datetime.utcnow()
            
            try:
                response = await self._llm_query_fn(prompt)
                
                end_time = datetime.utcnow()
                execution_time_ms = (end_time - start_time).total_seconds() * 1000
                
                child_call = ChildCall(
                    prompt=prompt[:1000] + "..." if len(prompt) > 1000 else prompt,
                    result=response.content[:1000] + "..." if len(response.content) > 1000 else response.content,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                    execution_time_ms=execution_time_ms,
                )
                self._child_calls.append(child_call)
                
                if self._on_child_call:
                    self._on_child_call(child_call)
                
                return response.content
                
            except Exception as e:
                self._output_log.append(f"[llm_query] ERROR: {str(e)}")
                raise
        
        # Return sync wrapper for use in exec()
        def llm_query(prompt: str) -> str:
            if self._loop is None:
                raise RuntimeError("No event loop")
            future = asyncio.run_coroutine_threadsafe(
                llm_query_async(prompt),
                self._loop
            )
            return future.result(timeout=120)
        
        return llm_query
