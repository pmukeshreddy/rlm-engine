"""REPL Executor for running agent-generated code.

Implements the persistent REPL environment from the RLM paper (Algorithm 1).
The REPL maintains state across multiple code executions, allowing the LLM
to iteratively probe, analyze, and process the context.
"""
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
class StepResult:
    """Result of a single REPL step (one code execution in the iterative loop)."""
    stdout: str  # Captured print output
    final_set: bool  # Whether Final was set this step
    final_value: Optional[str] = None  # The value of Final if set
    error: Optional[str] = None  # Error message if code failed
    child_calls_this_step: int = 0  # Number of llm_query calls in this step


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
    Persistent REPL environment for RLM execution.

    Implements the REPL from Algorithm 1 of the RLM paper:
    - Context is loaded as a variable (never in the LLM's context window)
    - State persists across multiple code executions (iterative loop)
    - llm_query() spawns sub-LLM calls
    - FINAL() / FINAL_VAR() signals completion
    - print() captures stdout for feedback to the root LLM
    """

    def __init__(
        self,
        context: str,
        memory: Dict[str, Any],
        llm_query_fn: Callable[[str], Any],
        on_child_call: Optional[Callable[[ChildCall], None]] = None,
        max_chunk_chars: int = 50000,
    ):
        self.context = context
        self.memory = memory.copy()
        self._initial_memory = memory.copy()
        self._llm_query_fn = llm_query_fn
        self._on_child_call = on_child_call
        self._max_chunk_chars = max_chunk_chars

        self._final_result: Optional[str] = None
        self._final_set: bool = False
        self._child_calls: List[ChildCall] = []
        self._output_log: List[str] = []
        self._stdout_buffer: List[str] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._memory_changes: Dict[str, Any] = {}

        # Persistent environment that survives across iterations
        self._env: Dict[str, Any] = self._create_base_env()

    def _create_base_env(self) -> Dict[str, Any]:
        """Create the persistent REPL environment with all available functions.

        Per the paper (Appendix C examples), the REPL supports:
        - import re (used heavily for regex-based context filtering)
        - import json
        - import collections (Counter, defaultdict)
        - import math
        These are pre-loaded so 'import X' statements work naturally.
        """
        import re as re_module
        import json as json_module
        import collections as collections_module
        import math as math_module

        # Create a safe __builtins__ that allows import of approved modules
        safe_modules = {
            're': re_module,
            'json': json_module,
            'collections': collections_module,
            'math': math_module,
        }

        def safe_import(name, *args, **kwargs):
            if name in safe_modules:
                return safe_modules[name]
            raise ImportError(f"Module '{name}' is not available in the REPL environment. Available: {list(safe_modules.keys())}")

        return {
            'context': self.context,
            'memory': self.memory,
            'MAX_CHUNK_CHARS': self._max_chunk_chars,
            'llm_query': self._create_llm_query_fn(),
            'FINAL': self._create_final_fn(),
            'FINAL_VAR': self._create_final_var_fn(),
            'set_memory': self._create_set_memory_fn(),
            'get_memory': self._create_get_memory_fn(),
            'print': self._create_print_fn(),
            # Pre-loaded modules (paper examples use these)
            're': re_module,
            'json': json_module,
            'collections': collections_module,
            'math': math_module,
            'Counter': collections_module.Counter,
            'defaultdict': collections_module.defaultdict,
            # Safe builtins
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
            'any': any,
            'all': all,
            'chr': chr,
            'ord': ord,
            'repr': repr,
            'True': True,
            'False': False,
            'None': None,
            '__builtins__': {'__import__': safe_import},
        }

    def _create_final_fn(self):
        """Create the FINAL function - signals completion with a direct value."""
        def final(result: Any) -> None:
            result_str = str(result) if not isinstance(result, str) else result
            self._final_result = result_str
            self._final_set = True
            raise FinalResultException(result_str)
        return final

    def _create_final_var_fn(self):
        """Create the FINAL_VAR function - signals completion with a variable from the REPL env."""
        def final_var(var_name: Any) -> None:
            # If passed a string, look up the variable name in the environment
            if isinstance(var_name, str) and var_name in self._env:
                result = str(self._env[var_name])
            else:
                # If passed the variable directly (not as a string), use its value
                result = str(var_name)
            self._final_result = result
            self._final_set = True
            raise FinalResultException(result)
        return final_var

    def _create_set_memory_fn(self):
        def set_memory(key: str, value: Any) -> None:
            self.memory[key] = value
            self._memory_changes[key] = value
            self._stdout_buffer.append(f"[memory] set '{key}'")
        return set_memory

    def _create_get_memory_fn(self):
        def get_memory(key: str, default: Any = None) -> Any:
            return self.memory.get(key, default)
        return get_memory

    def get_memory_changes(self) -> Dict[str, Any]:
        return self._memory_changes.copy()

    def _create_llm_query_fn(self):
        """Create the llm_query function for sub-LLM calls."""
        def llm_query(prompt: str) -> str:
            if self._loop is None:
                raise RuntimeError("No event loop available for llm_query")

            start_time = datetime.utcnow()

            future = asyncio.run_coroutine_threadsafe(
                self._llm_query_fn(prompt),
                self._loop
            )

            try:
                response = future.result(timeout=120)

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
        """Create a print function that captures to stdout buffer."""
        def custom_print(*args, **kwargs):
            message = " ".join(str(arg) for arg in args)
            self._stdout_buffer.append(message)
            self._output_log.append(f"[print] {message}")
        return custom_print

    def _sanitize_code(self, code: str) -> str:
        """Extract and sanitize code from LLM output."""
        # Extract code from ```repl or ```python blocks
        repl_match = re.search(r'```(?:repl|python)\s*\n(.*?)```', code, re.DOTALL)
        if repl_match:
            code = repl_match.group(1)
        else:
            # Try plain ``` blocks
            plain_match = re.search(r'```\s*\n(.*?)```', code, re.DOTALL)
            if plain_match:
                code = plain_match.group(1)
            else:
                # Remove any remaining markdown
                code = re.sub(r'^```python\s*\n?', '', code.strip())
                code = re.sub(r'^```\s*\n?', '', code.strip())
                code = re.sub(r'\n?```\s*$', '', code.strip())

        # Block dangerous imports/operations
        dangerous_patterns = [
            r'\bimport\s+os\b',
            r'\bimport\s+subprocess\b',
            r'\bimport\s+sys\b',
            r'\bimport\s+shutil\b',
            r'\bimport\s+socket\b',
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

    async def execute_step(self, code: str, timeout: int = 120) -> StepResult:
        """
        Execute one step of code in the persistent REPL.

        This is one iteration of the RLM loop (Algorithm 1):
        - Code runs in the persistent environment
        - Variables from previous steps are available
        - stdout is captured and returned
        - If FINAL/FINAL_VAR is called, final_set=True

        Args:
            code: Python code to execute (may contain ```repl blocks)
            timeout: Timeout for this step in seconds

        Returns:
            StepResult with stdout, final status, and any errors
        """
        self._loop = asyncio.get_event_loop()
        self._stdout_buffer = []  # Reset stdout for this step
        child_calls_before = len(self._child_calls)

        try:
            code = self._sanitize_code(code)

            def run_code():
                exec(code, self._env)

            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, run_code),
                timeout=timeout
            )

            # Code completed without FINAL — normal for iterative RLM
            stdout = "\n".join(self._stdout_buffer)
            child_calls_this_step = len(self._child_calls) - child_calls_before

            return StepResult(
                stdout=stdout,
                final_set=False,
                child_calls_this_step=child_calls_this_step,
            )

        except FinalResultException as e:
            stdout = "\n".join(self._stdout_buffer)
            child_calls_this_step = len(self._child_calls) - child_calls_before

            return StepResult(
                stdout=stdout,
                final_set=True,
                final_value=e.result,
                child_calls_this_step=child_calls_this_step,
            )

        except asyncio.TimeoutError:
            stdout = "\n".join(self._stdout_buffer)
            return StepResult(
                stdout=stdout,
                final_set=False,
                error=f"Code execution timed out after {timeout} seconds",
            )

        except Exception as e:
            stdout = "\n".join(self._stdout_buffer)
            return StepResult(
                stdout=stdout,
                final_set=False,
                error=f"{type(e).__name__}: {str(e)}",
            )

    def get_execution_summary(self) -> ExecutionResult:
        """Get the final execution result after all iterations complete."""
        return ExecutionResult(
            success=self._final_set,
            final_result=self._final_result,
            error=None if self._final_set else "RLM loop ended without setting Final",
            output_log=self._output_log,
            child_calls=[vars(c) for c in self._child_calls],
            execution_time_ms=0,  # Set by caller
            memory_changes=self._memory_changes,
        )

    # Legacy single-shot execute for backward compatibility
    async def execute(self, code: str, timeout: int = 300) -> ExecutionResult:
        """
        Execute code in a single shot (legacy interface).
        For the iterative RLM loop, use execute_step() instead.
        """
        start_time = datetime.utcnow()
        step_result = await self.execute_step(code, timeout)

        end_time = datetime.utcnow()
        exec_time = (end_time - start_time).total_seconds() * 1000

        if step_result.final_set:
            return ExecutionResult(
                success=True,
                final_result=step_result.final_value,
                output_log=self._output_log,
                child_calls=[vars(c) for c in self._child_calls],
                execution_time_ms=exec_time,
                memory_changes=self._memory_changes,
            )
        elif step_result.error:
            return ExecutionResult(
                success=False,
                error=step_result.error,
                output_log=self._output_log,
                child_calls=[vars(c) for c in self._child_calls],
                execution_time_ms=exec_time,
                memory_changes=self._memory_changes,
            )
        else:
            return ExecutionResult(
                success=False,
                error="Code completed without calling FINAL(result)",
                output_log=self._output_log,
                child_calls=[vars(c) for c in self._child_calls],
                execution_time_ms=exec_time,
                memory_changes=self._memory_changes,
            )
