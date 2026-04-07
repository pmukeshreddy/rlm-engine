"""Letta Agent - The core RLM (Recursive Language Model) orchestrator.

Implements Algorithm 1 from the RLM paper:
  1. Initialize REPL with context as variable + llm_query function
  2. Show LLM only metadata about context (length, preview)
  3. LLM generates code to probe/process context
  4. Execute code in persistent REPL
  5. Feed truncated stdout back to LLM
  6. Repeat until Final is set
"""
import hashlib
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from dataclasses import dataclass, field

from app.engine.llm import (
    LLMClient, LLMResponse, MODEL_CONTEXT_WINDOWS,
    _get_rlm_system_prompt,
)
from app.engine.repl import REPLExecutor, ExecutionResult, ChildCall, StepResult
from app.engine.metrics import MetricsEvaluator, ExecutionMetrics
from app.config import settings


def _max_chunk_chars_for_model(model: str) -> int:
    """Calculate the max chars per chunk that fits in the model's context window."""
    max_tokens = MODEL_CONTEXT_WINDOWS.get(model, 16384)
    # Reserve: system prompt (~150), memory context (~200), prompt text (~200), output (~1024)
    available_tokens = max_tokens - 1600
    # Use conservative 3.5 chars per token
    return int(available_tokens * 3.5)


# Max stdout chars to show back to the LLM per iteration
# (paper: "Only constant-size metadata about stdout is appended to M's history")
STDOUT_TRUNCATE_CHARS = 2000

# Max iterations of the RLM loop before forcing termination
MAX_RLM_ITERATIONS = 10


@dataclass
class AgentConfig:
    """Configuration for a Letta agent."""
    model: str = settings.default_model
    max_chunk_size: int = settings.default_chunk_size
    max_recursion_depth: int = settings.max_recursion_depth
    execution_timeout: int = settings.execution_timeout
    max_iterations: int = MAX_RLM_ITERATIONS


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

    # RLM loop iterations
    iterations: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics
    metrics: Optional[ExecutionMetrics] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
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
            "iterations": self.iterations,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


def _truncate_stdout(stdout: str, max_chars: int = STDOUT_TRUNCATE_CHARS) -> str:
    """
    Truncate stdout to constant-size metadata for the LLM's history.

    From the paper (footnote 1, page 3): "Only (constant-size) metadata about
    stdout, like a short prefix and length, is appended to M's history for the
    next iteration. This is key: it forces M to rely on variables and sub-calls
    to manage long strings instead of polluting its window."
    """
    if not stdout:
        return "(no output)"
    if len(stdout) <= max_chars:
        return stdout
    # Show prefix + length metadata, matching paper's approach
    half = max_chars // 2
    return (
        f"[stdout: {len(stdout)} total chars, showing first and last {half} chars]\n"
        + stdout[:half]
        + f"\n\n... [{len(stdout) - max_chars} chars truncated] ...\n\n"
        + stdout[-half:]
    )


def _extract_final_from_text(text: str) -> Optional[str]:
    """
    Check if the LLM responded with FINAL() or FINAL_VAR() outside of code blocks.

    Per the paper (Appendix C): "you MUST provide a final answer inside a FINAL
    function when you have completed your task, NOT in code."

    The LLM may output reasoning + code blocks + then FINAL() as text after.
    We strip code blocks and look for FINAL/FINAL_VAR in the remaining text.
    """
    import re
    # Remove all code blocks (```...```) to get just the prose/text parts
    text_no_code = re.sub(r'```[a-z]*\n.*?```', '', text, flags=re.DOTALL)
    text_no_code = text_no_code.strip()

    if not text_no_code:
        return None

    # Look for FINAL_VAR(var_name) — returns a variable from REPL env
    match = re.search(r'FINAL_VAR\((\w+)\)', text_no_code)
    if match:
        return f"__VAR__{match.group(1)}"  # Prefix to signal variable lookup

    # Look for FINAL(answer) with various quote styles
    # FINAL("answer"), FINAL('answer'), FINAL(answer)
    match = re.search(r'FINAL\((["\'])(.*?)\1\)', text_no_code, re.DOTALL)
    if match:
        return match.group(2).strip()

    # FINAL(unquoted answer)
    match = re.search(r'FINAL\(([^)]+)\)', text_no_code, re.DOTALL)
    if match:
        val = match.group(1).strip().strip('"\'')
        if val:
            return val

    return None


class LettaAgent:
    """
    The main RLM agent implementing Algorithm 1 from the paper.

    Key design principles (from the paper):
    1. Context lives in the REPL environment, NOT in the LLM's context window
    2. LLM only sees metadata about context (length, type, preview)
    3. LLM writes code iteratively, observing truncated stdout between steps
    4. llm_query() enables recursive sub-LLM calls within code
    5. Loop continues until FINAL/FINAL_VAR is called
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[AgentConfig] = None,
        on_node_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.config = config or AgentConfig()
        self.on_node_update = on_node_update

        self._current_trace: Optional[ExecutionTrace] = None
        self._child_sequence = 0
        self._current_depth = 0

    def _hash_context(self, context: str) -> str:
        return hashlib.sha256(context.encode()).hexdigest()

    def _get_context_info(self, context: str) -> Dict[str, Any]:
        return {
            "size": len(context),
            "hash": self._hash_context(context),
            "type": "text",
            "preview": context[:200] + "..." if len(context) > 200 else context,
        }

    async def _child_agent_query(self, prompt: str, memory: Dict[str, Any]) -> LLMResponse:
        """Execute a child agent query (called via llm_query in the REPL)."""
        self._child_sequence += 1

        response = await self.llm_client.child_agent_query(
            prompt=prompt,
            parent_memory=memory,
            model=self.config.model,
        )

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

        if self.on_node_update:
            self.on_node_update({
                "type": "child_complete",
                "data": child_trace,
            })

        return response

    async def compute_metrics(
        self,
        context: str,
        trace: ExecutionTrace,
        memory: Dict[str, Any],
        baseline_execution: Optional[Dict[str, Any]] = None,
    ) -> ExecutionMetrics:
        evaluator = MetricsEvaluator(llm_client=self.llm_client)
        metrics = ExecutionMetrics()

        final_result = trace.execution_result.final_result if trace.execution_result else None
        if not final_result or not trace.execution_result or not trace.execution_result.success:
            return metrics

        child_count = len(trace.child_traces)
        metrics.compression = evaluator.evaluate_compression(
            context=context,
            final_result=final_result,
            child_call_count=child_count,
        )

        execution_time_ms = trace.execution_result.execution_time_ms if trace.execution_result else 0
        metrics.memory_speedup = evaluator.evaluate_memory_speedup(
            current_tokens=trace.total_input_tokens + trace.total_output_tokens,
            current_cost_usd=trace.total_cost_usd,
            current_time_ms=execution_time_ms,
            current_child_calls=child_count,
            memory_keys=list(memory.keys()),
            baseline_execution=baseline_execution,
        )

        return metrics

    async def run(
        self,
        user_query: str,
        context: str,
        memory: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> ExecutionTrace:
        """
        Run the RLM agent — implements Algorithm 1 from the paper.

        The iterative loop:
        1. Build system prompt with context metadata (NOT the context itself)
        2. Send conversation history to LLM
        3. LLM generates code (wrapped in ```repl blocks)
        4. Execute code in persistent REPL
        5. Capture stdout, truncate to constant size
        6. Append code + truncated stdout to conversation history
        7. If FINAL was called, return the result
        8. Otherwise, loop back to step 2
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
            max_chunk_chars = _max_chunk_chars_for_model(self.config.model)

            # Build the system prompt (paper's Appendix C)
            system_prompt = _get_rlm_system_prompt(
                context_length=len(context),
                max_chunk_chars=max_chunk_chars,
                model=self.config.model,
            )

            # Initialize REPL with persistent state
            async def child_query(prompt: str) -> LLMResponse:
                return await self._child_agent_query(prompt, memory)

            repl = REPLExecutor(
                context=context,
                memory=memory,
                llm_query_fn=child_query,
                max_chunk_chars=max_chunk_chars,
            )

            # Initialize conversation history with just the user query
            # (Algorithm 1: hist <- [Metadata(state)])
            conversation_history = [
                {
                    "role": "user",
                    "content": f"Query: {user_query}\n\nThe context ({len(context):,} chars) is loaded in the REPL as the variable `context`. Use the REPL to explore and answer the query.",
                }
            ]

            all_code_blocks = []

            # === RLM LOOP (Algorithm 1) ===
            for iteration in range(self.config.max_iterations):
                if self.on_node_update:
                    self.on_node_update({
                        "type": "rlm_iteration",
                        "data": {"iteration": iteration + 1}
                    })

                # Step 1: Get code from LLM
                llm_response = await self.llm_client.rlm_iteration(
                    conversation_history=conversation_history,
                    system_prompt=system_prompt,
                    model=self.config.model,
                )

                # Track tokens/cost
                self._current_trace.total_input_tokens += llm_response.input_tokens
                self._current_trace.total_output_tokens += llm_response.output_tokens
                self._current_trace.total_cost_usd += llm_response.cost_usd

                llm_output = llm_response.content
                all_code_blocks.append(llm_output)

                # Check if LLM output contains code blocks to execute
                has_code = '```' in llm_output

                # If there's code, execute it first (it may set up variables
                # that FINAL_VAR references)
                if has_code:
                    # Step 2a: Execute code in persistent REPL
                    step_result = await repl.execute_step(
                        code=llm_output,
                        timeout=self.config.execution_timeout,
                    )

                    # If FINAL was called inside the code, we're done
                    if step_result.final_set:
                        self._current_trace.generated_code = "\n\n".join(all_code_blocks)
                        self._current_trace.execution_result = ExecutionResult(
                            success=True,
                            final_result=step_result.final_value,
                            output_log=repl._output_log,
                            child_calls=[vars(c) for c in repl._child_calls],
                            execution_time_ms=(datetime.utcnow() - self._current_trace.started_at).total_seconds() * 1000,
                            memory_changes=repl.get_memory_changes(),
                        )
                        self._current_trace.completed_at = datetime.utcnow()
                        self._current_trace.iterations.append({
                            "iteration": iteration + 1,
                            "code_preview": llm_output[:500],
                            "stdout_preview": step_result.stdout[:500],
                            "final_set": True,
                            "error": None,
                            "child_calls": step_result.child_calls_this_step,
                        })
                        if self.on_node_update:
                            self.on_node_update({
                                "type": "execution_complete",
                                "data": {
                                    "success": True,
                                    "final_result": step_result.final_value,
                                    "iterations": iteration + 1,
                                    "total_cost": self._current_trace.total_cost_usd,
                                }
                            })
                        return self._current_trace

                # Now check for FINAL/FINAL_VAR in the text (outside code blocks)
                # Per paper: "you MUST provide a final answer inside a FINAL function,
                # NOT in code"
                text_final = _extract_final_from_text(llm_output)
                if text_final:
                    # Resolve FINAL_VAR(variable_name)
                    if text_final.startswith("__VAR__"):
                        var_name = text_final[7:]
                        if var_name in repl._env:
                            final_value = str(repl._env[var_name])
                        else:
                            final_value = f"(variable '{var_name}' not found in REPL)"
                    else:
                        final_value = text_final

                    self._current_trace.generated_code = "\n\n".join(all_code_blocks)
                    self._current_trace.execution_result = ExecutionResult(
                        success=True,
                        final_result=final_value,
                        output_log=repl._output_log,
                        child_calls=[vars(c) for c in repl._child_calls],
                        execution_time_ms=(datetime.utcnow() - self._current_trace.started_at).total_seconds() * 1000,
                        memory_changes=repl.get_memory_changes(),
                    )
                    self._current_trace.completed_at = datetime.utcnow()
                    self._current_trace.iterations.append({
                        "iteration": iteration + 1,
                        "llm_output": llm_output[:500],
                        "type": "text_final",
                        "final_value": final_value[:200],
                    })

                    if self.on_node_update:
                        self.on_node_update({
                            "type": "execution_complete",
                            "data": {
                                "success": True,
                                "final_result": final_value,
                                "iterations": iteration + 1,
                                "total_cost": self._current_trace.total_cost_usd,
                            }
                        })
                    return self._current_trace

                # If we had code but no FINAL, continue the loop with stdout feedback
                if has_code:
                    # Truncate stdout for feedback (paper: constant-size metadata)
                    truncated_stdout = _truncate_stdout(step_result.stdout)

                    iter_record = {
                        "iteration": iteration + 1,
                        "code_preview": llm_output[:500],
                        "stdout_preview": truncated_stdout[:500],
                        "final_set": False,
                        "error": step_result.error,
                        "child_calls": step_result.child_calls_this_step,
                    }
                    self._current_trace.iterations.append(iter_record)

                    if self.on_node_update:
                        self.on_node_update({
                            "type": "rlm_step_complete",
                            "data": iter_record,
                        })

                    # Append code + stdout to conversation history
                    conversation_history.append({
                        "role": "assistant",
                        "content": llm_output,
                    })

                    if step_result.error:
                        feedback = f"REPL Error: {step_result.error}\n\nStdout before error ({len(step_result.stdout)} chars):\n{truncated_stdout}"
                    else:
                        feedback = f"REPL Output ({len(step_result.stdout)} chars):\n{truncated_stdout}"

                    conversation_history.append({
                        "role": "user",
                        "content": feedback + "\n\nContinue. When you have your final answer, call FINAL(answer) outside of code blocks.",
                    })
                    continue

                # No code and no FINAL — LLM is just reasoning in text
                # Append and continue (this shouldn't happen often)
                self._current_trace.iterations.append({
                    "iteration": iteration + 1,
                    "llm_output": llm_output[:500],
                    "type": "text_only",
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": llm_output,
                })
                conversation_history.append({
                    "role": "user",
                    "content": "Please write code in ```repl blocks to explore the context and work toward your answer. Call FINAL(answer) when done.",
                })
                continue

            # Exhausted iterations without FINAL
            self._current_trace.generated_code = "\n\n".join(all_code_blocks)
            self._current_trace.execution_result = ExecutionResult(
                success=False,
                error=f"RLM loop exhausted {self.config.max_iterations} iterations without calling FINAL",
                output_log=repl._output_log,
                child_calls=[vars(c) for c in repl._child_calls],
                execution_time_ms=(datetime.utcnow() - self._current_trace.started_at).total_seconds() * 1000,
                memory_changes=repl.get_memory_changes(),
            )
            self._current_trace.completed_at = datetime.utcnow()

            if self.on_node_update:
                self.on_node_update({
                    "type": "execution_error",
                    "data": {"error": "Max iterations reached"}
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
    Extended agent that supports recursive child agents.
    Child agents can also spawn their own children (up to max depth).
    """

    async def _child_agent_query(self, prompt: str, memory: Dict[str, Any]) -> LLMResponse:
        self._child_sequence += 1
        new_depth = self._current_depth + 1

        if new_depth >= self.config.max_recursion_depth:
            return await self.llm_client.child_agent_query(
                prompt=prompt,
                parent_memory=memory,
                model=self.config.model,
            )

        if len(prompt) > self.config.max_chunk_size:
            child_agent = RecursiveLettaAgent(
                llm_client=self.llm_client,
                config=self.config,
                on_node_update=self.on_node_update,
            )
            child_agent._current_depth = new_depth

            child_trace = await child_agent.run(
                user_query="Process and respond to this request",
                context=prompt,
                memory=memory,
            )

            if self._current_trace:
                self._current_trace.child_traces.append(child_trace.to_dict())
                self._current_trace.total_input_tokens += child_trace.total_input_tokens
                self._current_trace.total_output_tokens += child_trace.total_output_tokens
                self._current_trace.total_cost_usd += child_trace.total_cost_usd

            return LLMResponse(
                content=child_trace.execution_result.final_result if child_trace.execution_result and child_trace.execution_result.success else f"Error: {child_trace.execution_result.error if child_trace.execution_result else 'Unknown'}",
                model=self.config.model,
                input_tokens=child_trace.total_input_tokens,
                output_tokens=child_trace.total_output_tokens,
                cost_usd=child_trace.total_cost_usd,
            )
        else:
            return await super()._child_agent_query(prompt, memory)
