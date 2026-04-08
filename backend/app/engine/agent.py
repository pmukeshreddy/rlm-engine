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
import re
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
    available_tokens = max_tokens - 1600
    return int(available_tokens * 3.5)


# Max stdout chars to show back to the LLM per iteration
# Paper footnote 1: "Only (constant-size) metadata about stdout, like
# a short prefix and length". Keep this small to force the model to
# rely on llm_query() for semantic analysis rather than reading raw
# context through print() output.
STDOUT_TRUNCATE_CHARS = 1500

# Max iterations of the RLM loop
MAX_RLM_ITERATIONS = 30


@dataclass
class AgentConfig:
    """Configuration for a Letta agent."""
    model: str = settings.default_model
    sub_model: Optional[str] = None  # Model for sub-calls; defaults to model if None
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
    generated_code: str
    code_generation_response: Optional[LLMResponse] = None
    execution_result: Optional[ExecutionResult] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    child_traces: List[Dict[str, Any]] = field(default_factory=list)
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[ExecutionMetrics] = None
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
    """Truncate stdout to constant-size metadata for the LLM's history."""
    if not stdout:
        return "(no output)"
    if len(stdout) <= max_chars:
        return stdout
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
    """
    # Remove all code blocks to get just the prose/text parts
    text_no_code = re.sub(r'```[a-z]*\n.*?```', '', text, flags=re.DOTALL)
    text_no_code = text_no_code.strip()

    if not text_no_code:
        return None

    # FINAL_VAR(var_name)
    match = re.search(r'FINAL_VAR\((\w+)\)', text_no_code)
    if match:
        return f"__VAR__{match.group(1)}"

    # FINAL("answer") or FINAL('answer')
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


def _make_result(trace, repl, all_code_blocks, success, final_result=None, error=None):
    """Helper to build ExecutionResult and finalize trace."""
    trace.generated_code = "\n\n".join(all_code_blocks)
    trace.execution_result = ExecutionResult(
        success=success,
        final_result=final_result,
        error=error,
        output_log=repl._output_log,
        child_calls=[vars(c) for c in repl._child_calls],
        execution_time_ms=(datetime.utcnow() - trace.started_at).total_seconds() * 1000,
        memory_changes=repl.get_memory_changes(),
    )
    trace.completed_at = datetime.utcnow()


class LettaAgent:
    """
    The main RLM agent implementing Algorithm 1 from the paper.
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
        sub_model = self.config.sub_model or self.config.model
        response = await self.llm_client.child_agent_query(
            prompt=prompt,
            parent_memory=memory,
            model=sub_model,
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
            self.on_node_update({"type": "child_complete", "data": child_trace})

        return response

    async def compute_metrics(
        self, context: str, trace: ExecutionTrace,
        memory: Dict[str, Any], baseline_execution: Optional[Dict[str, Any]] = None,
    ) -> ExecutionMetrics:
        evaluator = MetricsEvaluator(llm_client=self.llm_client)
        metrics = ExecutionMetrics()
        final_result = trace.execution_result.final_result if trace.execution_result else None
        if not final_result or not trace.execution_result or not trace.execution_result.success:
            return metrics
        child_count = len(trace.child_traces)
        metrics.compression = evaluator.evaluate_compression(
            context=context, final_result=final_result, child_call_count=child_count,
        )
        execution_time_ms = trace.execution_result.execution_time_ms if trace.execution_result else 0
        metrics.memory_speedup = evaluator.evaluate_memory_speedup(
            current_tokens=trace.total_input_tokens + trace.total_output_tokens,
            current_cost_usd=trace.total_cost_usd, current_time_ms=execution_time_ms,
            current_child_calls=child_count, memory_keys=list(memory.keys()),
            baseline_execution=baseline_execution,
        )
        return metrics

    async def run(
        self, user_query: str, context: str,
        memory: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> ExecutionTrace:
        """
        Run the RLM agent — Algorithm 1 from the paper.

        state <- InitREPL(prompt=P)
        state <- AddFunction(state, sub_RLM_M)
        hist <- [Metadata(state)]
        while True do
            code <- LLM_M(hist)
            (state, stdout) <- REPL(state, code)
            hist <- hist || code || Metadata(stdout)
            if state[Final] is set then return state[Final]
        """
        from uuid import uuid4

        execution_id = execution_id or str(uuid4())
        root_node_id = str(uuid4())
        memory = memory or {}

        self._current_trace = ExecutionTrace(
            execution_id=execution_id, root_node_id=root_node_id,
            user_query=user_query, context_size=len(context),
            context_hash=self._hash_context(context),
            generated_code="", started_at=datetime.utcnow(),
        )
        self._child_sequence = 0
        self._current_depth = 0

        if self.on_node_update:
            self.on_node_update({"type": "execution_start", "data": {
                "execution_id": execution_id, "root_node_id": root_node_id,
                "user_query": user_query, "context_size": len(context),
            }})

        try:
            max_chunk_chars = _max_chunk_chars_for_model(self.config.model)

            # System prompt (paper's Appendix C)
            system_prompt = _get_rlm_system_prompt(
                context_length=len(context),
                max_chunk_chars=max_chunk_chars,
                model=self.config.model,
            )

            # Initialize persistent REPL
            async def child_query(prompt: str) -> LLMResponse:
                return await self._child_agent_query(prompt, memory)

            repl = REPLExecutor(
                context=context, memory=memory,
                llm_query_fn=child_query, max_chunk_chars=max_chunk_chars,
            )

            # hist <- [Metadata(state)]
            conversation_history = [{
                "role": "user",
                "content": f"Query: {user_query}\n\nThe context ({len(context):,} chars) is loaded in the REPL as the variable `context`. Use the REPL to explore and answer the query.",
            }]

            all_code_blocks = []

            # === RLM LOOP ===
            for iteration in range(self.config.max_iterations):

                # code <- LLM_M(hist)
                llm_response = await self.llm_client.rlm_iteration(
                    conversation_history=conversation_history,
                    system_prompt=system_prompt,
                    model=self.config.model,
                )

                self._current_trace.total_input_tokens += llm_response.input_tokens
                self._current_trace.total_output_tokens += llm_response.output_tokens
                self._current_trace.total_cost_usd += llm_response.cost_usd

                llm_output = llm_response.content
                all_code_blocks.append(llm_output)
                has_code = '```' in llm_output

                # (state, stdout) <- REPL(state, code)
                step_result = None
                if has_code:
                    step_result = await repl.execute_step(
                        code=llm_output, timeout=self.config.execution_timeout,
                    )
                    # if state[Final] is set then return state[Final]
                    if step_result.final_set and step_result.final_value and step_result.final_value.strip():
                        _make_result(self._current_trace, repl, all_code_blocks,
                                     success=True, final_result=step_result.final_value)
                        self._current_trace.iterations.append({
                            "iteration": iteration + 1, "final_set": True,
                            "child_calls": step_result.child_calls_this_step,
                        })
                        return self._current_trace
                    elif step_result.final_set:
                        # FINAL was called with empty value — reset and keep going
                        repl._final_set = False
                        repl._final_result = None

                # Check for FINAL/FINAL_VAR in text (outside code blocks)
                text_final = _extract_final_from_text(llm_output)
                if text_final:
                    if text_final.startswith("__VAR__"):
                        var_name = text_final[7:]
                        final_value = str(repl._env.get(var_name, f"(var '{var_name}' not found)"))
                    else:
                        final_value = text_final
                    _make_result(self._current_trace, repl, all_code_blocks,
                                 success=True, final_result=final_value)
                    self._current_trace.iterations.append({
                        "iteration": iteration + 1, "type": "text_final",
                    })
                    return self._current_trace

                # hist <- hist || code || Metadata(stdout)
                self._current_trace.iterations.append({
                    "iteration": iteration + 1,
                    "has_code": has_code,
                    "error": step_result.error if step_result else None,
                    "child_calls": step_result.child_calls_this_step if step_result else 0,
                })

                conversation_history.append({"role": "assistant", "content": llm_output})

                if has_code:
                    truncated_stdout = _truncate_stdout(step_result.stdout)
                    if step_result.error:
                        feedback = f"REPL Error: {step_result.error}\n\nStdout:\n{truncated_stdout}"
                    else:
                        feedback = f"REPL Output ({len(step_result.stdout)} chars):\n{truncated_stdout}"
                else:
                    feedback = (
                        "You did not write any executable code. You MUST write code in ```repl blocks.\n"
                        "Example:\n```repl\nchunk = context[:MAX_CHUNK_CHARS]\n"
                        "answer = llm_query(f\"Answer this question: {user_query}\\n\\nText: {chunk}\")\n"
                        "FINAL(answer)\n```"
                    )

                # Escalating urgency as iterations run out
                remaining = self.config.max_iterations - iteration - 1
                if remaining <= 0:
                    feedback += "\n\nFINAL WARNING: This is your LAST iteration. You MUST call FINAL(answer) RIGHT NOW with your best answer. If you have any partial findings, call FINAL with those."
                elif remaining <= 2:
                    feedback += f"\n\nCRITICAL: Only {remaining} iteration(s) left! You MUST call FINAL(answer) NOW. Use what you have gathered so far. Do NOT start new analysis — just call FINAL(your_best_answer)."
                elif remaining <= 4:
                    feedback += f"\n\nWARNING: {remaining} iterations remaining. Wrap up your analysis and call FINAL(answer) soon."
                else:
                    feedback += "\n\nContinue. Call FINAL(answer) when you have your answer."

                conversation_history.append({"role": "user", "content": feedback})

            # Exhausted iterations — ask the LLM to provide a final answer
            # from the full conversation history (matching reference _default_answer)
            default_prompt = conversation_history + [{
                "role": "user",
                "content": "You have run out of iterations. Based on everything you have learned, please provide a final answer to the original query now. Be concise and direct.",
            }]
            try:
                default_response = await self.llm_client.rlm_iteration(
                    conversation_history=default_prompt,
                    system_prompt=system_prompt,
                    model=self.config.model,
                )
                self._current_trace.total_input_tokens += default_response.input_tokens
                self._current_trace.total_output_tokens += default_response.output_tokens
                self._current_trace.total_cost_usd += default_response.cost_usd
                default_answer = default_response.content.strip()
                if default_answer:
                    _make_result(self._current_trace, repl, all_code_blocks,
                                 success=True, final_result=default_answer)
                    return self._current_trace
            except Exception:
                pass

            _make_result(self._current_trace, repl, all_code_blocks,
                         success=False,
                         error=f"RLM loop exhausted {self.config.max_iterations} iterations without calling FINAL")
            return self._current_trace

        except Exception as e:
            self._current_trace.completed_at = datetime.utcnow()
            self._current_trace.execution_result = ExecutionResult(success=False, error=str(e))
            return self._current_trace


class RecursiveLettaAgent(LettaAgent):
    """Extended agent with recursive child agents (up to max depth)."""

    async def _child_agent_query(self, prompt: str, memory: Dict[str, Any]) -> LLMResponse:
        self._child_sequence += 1
        new_depth = self._current_depth + 1

        if new_depth >= self.config.max_recursion_depth:
            return await self.llm_client.child_agent_query(
                prompt=prompt, parent_memory=memory, model=self.config.model,
            )

        if len(prompt) > self.config.max_chunk_size:
            child_agent = RecursiveLettaAgent(
                llm_client=self.llm_client, config=self.config,
                on_node_update=self.on_node_update,
            )
            child_agent._current_depth = new_depth
            child_trace = await child_agent.run(
                user_query="Process and respond to this request",
                context=prompt, memory=memory,
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
