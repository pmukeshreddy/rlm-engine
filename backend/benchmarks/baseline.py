"""Baselines: Direct LLM, CodeAct (+sub-calls), Summary agent."""
import time
from dataclasses import dataclass
from typing import Optional, List

from app.engine.llm import LLMClient, LLMResponse, count_tokens, MODEL_CONTEXT_WINDOWS


@dataclass
class BaselineResult:
    """Result from the direct LLM baseline."""
    answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    truncated: bool  # Whether context was truncated to fit
    context_tokens_used: int  # Actual tokens sent to the model


class DirectLLMBaseline:
    """
    Baseline approach: stuff the full context into a single LLM call.

    Sends the full context without truncation for a fair apple-to-apple
    comparison with RLM Engine. Both approaches see the same input.

    Only truncates if the context truly exceeds the model's context window.
    """

    # Context window limits (in tokens)
    # gpt-5-mini capped at 16K to force truncation (real window is 400K)
    MODEL_LIMITS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16384,
        "gpt-4.1-nano": 16384,
        "gpt-4.1-mini": 16384,
        "gpt-5-mini": 16384,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-sonnet-4-6": 200000,
    }

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def _truncate_context(self, context: str, model: str) -> tuple[str, bool]:
        """
        Truncate context ONLY if it exceeds the model's actual context window.

        For a fair comparison, we want the baseline to see the same full
        context that RLM processes. Truncation is a last resort.
        """
        max_tokens = self.MODEL_LIMITS.get(model, 16384)
        # Reserve tokens for system prompt, question, instructions, and output
        available_tokens = max_tokens - 3000

        token_count = count_tokens(context, model)
        if token_count <= available_tokens:
            return context, False

        # Estimate chars per token (~4 for English)
        chars_per_token = len(context) / token_count
        max_chars = int(available_tokens * chars_per_token)

        # Keep 60% from beginning, 40% from end
        head_chars = int(max_chars * 0.6)
        tail_chars = max_chars - head_chars

        truncated = (
            context[:head_chars]
            + "\n\n[... content truncated due to length ...]\n\n"
            + context[-tail_chars:]
        )
        return truncated, True

    async def run(
        self,
        context: str,
        question: str,
        model: str = "gpt-4o-mini",
    ) -> BaselineResult:
        """
        Run the baseline: single LLM call with full context.
        Only truncates if context exceeds the model's actual window.
        """
        context_for_prompt, truncated = self._truncate_context(context, model)
        context_tokens = count_tokens(context_for_prompt, model)

        truncation_note = ""
        if truncated:
            truncation_note = (
                "\n\nNOTE: The context above was truncated to fit the model's context window. "
                "You may be missing parts of the original document. If the answer is not in the "
                "provided context, say \"Insufficient context\" rather than guessing."
            )

        start = time.perf_counter()
        response = await self.llm_client.complete(
            messages=[{
                "role": "user",
                "content": (
                    f"Answer the following question based on the provided context.\n\n"
                    f"Context:\n{context_for_prompt}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer in 1-2 short sentences maximum. Give ONLY the answer, no explanation or preamble."
                    f"{truncation_note}"
                ),
            }],
            model=model,
            system_prompt="You answer questions based on provided context. Give short, direct answers — ideally under 20 words. Never repeat the question or add unnecessary context. If the context appears truncated and the answer is not present, say \"Insufficient context.\"",
            temperature=0.3,
            max_tokens=1024,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return BaselineResult(
            answer=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            latency_ms=latency_ms,
            truncated=truncated,
            context_tokens_used=context_tokens,
        )


class CodeActBaseline:
    """
    CodeAct baseline from the paper (Wang et al., 2024).

    Unlike RLM, CodeAct loads the context directly into the LLM's prompt
    (not as a REPL variable). It can execute code and optionally invoke
    sub-LLM calls, but the context itself fills the context window.

    Paper flaw #1: "puts P into the LLM context window and thus inherits
    the window limitations of M"
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    async def run(
        self,
        context: str,
        question: str,
        model: str = "gpt-4o-mini",
        with_sub_calls: bool = False,
    ) -> BaselineResult:
        max_tokens = MODEL_CONTEXT_WINDOWS.get(model, 16384)
        available_tokens = max_tokens - 3000
        context_tokens = count_tokens(context, model)
        truncated = context_tokens > available_tokens

        if truncated:
            chars_per_token = len(context) / context_tokens
            max_chars = int(available_tokens * chars_per_token)
            context_for_prompt = context[:max_chars]
        else:
            context_for_prompt = context

        sub_call_note = ""
        if with_sub_calls:
            sub_call_note = "\nYou can delegate sub-tasks by writing SUB_CALL(question) in your reasoning."

        start = time.perf_counter()
        response = await self.llm_client.complete(
            messages=[{
                "role": "user",
                "content": (
                    f"You are a CodeAct agent. You can execute Python code and search through "
                    f"the provided context to answer questions.{sub_call_note}\n\n"
                    f"Context:\n{context_for_prompt}\n\n"
                    f"Question: {question}\n\n"
                    f"ANSWER: [your final answer]"
                ),
            }],
            model=model,
            system_prompt="You are a CodeAct agent that answers questions using code execution and reasoning. Provide your final answer after ANSWER:",
            temperature=0.3,
            max_tokens=1024,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Extract answer after "ANSWER:" if present
        answer = response.content
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1].strip()

        return BaselineResult(
            answer=answer,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            latency_ms=latency_ms,
            truncated=truncated,
            context_tokens_used=count_tokens(context_for_prompt, model),
        )


class SummaryAgentBaseline:
    """
    Summary agent baseline from the paper (Sun et al., 2025; Wu et al., 2025).

    Iteratively accumulates context and summarizes when the context window
    is full. For contexts larger than the window, chunks and summarizes
    progressively.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    async def run(
        self,
        context: str,
        question: str,
        model: str = "gpt-4o-mini",
    ) -> BaselineResult:
        max_tokens = MODEL_CONTEXT_WINDOWS.get(model, 16384)
        # Reserve space for system prompt, question, and output
        available_tokens = max_tokens - 3000
        chars_per_token = 3.5
        chunk_chars = int(available_tokens * chars_per_token * 0.6)

        start = time.perf_counter()
        total_input = 0
        total_output = 0
        total_cost = 0.0

        # Chunk and iteratively summarize
        running_summary = ""
        chunks = [context[i:i+chunk_chars] for i in range(0, len(context), chunk_chars)]
        truncated = len(chunks) > 1

        for i, chunk in enumerate(chunks):
            if running_summary:
                prompt = (
                    f"Previous summary:\n{running_summary}\n\n"
                    f"New content (section {i+1}/{len(chunks)}):\n{chunk}\n\n"
                    f"Question: {question}\n\n"
                    f"Update the summary with any new relevant information for answering the question."
                )
            else:
                prompt = (
                    f"Content (section {i+1}/{len(chunks)}):\n{chunk}\n\n"
                    f"Question: {question}\n\n"
                    f"Summarize the relevant information for answering the question."
                )

            response = await self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                system_prompt="You are a summarization agent. Extract and accumulate information relevant to the question. Be concise.",
                temperature=0.3,
                max_tokens=1024,
            )
            running_summary = response.content
            total_input += response.input_tokens
            total_output += response.output_tokens
            total_cost += response.cost_usd

        # Final answer from summary
        final_response = await self.llm_client.complete(
            messages=[{
                "role": "user",
                "content": (
                    f"Based on this summary of the full document:\n{running_summary}\n\n"
                    f"Question: {question}\n\n"
                    f"Give a short, direct answer."
                ),
            }],
            model=model,
            system_prompt="Answer the question based on the provided summary. Be concise and direct.",
            temperature=0.3,
            max_tokens=256,
        )
        total_input += final_response.input_tokens
        total_output += final_response.output_tokens
        total_cost += final_response.cost_usd
        latency_ms = (time.perf_counter() - start) * 1000

        return BaselineResult(
            answer=final_response.content,
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=total_cost,
            latency_ms=latency_ms,
            truncated=truncated,
            context_tokens_used=total_input,
        )
