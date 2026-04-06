"""Baseline: direct single-call LLM approach (no chunking, no agents)."""
import time
from dataclasses import dataclass
from typing import Optional

from app.engine.llm import LLMClient, LLMResponse, count_tokens


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

    # Context window limits (in tokens) — full window, not reduced
    MODEL_LIMITS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16384,
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
        # Reserve tokens for system prompt, question, and output
        available_tokens = max_tokens - 2000

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

        start = time.perf_counter()
        response = await self.llm_client.complete(
            messages=[{
                "role": "user",
                "content": (
                    f"Answer the following question based on the provided context.\n\n"
                    f"Context:\n{context_for_prompt}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer in 1-2 short sentences maximum. Give ONLY the answer, no explanation or preamble."
                ),
            }],
            model=model,
            system_prompt="You answer questions based on provided context. Give short, direct answers — ideally under 20 words. Never repeat the question or add unnecessary context.",
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
