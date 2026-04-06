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


class DirectLLMBaseline:
    """
    Baseline approach: stuff the full context into a single LLM call.

    If the context exceeds the model's limit, it truncates from the middle
    (keeps beginning + end, which is standard for long-context baselines).
    """

    # Approximate context window limits (in tokens, leaving room for prompt + output)
    MODEL_LIMITS = {
        "gpt-4o": 120000,
        "gpt-4o-mini": 120000,
        "gpt-4-turbo-preview": 120000,
        "gpt-4-turbo": 120000,
        "gpt-4": 7000,
        "gpt-3.5-turbo": 14000,
        "claude-3-opus-20240229": 190000,
        "claude-3-sonnet-20240229": 190000,
        "claude-3-haiku-20240307": 190000,
        "claude-3-5-sonnet-20241022": 190000,
    }

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def _truncate_context(self, context: str, model: str) -> tuple[str, bool]:
        """
        Truncate context to fit model's window.

        Keeps beginning and end (middle-out truncation), which preserves
        document structure better than simple head truncation.
        """
        max_tokens = self.MODEL_LIMITS.get(model, 14000)
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
        Run the baseline: single LLM call with full (or truncated) context.
        """
        context_for_prompt, truncated = self._truncate_context(context, model)

        start = time.perf_counter()
        response = await self.llm_client.complete(
            messages=[{
                "role": "user",
                "content": (
                    f"Answer the following question based on the provided context.\n\n"
                    f"Context:\n{context_for_prompt}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer concisely and accurately based only on the context provided."
                ),
            }],
            model=model,
            system_prompt="You answer questions based on provided context. Be concise and accurate.",
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
        )
