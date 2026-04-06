"""Metrics for evaluating agent execution quality and efficiency."""
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from app.engine.llm import LLMClient, LLMResponse, count_tokens


@dataclass
class FaithfulnessResult:
    """Result of faithfulness evaluation."""
    score: float  # 0.0 to 1.0
    verdict: str  # "faithful", "partially_faithful", "unfaithful"
    claims_total: int
    claims_supported: int
    claims_unsupported: int
    unsupported_claims: List[str]
    eval_tokens_used: int
    eval_cost_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "claims_total": self.claims_total,
            "claims_supported": self.claims_supported,
            "claims_unsupported": self.claims_unsupported,
            "unsupported_claims": self.unsupported_claims,
            "eval_tokens_used": self.eval_tokens_used,
            "eval_cost_usd": self.eval_cost_usd,
        }


@dataclass
class CompressionResult:
    """Result of compression ratio analysis."""
    input_chars: int
    output_chars: int
    compression_ratio: float  # input / output (higher = more compression)
    input_tokens: int
    output_tokens: int
    token_compression_ratio: float
    info_density_score: float  # 0.0 to 1.0 — how much of the output is meaningful content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "compression_ratio": round(self.compression_ratio, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "token_compression_ratio": round(self.token_compression_ratio, 2),
            "info_density_score": round(self.info_density_score, 3),
        }


@dataclass
class MemorySpeedupResult:
    """Result of memory speedup comparison."""
    has_baseline: bool  # Whether a no-memory baseline exists
    baseline_execution_id: Optional[str]
    baseline_tokens: int
    baseline_cost_usd: float
    baseline_time_ms: float
    baseline_child_calls: int
    current_tokens: int
    current_cost_usd: float
    current_time_ms: float
    current_child_calls: int
    memory_keys_used: List[str]
    token_reduction_pct: float  # negative means increase
    cost_reduction_pct: float
    time_reduction_pct: float
    child_call_reduction_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_baseline": self.has_baseline,
            "baseline_execution_id": self.baseline_execution_id,
            "baseline": {
                "tokens": self.baseline_tokens,
                "cost_usd": round(self.baseline_cost_usd, 6),
                "time_ms": round(self.baseline_time_ms, 1),
                "child_calls": self.baseline_child_calls,
            },
            "current": {
                "tokens": self.current_tokens,
                "cost_usd": round(self.current_cost_usd, 6),
                "time_ms": round(self.current_time_ms, 1),
                "child_calls": self.current_child_calls,
            },
            "memory_keys_used": self.memory_keys_used,
            "improvements": {
                "token_reduction_pct": round(self.token_reduction_pct, 1),
                "cost_reduction_pct": round(self.cost_reduction_pct, 1),
                "time_reduction_pct": round(self.time_reduction_pct, 1),
                "child_call_reduction_pct": round(self.child_call_reduction_pct, 1),
            },
        }


@dataclass
class ExecutionMetrics:
    """All metrics for a single execution."""
    faithfulness: Optional[FaithfulnessResult] = None
    compression: Optional[CompressionResult] = None
    memory_speedup: Optional[MemorySpeedupResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": self.faithfulness.to_dict() if self.faithfulness else None,
            "compression": self.compression.to_dict() if self.compression else None,
            "memory_speedup": self.memory_speedup.to_dict() if self.memory_speedup else None,
        }


class MetricsEvaluator:
    """Evaluates execution quality and efficiency metrics."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    async def evaluate_faithfulness(
        self,
        context: str,
        final_result: str,
        model: str = "gpt-4o-mini",
    ) -> FaithfulnessResult:
        """
        Evaluate whether the final answer is faithful to the source context.

        Uses LLM-as-judge: samples chunks from the context, then asks a judge
        model to verify each claim in the answer against the source material.

        Steps:
        1. Extract claims from the final answer
        2. Sample relevant chunks from context
        3. Judge each claim against the context
        """
        # Step 1: Extract claims from the answer
        extract_response = await self.llm_client.complete(
            messages=[{
                "role": "user",
                "content": (
                    f"Extract all factual claims from this answer as a JSON list of strings. "
                    f"Each claim should be a single, verifiable statement.\n\n"
                    f"Answer:\n{final_result}\n\n"
                    f"Respond with ONLY a JSON array of strings, e.g. "
                    f'["claim 1", "claim 2"]. If there are no factual claims, respond with [].'
                ),
            }],
            model=model,
            system_prompt="You extract factual claims from text. Respond only with valid JSON.",
            temperature=0.0,
            max_tokens=2048,
        )

        total_eval_tokens = extract_response.input_tokens + extract_response.output_tokens
        total_eval_cost = extract_response.cost_usd

        try:
            claims = json.loads(extract_response.content.strip())
            if not isinstance(claims, list):
                claims = []
        except json.JSONDecodeError:
            claims = []

        if not claims:
            return FaithfulnessResult(
                score=1.0,
                verdict="faithful",
                claims_total=0,
                claims_supported=0,
                claims_unsupported=0,
                unsupported_claims=[],
                eval_tokens_used=total_eval_tokens,
                eval_cost_usd=total_eval_cost,
            )

        # Step 2: Sample context chunks around relevant areas
        # Use beginning, middle, end + keyword-matched sections
        context_sample = self._sample_context_for_verification(context, claims)

        # Step 3: Judge each claim against the context
        claims_json = json.dumps(claims, indent=2)
        judge_response = await self.llm_client.complete(
            messages=[{
                "role": "user",
                "content": (
                    f"You are a faithfulness judge. For each claim below, determine if it is "
                    f"SUPPORTED or UNSUPPORTED by the source context.\n\n"
                    f"Source Context (sampled):\n{context_sample}\n\n"
                    f"Claims to verify:\n{claims_json}\n\n"
                    f"Respond with a JSON object:\n"
                    f'{{"results": [{{"claim": "...", "supported": true/false, "reason": "..."}}]}}'
                ),
            }],
            model=model,
            system_prompt=(
                "You judge whether claims are supported by source text. "
                "Be strict: a claim is SUPPORTED only if the source text clearly contains "
                "the information. Respond only with valid JSON."
            ),
            temperature=0.0,
            max_tokens=4096,
        )

        total_eval_tokens += judge_response.input_tokens + judge_response.output_tokens
        total_eval_cost += judge_response.cost_usd

        # Parse results
        supported = 0
        unsupported_claims = []
        try:
            judge_result = json.loads(judge_response.content.strip())
            for item in judge_result.get("results", []):
                if item.get("supported", False):
                    supported += 1
                else:
                    unsupported_claims.append(item.get("claim", "unknown"))
        except json.JSONDecodeError:
            # If parsing fails, be conservative
            supported = 0
            unsupported_claims = claims

        unsupported = len(claims) - supported
        score = supported / len(claims) if claims else 1.0

        if score >= 0.9:
            verdict = "faithful"
        elif score >= 0.5:
            verdict = "partially_faithful"
        else:
            verdict = "unfaithful"

        return FaithfulnessResult(
            score=round(score, 3),
            verdict=verdict,
            claims_total=len(claims),
            claims_supported=supported,
            claims_unsupported=unsupported,
            unsupported_claims=unsupported_claims[:10],  # Cap at 10
            eval_tokens_used=total_eval_tokens,
            eval_cost_usd=total_eval_cost,
        )

    def _sample_context_for_verification(
        self,
        context: str,
        claims: List[str],
        max_sample_chars: int = 30000,
    ) -> str:
        """
        Intelligently sample context for claim verification.

        Takes beginning, end, and keyword-matched sections.
        """
        if len(context) <= max_sample_chars:
            return context

        samples = []
        per_section = max_sample_chars // 4

        # Beginning
        samples.append(context[:per_section])

        # End
        samples.append(context[-per_section:])

        # Keyword-matched middle sections
        keywords = set()
        for claim in claims:
            words = claim.lower().split()
            # Take significant words (>4 chars, skip common words)
            common = {"this", "that", "with", "from", "have", "been", "were", "their", "about", "which", "would", "there"}
            keywords.update(w for w in words if len(w) > 4 and w not in common)

        remaining_budget = max_sample_chars - (per_section * 2)
        chunk_size = 5000
        matched_chunks = []

        for i in range(0, len(context), chunk_size):
            chunk = context[i:i + chunk_size]
            chunk_lower = chunk.lower()
            matches = sum(1 for kw in keywords if kw in chunk_lower)
            if matches > 0:
                matched_chunks.append((matches, chunk))

        # Sort by relevance and take top chunks within budget
        matched_chunks.sort(key=lambda x: x[0], reverse=True)
        chars_used = 0
        for _, chunk in matched_chunks:
            if chars_used + len(chunk) > remaining_budget:
                break
            samples.append(chunk)
            chars_used += len(chunk)

        return "\n...\n".join(samples)

    def evaluate_compression(
        self,
        context: str,
        final_result: str,
        child_call_count: int,
    ) -> CompressionResult:
        """
        Evaluate the compression ratio of the execution.

        Measures how much the large context was distilled into the final answer,
        accounting for both raw size and information density.
        """
        input_chars = len(context)
        output_chars = len(final_result) if final_result else 0

        input_tokens = count_tokens(context)
        output_tokens = count_tokens(final_result) if final_result else 0

        compression_ratio = input_chars / output_chars if output_chars > 0 else 0
        token_compression_ratio = input_tokens / output_tokens if output_tokens > 0 else 0

        # Info density: ratio of non-whitespace, non-filler content in the output.
        # A good summary has high density (few filler phrases, high information per word).
        info_density_score = self._estimate_info_density(final_result, child_call_count)

        return CompressionResult(
            input_chars=input_chars,
            output_chars=output_chars,
            compression_ratio=compression_ratio,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_compression_ratio=token_compression_ratio,
            info_density_score=info_density_score,
        )

    def _estimate_info_density(self, text: str, child_call_count: int) -> float:
        """
        Estimate information density of the output.

        Heuristic based on:
        - Unique word ratio (diverse vocabulary = more information)
        - Sentence count relative to length (structured = denser)
        - Presence of specific entities (numbers, proper nouns)
        """
        if not text:
            return 0.0

        words = text.lower().split()
        if not words:
            return 0.0

        # Unique word ratio — higher means more diverse/informative
        unique_ratio = len(set(words)) / len(words)

        # Entity density — numbers, percentages, dates indicate factual content
        entity_count = 0
        for word in words:
            stripped = word.strip(".,;:!?()[]{}\"'")
            if any(c.isdigit() for c in stripped):
                entity_count += 1
            elif stripped.startswith("$") or stripped.endswith("%"):
                entity_count += 1
        entity_ratio = min(entity_count / len(words), 0.3) / 0.3  # Normalize to 0-1

        # Sentence structure — more sentences per word = more structured
        sentences = text.count(".") + text.count("!") + text.count("?")
        avg_sentence_len = len(words) / max(sentences, 1)
        # Optimal sentence length ~15-25 words
        structure_score = 1.0 - min(abs(avg_sentence_len - 20) / 20, 1.0)

        # Weighted combination
        score = (unique_ratio * 0.4) + (entity_ratio * 0.35) + (structure_score * 0.25)
        return min(max(score, 0.0), 1.0)

    def evaluate_memory_speedup(
        self,
        current_tokens: int,
        current_cost_usd: float,
        current_time_ms: float,
        current_child_calls: int,
        memory_keys: List[str],
        baseline_execution: Optional[Dict[str, Any]] = None,
    ) -> MemorySpeedupResult:
        """
        Evaluate how much memory improved this execution vs a baseline.

        The baseline is the first execution on the same context_hash
        (when memory was empty). If no baseline exists, reports current
        metrics and flags has_baseline=False.
        """
        if not baseline_execution:
            return MemorySpeedupResult(
                has_baseline=False,
                baseline_execution_id=None,
                baseline_tokens=0,
                baseline_cost_usd=0.0,
                baseline_time_ms=0.0,
                baseline_child_calls=0,
                current_tokens=current_tokens,
                current_cost_usd=current_cost_usd,
                current_time_ms=current_time_ms,
                current_child_calls=current_child_calls,
                memory_keys_used=memory_keys,
                token_reduction_pct=0.0,
                cost_reduction_pct=0.0,
                time_reduction_pct=0.0,
                child_call_reduction_pct=0.0,
            )

        b_tokens = baseline_execution["total_tokens"]
        b_cost = baseline_execution["total_cost_usd"]
        b_time = baseline_execution["time_ms"]
        b_children = baseline_execution["child_calls"]

        def pct_reduction(baseline_val: float, current_val: float) -> float:
            if baseline_val == 0:
                return 0.0
            return ((baseline_val - current_val) / baseline_val) * 100

        return MemorySpeedupResult(
            has_baseline=True,
            baseline_execution_id=baseline_execution.get("execution_id"),
            baseline_tokens=b_tokens,
            baseline_cost_usd=b_cost,
            baseline_time_ms=b_time,
            baseline_child_calls=b_children,
            current_tokens=current_tokens,
            current_cost_usd=current_cost_usd,
            current_time_ms=current_time_ms,
            current_child_calls=current_child_calls,
            memory_keys_used=memory_keys,
            token_reduction_pct=pct_reduction(b_tokens, current_tokens),
            cost_reduction_pct=pct_reduction(b_cost, current_cost_usd),
            time_reduction_pct=pct_reduction(b_time, current_time_ms),
            child_call_reduction_pct=pct_reduction(b_children, current_child_calls),
        )
