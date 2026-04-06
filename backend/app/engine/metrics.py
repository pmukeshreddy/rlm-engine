"""Metrics for evaluating agent execution quality and efficiency."""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from app.engine.llm import LLMClient, count_tokens


@dataclass
class CompressionResult:
    """Result of compression ratio analysis."""
    input_chars: int
    output_chars: int
    compression_ratio: float  # input / output (higher = more compression)
    input_tokens: int
    output_tokens: int
    token_compression_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "compression_ratio": round(self.compression_ratio, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "token_compression_ratio": round(self.token_compression_ratio, 2),
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
    compression: Optional[CompressionResult] = None
    memory_speedup: Optional[MemorySpeedupResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compression": self.compression.to_dict() if self.compression else None,
            "memory_speedup": self.memory_speedup.to_dict() if self.memory_speedup else None,
        }


class MetricsEvaluator:
    """Evaluates execution quality and efficiency metrics."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def evaluate_compression(
        self,
        context: str,
        final_result: str,
        child_call_count: int,
    ) -> CompressionResult:
        """
        Evaluate the compression ratio of the execution.

        Measures how much the large context was distilled into the final answer.
        """
        input_chars = len(context)
        output_chars = len(final_result) if final_result else 0

        input_tokens = count_tokens(context)
        output_tokens = count_tokens(final_result) if final_result else 0

        compression_ratio = input_chars / output_chars if output_chars > 0 else 0
        token_compression_ratio = input_tokens / output_tokens if output_tokens > 0 else 0

        return CompressionResult(
            input_chars=input_chars,
            output_chars=output_chars,
            compression_ratio=compression_ratio,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_compression_ratio=token_compression_ratio,
        )

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
