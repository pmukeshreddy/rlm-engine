"""
RLM Engine Benchmark — Compare recursive agent vs direct LLM on standard datasets.

Usage:
    python -m benchmarks.run --dataset narrativeqa --samples 10 --model gpt-4o-mini
    python -m benchmarks.run --dataset quality --samples 20
    python -m benchmarks.run --dataset longbench --samples 15
    python -m benchmarks.run --dataset scrolls_qmsum --samples 10
    python -m benchmarks.run --dataset narrativeqa --samples 5 --output results.json
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.agent import LettaAgent, AgentConfig
from app.engine.llm import LLMClient
from app.engine.metrics import MetricsEvaluator

from benchmarks.datasets import load_benchmark, BenchmarkSample, DATASET_REGISTRY
from benchmarks.baseline import DirectLLMBaseline
from benchmarks.evaluate import score_prediction


def print_header(dataset: str, n_samples: int, model: str):
    w = 72
    print()
    print("=" * w)
    print(f"  RLM ENGINE BENCHMARK")
    print(f"  Dataset:  {dataset}")
    print(f"  Samples:  {n_samples}")
    print(f"  Model:    {model}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * w)
    print()


def print_progress(i: int, total: int, sample_id: str, approach: str):
    pct = (i + 1) / total * 100
    print(f"  [{i+1}/{total}] ({pct:5.1f}%) {approach:12s} | {sample_id}")


def print_comparison_table(results: Dict[str, Any]):
    w = 72
    rlm = results["rlm_aggregate"]
    direct = results["direct_aggregate"]

    def winner(rlm_val, direct_val, higher_is_better=True):
        if rlm_val == direct_val:
            return "  TIE"
        if higher_is_better:
            return "< RLM" if rlm_val > direct_val else "  Direct"
        else:
            return "< RLM" if rlm_val < direct_val else "  Direct"

    print()
    print("=" * w)
    print(f"  {'METRIC':<28} {'RLM Engine':>12}  {'Direct LLM':>12}  {'Winner':>8}")
    print("-" * w)

    # Accuracy metrics
    for metric in ["f1", "rouge1", "rouge2", "rougeL", "exact_match"]:
        rv = rlm.get(metric, 0)
        dv = direct.get(metric, 0)
        print(f"  {metric:<28} {rv:>12.4f}  {dv:>12.4f}  {winner(rv, dv):>8}")

    print("-" * w)

    # Faithfulness
    rv = rlm.get("faithfulness", 0)
    dv = direct.get("faithfulness", 0)
    print(f"  {'faithfulness_score':<28} {rv:>12.3f}  {dv:>12.3f}  {winner(rv, dv):>8}")

    print("-" * w)

    # Efficiency metrics
    rv = rlm.get("avg_tokens", 0)
    dv = direct.get("avg_tokens", 0)
    print(f"  {'avg_tokens_used':<28} {rv:>12,.0f}  {dv:>12,.0f}  {winner(rv, dv, False):>8}")

    rv = rlm.get("avg_cost", 0)
    dv = direct.get("avg_cost", 0)
    print(f"  {'avg_cost_per_query ($)':<28} {rv:>12.5f}  {dv:>12.5f}  {winner(rv, dv, False):>8}")

    rv = rlm.get("avg_latency_ms", 0)
    dv = direct.get("avg_latency_ms", 0)
    print(f"  {'avg_latency (ms)':<28} {rv:>12,.0f}  {dv:>12,.0f}  {winner(rv, dv, False):>8}")

    rv = rlm.get("total_cost", 0)
    dv = direct.get("total_cost", 0)
    print(f"  {'total_cost ($)':<28} {rv:>12.5f}  {dv:>12.5f}  {winner(rv, dv, False):>8}")

    print("-" * w)

    # RLM-specific
    print(f"  {'avg_compression_ratio':<28} {rlm.get('avg_compression', 0):>12.1f}x {'N/A':>12}  {'':>8}")
    print(f"  {'avg_child_calls':<28} {rlm.get('avg_child_calls', 0):>12.1f}  {'N/A':>12}  {'':>8}")
    print(f"  {'context_truncated (direct)':<28} {'N/A':>12}  {direct.get('truncated_pct', 0):>11.0f}%  {'':>8}")

    print("=" * w)
    print()


def aggregate_results(per_sample: List[Dict[str, Any]], approach: str) -> Dict[str, Any]:
    n = len(per_sample)
    if n == 0:
        return {}

    def avg(key):
        vals = [s[approach].get(key, 0) for s in per_sample if s.get(approach)]
        return sum(vals) / len(vals) if vals else 0

    def avg_score(key):
        vals = [s[approach]["scores"].get(key, 0) for s in per_sample if s.get(approach) and s[approach].get("scores")]
        return sum(vals) / len(vals) if vals else 0

    agg = {
        "f1": avg_score("f1"),
        "exact_match": avg_score("exact_match"),
        "rouge1": avg_score("rouge1"),
        "rouge2": avg_score("rouge2"),
        "rougeL": avg_score("rougeL"),
        "faithfulness": avg("faithfulness_score"),
        "avg_tokens": avg("total_tokens"),
        "avg_cost": avg("cost_usd"),
        "avg_latency_ms": avg("latency_ms"),
        "total_cost": sum(s[approach].get("cost_usd", 0) for s in per_sample if s.get(approach)),
    }

    if approach == "rlm":
        agg["avg_compression"] = avg("compression_ratio")
        agg["avg_child_calls"] = avg("child_calls")
    elif approach == "direct":
        truncated = sum(1 for s in per_sample if s.get(approach, {}).get("truncated", False))
        agg["truncated_pct"] = (truncated / n) * 100

    return agg


async def run_benchmark(
    dataset_name: str,
    n_samples: int,
    model: str,
    output_path: str = None,
    eval_model: str = "gpt-4o-mini",
    **dataset_kwargs,
):
    # Load dataset
    print(f"Loading {dataset_name} ({n_samples} samples)...")
    samples = load_benchmark(dataset_name, n_samples=n_samples, **dataset_kwargs)
    print(f"  Loaded {len(samples)} samples")
    print(f"  Avg context size: {sum(len(s.context) for s in samples) // len(samples):,} chars")
    print()

    # Initialize
    llm_client = LLMClient()
    agent = LettaAgent(llm_client=llm_client, config=AgentConfig(model=model))
    baseline = DirectLLMBaseline(llm_client=llm_client)
    evaluator = MetricsEvaluator(llm_client=llm_client)

    per_sample_results = []

    for i, sample in enumerate(samples):
        result = {
            "id": sample.id,
            "dataset": sample.dataset,
            "question": sample.question,
            "reference_answers": sample.reference_answers,
            "context_chars": len(sample.context),
        }

        # --- RLM Engine ---
        print_progress(i, len(samples), sample.id, "RLM Engine")
        try:
            start = time.perf_counter()
            trace = await agent.run(
                user_query=sample.question,
                context=sample.context,
                memory={},
            )
            rlm_latency = (time.perf_counter() - start) * 1000

            rlm_answer = ""
            if trace.execution_result and trace.execution_result.success:
                rlm_answer = trace.execution_result.final_result or ""

            # Score against references
            rlm_scores = score_prediction(rlm_answer, sample.reference_answers)

            # Faithfulness
            faith_result = None
            if rlm_answer:
                faith_result = await evaluator.evaluate_faithfulness(
                    context=sample.context,
                    final_result=rlm_answer,
                    model=eval_model,
                )

            # Compression
            comp_result = None
            if rlm_answer:
                comp_result = evaluator.evaluate_compression(
                    context=sample.context,
                    final_result=rlm_answer,
                    child_call_count=len(trace.child_traces),
                )

            result["rlm"] = {
                "answer": rlm_answer,
                "scores": rlm_scores,
                "total_tokens": trace.total_input_tokens + trace.total_output_tokens,
                "cost_usd": trace.total_cost_usd,
                "latency_ms": rlm_latency,
                "child_calls": len(trace.child_traces),
                "faithfulness_score": faith_result.score if faith_result else 0,
                "faithfulness_verdict": faith_result.verdict if faith_result else "N/A",
                "compression_ratio": comp_result.compression_ratio if comp_result else 0,
                "info_density": comp_result.info_density_score if comp_result else 0,
                "success": trace.execution_result.success if trace.execution_result else False,
            }

        except Exception as e:
            print(f"    [ERROR] RLM failed: {e}")
            result["rlm"] = {"answer": "", "scores": score_prediction("", sample.reference_answers),
                             "total_tokens": 0, "cost_usd": 0, "latency_ms": 0, "child_calls": 0,
                             "faithfulness_score": 0, "compression_ratio": 0, "success": False, "error": str(e)}

        # --- Direct LLM Baseline ---
        print_progress(i, len(samples), sample.id, "Direct LLM")
        try:
            baseline_result = await baseline.run(
                context=sample.context,
                question=sample.question,
                model=model,
            )

            direct_scores = score_prediction(baseline_result.answer, sample.reference_answers)

            # Faithfulness for baseline too
            direct_faith = None
            if baseline_result.answer:
                direct_faith = await evaluator.evaluate_faithfulness(
                    context=sample.context,
                    final_result=baseline_result.answer,
                    model=eval_model,
                )

            result["direct"] = {
                "answer": baseline_result.answer,
                "scores": direct_scores,
                "total_tokens": baseline_result.input_tokens + baseline_result.output_tokens,
                "cost_usd": baseline_result.cost_usd,
                "latency_ms": baseline_result.latency_ms,
                "truncated": baseline_result.truncated,
                "faithfulness_score": direct_faith.score if direct_faith else 0,
                "faithfulness_verdict": direct_faith.verdict if direct_faith else "N/A",
            }

        except Exception as e:
            print(f"    [ERROR] Direct failed: {e}")
            result["direct"] = {"answer": "", "scores": score_prediction("", sample.reference_answers),
                                "total_tokens": 0, "cost_usd": 0, "latency_ms": 0, "truncated": False,
                                "faithfulness_score": 0, "error": str(e)}

        per_sample_results.append(result)

    # Aggregate
    rlm_agg = aggregate_results(per_sample_results, "rlm")
    direct_agg = aggregate_results(per_sample_results, "direct")

    full_results = {
        "metadata": {
            "dataset": dataset_name,
            "model": model,
            "eval_model": eval_model,
            "n_samples": len(samples),
            "timestamp": datetime.now().isoformat(),
        },
        "rlm_aggregate": rlm_agg,
        "direct_aggregate": direct_agg,
        "per_sample": per_sample_results,
    }

    # Print table
    print_header(dataset_name, len(samples), model)
    print_comparison_table(full_results)

    # Save to file
    if output_path:
        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        print(f"  Results saved to {output_path}")
        print()

    return full_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RLM Engine vs Direct LLM on standard datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
datasets:
  narrativeqa     QA over books and movie scripts (Kocisky et al., 2018)
  quality         Multiple-choice long-document QA (Pang et al., 2022)
  longbench       Comprehensive long-context benchmark (Bai et al., 2023)
  scrolls_qmsum   Query-based meeting summarization (Shaham et al., 2022)

examples:
  python -m benchmarks.run --dataset narrativeqa --samples 10
  python -m benchmarks.run --dataset quality --samples 20 --model gpt-4o
  python -m benchmarks.run --dataset longbench --samples 15 --output results.json
        """,
    )

    parser.add_argument(
        "--dataset", "-d",
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Benchmark dataset to use",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="Model for both RLM and baseline (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--eval-model",
        default="gpt-4o-mini",
        help="Model for faithfulness evaluation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save full results JSON",
    )
    parser.add_argument(
        "--task",
        default="narrativeqa",
        help="Sub-task for LongBench (default: narrativeqa)",
    )

    args = parser.parse_args()

    kwargs = {}
    if args.dataset == "longbench":
        kwargs["task"] = args.task

    asyncio.run(run_benchmark(
        dataset_name=args.dataset,
        n_samples=args.samples,
        model=args.model,
        output_path=args.output,
        eval_model=args.eval_model,
        **kwargs,
    ))


if __name__ == "__main__":
    main()
