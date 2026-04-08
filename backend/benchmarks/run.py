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

from app.engine.agent import RecursiveLettaAgent, AgentConfig
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
    for metric in ["f1", "rouge1", "rouge2", "rougeL", "exact_match", "bertscore"]:
        rv = rlm.get(metric, 0)
        dv = direct.get(metric, 0)
        print(f"  {metric:<28} {rv:>12.4f}  {dv:>12.4f}  {winner(rv, dv):>8}")

    print("-" * w)

    # Answer length ratio (closer to 1.0 is better)
    rv = rlm.get("length_ratio", 0)
    dv = direct.get("length_ratio", 0)
    print(f"  {'answer_length_ratio':<28} {rv:>12.2f}  {dv:>12.2f}  {'':>8}")

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
    print(f"  {'avg_rlm_iterations':<28} {rlm.get('avg_iterations', 0):>12.1f}  {'N/A':>12}  {'':>8}")
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
        "bertscore": avg_score("bertscore"),
        "length_ratio": avg_score("length_ratio"),
        "avg_tokens": avg("total_tokens"),
        "avg_cost": avg("cost_usd"),
        "avg_latency_ms": avg("latency_ms"),
        "total_cost": sum(s[approach].get("cost_usd", 0) for s in per_sample if s.get(approach)),
    }

    if approach == "rlm":
        agg["avg_compression"] = avg("compression_ratio")
        agg["avg_child_calls"] = avg("child_calls")
        agg["avg_iterations"] = avg("iterations")
    elif approach == "direct":
        truncated = sum(1 for s in per_sample if s.get(approach, {}).get("truncated", False))
        agg["truncated_pct"] = (truncated / n) * 100

    return agg


async def run_benchmark(
    dataset_name: str,
    n_samples: int,
    model: str,
    output_path: str = None,
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
    agent = RecursiveLettaAgent(llm_client=llm_client, config=AgentConfig(model=model))
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

        # Debug: show reference answers
        print(f"    [REF] {sample.reference_answers}")

        # --- RLM Engine (Iterative REPL Loop - Algorithm 1) ---
        from app.engine.agent import _max_chunk_chars_for_model
        max_cc = _max_chunk_chars_for_model(model)
        print(f"    [RLM INFO] MAX_CHUNK_CHARS={max_cc}, context={len(sample.context)} chars")
        print_progress(i, len(samples), sample.id, "RLM Engine")
        try:
            start = time.perf_counter()
            trace = await agent.run(
                user_query=sample.question,
                context=sample.context,
                memory={},
            )
            rlm_latency = (time.perf_counter() - start) * 1000

            n_iterations = len(trace.iterations)
            rlm_answer = ""
            if trace.execution_result and trace.execution_result.success:
                rlm_answer = trace.execution_result.final_result or ""
                print(f"    [RLM OK] {n_iterations} iterations, {len(trace.child_traces)} sub-calls")
            else:
                err = trace.execution_result.error if trace.execution_result else "No execution result"
                print(f"    [RLM FAIL] {n_iterations} iterations | {err[:300]}")
                if trace.execution_result and trace.execution_result.output_log:
                    for log in trace.execution_result.output_log[-3:]:
                        print(f"    [RLM LOG] {log[:200]}")

            # Debug: show RLM answer vs reference
            rlm_words = len(rlm_answer.split()) if rlm_answer else 0
            ref_words = len(sample.reference_answers[0].split()) if sample.reference_answers else 0
            print(f"    [RLM ANS] ({rlm_words}w vs ref {ref_words}w): {rlm_answer[:200]}{'...' if len(rlm_answer) > 200 else ''}")

            rlm_scores = score_prediction(rlm_answer, sample.reference_answers)
            print(f"    [RLM SCORES] f1={rlm_scores['f1']:.4f} bertscore={rlm_scores['bertscore']:.4f} len_ratio={rlm_scores['length_ratio']:.1f}")

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
                "iterations": n_iterations,
                "compression_ratio": comp_result.compression_ratio if comp_result else 0,
                "token_compression_ratio": comp_result.token_compression_ratio if comp_result else 0,
                "success": trace.execution_result.success if trace.execution_result else False,
            }

        except Exception as e:
            print(f"    [ERROR] RLM failed: {e}")
            result["rlm"] = {"answer": "", "scores": score_prediction("", sample.reference_answers),
                             "total_tokens": 0, "cost_usd": 0, "latency_ms": 0, "child_calls": 0,
                             "compression_ratio": 0, "success": False, "error": str(e)}

        # --- Direct LLM Baseline ---
        print_progress(i, len(samples), sample.id, "Direct LLM")
        try:
            baseline_result = await baseline.run(
                context=sample.context,
                question=sample.question,
                model=model,
            )

            direct_scores = score_prediction(baseline_result.answer, sample.reference_answers)

            # Debug: show Direct answer
            direct_words = len(baseline_result.answer.split()) if baseline_result.answer else 0
            print(f"    [DIRECT ANS] ({direct_words} words, truncated={baseline_result.truncated}): {baseline_result.answer[:200]}{'...' if len(baseline_result.answer) > 200 else ''}")
            print(f"    [DIRECT SCORES] f1={direct_scores['f1']:.4f} bertscore={direct_scores['bertscore']:.4f} len_ratio={direct_scores['length_ratio']:.1f}")

            result["direct"] = {
                "answer": baseline_result.answer,
                "scores": direct_scores,
                "total_tokens": baseline_result.input_tokens + baseline_result.output_tokens,
                "cost_usd": baseline_result.cost_usd,
                "latency_ms": baseline_result.latency_ms,
                "truncated": baseline_result.truncated,
                "context_tokens_used": baseline_result.context_tokens_used,
            }

        except Exception as e:
            print(f"    [ERROR] Direct failed: {e}")
            result["direct"] = {"answer": "", "scores": score_prediction("", sample.reference_answers),
                                "total_tokens": 0, "cost_usd": 0, "latency_ms": 0, "truncated": False,
                                "error": str(e)}

        per_sample_results.append(result)

    # Aggregate
    rlm_agg = aggregate_results(per_sample_results, "rlm")
    direct_agg = aggregate_results(per_sample_results, "direct")

    full_results = {
        "metadata": {
            "dataset": dataset_name,
            "model": model,
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
        **kwargs,
    ))


if __name__ == "__main__":
    main()
