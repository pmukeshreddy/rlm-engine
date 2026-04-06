"""Load industry-standard long-context benchmarks from HuggingFace."""
from dataclasses import dataclass
from typing import List, Optional
from datasets import load_dataset


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""
    id: str
    context: str
    question: str
    reference_answers: List[str]  # One or more gold answers
    dataset: str
    metadata: Optional[dict] = None


def load_narrativeqa(n_samples: int = 20, split: str = "test") -> List[BenchmarkSample]:
    """
    Load NarrativeQA — QA over full books and movie scripts.

    Paper: "The NarrativeQA Reading Comprehension Challenge" (Kocisky et al., 2018)
    Source: https://huggingface.co/datasets/deepmind/narrativeqa

    Each sample has a long document (~60K+ chars), a question,
    and 2 human-written reference answers.
    """
    ds = load_dataset("deepmind/narrativeqa", split=split, trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    samples = []
    for i, row in enumerate(ds):
        context = row["document"]["text"]
        if not context or len(context) < 1000:
            continue

        samples.append(BenchmarkSample(
            id=f"narrativeqa_{i}",
            context=context,
            question=row["question"]["text"],
            reference_answers=[
                row["answers"][0]["text"],
                row["answers"][1]["text"],
            ],
            dataset="narrativeqa",
            metadata={"doc_id": row["document"]["id"]},
        ))

    return samples


def load_quality(n_samples: int = 20, split: str = "validation") -> List[BenchmarkSample]:
    """
    Load QuALITY — multiple-choice QA on long articles (avg ~5K words).

    Paper: "QuALITY: Question Answering with Long Input Texts, Yes!" (Pang et al., 2022)
    Source: https://huggingface.co/datasets/emozilla/quality

    We convert multiple-choice to open-ended by using the correct
    answer option as the reference answer.
    """
    ds = load_dataset("emozilla/quality", split=split, trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    samples = []
    for i, row in enumerate(ds):
        context = row["article"]
        if not context or len(context) < 500:
            continue

        options = row["options"]
        gold_idx = row["gold_label"] - 1  # 1-indexed
        correct_answer = options[gold_idx] if 0 <= gold_idx < len(options) else options[0]

        samples.append(BenchmarkSample(
            id=f"quality_{i}",
            context=context,
            question=row["question"],
            reference_answers=[correct_answer],
            dataset="quality",
            metadata={
                "options": options,
                "gold_label": row["gold_label"],
                "hard": row.get("difficult", False),
            },
        ))

    return samples


def load_longbench(
    n_samples: int = 20,
    task: str = "narrativeqa",
) -> List[BenchmarkSample]:
    """
    Load LongBench — comprehensive long-context benchmark.

    Paper: "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"
           (Bai et al., 2023)
    Source: https://huggingface.co/datasets/THUDM/LongBench

    Tasks: narrativeqa, qasper, multifieldqa_en, hotpotqa, musique,
           gov_report, multi_news, trec, triviaqa, samsum, passage_count,
           passage_retrieval_en, lcc, repobench-p
    """
    ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    samples = []
    for i, row in enumerate(ds):
        context = row["context"]
        if not context or len(context) < 500:
            continue

        answers = row["answers"]
        if isinstance(answers, str):
            answers = [answers]

        samples.append(BenchmarkSample(
            id=f"longbench_{task}_{i}",
            context=context,
            question=row["input"],
            reference_answers=answers,
            dataset=f"longbench/{task}",
            metadata={"length": row.get("length"), "all_classes": row.get("all_classes")},
        ))

    return samples


def load_scrolls_qmsum(n_samples: int = 20, split: str = "validation") -> List[BenchmarkSample]:
    """
    Load SCROLLS/QMSum — query-based meeting summarization.

    Paper: "SCROLLS: Standardized CompaRison Over Long Language Sequences" (Shaham et al., 2022)
    Source: https://huggingface.co/datasets/tau/scrolls

    Long meeting transcripts with specific queries about the meeting.
    """
    ds = load_dataset("tau/scrolls", "qmsum", split=split, trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    samples = []
    for i, row in enumerate(ds):
        context = row["input"]
        if not context or len(context) < 500:
            continue

        # SCROLLS format: the query is embedded in the input, output is the reference
        samples.append(BenchmarkSample(
            id=f"scrolls_qmsum_{i}",
            context=context,
            question="Summarize the key points discussed in this meeting transcript.",
            reference_answers=[row["output"]] if row.get("output") else [""],
            dataset="scrolls/qmsum",
            metadata={"pid": row.get("pid")},
        ))

    return samples


DATASET_REGISTRY = {
    "narrativeqa": load_narrativeqa,
    "quality": load_quality,
    "longbench": load_longbench,
    "scrolls_qmsum": load_scrolls_qmsum,
}


def load_benchmark(name: str, n_samples: int = 20, **kwargs) -> List[BenchmarkSample]:
    """Load a benchmark dataset by name."""
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name](n_samples=n_samples, **kwargs)
