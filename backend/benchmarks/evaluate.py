"""Evaluation metrics for comparing predicted answers against references."""
import re
import string
from collections import Counter
from typing import List

from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation/articles/whitespace."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score between prediction and reference.

    Standard metric from SQuAD (Rajpurkar et al., 2016).
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * (precision * recall) / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """Exact match after normalization."""
    return float(normalize_text(prediction) == normalize_text(reference))


def compute_rouge(prediction: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Standard metric for summarization (Lin, 2004).
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_bertscore(prediction: str, reference: str) -> float:
    """
    Compute BERTScore F1 between prediction and reference.

    Uses embedding-based similarity to catch semantic matches
    that token-level F1 misses (e.g. "he died" vs "he passed away").
    """
    if not prediction or not reference:
        return 0.0

    P, R, F1 = bert_score_fn(
        [prediction], [reference],
        lang="en",
        verbose=False,
    )
    return F1.item()


def answer_length_ratio(prediction: str, reference_answers: List[str]) -> float:
    """
    Compute the length ratio of prediction vs average reference length.

    Returns pred_len / ref_len. Ideal is ~1.0.
    Values >> 1.0 mean the model is too verbose.
    Values << 1.0 mean the model is too terse.
    """
    ref_lengths = [len(r.split()) for r in reference_answers if r]
    if not ref_lengths:
        return 0.0
    avg_ref_len = sum(ref_lengths) / len(ref_lengths)
    pred_len = len(prediction.split()) if prediction else 0
    if avg_ref_len == 0:
        return 0.0
    return pred_len / avg_ref_len


def score_prediction(prediction: str, reference_answers: List[str]) -> dict:
    """
    Score a prediction against one or more reference answers.

    Takes the max score across all references (standard practice for
    datasets with multiple valid answers like NarrativeQA).
    """
    best_f1 = 0.0
    best_em = 0.0
    best_rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    best_bertscore = 0.0

    for ref in reference_answers:
        if not ref:
            continue

        f1 = token_f1(prediction, ref)
        em = exact_match(prediction, ref)
        rouge = compute_rouge(prediction, ref)
        bs = compute_bertscore(prediction, ref)

        if f1 > best_f1:
            best_f1 = f1
        if em > best_em:
            best_em = em
        if bs > best_bertscore:
            best_bertscore = bs
        for key in best_rouge:
            if rouge[key] > best_rouge[key]:
                best_rouge[key] = rouge[key]

    length_ratio = answer_length_ratio(prediction, reference_answers)

    return {
        "f1": round(best_f1, 4),
        "exact_match": round(best_em, 4),
        **{k: round(v, 4) for k, v in best_rouge.items()},
        "bertscore": round(best_bertscore, 4),
        "length_ratio": round(length_ratio, 2),
    }
