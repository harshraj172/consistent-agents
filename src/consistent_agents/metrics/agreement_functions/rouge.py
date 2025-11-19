from typing import Optional

__all__ = ["rouge"]


def rouge(
    output1: str,
    output2: str,
    question: Optional[str] = None,
    rouge_type: str = "rougeL",
    use_stemmer: bool = True,
    **kwargs
) -> float:
    """ROUGE-based agreement function.
    
    Uses ROUGE scores to compute overlap between two outputs.
    Returns the F1 score from ROUGE, which ranges from 0.0 to 1.0.
    
    Args:
        output1: First output text (used as reference)
        output2: Second output text (used as candidate)
        question: Ignored (not used by ROUGE)
        rouge_type: One of {"rouge1", "rouge2", "rougeL", "rougeLsum"}
        use_stemmer: Apply Porter stemming
        **kwargs: Ignored
        
    Returns:
        ROUGE F1 score (0.0 to 1.0)
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score package not installed. Install with: pip install rouge-score"
        )
    
    try:
        scorer = rouge_scorer.RougeScorer(
            [rouge_type], use_stemmer=use_stemmer
        )
        scores = scorer.score(output1, output2)
        return scores[rouge_type].fmeasure
    except Exception as e:
        print(f"Error computing ROUGE agreement: {e}")
        return 0.0

