from typing import Optional

__all__ = ["bertscore"]


def bertscore(
    output1: str,
    output2: str,
    question: Optional[str] = None,
    model_type: str = "bert-base-uncased",
    lang: str = "en",
    rescale_with_baseline: bool = True,
    **kwargs
) -> float:
    """BERTScore-based agreement function.
    
    Uses BERT embeddings to compute semantic similarity between two outputs.
    Returns the F1 score from BERTScore, which ranges from 0.0 to 1.0.
    
    Args:
        output1: First output text
        output2: Second output text
        question: Ignored (not used by BERTScore)
        model_type: BERT model to use for computing embeddings
        lang: Language code for the text
        rescale_with_baseline: Whether to rescale scores with baseline
        **kwargs: Ignored
        
    Returns:
        BERTScore F1 score (0.0 to 1.0)
    """
    try:
        from bert_score import BERTScorer
    except ImportError:
        raise ImportError(
            "bert-score package not installed. Install with: pip install bert-score"
        )
    
    try:
        scorer = BERTScorer(
            model_type=model_type,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline
        )
        P, R, F1 = scorer.score([output1], [output2])
        return F1.item()
    except Exception as e:
        print(f"Error computing BERTScore agreement: {e}")
        return 0.0

