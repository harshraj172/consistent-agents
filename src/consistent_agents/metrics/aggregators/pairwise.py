from typing import List, Optional, Callable

__all__ = ["pairwise"]


def pairwise(
    outputs: List[str],
    agreement_fn: Callable,
    question: Optional[str] = None,
    **kwargs
) -> float:
    """Pairwise aggregation: average agreement across all pairs of outputs.
    
    Computes agreement between all pairs of outputs and returns the average.
    This is the standard pairwise consistency metric.
    
    Args:
        outputs: List of output texts
        agreement_fn: AgreementFunction to use for pairwise comparisons
        question: Optional question context (passed to agreement function)
        **kwargs: Additional context (passed to agreement function)
        
    Returns:
        Average agreement score across all pairs (0.0 to 1.0)
    """
    n = len(outputs)
    if n < 2:
        return 1.0  # Single output is trivially consistent
    
    total_agreement = 0.0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            score = agreement_fn(outputs[i], outputs[j], question=question, **kwargs)
            total_agreement += score
            total_pairs += 1
    
    return total_agreement / total_pairs if total_pairs > 0 else 0.0

