from typing import List, Optional
import numpy as np
from scipy.stats import entropy
from typing import Callable

__all__ = ["entropy", "semantic_clustering"]


def entropy(
    outputs: List[str], 
    agreement_fn: Callable,
    question: Optional[str] = None,
    threshold: float = 0.5,
    **kwargs
) -> float:
    """Entropy-based aggregation: cluster outputs and compute entropy of cluster distribution.
    
    Clusters outputs based on agreement scores, then computes the entropy
    of the cluster size distribution. Higher entropy indicates more diversity.
    
    Args:
        outputs: List of output texts
        agreement_fn: AgreementFunction to use for clustering
        question: Optional question context (passed to agreement function)
        threshold: Agreement score threshold for clustering (default: 0.5)
        **kwargs: Additional context (passed to agreement function)
        
    Returns:
        Entropy value (higher = more diversity)
    """
    if not outputs:
        return 0.0
    
    clusters = _semantic_clustering(outputs, agreement_fn, question, threshold, **kwargs)
    
    if not clusters:
        return 0.0
    
    cluster_sizes = np.array([len(c) for c in clusters])
    pk = cluster_sizes / cluster_sizes.sum()
    H = entropy(pk, base=2)
    
    return H


def _semantic_clustering(
    outputs: List[str], 
    agreement_fn: Callable,
    question: Optional[str],
    threshold: float,
    **kwargs
) -> List[List[str]]:
    """
    Organize similar outputs into clusters based on agreement scores.

    Parameters:
        outputs: A list of outputs to be clustered
        agreement_fn: Agreement function to use for comparisons
        question: The question context for consistency evaluation
        threshold: Agreement score threshold for clustering
        **kwargs: Additional context passed to agreement function

    Returns:
        A list of clusters, where each cluster is a list of consistent outputs.
    """
    if not outputs:
        return []
    
    C = [[outputs[0]]]
    for i in range(1, len(outputs)):
        stored = False
        for j in range(len(C)):
            s_c = C[j][0]
            
            score = agreement_fn(s_c, outputs[i], question=question, **kwargs)
            if score >= threshold:
                stored = True
                C[j].append(outputs[i])
                break
        
        if not stored:
            C.append([outputs[i]])
    
    return C

