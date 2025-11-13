from typing import List, Optional
import numpy as np
from scipy.stats import entropy
from consistent_agents.metrics.aggregators import Aggregator
from consistent_agents.metrics.agreement_functions import AgreementFunction

__all__ = ["EntropyAggregator"]


class EntropyAggregator:
    """Entropy-based aggregation: cluster outputs and compute entropy of cluster distribution.
    
    Clusters outputs based on agreement scores, then computes the entropy
    of the cluster size distribution. Higher entropy indicates more diversity.
    """
    
    def __init__(self, threshold: float = 0.5):
        """Initialize the entropy aggregator.
        
        Args:
            threshold: Agreement score threshold for clustering (default: 0.5)
        """
        self.threshold = threshold
    
    def __call__(self, 
                 outputs: List[str], 
                 agreement_fn: AgreementFunction,
                 question: Optional[str] = None,
                 **kwargs) -> float:
        """Compute entropy of clustered outputs.
        
        Args:
            outputs: List of output texts
            agreement_fn: AgreementFunction to use for clustering
            question: Optional question context (passed to agreement function)
            **kwargs: Additional context (e.g., threshold to override default)
            
        Returns:
            Entropy value (higher = more diversity)
        """
        if not outputs:
            return 0.0
        
        threshold = kwargs.get("threshold", self.threshold)
        clusters = self._semantic_clustering(outputs, agreement_fn, question, threshold, **kwargs)
        
        if not clusters:
            return 0.0
        
        cluster_sizes = np.array([len(c) for c in clusters])
        pk = cluster_sizes / cluster_sizes.sum()
        H = entropy(pk, base=2)
        
        return H
    
    def _semantic_clustering(self, 
                            outputs: List[str], 
                            agreement_fn: AgreementFunction,
                            question: Optional[str],
                            threshold: float,
                            **kwargs) -> List[List[str]]:
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

