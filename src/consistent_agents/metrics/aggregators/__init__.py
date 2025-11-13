from typing import Protocol, List, Optional
from typing_extensions import runtime_checkable
from consistent_agents.metrics.agreement_functions import AgreementFunction
__all__ = ["Aggregator"]


@runtime_checkable
class Aggregator(Protocol):
    """Protocol for aggregators that combine agreement scores into a single metric.
    
    Aggregators take a list of outputs and an agreement function, then compute
    a single aggregated score representing the overall consistency/diversity.
    """
    
    def __call__(self, 
                 outputs: List[str], 
                 agreement_fn: AgreementFunction,
                 question: Optional[str] = None,
                 **kwargs) -> float:
        """
        Aggregate agreement scores across outputs.
        
        Args:
            outputs: List of output texts to aggregate
            agreement_fn: AgreementFunction to use for pairwise comparisons
            question: Optional question context (passed to agreement function)
            **kwargs: Additional context (e.g., threshold for entropy clustering)
            
        Returns:
            Aggregated score (typically 0.0 to 1.0, or entropy value)
        """
        ...

