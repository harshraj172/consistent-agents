from typing import Protocol, Optional
from typing_extensions import runtime_checkable
__all__ = ["AgreementFunction"]


@runtime_checkable
class AgreementFunction(Protocol):
    """Protocol for agreement functions that compute similarity/agreement between two outputs.
    
    Agreement functions compute a score indicating how similar or consistent two outputs are.
    The score is typically in the range [0.0, 1.0], where 1.0 indicates perfect agreement.
    """
    
    def __call__(self, output1: str, output2: str, question: Optional[str] = None, **kwargs) -> float:
        """
        Compute agreement score between two outputs.
        
        Args:
            output1: First output text
            output2: Second output text
            question: Optional question context (for LLM-based judges)
            **kwargs: Additional context (e.g., prompt_template for LLM judges)
            
        Returns:
            Agreement score (typically 0.0 to 1.0, where 1.0 = perfect agreement)
        """
        ...

