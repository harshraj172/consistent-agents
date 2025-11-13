from typing import Optional
from consistent_agents.metrics.agreement_functions import AgreementFunction

__all__ = ["ROUGEAgreement"]


class ROUGEAgreement:
    """ROUGE-based agreement function.
    
    Uses ROUGE scores to compute overlap between two outputs.
    Returns the F1 score from ROUGE, which ranges from 0.0 to 1.0.
    """
    
    def __init__(self, rouge_type: str = "rougeL", use_stemmer: bool = True):
        """Initialize the ROUGE agreement function.
        
        Args:
            rouge_type: One of {"rouge1", "rouge2", "rougeL", "rougeLsum"}
            use_stemmer: Apply Porter stemming
        """
        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer
        
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(
                [self.rouge_type], use_stemmer=self.use_stemmer
            )
        except ImportError:
            raise ImportError(
                "rouge-score package not installed. Install with: pip install rouge-score"
            )
    
    def __call__(self, output1: str, output2: str, question: Optional[str] = None, **kwargs) -> float:
        """Compute ROUGE F1 between two outputs.
        
        Args:
            output1: First output text (used as reference)
            output2: Second output text (used as candidate)
            question: Ignored (not used by ROUGE)
            **kwargs: Ignored
            
        Returns:
            ROUGE F1 score (0.0 to 1.0)
        """
        try:
            scores = self.scorer.score(output1, output2)
            return scores[self.rouge_type].fmeasure
        except Exception as e:
            print(f"Error computing ROUGE agreement: {e}")
            return 0.0

