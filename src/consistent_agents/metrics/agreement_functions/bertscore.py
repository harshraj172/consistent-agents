from typing import Optional
from consistent_agents.metrics.agreement_functions import AgreementFunction

__all__ = ["BERTScoreAgreement"]


class BERTScoreAgreement:
    """BERTScore-based agreement function.
    
    Uses BERT embeddings to compute semantic similarity between two outputs.
    Returns the F1 score from BERTScore, which ranges from 0.0 to 1.0.
    """
    
    def __init__(self, 
                 model_type: str = "bert-base-uncased", 
                 lang: str = "en", 
                 rescale_with_baseline: bool = True):
        """Initialize the BERTScore agreement function.
        
        Args:
            model_type: BERT model to use for computing embeddings
            lang: Language code for the text
            rescale_with_baseline: Whether to rescale scores with baseline
        """
        self.model_type = model_type
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        
        try:
            from bert_score import BERTScorer
            self.scorer = BERTScorer(
                model_type=self.model_type,
                lang=self.lang,
                rescale_with_baseline=self.rescale_with_baseline
            )
        except ImportError:
            raise ImportError(
                "bert-score package not installed. Install with: pip install bert-score"
            )
    
    def __call__(self, output1: str, output2: str, question: Optional[str] = None, **kwargs) -> float:
        """Compute BERTScore F1 between two outputs.
        
        Args:
            output1: First output text
            output2: Second output text
            question: Ignored (not used by BERTScore)
            **kwargs: Ignored
            
        Returns:
            BERTScore F1 score (0.0 to 1.0)
        """
        try:
            P, R, F1 = self.scorer.score([output1], [output2])
            return F1.item()
        except Exception as e:
            print(f"Error computing BERTScore agreement: {e}")
            return 0.0

