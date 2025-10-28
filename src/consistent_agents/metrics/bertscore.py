from typing import Dict, Any, List
from consistent_agents.metrics.base import BaseMetric
from consistent_agents.data_models import BenchmarkItem


class BERTScoreMetric(BaseMetric):
    """BERTScore metric that evaluates semantic similarity using BERT embeddings."""

    def __init__(self, 
                 model_type: str = "bert-base-uncased",
                 lang: str = "en",
                 rescale_with_baseline: bool = True,
                 **kwargs):
        """Initialize the BERTScore metric.
        
        Args:
            model_type: BERT model to use for computing embeddings
            lang: Language code for the text
            rescale_with_baseline: Whether to rescale scores with baseline
        """
        super().__init__(**kwargs)
        self.model_type = model_type
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        self.scores = []
        
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

    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> float:
        """Calculate BERTScore F1 for a single benchmark item.
        
        Args:
            item: The benchmark item containing reference text
            perturbed_outputs: List of perturbed outputs to evaluate
            
        Returns:
            F1 score as a float
        """
        reference = item['question']
        candidates = [perturbation['output'] for perturbation in perturbed_outputs]
        
        try:
            P, R, F1 = self.scorer.score(candidates, [reference] * len(candidates))
            
            avg_f1 = F1.mean().item()
            
            self.scores.append(avg_f1)
            
            return avg_f1
            
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return 0.0

    def total_score(self) -> float:
        """Calculate total BERTScore F1 across all processed items.
        
        Returns:
            Average F1 score as a float
        """
        if not self.scores:
            return 0.0
        
        avg_f1 = sum(self.scores) / len(self.scores)
        
        return avg_f1

    def name(self) -> str:
        """Return the name of this metric."""
        return "bertscore"