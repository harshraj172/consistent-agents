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

    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate BERTScore for a single benchmark item.
        
        Args:
            item: The benchmark item containing reference text
            perturbed_outputs: List of perturbed outputs to evaluate
            
        Returns:
            Dictionary containing BERTScore metrics (precision, recall, f1)
        """
        if item.get('question') is None:
            # If no reference question, return zero scores
            print("empty question")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        reference = item['question']
        candidates = [perturbation.get('output', '') for perturbation in perturbed_outputs]
        
        # Filter out empty candidates
        candidates = [cand for cand in candidates if cand.strip()]
        if not candidates:
            print("empty candidates")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            # Compute BERTScore for all candidates against the reference
            P, R, F1 = self.scorer.score(candidates, [reference] * len(candidates))
            
            # Average the scores across all candidates
            avg_precision = P.mean().item()
            avg_recall = R.mean().item()
            avg_f1 = F1.mean().item()
            
            # Store individual scores for total calculation
            self.scores.append({
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            })
            
            return {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            }
            
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def total_score(self) -> Dict[str, float]:
        """Calculate total BERTScore across all processed items.
        
        Returns:
            Dictionary containing average BERTScore metrics
        """
        if not self.scores:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        avg_precision = sum(score["precision"] for score in self.scores) / len(self.scores)
        avg_recall = sum(score["recall"] for score in self.scores) / len(self.scores)
        avg_f1 = sum(score["f1"] for score in self.scores) / len(self.scores)
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }

    def name(self) -> str:
        """Return the name of this metric."""
        return "bertscore"