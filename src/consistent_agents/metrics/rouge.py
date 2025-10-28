from typing import Dict, Any, List
from consistent_agents.metrics.base import BaseMetric
from consistent_agents.data_models import BenchmarkItem


class ROUGEScoreMetric(BaseMetric):
    """ROUGE metric (default: ROUGE-L F1) for reference–candidate overlap."""

    def __init__(
        self,
        rouge_type: str = "rougeL",
        use_stemmer: bool = True,
        **kwargs,
    ):
        """
        Args:
            rouge_type: One of {"rouge1","rouge2","rougeL","rougeLsum"}
            use_stemmer: Apply Porter stemming
        """
        super().__init__(**kwargs)
        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer
        self.scores: List[float] = []

        try:
            from rouge_score import rouge_scorer  # noqa: F401
            self._scorer = rouge_scorer.RougeScorer(
                [self.rouge_type], use_stemmer=self.use_stemmer
            )
        except ImportError:
            raise ImportError(
                "rouge-score package not installed. Install with: pip install rouge-score"
            )

    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> float:
        """
        Returns:
            Single scalar: ROUGE F1 for the configured rouge_type, averaged over candidates.
        """
        reference = item["question"]
        candidates = [p["output"] for p in perturbed_outputs]

        try:
            vals = []
            for cand in candidates:
                res = self._scorer.score(reference, cand)[self.rouge_type]
                vals.append(res.fmeasure)
            avg = sum(vals) / len(vals)
            self.scores.append(avg)
            return avg
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            return 0.0

    def total_score(self) -> float:
        """Calculate total ROUGE score across all processed items.
        
        Returns:
            Average F1 score as a float
        """
        if not self.scores:
            return 0.0
        avg_f1 = sum(self.scores) / len(self.scores)
        return avg_f1

    def name(self) -> str:
        return "rouge"
