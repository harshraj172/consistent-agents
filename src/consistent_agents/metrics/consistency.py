from typing import Dict, Any, List, Optional
from consistent_agents.metrics.base import BaseMetric
from consistent_agents.metrics.agreement_functions import AgreementFunction
from consistent_agents.metrics.agreement_functions.llm_judge import LLMJudgeAgreement
from consistent_agents.metrics.aggregators import Aggregator
from consistent_agents.metrics.aggregators.pairwise import PairwiseAggregator
from consistent_agents.data_models import BenchmarkItem
from consistent_agents.utils import resolve_object, maybe_instantiate


class ConsistencyMetric(BaseMetric):
    """Consistency metric using agreement function + aggregator pattern.
    
    This metric evaluates how consistent model outputs are across different perturbations.
    It uses an agreement function to compute similarity between outputs and an aggregator
    to combine these scores into a single metric.
    
    By default, uses LLM judge for agreement and pairwise aggregation.
    """

    def __init__(self, 
                 agreement_function_path: Optional[str] = None,
                 agreement_function_params: Optional[Dict[str, Any]] = None,
                 aggregator_path: Optional[str] = None,
                 aggregator_params: Optional[Dict[str, Any]] = None,
                 judge_model: str = "gpt-4o-mini",
                 **kwargs):
        """Initialize the consistency metric.
        
        Args:
            agreement_function_path: Path to agreement function class.
            agreement_function_params: Parameters for agreement function instantiation.
            aggregator_path: Path to aggregator class.
            aggregator_params: Parameters for aggregator instantiation.
            judge_model: Model to use for LLM judge (used as default if paths not provided).
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(**kwargs)
        
        if agreement_function_path:
            agreement_function_cls = resolve_object(agreement_function_path)
            self.agreement_function = maybe_instantiate(agreement_function_cls, agreement_function_params or {})
        else:
            self.agreement_function = LLMJudgeAgreement(judge_model=judge_model)
        
        if aggregator_path:
            aggregator_cls = resolve_object(aggregator_path)
            self.aggregator = maybe_instantiate(aggregator_cls, aggregator_params or {})
        else:
            self.aggregator = PairwiseAggregator()
        
        self.scores: List[float] = []

    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> float:
        """Calculate consistency score for a single benchmark item."""
        question = item.get("question", "")
        outputs = [perturbation['output'] for perturbation in perturbed_outputs]
        
        score = self.aggregator(outputs, self.agreement_function, question=question)
        self.scores.append(score)
        
        return score

    def total_score(self) -> float:
        """Calculate total score across all processed items."""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def name(self) -> str:
        """Return metric name, including agreement function and aggregator names."""
        agreement_name = self.agreement_function.__class__.__name__.replace("Agreement", "").lower()
        aggregator_name = self.aggregator.__class__.__name__.replace("Aggregator", "").lower()

        return aggregator_name, agreement_name
