from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from consistent_agents.data_models import BenchmarkItem
from consistent_agents.benchmarks import BaseBenchmark

class BaseMetric(ABC):
    """Base class for all metrics."""

    def __init__(self, **kwargs):
        """Initialize the metric with optional configuration parameters."""
        self.kwargs = kwargs

    def set_benchmark(self, benchmark: BaseBenchmark):
        pass
    
    @abstractmethod
    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> float:
        """Calculate score for a single benchmark item."""
        raise NotImplementedError

    @abstractmethod
    def total_score(self) -> float:
        """Calculate total score across all processed items."""
        raise NotImplementedError
    
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError