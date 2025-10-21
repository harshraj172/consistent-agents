from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from consistent_agents.data_models import BenchmarkItem


class BaseMetric(ABC):
    """Base class for all metrics."""

    def __init__(self, **kwargs):
        """Initialize the metric with optional configuration parameters."""
        self.kwargs = kwargs

    @abstractmethod
    def item_score(self, item: BenchmarkItem) -> Dict[str, float]:
        """Calculate score for a single benchmark item."""
        raise NotImplementedError

    @abstractmethod
    def total_score(self) -> Dict[str, float]:
        """Calculate total score across all processed items."""
        raise NotImplementedError
    
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError