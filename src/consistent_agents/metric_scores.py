from typing import List, Dict, Any
from consistent_agents.metrics.base import BaseMetric
from consistent_agents.metrics.consistency import ConsistencyMetric


class MetricScores:
    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics
        self.scores: Dict[str, Any] = {}
        self.setup_scores()

    def _set_metric_entry(
        self,
        target: Dict[str, Any],
        metric: BaseMetric,
        value: Any,
    ) -> None:
        """Place a metric value into the given target dict in the right shape."""
        if isinstance(metric, ConsistencyMetric):
            aggregator, agreement = metric.name()
            (
                target
                .setdefault("consistency", {})
                .setdefault(aggregator, {})
            )[agreement] = value
        else:
            target[metric.name()] = value

    def setup_scores(self) -> None:
        for metric in self.metrics:
            self._set_metric_entry(self.scores, metric, None)

    def add_score(self, metric: BaseMetric, score: float) -> None:
        self._set_metric_entry(self.scores, metric, score)

    def get_scores(self) -> Dict[str, Any]:
        return self.scores

    def get_total_scores(self) -> Dict[str, Any]:
        """Get total/aggregated scores from all metrics."""
        total_scores: Dict[str, Any] = {}
        for metric in self.metrics:
            self._set_metric_entry(total_scores, metric, metric.total_score())
        return total_scores
