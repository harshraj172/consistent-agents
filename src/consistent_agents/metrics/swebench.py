from pathlib import Path
from typing import Dict, Any, List, Optional
from consistent_agents.metrics.base import BaseMetric
from consistent_agents.data_models import BenchmarkItem
from consistent_agents.benchmarks.swebench.swebench_benchmark import SWEBenchBenchmark


class SWEBenchMetric(BaseMetric):
    """SWEBench metric that evaluates predictions by applying patches and running tests."""

    def __init__(self, 
                 benchmark: Optional[SWEBenchBenchmark] = None,
                 **kwargs):
        """Initialize the SWEBench metric.
        
        Args:
            benchmark: The SWEBenchBenchmark instance to use for scoring.
                       If None, must be set later via set_benchmark().
        """
        super().__init__(**kwargs)
        self.benchmark = benchmark
        self.total_consistent = 0
        self.total_correct = 0
        self.total_predictions = 0
        self.total_base_passed = 0
        self.total_items = 0

    def set_benchmark(self, benchmark: SWEBenchBenchmark) -> None:
        """Set the benchmark instance if not provided in constructor."""
        self.benchmark = benchmark

    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate SWEBench score for a single benchmark item."""
        if self.benchmark is None:
            raise ValueError("SWEBenchMetric requires a benchmark instance. Set it via constructor or set_benchmark().")
        
        idx = self.benchmark.get_index(item["id"])
        if idx is None:
            raise ValueError(f"Could not find index for instance_id: {item.id}")
        
        state = self.benchmark._prepared.get(idx)
        if state is None:
            raise ValueError(f"Benchmark state not prepared for index: {idx}")
        
        env = state["env"]
        
        # Upload tests directory
        env.upload(str(state["tests_dir"]), "/")
        
        # Test base output
        base_passed = self.benchmark._apply_patch_and_test(env, str(item.get("base_output", "")))
        
        # Test each perturbed output
        outcomes: List[bool] = []
        for perturbation in perturbed_outputs:
            pred_passed = self.benchmark._apply_patch_and_test(env, str(perturbation["output"]))
            outcomes.append(pred_passed)
        
        consistent_count = correct_count = sum(1 for passed in outcomes if passed)
        total = len(outcomes) if outcomes else 1
        
        # Update totals
        self.total_consistent += consistent_count
        self.total_correct += correct_count
        self.total_predictions += total
        self.total_base_passed += 1 if base_passed else 0
        self.total_items += 1
        
        return {
            "consistent_count": float(consistent_count),
            "correct_count": float(correct_count),
            "total": float(total),
            "base_passed": float(base_passed),
        }

    def total_score(self) -> Dict[str, float]:
        """Calculate total score across all processed items."""
        if self.total_items == 0:
            return {
                "consistent_count": 0.0,
                "correct_count": 0.0,
                "total": 0.0,
                "base_passed": 0.0,
            }
        
        return {
            "consistent_count": float(self.total_consistent),
            "correct_count": float(self.total_correct),
            "total": float(self.total_predictions),
            "base_passed": float(self.total_base_passed) / float(self.total_items),
        }

    def name(self) -> str:
        return "swebench"
