"""
BFCL Benchmark for consistent-agents.

Berkeley Function Call Leaderboard (BFCL) v4 evaluation benchmark.
Evaluates function calling capability across Python, Java, JavaScript,
multiple functions, parallel execution, and irrelevance detection.

Total: 3,641 tasks across 13 categories (excludes multi-turn and agentic).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from consistent_agents.benchmarks.base import BaseBenchmark
from consistent_agents.environments import DockerEnvironment

from .adapter import BfclAdapter


class BFCLBenchmark(BaseBenchmark):
    """BFCL Benchmark for evaluating function calling capabilities.

    This benchmark loads BFCL v4 tasks and evaluates agent outputs
    against ground truth function calls.
    """

    def __init__(
        self,
        split: str = "test",
        benchmark_root: Optional[str] = None,
        tasks_root: str = ".bfcl_tasks",
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> None:
        """Initialize the BFCL benchmark.

        Args:
            split: Dataset split (currently only "test" is supported)
            benchmark_root: Path to the BFCL benchmark repository
                           (berkeley-function-call-leaderboard)
            tasks_root: Directory where generated tasks will be stored
            categories: List of categories to include (default: all)
            limit: Maximum number of tasks to load (for testing)
        """
        super().__init__(split=split)
        self.benchmark_root = Path(benchmark_root) if benchmark_root else None
        self.tasks_root = Path(tasks_root)
        self.categories = categories
        self.limit = limit

        self.adapter: Optional[BfclAdapter] = None
        self.task_ids: List[str] = []
        self._prepared: Dict[int, Dict[str, Any]] = {}

        self.item_scores: Dict[str, float] = {}
        self.total_scores: Dict[str, int] = {
            "correct_count": 0,
            "total": 0,
        }

    def load(self) -> None:
        """Load the BFCL benchmark data and prepare tasks."""
        if self.benchmark_root is None:
            raise ValueError(
                "benchmark_root must be provided. "
                "Clone the BFCL repo: git clone https://github.com/ShishirPatil/gorilla.git"
            )

        self.tasks_root.mkdir(parents=True, exist_ok=True)

        self.adapter = BfclAdapter(
            task_dir=self.tasks_root,
            benchmark_root=self.benchmark_root,
            categories=self.categories,
        )

        self.task_ids = self.adapter.list_available_tasks()

        if self.limit:
            self.task_ids = self.task_ids[: self.limit]

        # Pre-generate task directories
        for source_id in self.task_ids:
            local_task_id = BfclAdapter.make_local_task_id(source_id)
            task_dir = self.tasks_root / local_task_id
            if not task_dir.exists():
                self.adapter.generate_task(source_id, local_task_id)

    def _prepare_instance(self, idx: int, source_id: str) -> Dict[str, Any]:
        """Prepare a single BFCL task instance."""
        local_task_id = BfclAdapter.make_local_task_id(source_id)
        task_dir = self.tasks_root / local_task_id

        # Get the instruction/prompt
        prompt = self.adapter.get_prompt(source_id)

        # Get ground truth for scoring
        ground_truth = self.adapter.get_ground_truth(source_id)

        # Create Docker environment
        dockerfile_path = task_dir / "environment" / "Dockerfile"
        env = DockerEnvironment()
        env.start(str(dockerfile_path), f"bfcl-{local_task_id}")

        state = {
            "source_id": source_id,
            "local_task_id": local_task_id,
            "task_dir": task_dir,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "env": env,
        }
        self._prepared[idx] = state
        return state

    def iter(self) -> Iterator[Dict[str, Any]]:
        """Yield benchmark items."""
        if self.adapter is None:
            self.load()

        for idx, source_id in enumerate(self.task_ids):
            state = self._prepared.get(idx)
            if state is None:
                state = self._prepare_instance(idx, source_id)

            yield {
                "instance_id": source_id,
                "prompt": state["prompt"],
                "env": state["env"],
                "label": json.dumps(state["ground_truth"]),
                "task_dir": state["task_dir"],
            }

    def _parse_agent_output(self, output: str) -> List[dict]:
        """Parse agent output as JSON array of function calls."""
        try:
            # Try to parse directly
            result = json.loads(output)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError:
            # Try to extract JSON from the output
            import re

            json_match = re.search(r'\[.*\]', output, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    pass
            return []

    def _compare_function_calls(
        self, predicted: List[dict], ground_truth: List[dict]
    ) -> bool:
        """Compare predicted function calls against ground truth."""
        if len(predicted) == 0 and len(ground_truth) == 0:
            return True

        if len(predicted) != len(ground_truth):
            return False

        for i, pred_call in enumerate(predicted):
            if not isinstance(pred_call, dict) or len(pred_call) != 1:
                return False

            pred_func_name = list(pred_call.keys())[0]
            pred_params = pred_call[pred_func_name]

            if i >= len(ground_truth):
                return False

            gt_call = ground_truth[i]
            if not isinstance(gt_call, dict) or len(gt_call) != 1:
                return False

            gt_func_name = list(gt_call.keys())[0]
            gt_params = gt_call[gt_func_name]

            # Normalize function names
            pred_norm = pred_func_name.replace(".", "_")
            gt_norm = gt_func_name.replace(".", "_")

            if pred_norm != gt_norm:
                return False

            if not self._compare_parameters(pred_params, gt_params):
                return False

        return True

    def _compare_parameters(
        self, pred_params: dict, gt_params: dict
    ) -> bool:
        """Compare predicted parameters against ground truth."""
        if not isinstance(pred_params, dict) or not isinstance(gt_params, dict):
            return False

        for param_name, acceptable_values in gt_params.items():
            if param_name not in pred_params:
                if not isinstance(acceptable_values, list):
                    acceptable_values = [acceptable_values]
                if "" not in acceptable_values and None not in acceptable_values:
                    return False
                continue

            pred_value = pred_params[param_name]

            if not isinstance(acceptable_values, list):
                acceptable_values = [acceptable_values]

            if len(acceptable_values) == 0:
                if pred_value != []:
                    return False
                continue

            matched = False
            for acceptable_value in acceptable_values:
                if self._values_equal(pred_value, acceptable_value):
                    matched = True
                    break

            if not matched:
                return False

        return True

    def _values_equal(self, v1: Any, v2: Any) -> bool:
        """Check if two values are equal with type flexibility."""
        if v2 == "" or v2 is None:
            return True

        if v1 == v2:
            return True

        try:
            if float(v1) == float(v2):
                return True
        except (ValueError, TypeError):
            pass

        if str(v1).lower() == str(v2).lower():
            return True

        if isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                return False
            return all(self._values_equal(a, b) for a, b in zip(v1, v2))

        return False

    def score(
        self,
        idx: int,
        base_output: str,
        predictions: List[Any],
    ) -> Dict[str, float]:
        """Score predictions against ground truth."""
        state = self._prepared[idx]
        ground_truth = state["ground_truth"]

        # Parse and evaluate base output
        base_parsed = self._parse_agent_output(base_output)
        base_correct = self._compare_function_calls(base_parsed, ground_truth)

        outcomes: List[bool] = [base_correct]

        # Evaluate perturbation outputs
        for prediction in predictions:
            pred_parsed = self._parse_agent_output(str(prediction))
            pred_correct = self._compare_function_calls(pred_parsed, ground_truth)
            outcomes.append(pred_correct)

        correct_count = sum(1 for passed in outcomes if passed)
        total = len(outcomes) if outcomes else 1

        self.item_scores = {
            "correct_count": correct_count,
            "total": total,
        }
        self.total_scores["correct_count"] += correct_count
        self.total_scores["total"] += total

        return self.item_score()

    def item_score(self) -> Dict[str, float]:
        """Get the scores for the current item."""
        accuracy = self.item_scores["correct_count"] / self.item_scores["total"]
        return {
            "accuracy": accuracy,
            "consistency": accuracy,
        }

    def total_score(self) -> Dict[str, float]:
        """Get the total scores for the benchmark."""
        if self.total_scores["total"] == 0:
            return {"accuracy": 0.0, "consistency": 0.0, "total": 0}

        accuracy = self.total_scores["correct_count"] / self.total_scores["total"]
        return {
            "accuracy": accuracy,
            "consistency": accuracy,
            "total": self.total_scores["total"],
        }

    def __len__(self) -> int:
        """Return the number of tasks in the benchmark."""
        if self.adapter is None:
            self.load()
        return len(self.task_ids)

    def get_index(self, source_id: str) -> Optional[int]:
        """Return the index for a given source_id."""
        if self.adapter is None:
            self.load()
        try:
            return self.task_ids.index(source_id)
        except ValueError:
            return None
