"""
Harbor-backed BFCL Benchmark.

Berkeley Function Call Leaderboard (BFCL) v4 evaluation benchmark via Harbor.
Evaluates function calling capability across Python, Java, JavaScript,
multiple functions, parallel execution, and irrelevance detection.

Total: 3,641 tasks across 13 categories (excludes multi-turn and agentic).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from consistent_agents.benchmarks.base import BaseBenchmark
from consistent_agents.benchmarks.bfcl.adapter import BfclAdapter
from consistent_agents.environments import LocalEnvironment


class HarborBFCLBenchmark(BaseBenchmark):
    """
    Harbor-backed BFCL benchmark.

    Generates Harbor task directories from the BFCL dataset and yields items
    whose `task_dir` points to the prepared Harbor task. Scoring uses the Harbor
    verifier reward (1 = pass, 0 = fail) across base + perturbation runs.
    """

    def __init__(
        self,
        split: str = "test",
        benchmark_root: Optional[str] = None,
        tasks_root: str | Path = ".harbor_tasks/bfcl",
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize the Harbor BFCL benchmark.

        Args:
            split: Dataset split (currently only "test" is supported)
            benchmark_root: Path to the BFCL benchmark repository
                           (berkeley-function-call-leaderboard)
            tasks_root: Directory where generated Harbor tasks will be stored
            categories: List of categories to include (default: all)
            limit: Maximum number of tasks to load (for testing)
            overwrite: If True, regenerate task directories even if they exist
        """
        super().__init__(split=split)
        self.benchmark_root = Path(benchmark_root) if benchmark_root else None
        self.tasks_root = Path(tasks_root)
        self.categories = categories
        self.limit = limit
        self.overwrite = overwrite

        self.adapter: Optional[BfclAdapter] = None
        self.task_ids: List[str] = []
        self._prepared: Dict[int, Dict[str, Any]] = {}

        self.item_scores: Dict[str, float] = {}
        self.total_scores: Dict[str, int] = {"success": 0, "total": 0}

    def load(self) -> None:
        """Load the BFCL benchmark data and prepare Harbor tasks."""
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

        # Pre-generate Harbor task directories
        for idx, source_id in enumerate(self.task_ids):
            local_task_id = BfclAdapter.make_local_task_id(source_id)
            task_dir = self.tasks_root / local_task_id

            if not task_dir.exists() or self.overwrite:
                self.adapter.generate_task(source_id, local_task_id)

            # Read instruction from generated task
            instruction_path = task_dir / "instruction.md"
            if instruction_path.is_file():
                prompt = instruction_path.read_text(encoding="utf-8").strip()
            else:
                prompt = self.adapter.get_prompt(source_id)

            state = {
                "source_id": source_id,
                "local_task_id": local_task_id,
                "task_dir": task_dir,
                "prompt": prompt,
                "ground_truth": self.adapter.get_ground_truth(source_id),
            }
            self._prepared[idx] = state

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.adapter is None:
            self.load()

        env = LocalEnvironment(name="harbor-local")
        for idx in range(len(self.task_ids)):
            state = self._prepared[idx]
            yield {
                "instance_id": state["source_id"],
                "prompt": state["prompt"],
                "env": env,
                "label": state.get("ground_truth"),
                "task_dir": state["task_dir"],
            }

    def score(self, idx: int, base_output: str, predictions: List[Any]) -> Dict[str, float]:
        """Interpret Harbor verifier reward > 0 as success."""
        outputs = [base_output, *predictions]

        def _is_success(text: str) -> bool:
            try:
                return float(text) > 0.0
            except (TypeError, ValueError):
                return False

        success_count = sum(1 for out in outputs if _is_success(out))
        total = len(outputs) or 1

        self.item_scores = {"success": success_count, "total": total}
        self.total_scores["success"] += success_count
        self.total_scores["total"] += total
        return self.item_score()

    def item_score(self) -> Dict[str, float]:
        accuracy = self.item_scores["success"] / self.item_scores["total"]
        return {
            "accuracy": accuracy,
            "consistency": None,
        }

    def total_score(self) -> Dict[str, float]:
        if self.total_scores["total"] == 0:
            return {"accuracy": 0.0, "consistency": None, "total": 0}

        accuracy = self.total_scores["success"] / self.total_scores["total"]
        return {
            "accuracy": accuracy,
            "consistency": None,
            "total": self.total_scores["total"],
        }

    def __len__(self) -> int:
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
