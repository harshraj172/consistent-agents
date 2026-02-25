from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import datasets

from consistent_agents.benchmarks.base import BaseBenchmark
from consistent_agents.environments import LocalEnvironment

from .adapter import SWEBenchToHarbor


class HarborSWEBenchBenchmark(BaseBenchmark):
    """
    Harbor-backed SWEBench benchmark.

    Generates Harbor task directories from the SWEBench dataset and yields items
    whose `task_dir` points to the prepared Harbor task. Scoring uses the Harbor
    verifier reward (1 = pass, 0 = fail) across base + perturbation runs.
    """

    def __init__(
        self,
        split: str = "test[:2]",
        dataset_name: str = "princeton-nlp/SWE-bench_Verified",
        tasks_root: str | Path = ".harbor_tasks/swebench",
        template_dir: str | Path | None = None,
        max_timeout_sec: float = 3000.0,
        overwrite: bool = False,
    ) -> None:
        super().__init__(split=split)
        self.dataset_name = dataset_name
        self.tasks_root = Path(tasks_root)
        self.template_dir = Path(template_dir) if template_dir is not None else None
        self.max_timeout = max_timeout_sec
        self.overwrite = overwrite

        self.dataset: Optional[datasets.Dataset] = None
        self._id_to_index: Dict[str, int] = {}
        self._prepared: Dict[int, Dict[str, Any]] = {}

        self.item_scores: Dict[str, float] = {}
        self.total_scores: Dict[str, int] = {"success": 0, "total": 0}
        self._consistency_scores: List[float] = []

    def load(self) -> None:
        """Load SWEBench split and materialize Harbor tasks."""
        self.dataset = datasets.load_dataset(self.dataset_name, split=self.split)
        self._id_to_index = {
            example["instance_id"]: idx for idx, example in enumerate(self.dataset)
        }

        records = [dict(ex) for ex in self.dataset]
        converter = SWEBenchToHarbor(
            harbor_tasks_root=self.tasks_root,
            max_timeout_sec=self.max_timeout,
            template_dir=self.template_dir,
            records=records,
            dataset_name=self.dataset_name,
            split=self.split,
        )

        for idx, example in enumerate(self.dataset):
            instance_id = example["instance_id"]
            task_dir = self.tasks_root / instance_id

            if not task_dir.exists() or self.overwrite:
                try:
                    converter.generate_task(instance_id, instance_id, overwrite=self.overwrite)
                except FileExistsError:
                    # If overwrite=False and task exists, reuse it.
                    pass

            instruction_path = task_dir / "instruction.md"
            if instruction_path.is_file():
                prompt = instruction_path.read_text(encoding="utf-8").strip()
            else:
                prompt = str(example.get("problem_statement", ""))

            state = {
                "instance_id": instance_id,
                "task_dir": task_dir,
                "prompt": prompt,
                "label": example.get("patch"),
                "base_commit": example.get("base_commit"),
                "repo": example.get("repo"),
                "version": example.get("version"),
                "test_patch": example.get("test_patch"),
                "problem_statement": example.get("problem_statement"),
            }
            self._prepared[idx] = state

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.dataset is None:
            self.load()
        assert self.dataset is not None

        env = LocalEnvironment(name="harbor-local")
        for idx in range(len(self.dataset)):
            state = self._prepared[idx]
            example = self.dataset[idx]
            yield {
                "instance_id": state["instance_id"],
                "prompt": state["prompt"],
                "env": env,
                "label": state.get("label"),
                "task_dir": state["task_dir"],
                "base_commit": example.get("base_commit"),
                "repo": example.get("repo"),
                "version": example.get("version"),
                "test_patch": example.get("test_patch"),
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

        # Compute consistency: 1.0 if all runs agree (all pass or all fail), 0.0 otherwise
        if predictions:
            results = [_is_success(o) for o in outputs]
            consistent = 1.0 if len(set(results)) == 1 else 0.0
            self._item_consistency = consistent
            self._consistency_scores.append(consistent)
        else:
            self._item_consistency = None

        return self.item_score()

    def item_score(self) -> Dict[str, float]:
        accuracy = self.item_scores["success"] / self.item_scores["total"]
        return {
            "accuracy": accuracy,
            "consistency": self._item_consistency,
        }

    def total_score(self) -> Dict[str, float]:
        accuracy = self.total_scores["success"] / self.total_scores["total"]
        consistency = None
        if self._consistency_scores:
            consistency = sum(self._consistency_scores) / len(self._consistency_scores)
        return {
            "accuracy": accuracy,
            "consistency": consistency,
            "total": self.total_scores["total"],
        }

    def __len__(self) -> int:
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        return len(self.dataset)

    def get_index(self, instance_id: str) -> Optional[int]:
        if self.dataset is None:
            self.load()
        return self._id_to_index.get(instance_id)
