from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from consistent_agents.benchmarks.base import BaseBenchmark
from consistent_agents.environments import LocalEnvironment

from .adapter import Spider2DBTAdapter
from .dataset_setup import SPIDER2_REPO_URL, ensure_spider2_dbt_root
import shutil


class HarborSpider2DBTBenchmark(BaseBenchmark):
    """
    Harbor-backed Spider2-DBT benchmark.

    Materializes Harbor task directories from a local Spider2-DBT dataset checkout
    and yields items whose `task_dir` points to the prepared Harbor task.

    Scoring interprets Harbor verifier reward > 0 as success across base +
    perturbation runs, consistent with other Harbor-backed benchmarks here.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path = ".cache",
        spider2_repo_url: str = SPIDER2_REPO_URL,
        force_reclone: bool = False,
        tasks_root: str | Path = ".harbor_tasks/spider2_dbt",
        template_dir: str | Path | None = None,
        sample_tasks: int | float | None = None,
        overwrite: bool = False,
        split: str = "all",
    ) -> None:
        super().__init__(split=split)
        self.cache_dir = Path(cache_dir)
        self.spider2_repo_url = str(spider2_repo_url)
        self.force_reclone = bool(force_reclone)

        self.spider2_dbt_root: Optional[Path] = None
        self.tasks_root = Path(tasks_root)
        self.template_dir = (
            Path(template_dir)
            if template_dir is not None
            else Path(__file__).parent / "template"
        )
        self.sample_tasks = sample_tasks
        self.overwrite = bool(overwrite)

        self._all_ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        self._prepared: Dict[int, Dict[str, Any]] = {}

        self.item_scores: Dict[str, float] = {}
        self.total_scores: Dict[str, int] = {"success": 0, "total": 0}

    def _sample_task_ids(self, all_ids: List[str]) -> List[str]:
        st = self.sample_tasks
        if st is None:
            return list(all_ids)
        n = len(all_ids)
        if n == 0:
            return []
        k = int(st) if isinstance(st, int) else int(round(n * (float(st) / 100.0 if float(st) > 1 else float(st))))
        k = max(0, min(k, n))
        if k == n:
            return list(all_ids)
        picked = set(random.sample(all_ids, k)) if k else set()
        return [tid for tid in all_ids if tid in picked]

    def load(self) -> None:
        """Ensure Spider2-DBT is available locally, then materialize Harbor tasks."""
        if not self.template_dir.is_dir():
            raise FileNotFoundError(f"Template dir not found: {self.template_dir}")

        self.tasks_root.mkdir(parents=True, exist_ok=True)

        spider2_dbt_root = ensure_spider2_dbt_root(
            cache_dir=self.cache_dir,
            spider2_repo_url=self.spider2_repo_url,
            force_reclone=self.force_reclone,
        )
        self.spider2_dbt_root = spider2_dbt_root

        adapter = Spider2DBTAdapter(
            template_dir=self.template_dir,
            output_dir=self.tasks_root,
            spider2_dbt_root=spider2_dbt_root,
        )

        all_ids = adapter.get_all_task_ids()
        self._all_ids = self._sample_task_ids(all_ids)
        self._id_to_index = {tid: idx for idx, tid in enumerate(self._all_ids)}

        for idx, task_id in enumerate(self._all_ids):
            task_dir = self.tasks_root / task_id
            if self.overwrite and task_dir.exists():
                shutil.rmtree(task_dir, ignore_errors=True)

            if not task_dir.exists():
                adapter.generate_task(task_id, local_task_id=task_id)

            instruction_path = task_dir / "instruction.md"
            prompt = instruction_path.read_text(encoding="utf-8").strip()

            state = {
                "instance_id": task_id,
                "task_dir": task_dir,
                "prompt": prompt,
                "label": None,
            }
            self._prepared[idx] = state

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not self._prepared:
            self.load()

        env = LocalEnvironment(name="harbor-local")
        for idx in range(len(self._all_ids)):
            state = self._prepared[idx]
            yield {
                "instance_id": state["instance_id"],
                "prompt": state["prompt"],
                "env": env,
                "label": state.get("label"),
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

        self.item_scores = {"success": float(success_count), "total": float(total)}
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
        accuracy = self.total_scores["success"] / self.total_scores["total"]
        return {
            "accuracy": accuracy,
            "consistency": None,
            "total": self.total_scores["total"],
        }

    def __len__(self) -> int:
        if not self._all_ids:
            self.load()
        return len(self._all_ids)

    def get_index(self, instance_id: str) -> Optional[int]:
        if not self._id_to_index:
            self.load()
        return self._id_to_index.get(instance_id)

