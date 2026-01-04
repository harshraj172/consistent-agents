from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from consistent_agents.benchmarks.base import BaseBenchmark
from consistent_agents.environments import LocalEnvironment


class HarborPromptBenchmark(BaseBenchmark):
    """
    Minimal benchmark that feeds a fixed prompt through Harbor.

    It is intended for synthetic or template-based Harbor tasks where the verifier
    emits a scalar reward (e.g., 1.0 for pass, 0.0 for fail). Accuracy is computed
    as the mean success rate across the base output plus any perturbation runs.
    """

    def __init__(self, template_path: str, prompt: Optional[str] = None, split: str = "single"):
        super().__init__(split=split)
        self.template_path = Path(template_path)
        self.prompt = prompt
        self.item_scores: Dict[str, Any] = {}
        self.total_scores: Dict[str, int] = {"success": 0, "total": 0}

    def load(self) -> None:
        """Validate template path and load prompt text if not provided."""
        if not self.template_path.is_dir():
            raise FileNotFoundError(f"Harbor template path not found: {self.template_path}")
        if self.prompt is None:
            instruction_path = self.template_path / "instruction.md"
            if not instruction_path.is_file():
                raise FileNotFoundError(f"Instruction file missing at {instruction_path}")
            self.prompt = instruction_path.read_text(encoding="utf-8").strip()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.prompt is None:
            self.load()
        assert self.prompt is not None

        # LocalEnvironment is unused by eval_harbor but satisfies the BenchmarkItem contract.
        env = LocalEnvironment(name="harbor-local")
        yield {
            "prompt": self.prompt,
            "env": env,
        }

    def score(self, idx: int, base_output: str, predictions: List[str]) -> Dict[str, float]:
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
        accuracy = self.total_scores["success"] / self.total_scores["total"]
        return {
            "accuracy": accuracy,
            "consistency": None,
            "total": self.total_scores["total"],
        }
