from pathlib import Path
from typing import Dict, Any, List
from consistent_agents.metrics.base import BaseMetric
from consistent_agents.data_models import BenchmarkItem

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class ContradictionMetric(BaseMetric):
    """Contradiction metric using LLM-as-judge.

    Uses a contradiction-judge prompt where the model answers "Yes" if A and B contradict,
    and "No" otherwise. Scoring is inverted as requested:
    - "Yes" (contradiction)  -> score 0.0
    - "No"  (no contradiction)-> score 1.0

    Final item score is the average across perturbations; total score is the average over items.
    """

    def __init__(self, judge_model: str = "gpt-4o-mini", **kwargs):
        super().__init__(**kwargs)
        self.judge_model = judge_model
        self.total_correct = 0
        self.total_comparisons = 0

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        self.client = OpenAI()

        # Template expected to exist
        self.template_path = (
            REPO_ROOT
            / "src"
            / "consistent_agents"
            / "metrics"
            / "prompt-templates"
            / "contradiction-judge.txt"
        )
        self.prompt_template = self.template_path.read_text()

    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> float:
        """Calculate non-contradiction score for a single benchmark item."""
        base_output = item["base_output"]

        correct_count = 0
        for p in perturbed_outputs:
            perturbed_output = p["output"]

            is_contradiction = self._judge_contradiction(
                sentence_a=base_output,
                sentence_b=perturbed_output,
                prompt_template=self.prompt_template,
            )

            if is_contradiction:
                correct_count += 1

        total_perturbations = len(perturbed_outputs)
        item_score = correct_count / total_perturbations if total_perturbations > 0 else 0.0

        self.total_correct += correct_count
        self.total_comparisons += total_perturbations

        return item_score

    def total_score(self) -> float:
        """Average non-contradiction score across all processed items."""
        return self.total_correct / self.total_comparisons if self.total_comparisons > 0 else 0.0

    def name(self) -> str:
        # Name reflects that higher is "non-contradiction" despite using a contradiction judge.
        return "contradiction"

    def _judge_contradiction(self, sentence_a: str, sentence_b: str, prompt_template: str) -> bool:
        """Use LLM to judge if A and B contradict. Returns True if 'Yes' (contradiction)."""
        try:
            prompt = prompt_template.format(sentence_a=sentence_a, sentence_b=sentence_b)
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=5,
            )
            judgment = (response.choices[0].message.content or "").strip().lower()
            return judgment.startswith("yes")
        except Exception as e:
            print(f"Error during contradiction judging: {e}")
            return False
