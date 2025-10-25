from pathlib import Path
from typing import Dict, Any, List
from consistent_agents.metrics.base import BaseMetric
from consistent_agents.data_models import BenchmarkItem

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class EntailmentMetric(BaseMetric):
    """Bidirectional entailment metric using LLM-as-judge.
    Scores 1.0 if A↔B is 'Yes', else 0.0, averaged across perturbations.
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

        self.template_path = (
            REPO_ROOT
            / "src"
            / "consistent_agents"
            / "metrics"
            / "prompt-templates"
            / "entailment-judge.txt"
        )
        self.prompt_template = self.template_path.read_text()

    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate entailment score for a single benchmark item."""
        question = item.get("base_output", "")
        if not question:
            return 0.0

        correct_count = 0
        for perturbation in perturbed_outputs:
            prediction = perturbation.get("output", "").strip()
            if not prediction:
                continue

            is_entailed = self._judge_entailment(
                sentence_a=question,
                sentence_b=prediction,
                prompt_template=self.prompt_template
            )

            if is_entailed:
                correct_count += 1

        total_perturbations = len(perturbed_outputs)
        item_score = correct_count / total_perturbations if total_perturbations > 0 else 0.0

        self.total_correct += correct_count
        self.total_comparisons += total_perturbations

        return item_score

    def total_score(self) -> float:
        """Calculate total entailment accuracy across all processed items."""
        total_entailment_score = self.total_correct / self.total_comparisons if self.total_comparisons > 0 else 0.0
        return total_entailment_score

    def name(self) -> str:
        return "entailment"

    def _judge_entailment(self, sentence_a: str, sentence_b: str, prompt_template: str) -> bool:
        """Use LLM to judge if sentence_b is entailed by sentence_a."""
        try:
            prompt = prompt_template.format(sentence_a=sentence_a, sentence_b=sentence_b)
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  ,
                max_tokens=5
            )
            judgment = (response.choices[0].message.content or "").strip().lower()
            return judgment.startswith("yes")
        except Exception as e:
            print(f"Error during entailment judging: {e}")
            return False
