from pathlib import Path
from typing import Dict, Any, List
from consistent_agents.metrics.base import BaseMetric
from consistent_agents.data_models import BenchmarkItem

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class AccuracyMetric(BaseMetric):
    """Accuracy metric that evaluates how accurate model outputs are using LLM-as-judge."""

    def __init__(self, 
                 judge_model: str = "gpt-4o-mini",
                 **kwargs):
        """Initialize the accuracy metric."""
        super().__init__(**kwargs)
        self.judge_model = judge_model
        self.total_correct = 0
        self.total_comparisons = 0
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        self.client = OpenAI()

    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate accuracy score for a single benchmark item."""
        prompt_template = Path(REPO_ROOT / "src" / "consistent_agents" / "metrics" / 
                              "prompt-templates" / "accuracy-judge.txt").read_text()
        
        question = item["question"]
        correct_answers = getattr(item, 'correct_answers', [])
        incorrect_answers = getattr(item, 'incorrect_answers', [])
        
        correct_count = 0
        for perturbation in perturbed_outputs:
            prediction = perturbation.get('output', '')
            if not prediction:
                continue
            
            is_correct = self._judge_accuracy(
                question=question,
                prediction=prediction,
                correct_answers=correct_answers,
                incorrect_answers=incorrect_answers,
                prompt_template=prompt_template
            )
            
            if is_correct:
                correct_count += 1
        
        total_perturbations = len(perturbed_outputs)
        item_score = correct_count / total_perturbations if total_perturbations > 0 else 0.0
        
        self.total_correct += correct_count
        self.total_comparisons += total_perturbations
        
        return item_score

    def total_score(self) -> float:
        """Calculate total score across all processed items."""
        total_accuracy_score = self.total_correct / self.total_comparisons if self.total_comparisons > 0 else 0.0
        return total_accuracy_score

    def name(self) -> str:
        return "accuracy"

    def _judge_accuracy(self, 
                       question: str, 
                       prediction: str,
                       correct_answers: list,
                       incorrect_answers: list,
                       prompt_template: str) -> bool:
        """Use LLM to judge if prediction is accurate."""
        try:
            prompt = prompt_template.format(
                question=question,
                prediction=prediction,
                correct_answers=','.join(correct_answers),
                incorrect_answers=','.join(incorrect_answers)
            )
            
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            judgment = response.choices[0].message.content.strip()
            return judgment.lower().startswith('yes')
            
        except Exception as e:
            print(f"Error during accuracy judging: {e}")
            return False
