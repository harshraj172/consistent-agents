from pathlib import Path
from typing import Dict, Any, List
from consistent_agents.metrics.base import BaseMetric
from consistent_agents.data_models import BenchmarkItem

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class ConsistencyMetric(BaseMetric):
    """Consistency metric that evaluates how consistent model outputs are across different perturbations."""

    def __init__(self, 
                 judge_model: str = "gpt-4o-mini",
                 **kwargs):
        """Initialize the consistency metric."""
        super().__init__(**kwargs)
        self.judge_model = judge_model
        self.total_consistent = 0
        self.total_comparisons = 0
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        self.client = OpenAI()

    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate consistency score for a single benchmark item."""
        prompt_template = Path(REPO_ROOT / "src" / "consistent_agents" / "metrics" / 
                              "prompt-templates" / "consistency-judge.txt").read_text()
        
        reference_answer = item["base_output"]
        question = item["question"]
        
        consistent_count = 0
        for perturbation in perturbed_outputs:
            prediction = perturbation['output']
            
            is_consistent = self._judge_consistency(
                question=question,
                reference_answer=reference_answer,
                prediction=prediction,
                prompt_template=prompt_template
            )
            if is_consistent:
                consistent_count += 1
        
        total_perturbations = len(perturbed_outputs)
        item_score = consistent_count / total_perturbations if total_perturbations > 0 else 0.0
        
        self.total_consistent += consistent_count
        self.total_comparisons += total_perturbations
        
        return item_score

    def total_score(self) -> float:
        """Calculate total score across all processed items."""
        total_consistency_score = self.total_consistent / self.total_comparisons if self.total_comparisons > 0 else 0.0
        return total_consistency_score

    def name(self) -> str:
        return "consistency"

    def _judge_consistency(self, 
                          question: str, 
                          reference_answer: str, 
                          prediction: str,
                          prompt_template: str) -> bool:
        """Use LLM to judge if prediction is consistent with reference answer."""
        try:
            prompt = prompt_template.format(
                question=question,
                reference_answer=reference_answer,
                prediction=prediction
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
            print(f"Error during consistency judging: {e}")
            return False
