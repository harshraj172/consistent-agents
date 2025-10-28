from pathlib import Path
from typing import Dict, Any, List
from consistent_agents.metrics.base import BaseMetric
from consistent_agents.data_models import BenchmarkItem
import numpy as np
from scipy.stats import entropy

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class EntropyMetric(BaseMetric):
    """Entropy metric that evaluates diversity of model outputs across different perturbations."""


    def __init__(self, 
                 judge_model: str = "gpt-4o-mini",
                 **kwargs):
        """Initialize the entropy metric."""
        super().__init__(**kwargs)
        self.judge_model = judge_model
        self.scores: List[float] = []  # Store individual entropy scores
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        self.client = OpenAI()

    def item_score(self, item: BenchmarkItem, perturbed_outputs: List[Dict[str, Any]]) -> float:
        """Calculate entropy score for a single benchmark item."""
        prompt_template = Path(REPO_ROOT / "src" / "consistent_agents" / "metrics" / 
                              "prompt-templates" / "consistency-judge.txt").read_text()
        
        question = item["question"]
        outputs = [perturbation['output'] for perturbation in perturbed_outputs]
        
        clusters = self._semantic_clustering(outputs, question, prompt_template)
        
        cluster_sizes = np.array([len(c) for c in clusters])
        pk = cluster_sizes / cluster_sizes.sum()
        H = entropy(pk, base=2)
        
        self.scores.append(H)
        
        return H

    def total_score(self) -> float:
        """Calculate mean entropy score across all processed items."""
        if not self.scores:
            return 0.0
        
        mean_entropy = sum(self.scores) / len(self.scores)
        return mean_entropy

    def name(self) -> str:
        return "entropy_consistency"

    def _semantic_clustering(self, outputs: List[str], question: str, prompt_template: str) -> List[List[str]]:
        """
        Organizes similar outputs into clusters based on consistency.

        Parameters:
            outputs (List[str]): A list of outputs to be clustered.
            question (str): The question context for consistency evaluation.
            prompt_template (str): The prompt template for consistency judging.

        Returns:
            List[List[str]]: A list of clusters, where each cluster is a list of consistent outputs.
        """
        if not outputs:
            return []
        
        C = [[outputs[0]]]
        for i in range(1, len(outputs)):
            stored = False
            for j in range(len(C)):
                s_c = C[j][0]

                is_consistent = self._judge_consistency(
                    question=question,
                    reference_answer=s_c,
                    prediction=outputs[i],
                    prompt_template=prompt_template
                )
                if is_consistent:
                    stored = True
                    C[j].append(outputs[i])
                    break
            if not stored:
                C.append([outputs[i]])
        
        return C

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
