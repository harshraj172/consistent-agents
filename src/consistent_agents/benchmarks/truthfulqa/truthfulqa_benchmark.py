from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional

import datasets
from consistent_agents.benchmarks.base import BaseBenchmark
from consistent_agents.environments import DockerEnvironment
from consistent_agents.metrics.accuracy import score as accuracy_score
from consistent_agents.metrics.consistency import score as consistency_score
from consistent_agents.metrics.entailment import score as entailment_score
from consistent_agents.metrics.contradiction import score as contradiction_score


class TruthfulQABenchmark(BaseBenchmark):
    """
    TruthfulQA is a benchmark to measure whether a language model is truthful in
    generating answers to questions. The benchmark includes questions that some
    humans would answer falsely due to misconceptions or false beliefs.
    
    TruthfulQA has two tasks:
    - Generation: Open-ended generation evaluated for truthfulness and informativeness
    - Multiple-choice: Select the correct answer from multiple options
    
    But we support "generation" currently
    """
    
    def __init__(self, 
                 data_dir: Optional[str] = None, 
                 split: str = "validation",
                 task: str = "generation",
                 judge_model: str = "gpt-4o-mini"):
        """Initialize TruthfulQA benchmark. """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.task = task
        self.dataset = None
        self._prepared: Dict[int, Dict[str, Any]] = {}
        self.template_dir = Path(__file__).parent / "prompt-templates"
        
        if task not in ["generation"]:
            raise ValueError(f"Task must be 'generation', got {task}")
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        self.judge_model = judge_model
        self.client = OpenAI()

        self.item_scores: Dict[int, Dict[str, float]] = {}
        self.total_scores: Dict[str, int] = {
            "consistent_count": 0,
            "accuracy_count": 0,
            "entailment_count": 0,
            "contradiction_count": 0,
            "total_pairs": 0,
            "total": 0,
        }
        
    def load(self) -> None:
        """
        Load the TruthfulQA dataset from HuggingFace datasets.
        """
        try:
            self.dataset = datasets.load_dataset(
                "truthful_qa", 
                self.task,
                split=self.split, 
                cache_dir=self.data_dir
            )
            print(f"Loaded TruthfulQA ({self.task}) {self.split} split with {len(self.dataset)} examples")
        except Exception as e:
            raise RuntimeError(f"Failed to load TruthfulQA dataset: {e}")
    
    def format_prompt(self, question: str) -> str:
        """Format the prompt for generation evaluation."""
        prompt = f"Question: {question}\nAnswer:"
        return prompt
    
    def _prepare_instance(self, idx: int, example: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare environment for a TruthfulQA example."""
        if idx in self._prepared:
            return self._prepared[idx]

        env = DockerEnvironment()
        env.start()
        state = {"env": env}
        self._prepared[idx] = state
        return state
    
    def iter(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset examples."""
        if self.dataset is None:
            self.load()
        for idx, example in enumerate(self.dataset):
            state = self._prepare_instance(idx, example)
            yield {
                "question": example["question"],
                "prompt": self.format_prompt(example["question"]),
                "correct_answers": example["correct_answers"],
                "incorrect_answers": example["incorrect_answers"],
                "env": state["env"],
            }
    
    def score(self, idx: int, base_output: str, predictions: List[str]) -> Dict[str, float]:
        """Calculate scores for TruthfulQA predictions."""
        row = self.dataset[idx]
        question = row["question"]
        correct_answers = row.get("correct_answers", [])
        incorrect_answers = row.get("incorrect_answers", [])
        
        total_pairs = len(predictions) * (len(predictions) - 1) // 2
        total_predictions = len(predictions)
        
        # Calculate consistency score (pairwise)
        consistency_proportion = consistency_score(
            outputs=predictions,
            question=question,
            agreement="llm_judge",
            agreement_params={"judge_model": self.judge_model},
            aggregator="pairwise"
        )
        consistent_count = int(consistency_proportion * total_pairs)
        
        # Calculate accuracy score
        accuracy_proportion = accuracy_score(
            question=question,
            predictions=predictions,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers,
            judge_model=self.judge_model
        )
        accuracy_count = int(accuracy_proportion * total_predictions)
        
        # Calculate entailment score
        entailment_proportion = entailment_score(
            base_output=base_output,
            predictions=predictions,
            judge_model=self.judge_model
        )
        entailment_count = int(entailment_proportion * total_predictions)
        
        # Calculate contradiction score (non-contradiction)
        contradiction_proportion = contradiction_score(
            base_output=base_output,
            predictions=predictions,
            judge_model=self.judge_model
        )
        contradiction_count = int(contradiction_proportion * total_predictions)
        
        self.item_scores = {
            "consistent_count": consistent_count,
            "accuracy_count": accuracy_count,
            "entailment_count": entailment_count,
            "contradiction_count": contradiction_count,
            "total_pairs": total_pairs,
            "total_predictions": total_predictions,
        }
        self.total_scores["consistent_count"] += consistent_count
        self.total_scores["accuracy_count"] += accuracy_count
        self.total_scores["entailment_count"] += entailment_count
        self.total_scores["contradiction_count"] += contradiction_count
        self.total_scores["total_pairs"] += total_pairs
        self.total_scores["total"] += total_predictions

    def item_score(self) -> Dict[str, float]:
        """Get the scores for a specific item."""
        total_pairs = self.item_scores.get("total_pairs", 1)
        total_predictions = self.item_scores.get("total_predictions", 1)
        
        return {
            "consistency": self.item_scores["consistent_count"] / total_pairs if total_pairs > 0 else 0.0,
            "accuracy": self.item_scores["accuracy_count"] / total_predictions if total_predictions > 0 else 0.0,
            "entailment": self.item_scores["entailment_count"] / total_predictions if total_predictions > 0 else 0.0,
            "contradiction": self.item_scores["contradiction_count"] / total_predictions if total_predictions > 0 else 0.0,
        }
    
    def total_score(self) -> Dict[str, float]:
        """Get the total scores for the benchmark."""
        total_pairs = self.total_scores.get("total_pairs", 1)
        total = self.total_scores["total"]
        
        return {
            "consistency": self.total_scores["consistent_count"] / total_pairs if total_pairs > 0 else 0.0,
            "accuracy": self.total_scores["accuracy_count"] / total if total > 0 else 0.0,
            "entailment": self.total_scores["entailment_count"] / total if total > 0 else 0.0,
            "contradiction": self.total_scores["contradiction_count"] / total if total > 0 else 0.0,
            "total": total
        }
        
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        if self.dataset is None:
            self.load()
        return len(self.dataset)