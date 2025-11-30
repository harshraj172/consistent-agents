from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional

import datasets
from consistent_agents.benchmarks.base import BaseBenchmark
from consistent_agents.environments import DockerEnvironment
from consistent_agents.metrics.accuracy import score as accuracy_score
from consistent_agents.metrics.consistency import score as consistency_score


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
        
        from openai import OpenAI
        self.judge_model = judge_model
        self.client = OpenAI()

        self.item_scores: Dict[str, Any] = {}

        self.total_scores = {
            "accuracy": None,
            "consistency": None,
            "total":0
        }

        self.consistency_configs = [
            ("consistency", "pairwise"),
            ("contradiction", "pairwise"),
            ("entailment", "pairwise"),
        ]

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
        except Exception as e:
            raise RuntimeError(f"Failed to load TruthfulQA dataset: {e}")
    
    def format_prompt(self, question: str) -> str:
        """Format the prompt for generation evaluation."""
        return f"Question: {question}\nAnswer:"
    
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
        
        total_predictions = len(predictions)
        
        consistency_results = {}
        for agreement, aggregator in self.consistency_configs:
            params = {"judge_model": self.judge_model} if agreement in ["consistency", "contradiction", "entailment"] else {}
            proportion = consistency_score(
                outputs=predictions,
                question=question,
                agreement=agreement,
                agreement_params=params,
                aggregator=aggregator
            )
            if aggregator == "pairwise":
                score, total_pairs = proportion
            else:
                score, total_pairs = proportion, 1
            key = f"{agreement}_{aggregator}"
            consistency_results[key] = {"score": score, "total_pairs": total_pairs}
        
        accuracy_count, accuracy_pairs = accuracy_score(
            question=question,
            predictions=predictions,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers,
            judge_model=self.judge_model
        )
        
        self.item_scores = {
            "accuracy": {"accuracy_count": accuracy_count, "accuracy_pairs": accuracy_pairs},
            "consistency":consistency_results,
            "total": total_predictions,
        }
        self._update_totals()

    
    def item_score(self) -> Dict[str, Any]:
        """Get the scores for a specific item."""

        accuracy = self.item_scores["accuracy"]["accuracy_count"] / self.item_scores["accuracy"]["accuracy_pairs"]
        consistency = []
        for key, result in self.item_scores["consistency"].items():
            agreement, aggregator = key.split("_", 1)  # Split "consistency_pairwise" into ("consistency", "pairwise")
            score = result["score"] / result["total_pairs"]
            consistency.append({
                "aggregator": aggregator,
                "agreement_function": agreement,
                "score": score
            })
        return {
            "accuracy": accuracy,
            "consistency": consistency
        }
    
    def total_score(self) -> Dict[str, Any]:
        """Get the total scores for the benchmark."""
        accuracy = self.total_scores["accuracy"]["accuracy_count"] / self.total_scores["accuracy"]["accuracy_pairs"]
        consistency = []
        for key, result in self.total_scores["consistency"].items():
            agreement, aggregator = key.split("_", 1)  # Split "consistency_pairwise" into ("consistency", "pairwise")
            score = result["score"] / result["total_pairs"]
            consistency.append({
                "aggregator": aggregator,
                "agreement_function": agreement,
                "score": score
            })
        return {
            "accuracy": accuracy,
            "consistency": consistency,
            "total": self.total_scores["total"]
        }

    def _update_totals(self):
        if self.total_scores["accuracy"] is None:
            self.total_scores["accuracy"] = self.item_scores["accuracy"]
        else:
            self.total_scores["accuracy"]["accuracy_pairs"] += self.item_scores["accuracy"]["accuracy_pairs"]
            self.total_scores["accuracy"]["accuracy_count"] += self.item_scores["accuracy"]["accuracy_count"]
        
        if self.total_scores["consistency"] is None:
            self.total_scores["consistency"] = self.item_scores["consistency"]
        else:
            for key, result in self.item_scores["consistency"].items():
                self.total_scores["consistency"][key]["score"] += result["score"]
                self.total_scores["consistency"][key]["total_pairs"] += result["total_pairs"]
        
        self.total_scores["total"] += self.item_scores["total"]
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        if self.dataset is None:
            self.load()
        return len(self.dataset)
