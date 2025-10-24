from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional

import datasets
from consistent_agents.benchmarks.base import BaseBenchmark

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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
    
    def iter(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset examples."""
        if self.dataset is None:
            self.load()
        for idx, example in enumerate(self.dataset):
            yield {
                "id": idx,
                "question": example["question"],
                "prompt": self.format_prompt(example["question"]),
                "correct_answers": example["correct_answers"],
                "incorrect_answers": example["incorrect_answers"]
            }
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        if self.dataset is None:
            self.load()
        return len(self.dataset)