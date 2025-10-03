import json
import os
from typing import Iterator, Dict, Any, List, Optional
import datasets
from consistent_agents.benchmarks.base import BaseBenchmark


class HellaSwagBenchmark(BaseBenchmark):
    """
    HellaSwag is a challenge dataset for commonsense natural language inference.
    It consists of multiple-choice questions where models must select the most
    plausible continuation of a given context.
    """
    
    def __init__(self, data_dir: Optional[str] = None, split: str = "validation"):
        """Initialize HellaSwag benchmark."""
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.dataset = None
        
    def load(self) -> None:
        """
        Load the HellaSwag dataset from HuggingFace datasets.
        """
        try:
            self.dataset = datasets.load_dataset("hellaswag", split=self.split, cache_dir=self.data_dir)
            print(f"Loaded HellaSwag {self.split} split with {len(self.dataset)} examples")
        except Exception as e:
            raise RuntimeError(f"Failed to load HellaSwag dataset: {e}")
        
    def format_prompt(self, context: str, choices: List[str]) -> str:
        """Format the prompt for language model evaluation."""
        prompt = f"{context}\n\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "\nAnswer:"
        prompt += "\n\n Answer this by printing the correct sentence in the format 'The answer is '<sentence>''."
        return prompt
    
    def iter(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset examples."""
        if self.dataset is None:
            self.load()
            
        for example in self.dataset:
            yield {
                "prompt": self.format_prompt(example["ctx"], example["endings"]),
                "label": example["label"]
            }
    
    def score(self, predictions: List[int], references: List[int]) -> Dict[str, float]:
        """Calculate accuracy score for HellaSwag predictions."""
        if len(predictions) != len(references):
            raise ValueError(f"Predictions length ({len(predictions)}) != references length ({len(references)})")
        
        # Calculate accuracy
        correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        accuracy = correct / len(predictions)
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(predictions)
        }
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        if self.dataset is None:
            self.load()
        return len(self.dataset)