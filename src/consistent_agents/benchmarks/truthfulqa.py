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
        
        for example in self.dataset:
            yield {
                "question": example["question"],
                "prompt": self.format_prompt(example["question"]),
                "correct_answers": example["correct_answers"],
                "incorrect_answers": example["incorrect_answers"]
            }
    
    def score(self, base_output, predictions: List[Any]) -> Dict[str, float]:
        """Calculate scores for TruthfulQA predictions."""
        return {
            "consistent_count": self._score_consistency(base_output, predictions),
            "correct_count": self._score_accuracy(predictions),
            "total": len(predictions)
        }

    def _score_accuracy(self, predictions: List[str]) -> int:
        """Score generation predictions using LLM-as-judge"""
        
        # Load prompt template
        prompt_template = Path(REPO_ROOT / "src" / "consistent_agents" / "benchmarks" / 
                            "prompt-templates" / "truthfulqa-accuracy-judge.txt").read_text()
        correct_count = 0
        
        for (row, pred) in zip(self.dataset, predictions):
            question = row["question"]
            correct_answers = row.get("correct_answers", [])
            incorrect_answers = row.get("incorrect_answers", [])
            prompt = prompt_template.format(
                question=question,
                prediction=pred,
                correct_answers=','.join(correct_answers),
                incorrect_answers=','.join(incorrect_answers)
            )
            is_correct = self._judge_answers(
                prompt=prompt,
            )
            if is_correct:
                correct_count += 1
        
        return correct_count
        
    def _score_consistency(self, 
            base_output: str, 
            predictions: List[Dict[str, List[str]]]) -> Dict[str, float]:
        """Score generation predictions using LLM-as-judge."""
        
        prompt_template = Path(REPO_ROOT / "src" / "consistent_agents" / "benchmarks" / 
                    "prompt-templates" / "truthfulqa-consistency-judge.txt").read_text()
        consistent_count = 0
        for row, pred in zip(self.dataset, predictions):
            prompt = prompt_template.format(
                question=row["question"],
                reference_answer=base_output,
                prediction=pred
            )
            is_same = self._judge_answers(
                prompt=prompt
            )
            if is_same:
                consistent_count += 1
        
        return consistent_count
    
    def _judge_answers(self, prompt: str) -> bool:
        """Use LLM to judge if prediction matches the reference answer"""
        try:
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
            print(f"Error during LLM judging: {e}")
            return False
        
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        if self.dataset is None:
            self.load()
        return len(self.dataset)