import re
import random
from datetime import datetime
from typing import Callable, Iterable
import consistent_agents.perturbations.injection as inj_dir
import re
import random
from datetime import datetime
from typing import Callable, Iterable, List, Optional
from consistent_agents.perturbations.base import BasePerturbation


def inject_time_limit(text: str, rng: random.Random, **kwargs) -> str:
    """Injects a random time limit to the task description."""
    if not text:
        return text
    
    time_limits = [2, 5, 10, 20, 30]
    gen_pos = ["prefix", "suffix"]
    
    set_time_limit = kwargs.get("set_time_limit")
    if set_time_limit is None:
        set_time_limit = rng.choice(time_limits)
    else:
        set_time_limit = int(set_time_limit)
    
    pos = kwargs.get("position", "")
    if pos not in gen_pos:
        # Try to match partial position name
        for p in gen_pos:
            if pos in p:
                pos = p
                break
        if pos not in gen_pos:
            pos = rng.choice(gen_pos)

    time_instruction = f"You have {set_time_limit} minutes to solve this task."
    
    if pos == "prefix":
        return f"{time_instruction}\n\n{text}"
    return f"{text}\n\n{time_instruction}"


def inject_coding_practices(text: str, rng: random.Random, **kwargs) -> str:
    """Injects random instructions for maintaining generic coding practices."""
    if not text:
        return text

    pos_instruction = kwargs.get("position", "prefix")
    
    code_instructions = [
        "Ensure all function definitions use Python type annotations (PEP 484 / PEP 604 style).",
        "Prioritize optimizing space complexity over time.",
        "Please include proper docstrings and preserve all indentation and line breaks.",
        "Consider at least two possible solutions before attempting to resolve the issue.",
        "Add inline comments explaining your reasoning for non-trivial changes.",
        "Ensure backward compatibility with Python 3.8+.",
    ]

    instr_add = rng.choice(code_instructions)
    
    if pos_instruction == "prefix":
        return f"Read the description below. {instr_add}\n\n{text}"
    return f"{text}\n\nFollow best practices. {instr_add}"


def inject_irrelevant_texts(text: str, rng: random.Random, **kwargs) -> str:
    """Injects an irrelevant emotional, factual, or question statement."""
    if not text:
        return text
    
    noise_select = kwargs.get("noise_type")
    if noise_select is None:
        noise_select = rng.choice(['emotion', 'fact', 'question'])

    # Split into sentences more robustly
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        sentences = [text]
    
    emotions = ['happy', 'angry', 'lonely', 'exhausted', 'stoic', 'ecstatic', 'determined', 'curious', 'focused']
    today = datetime.today()
    
    pos = rng.randint(0, len(sentences))

    if noise_select == "emotion":
        insert_noise = f"I am feeling very {rng.choice(emotions)} today."
    elif noise_select == "fact":
        insert_noise = f"Today is {today.strftime('%A, %B %d, %Y')}."
    elif noise_select == "question":
        insert_noise = "How are you doing today?"
    else:
        insert_noise = "Carpe Diem!"

    sentences.insert(pos, insert_noise)
    return " ".join(sentences)


# Default injection functions for this SWE-specific perturbation
SWE_INJECTION_FUNCS = (
    inject_time_limit,
    inject_coding_practices,
    inject_irrelevant_texts,
)


class LLMInjectionSWEPerturbation(BasePerturbation):
    """
    SWEBench-specific injection perturbation.
    """
    
    def __init__(
        self, 
        seed: Optional[int] = None, 
        funcs: Optional[Iterable[Callable]] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name or "injection_swe", **kwargs)
        
        self.seed = seed
        self.rng = random.Random(seed if seed is not None else 42)
        
        # Use provided functions or defaults
        if funcs is not None:
            self.funcs: List[Callable] = list(funcs)
        else:
            self.funcs = list(SWE_INJECTION_FUNCS)
        
        self.last_applied_func: Optional[str] = None
    
    def apply(self, text: str, **kwargs) -> str:
        """
        Apply a randomly selected injection to the text.
        """
        if not text or not self.funcs:
            return text
        
        func = self.rng.choice(self.funcs)
        self.last_applied_func = func.__name__
        
        try:
            return func(text, self.rng, **kwargs)
        except Exception as e:
            self.logger.warning(f"Injection {func.__name__} failed: {e}")
            return text
    
    def apply_multiple(self, text: str, n: int = 1, **kwargs) -> str:
        """
        Apply multiple injections sequentially.
        """
        result = text
        for _ in range(n):
            result = self.apply(result, **kwargs)
        return result
    
    def get_config(self) -> dict:
        """Get perturbation configuration."""
        config = super().get_config()
        config.update({
            "seed": self.seed,
            "funcs": [f.__name__ for f in self.funcs],
            "last_applied": self.last_applied_func,
        })
        return config




if __name__ == "__main__":
    # Test the perturbations
    task = """I need a Python function that parses CSV files and extracts specific columns.
    
Technical context:
- Python 3.10+
- Using standard library only (no pandas)
- Will process files up to 1GB in size

The function should handle edge cases like quoted fields and escape characters."""

    print("--- Original Task ---")
    print(task)
    print()
    
    ip = LLMInjectionSWEPerturbation(seed=12)
    
    print("--- With Random Injection ---")
    result = ip.apply(task)
    print(result)
    print(f"\nApplied function: {ip.last_applied_func}")
    print()
    
    print("--- With 3 Injections ---")
    ip2 = LLMInjectionSWEPerturbation(seed=42)
    result2 = ip2.apply_multiple(task, n=3)
    print(result2)
    print()
    
