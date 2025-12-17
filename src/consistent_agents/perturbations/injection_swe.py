import random
from typing import Callable, Iterable
import consistent_agents.perturbations.injection as inj_dir

def inject_time_limit(text: str, **kwargs) -> str:
    """Injects a random time limit to the tasks"""
    if not text:
        return text
    
    time_limits = [2, 5, 10, 20, 30]
    gen_pos = ["prefix", "suffix"]
    set_time_limit = int(kwargs.get("set_time_limit", random.choice(time_limits)))
    pos = kwargs.get("position", "")

    if pos not in gen_pos:
        for p in gen_pos:
            if pos in p:
                pos = p
        if not pos:
            pos = random.choice(gen_pos)

    if pos == "suffix":
        return f"You have {set_time_limit} minutes to solve this task.\n" + text
    return text + f"\nYou have {set_time_limit} minutes to solve this task."

def inject_coding_practices(text: str, **kwargs):
    """Injects random instructions for maintaining generic coding practices"""
    if not text:
        return text

    pos_instruction = kwargs.get("position", "prefix")
    code_instructions = [
        "Ensure all function definitions use Python type annotations (PEP 484 / PEP 604 style).",
        "Prioritize optimizing space complexity over time.",
        "Please include proper docstrings and preserve all indentation and line breaks.",
        "Consider at least two possible solutions before attempting to resolve the issue."
    ]

    instr_add = random.choice(code_instructions)
    if pos_instruction == "prefix":
        return f"Read the description below. " + instr_add + "\n" + text
    return text + "\nFollow best practices. " + instr_add

inj_dir.INJECTION_FUNCS = (*inj_dir.INJECTION_FUNCS, 
                   inject_time_limit,
                   inject_coding_practices
                   )

class LLMInjectionSWEPerturbation(inj_dir.LLMInjectionPerturbation):
    def __init__(self, seed: int | None = None, funcs: Iterable[Callable] | None = None, **kwargs):
        super().__init__(seed, funcs, **kwargs)


if __name__ == "__main__":
    ip = LLMInjectionSWEPerturbation(seed=12)
    task = """I need a Python function that parses CSV files and extracts specific columns.
    Technical context:
    - Python 3.10+
    - Using standard library only (no pandas)
    - Will process files up to 1GB in size
    """
    print(ip.apply(task))