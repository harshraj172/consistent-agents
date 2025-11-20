from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

__all__ = ["entailment"]


def entailment(
    output1: str,
    output2: str,
    question: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
    prompt_template_path: Optional[Path] = None,
    **kwargs
) -> float:
    """LLM-based agreement function using bidirectional entailment judgment.
    
    Uses an LLM to judge if two outputs mutually entail each other.
    Returns 1.0 if they mutually entail, 0.0 otherwise.
    
    Args:
        output1: First output text (sentence A)
        output2: Second output text (sentence B)
        question: Ignored (not used by entailment judge)
        judge_model: Model to use for judging (default: "gpt-4o-mini")
        prompt_template_path: Optional path to prompt template. If None, uses default.
        **kwargs: Additional context (e.g., prompt_template to override default)
        
    Returns:
        1.0 if mutually entailed, 0.0 otherwise
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    if prompt_template_path is None:
        prompt_template_path = (
            REPO_ROOT
            / "src"
            / "consistent_agents"
            / "benchmarks"
            / "truthfulqa"
            / "prompt-templates"
            / "truthfulqa-entailment-judge.txt"
        )
    
    prompt_template = kwargs.get("prompt_template")
    if prompt_template is None:
        prompt_template = prompt_template_path.read_text()
    
    client = OpenAI()
    
    try:
        prompt = prompt_template.format(sentence_a=output1, sentence_b=output2)
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5
        )
        judgment = (response.choices[0].message.content or "").strip().lower()
        return 1.0 if judgment.startswith("yes") else 0.0
    except Exception as e:
        print(f"Error during entailment judging: {e}")
        return 0.0
