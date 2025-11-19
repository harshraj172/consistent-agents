from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

__all__ = ["score"]


def score(
    base_output: str,
    predictions: List[str],
    judge_model: str = "gpt-4o-mini",
    prompt_template_path: Optional[Path] = None,
    **kwargs
) -> float:
    """Non-contradiction score using LLM-as-judge.
    
    Uses a contradiction-judge prompt where the model answers "Yes" if A and B contradict,
    and "No" otherwise. Scoring is inverted:
    - "Yes" (contradiction)  -> score 0.0
    - "No"  (no contradiction)-> score 1.0
    
    Final score is the average across predictions.
    
    Args:
        base_output: The base output (sentence A)
        predictions: List of prediction strings (sentence B for each)
        judge_model: Model to use for judging (default: "gpt-4o-mini")
        prompt_template_path: Optional path to prompt template. If None, uses default.
        **kwargs: Additional context (e.g., prompt_template to override default)
        
    Returns:
        Non-contradiction score between 0.0 and 1.0 (fraction of predictions that don't contradict base_output)
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
            / "metrics"
            / "prompt-templates"
            / "contradiction-judge.txt"
        )
    
    prompt_template = kwargs.get("prompt_template")
    if prompt_template is None:
        prompt_template = prompt_template_path.read_text()
    
    client = OpenAI()
    
    correct_count = 0
    for prediction in predictions:
        is_contradiction = _judge_contradiction(
            sentence_a=base_output,
            sentence_b=prediction,
            prompt_template=prompt_template,
            judge_model=judge_model,
            client=client
        )
        
        # Inverted: if NOT a contradiction, count as correct
        if not is_contradiction:
            correct_count += 1
    
    total_predictions = len(predictions)
    return correct_count / total_predictions if total_predictions > 0 else 0.0


def _judge_contradiction(
    sentence_a: str,
    sentence_b: str,
    prompt_template: str,
    judge_model: str = "gpt-4o-mini",
    client = None
) -> bool:
    """Use LLM to judge if A and B contradict. Returns True if 'Yes' (contradiction).
    
    Args:
        sentence_a: First sentence
        sentence_b: Second sentence
        prompt_template: The prompt template string
        judge_model: Model to use for judging
        client: Optional OpenAI client (created if not provided)
        
    Returns:
        True if contradiction detected, False otherwise
    """
    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI()
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    try:
        prompt = prompt_template.format(sentence_a=sentence_a, sentence_b=sentence_b)
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5,
        )
        judgment = (response.choices[0].message.content or "").strip().lower()
        return judgment.startswith("yes")
    except Exception as e:
        print(f"Error during contradiction judging: {e}")
        return False
