from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

__all__ = ["llm_as_judge"]


def llm_as_judge(
    output1: str,
    output2: str,
    question: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
    prompt_template_path: Optional[Path] = None,
    **kwargs
) -> float:
    """LLM-based agreement function using consistency judgment.
    
    Uses an LLM to judge if two outputs are consistent with each other.
    Returns 1.0 if consistent, 0.0 if not consistent.
    
    Args:
        output1: First output text (used as reference_answer)
        output2: Second output text (used as prediction)
        question: Question context (required for LLM judge)
        judge_model: Model to use for judging (e.g., "gpt-4o-mini")
        prompt_template_path: Optional path to prompt template. If None, uses default.
        **kwargs: Additional context (e.g., prompt_template to override default)
        
    Returns:
        1.0 if consistent, 0.0 if not consistent
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI package not installed. Install with: pip install openai"
        )
    
    if prompt_template_path is None:
        raise ValueError(
            "prompt_template_path must be provided. "
            "Each benchmark may require a different prompt template for the judge."
        )
    
    prompt_template = prompt_template_path.read_text()
    
    if question is None:
        question = ""  # Some prompts might work without question
    
    client = OpenAI()
    
    try:
        prompt = prompt_template.format(
            question=question,
            prediction1=output1,
            prediction2=output2
        )
        
        response = client.chat.completions.create(
            model=judge_model,
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
        return 1.0 if judgment.lower().startswith('yes') else 0.0
        
    except Exception as e:
        print(f"Error during LLM consistency judging: {e}")
        return 0.0
