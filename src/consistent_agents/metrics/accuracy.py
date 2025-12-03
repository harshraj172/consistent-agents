from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def score(
    question: str,
    predictions: List[str],
    correct_answers: List[str],
    incorrect_answers: List[str],
    judge_model: str = "gpt-4o-mini",
    prompt_template_path: Optional[Path] = None,

) -> float:
    """Calculate accuracy score for predictions using LLM-as-judge.
    
    Args:
        question: The question being answered
        predictions: List of prediction strings to evaluate
        correct_answers: List of correct answer strings
        incorrect_answers: List of incorrect answer strings
        judge_model: Model to use for judging (default: "gpt-4o-mini")
    
    Returns:
        Accuracy score between 0.0 and 1.0 (fraction of correct predictions)
    """
    if prompt_template_path is None:
        raise ValueError(
            "prompt_template_path must be provided. "
            "Each benchmark may require a different prompt template for the judge."
        )
    
    prompt_template = prompt_template_path.read_text()
        
    correct_count = 0
    for prediction in predictions:
        is_correct = _judge_accuracy(
            question=question,
            prediction=prediction,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers,
            prompt_template=prompt_template,
            judge_model=judge_model
        )
        
        if is_correct:
            correct_count += 1
    
    return correct_count


def _judge_accuracy(question: str, prediction: str, correct_answers: List[str], incorrect_answers: List[str], prompt_template: str, judge_model: str = "gpt-4o-mini") -> bool:
    """Use LLM to judge if prediction is accurate.
    
    Args:
        question: The question being answered
        prediction: The prediction to judge
        correct_answers: List of correct answer strings
        incorrect_answers: List of incorrect answer strings
        prompt_template: The prompt template string
        judge_model: Model to use for judging
    
    Returns:
        True if prediction is judged as accurate, False otherwise
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm package not installed. Install with: pip install litellm"
        )
    
    try:
        prompt = prompt_template.format(
            question=question,
            prediction=prediction,
            correct_answers=','.join(correct_answers),
            incorrect_answers=','.join(incorrect_answers)
        )
        
        response = litellm.completion(
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
        
        judgment = (response.choices[0].message.content or "").strip()
        return judgment.lower().startswith('yes')
        
    except Exception as e:
        print(f"Error during accuracy judging: {e}")
        return False
