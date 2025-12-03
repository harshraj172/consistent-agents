from pathlib import Path
from typing import Optional
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

__all__ = ["contradiction", "contradiction_bert", "contradiction_openai"]

from consistent_agents.models.hfmodel import get_huggingface_model


def contradiction(
    output1: str,
    output2: str,
    question: Optional[str] = None,
    judge_model: str = "microsoft/deberta-base-mnli",
    prompt_template_path: Optional[Path] = None,
    **kwargs
) -> float:
    """
    Wrapper function for contradiction that routes to BERT or OpenAI based on judge_model.
    If judge_model is microsoft/deberta-base-mnli (default BERT option), uses contradiction_bert.
    If judge_model is gpt-5 or other OpenAI models, uses contradiction_openai.
    
    Args:
        output1: First output text (sentence A)
        output2: Second output text (sentence B)
        question: Ignored (not used by contradiction judge)
        judge_model: Model to use for judging (default: "microsoft/deberta-base-mnli")
        prompt_template_path: Optional path to prompt template. If None, uses default.
        **kwargs: Additional context (e.g., prompt_template to override default)
        
    Returns:
        0.0 if they contradict, 1.0 otherwise
    """

    if judge_model == "microsoft/deberta-base-mnli":
        return contradiction_bert(
            output1=output1,
            output2=output2,
            question=question,
            judge_model=judge_model,
            **kwargs
        )
    else:
        return contradiction_openai(
            output1=output1,
            output2=output2,
            question=question,
            judge_model=judge_model,
            prompt_template_path=prompt_template_path,
            **kwargs
        )


def contradiction_bert(
    output1: str,
    output2: str,
    question: Optional[str] = None,
    judge_model: str = "microsoft/deberta-base-mnli",
    **kwargs
) -> float:
    """
    BERT-based contradiction function using MNLI.
    Returns 0.0 if they contradict, 1.0 otherwise.
    """
    try:    
        nli = get_huggingface_model(judge_model, model_type="sequence_classification")
        inputs = nli.tokenizer(
            output1, output2, return_tensors="pt", padding=True
        ).to(nli.device)
        with torch.no_grad():
            outputs = nli.model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[0].item()
    except Exception as e:
        print(f"Error during BERT contradiction judging: {e}")
        return 0.0


def contradiction_openai(
    output1: str,
    output2: str,
    question: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
    prompt_template_path: Optional[Path] = None,
    **kwargs
) -> float:
    """LLM-based agreement function using contradiction judgment.
    
    Uses an LLM to judge if two outputs contradict each other.
    Returns 0.0 if they contradict, 1.0 otherwise.
    """
    try:
        import litellm
    except ImportError:
        raise ImportError("litellm package not installed. Install with: pip install litellm")
    
    if prompt_template_path is None:
        raise ValueError(
            "prompt_template_path must be provided. "
            "Each benchmark may require a different prompt template for the judge."
        )
    
    prompt_template = prompt_template_path.read_text()
    
    try:
        prompt = prompt_template.format(sentence_a=output1, sentence_b=output2)
        response = litellm.completion(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5
        )
        judgment = (response.choices[0].message.content or "").strip().lower()
        return 0.0 if judgment.startswith("yes") else 1.0
    except Exception as e:
        print(f"Error during litellm contradiction judging: {e}")
        return 0.0
