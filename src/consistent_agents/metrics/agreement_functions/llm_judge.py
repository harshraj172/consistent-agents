from pathlib import Path
from typing import Optional
from consistent_agents.metrics.agreement_functions import AgreementFunction

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

__all__ = ["LLMJudgeAgreement"]


class LLMJudgeAgreement:
    """LLM-based agreement function using consistency judgment.
    
    Uses an LLM to judge if two outputs are consistent with each other.
    Returns 1.0 if consistent, 0.0 if not consistent.
    """
    
    def __init__(self, 
                 judge_model: str = "gpt-4o-mini",
                 prompt_template_path: Optional[Path] = None):
        """Initialize the LLM judge agreement function.
        
        Args:
            judge_model: Model to use for judging (e.g., "gpt-4o-mini")
            prompt_template_path: Optional path to prompt template. If None, uses default.
        """
        self.judge_model = judge_model
        
        if prompt_template_path is None:
            prompt_template_path = (
                REPO_ROOT / "src" / "consistent_agents" / "metrics" / 
                "prompt-templates" / "consistency-judge.txt"
            )
        
        self.prompt_template = prompt_template_path.read_text()
        
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
    
    def __call__(self, output1: str, output2: str, question: Optional[str] = None, **kwargs) -> float:
        """Judge if two outputs are consistent using LLM.
        
        Args:
            output1: First output text (used as reference_answer)
            output2: Second output text (used as prediction)
            question: Question context (required for LLM judge)
            **kwargs: Additional context (e.g., prompt_template to override default)
            
        Returns:
            1.0 if consistent, 0.0 if not consistent
        """
        prompt_template = kwargs.get("prompt_template", self.prompt_template)
        
        if question is None:
            question = ""  # Some prompts might work without question
        
        try:
            prompt = prompt_template.format(
                question=question,
                reference_answer=output1,
                prediction=output2
            )
            
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
            return 1.0 if judgment.lower().startswith('yes') else 0.0
            
        except Exception as e:
            print(f"Error during LLM consistency judging: {e}")
            return 0.0

