import random
from pathlib import Path

from consistent_agents.perturbations.base import BasePerturbation
from consistent_agents.models.litellm import LitellmModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class LLMParaphrasePerturbation(BasePerturbation):
    """Paraphrase text using openai llm"""
    
    def __init__(self,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 seed: int = None,
                 reasoning_effort: str = None,
                 **kwargs):
        """Initialize LLM-based paraphrase perturbation."""
        super().__init__(**kwargs)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

        model_kwargs = {}
        if seed is not None:
            model_kwargs["seed"] = seed
        if reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = reasoning_effort
        self.litellm_model = LitellmModel(
            model_name=model,
            model_kwargs=model_kwargs
        )
    
    def apply(self, text: str, **kwargs) -> str:
        """Apply LLM-based paraphrasing to the text."""
        if not text:
            return text
        
        temperature = kwargs.get('temperature', self.temperature)
        prompt = Path(REPO_ROOT / "src" / "consistent_agents" / "perturbations" / 
                      "prompt-templates" / "paraphrase.txt").read_text()
        prompt = prompt.replace("{sentence}", text)
        
        try:
            response = self.litellm_model.query(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
            
            paraphrased = response["content"].strip()
            return paraphrased
            
        except Exception as e:
            print(f"Error during LLM paraphrasing: {e}")
            return text  


if __name__ == "__main__":
    perturbation = LLMParaphrasePerturbation(
        model="gpt-4o-mini",
        temperature=0.7,
        seed=42
    )
    
    original = "The quick brown fox jumps over the lazy dog."
    perturbed = perturbation.apply(original)
    
    print(f"Original:  {original}")
    print(f"Perturbed: {perturbed}")