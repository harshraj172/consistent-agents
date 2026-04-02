import random
from pathlib import Path
from consistent_agents.perturbations.base import BasePerturbation
from consistent_agents.models.litellm import LitellmModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class LLMBackTranslationPerturbation(BasePerturbation):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        languages: list[str] | None = None,
        seed: int | None = None,
        reasoning_effort: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

        self.languages = languages or ["French", "Spanish", "German", "Hindi", "Japanese", "Arabic"]

        model_kwargs = {}
        if seed is not None:
            model_kwargs["seed"] = seed
        if reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = reasoning_effort
        self.litellm_model = LitellmModel(
            model_name=model,
            model_kwargs=model_kwargs
        )

        tmpl_dir = (
            REPO_ROOT
            / "src"
            / "consistent_agents"
            / "perturbations"
            / "prompt-templates"
        )
        self.en_to_lang_tmpl = (tmpl_dir / "translate-en-to-lang.txt").read_text(encoding="utf-8")
        self.lang_to_en_tmpl = (tmpl_dir / "translate-lang-to-en.txt").read_text(encoding="utf-8")

    def _chat(self, prompt: str) -> str:
        response = self.litellm_model.query(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response["content"].strip()

    def apply(self, text: str, **kwargs) -> str:
        if not text:
            return text

        language = random.choice(self.languages)

        try:
            p1 = self.en_to_lang_tmpl.format(language=language, text=text)
            t_lang = self._chat(p1)

            p2 = self.lang_to_en_tmpl.format(language=language, text=t_lang)
            return self._chat(p2)
        except Exception as e:
            print(f"[translation] Back-translation failed ({type(e).__name__}: {e}), returning original text")
            return text



if __name__ == "__main__":
    bt = LLMBackTranslationPerturbation(model="gpt-4o-mini", temperature=0.3)
    print(bt.apply("Robots should operate safely around humans."))