import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from consistent_agents.models import GLOBAL_MODEL_STATS
from consistent_agents.models.base import BaseModel, BaseModelConfig


logger = logging.getLogger("litellm_model")


@dataclass
class LitellmModelConfig(BaseModelConfig):
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")


class LitellmModel(BaseModel):
    
    def _create_config(self, **kwargs) -> LitellmModelConfig:
        return LitellmModelConfig(**kwargs)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))
    
    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: List[Dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name, 
                messages=messages, 
                **(self.config.model_kwargs | kwargs)
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
    
    def query(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        response = self._query(messages, **kwargs)
        
        try:
            cost = litellm.cost_calculator.completion_cost(response)
        except Exception as e:
            logger.critical(
                f"Error calculating cost for model {self.config.model_name}: {e}. "
                "Please check the 'Updating the model registry' section in the documentation at "
                "https://klieret.short.gy/litellm-model-registry Still stuck? Please open a github issue for help!"
            )
            raise
        
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        
        return {
            "content": response.choices[0].message.content or "",
            "extra": {
                "response": response.model_dump(),
            },
        }
    
    def get_template_vars(self) -> Dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}