from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BaseModelConfig:
    """Configuration for base model."""
    model_name: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    def __init__(self, **kwargs):
        self.config = self._create_config(**kwargs)
        self.n_calls = 0
        self.cost = 0.0
    
    @abstractmethod
    def _create_config(self, **kwargs) -> BaseModelConfig:
        """Create and return model-specific configuration."""
        pass
    
    @abstractmethod
    def query(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Query the model with messages"""
        pass
    
    def get_template_vars(self) -> Dict[str, Any]:
        """Return variables for templates/logging."""
        return {
            "model_name": self.config.model_name,
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }